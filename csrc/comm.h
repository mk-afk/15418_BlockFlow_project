// comm.h - process layout and gradient averaging for hybrid block/data parallelism.
//
// P ranks form G = P / D groups of D replicas. Group g trains every block b with b % G == g
// (G == B: one block per group; G == 1: every block on every rank, one after another).
// Replicas of a block average their gradients with one all-reduce over the block's flat gradient
// buffer: NCCL when built with USE_NCCL, otherwise MPI through host memory, which also works between
// machines. Different groups never communicate.
//
// Each rank uses one GPU. The ranks on a machine take its GPUs in turn, so a multi-GPU run is one rank
// per GPU on one node, or one rank on each of several one-GPU machines (mpirun --host a,b). With fewer
// GPUs than ranks the ranks share a GPU, which is enough to test the parallel code but not to time it.
#pragma once
#include "common.h"
#ifdef USE_MPI
#include <mpi.h>
#include <unistd.h>
#endif
#ifdef USE_NCCL
#include <nccl.h>
#define ncclCheck(call)                                                                                         \
    do {                                                                                                        \
        ncclResult_t r_ = (call);                                                                               \
        if (r_ != ncclSuccess) {                                                                                \
            fprintf(stderr, "[NCCL ERROR] %s at %s:%d: %s\n", #call, __FILE__, __LINE__, ncclGetErrorString(r_)); \
            exit(EXIT_FAILURE);                                                                                 \
        }                                                                                                       \
    } while (0)
#endif
#if defined(USE_NCCL)
#define COMM_BACKEND "nccl"
#elif defined(USE_MPI)
#define COMM_BACKEND "mpi"
#else
#define COMM_BACKEND "none"
#endif

#define MAX_BLOCKS 64

typedef struct {
    int world_rank, world_size;
    int device, hosts, gpus;  // this rank's GPU; machines and distinct GPUs used by all ranks
    int B, D, G, group, replica;
    int nowned, owned[MAX_BLOCKS];
#ifdef USE_MPI
    MPI_Comm group_comm;
#endif
#ifdef USE_NCCL
    ncclComm_t nccl;
    int have_nccl;
#endif
    float* stage;  // host staging buffer for the MPI all-reduce on GPU builds
    size_t stage_n;
} Comm;

// Starts MPI and selects this rank's GPU, before anything creates a CUDA context or an NCCL communicator.
// Also counts the machines and the distinct GPUs (machine and PCI address) that the ranks run on.
static void comm_init(Comm* c, int* argc, char*** argv) {
    memset(c, 0, sizeof(*c));
    c->world_size = c->hosts = c->gpus = 1;
    int local_rank = 0, ndev = 1;
#ifdef USE_MPI
    MPI_Init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &c->world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &c->world_size);
    MPI_Comm node;  // the ranks on this machine
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, c->world_rank, MPI_INFO_NULL, &node);
    MPI_Comm_rank(node, &local_rank);
    MPI_Comm_free(&node);
#else
    (void)argc;
    (void)argv;
#endif
    cudaCheck(cudaGetDeviceCount(&ndev));
    c->device = local_rank % ndev;
    cudaCheck(cudaSetDevice(c->device));
#ifdef USE_MPI
    cudaDeviceProp prop;
    cudaCheck(cudaGetDeviceProperties(&prop, c->device));
    char host[256] = "";
    gethostname(host, sizeof(host) - 1);
    unsigned long long me[2] = {0, ((unsigned long long)prop.pciDomainID << 16) | (prop.pciBusID << 8) | prop.pciDeviceID};
    for (const char* ch = host; *ch; ch++) me[0] = me[0] * 131 + (unsigned char)*ch;
    std::vector<unsigned long long> all(2 * c->world_size);
    MPI_Allgather(me, 2, MPI_UNSIGNED_LONG_LONG, all.data(), 2, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
    c->hosts = c->gpus = 0;
    for (int i = 0; i < c->world_size; i++) {
        int new_host = 1, new_gpu = 1;
        for (int j = 0; j < i; j++) {
            if (all[2 * j] != all[2 * i]) continue;
            new_host = 0;
            if (all[2 * j + 1] == all[2 * i + 1]) new_gpu = 0;
        }
        c->hosts += new_host;
        c->gpus += new_gpu;
    }
#endif
}

// Ranks rank/D .. form group rank / D. Returns 0, or prints the problem and returns -1.
static int comm_layout(Comm* c, int B, int D) {
    if (B < 1 || B > MAX_BLOCKS || D < 1 || c->world_size % D || c->world_size / D > B) {
        if (c->world_rank == 0)
            fprintf(stderr, "layout: %d ranks cannot hold %d blocks x %d replicas (need ranks / D <= B)\n",
                    c->world_size, B, D);
        return -1;
    }
    c->B = B;
    c->D = D;
    c->G = c->world_size / D;
    c->group = c->world_rank / D;
    c->replica = c->world_rank % D;
    c->nowned = 0;
    for (int b = 0; b < B; b++)
        if (b % c->G == c->group) c->owned[c->nowned++] = b;
#ifdef USE_MPI
    MPI_Comm_split(MPI_COMM_WORLD, c->group, c->replica, &c->group_comm);
#else
    if (D > 1) {
        fprintf(stderr, "layout: D = %d needs an MPI build\n", D);
        return -1;
    }
#endif
#ifdef USE_NCCL
    c->have_nccl = 0;
    if (D > 1) {
        ncclUniqueId id;
        if (c->replica == 0) ncclCheck(ncclGetUniqueId(&id));
        MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, c->group_comm);
        ncclCheck(ncclCommInitRank(&c->nccl, D, id, c->replica));
        c->have_nccl = 1;
    }
#endif
    return 0;
}

// Bytes each rank sends (and receives) in a ring all-reduce of n floats over D ranks.
static double ring_allreduce_bytes(size_t n, int D) { return D > 1 ? 2.0 * (D - 1) / D * (double)n * sizeof(float) : 0.0; }

__global__ void k_comm_scale(float* x, long n, float s) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

// In-place mean of a device buffer over the D replicas of this rank's group.
static void comm_allreduce_mean(Comm* c, float* buf, size_t n, cudaStream_t s) {
    if (c->D == 1) return;
#if defined(USE_NCCL)
    ncclCheck(ncclAllReduce(buf, buf, n, ncclFloat, ncclAvg, c->nccl, s));
#elif defined(USE_MPI)
#ifdef BLOCKFLOW_EMU
    MPI_Allreduce(MPI_IN_PLACE, buf, (int)n, MPI_FLOAT, MPI_SUM, c->group_comm);  // emulated device memory is host memory
#else
    if (c->stage_n < n) {
        if (c->stage) cudaCheck(cudaFreeHost(c->stage));
        cudaCheck(cudaMallocHost(&c->stage, n * sizeof(float)));
        c->stage_n = n;
    }
    cudaCheck(cudaMemcpyAsync(c->stage, buf, n * sizeof(float), cudaMemcpyDeviceToHost, s));
    cudaCheck(cudaStreamSynchronize(s));
    MPI_Allreduce(MPI_IN_PLACE, c->stage, (int)n, MPI_FLOAT, MPI_SUM, c->group_comm);
    cudaCheck(cudaMemcpyAsync(buf, c->stage, n * sizeof(float), cudaMemcpyHostToDevice, s));
#endif
    LAUNCH(k_comm_scale, ceil_div((long)n, 256), 256, 0, s)(buf, (long)n, 1.0f / c->D);
#endif
}

// Small host-side reductions (losses, timings).
static void comm_sum_world(Comm* c, double* v, int n) {
#ifdef USE_MPI
    if (c->world_size > 1) MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
    (void)c; (void)v; (void)n;
#endif
}
static void comm_max_world(Comm* c, double* v, int n) {
#ifdef USE_MPI
    if (c->world_size > 1) MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#else
    (void)c; (void)v; (void)n;
#endif
}
static void comm_sum_group(Comm* c, double* v, int n) {
#ifdef USE_MPI
    if (c->D > 1) MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_DOUBLE, MPI_SUM, c->group_comm);
#else
    (void)c; (void)v; (void)n;
#endif
}
static void comm_barrier_world(Comm* c) {
#ifdef USE_MPI
    if (c->world_size > 1) MPI_Barrier(MPI_COMM_WORLD);
#else
    (void)c;
#endif
}

// A single-process view (D = 1, no communication) that trains one given block (test reference).
static void comm_local(Comm* c, int B, int block) {
    memset(c, 0, sizeof(*c));
    c->world_size = 1;
    c->B = B;
    c->D = c->G = 1;
    c->nowned = 1;
    c->owned[0] = block;
}

// Releases what comm_layout created (MPI itself stays initialised).
static void comm_layout_free(Comm* c) {
#ifdef USE_NCCL
    if (c->have_nccl) ncclCommDestroy(c->nccl);
    c->have_nccl = 0;
#endif
#ifdef USE_MPI
    if (c->stage) cudaFreeHost(c->stage);
    c->stage = NULL;
    c->stage_n = 0;
    if (c->B) MPI_Comm_free(&c->group_comm);
#endif
    c->B = 0;
}

static void comm_finalize(Comm* c) {
    comm_layout_free(c);
#ifdef USE_MPI
    MPI_Finalize();
#endif
}
