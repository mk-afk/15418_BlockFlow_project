// trainer.cuh - the dataset on the GPU and one optimizer step of hybrid block/data-parallel training.
// Shared by train.cu and test_comm.cu, so the parallel test runs the real training step.
#pragma once
#include "comm.h"
#include "dit.cuh"

// Training hyperparameters (DiT defaults; python/train_ref.py uses the same values).
#define LR 1e-4f
#define ADAM_BETA1 0.9f
#define ADAM_BETA2 0.999f
#define ADAM_EPS 1e-8f
#define WEIGHT_DECAY 0.0f
#define EMA_MAX_DECAY 0.9999f  // EMA decay after optimizer step s: min(EMA_MAX_DECAY, (1 + s) / (10 + s))
#define LOGNORM_MEAN 0.0       // training times t = sigmoid(z), z ~ N(LOGNORM_MEAN, LOGNORM_STD^2)
#define LOGNORM_STD 1.0
#define TRAIN_SEED 0ULL

// ------------------------------------------------------------------ dataset (format: python/blockflow/data.py)
#define DATA_MAGIC 4180001
typedef struct {
    int N, C, H, W, num_classes;
    unsigned char* d_images;  // (N, C, H, W) uint8 on the device
    unsigned char* labels;    // (N) on the host
} Dataset;

static void dataset_load(Dataset* d, const char* path) {
    int hdr[16];
    FILE* f = fopen_check(path, "rb");
    fread_check(hdr, sizeof(int), 16, f, "dataset header");
    if (hdr[0] != DATA_MAGIC) {
        fprintf(stderr, "%s: not a BlockFlow dataset (see python/prepare_data.py)\n", path);
        exit(EXIT_FAILURE);
    }
    d->N = hdr[2]; d->C = hdr[3]; d->H = hdr[4]; d->W = hdr[5]; d->num_classes = hdr[6];
    size_t nimg = (size_t)d->N * d->C * d->H * d->W;
    d->labels = (unsigned char*)malloc_check(d->N);
    unsigned char* tmp = (unsigned char*)malloc_check(nimg);
    fread_check(d->labels, 1, d->N, f, "labels");
    fread_check(tmp, 1, nimg, f, "images");
    fclose(f);
    cudaCheck(cudaMalloc(&d->d_images, nimg));
    cudaCheck(cudaMemcpy(d->d_images, tmp, nimg, cudaMemcpyHostToDevice));
    free(tmp);
}
static void dataset_free(Dataset* d) {
    cudaCheck(cudaFree(d->d_images));
    free(d->labels);
}

// ------------------------------------------------------------------ trainer
typedef struct {  // fixed global batch for tests: replica r uses samples [r*micro*accum, (r+1)*micro*accum)
    const int *idx, *y;
    const float* t;
    const float* d_noise;  // device, (global batch, C, H, W)
} BatchOverride;

typedef struct {  // milliseconds summed over steps and over this rank's blocks
    double compute, allreduce, optimizer, step_wall;
    long steps;
} PhaseTimes;

typedef struct {
    Comm* comm;
    Dataset* data;
    DiTConfig cfg;  // specialist config (depth = total depth / B)
    int global_batch, micro, accum, nblocks;
    DiT models[MAX_BLOCKS];
    Acts acts;
    Rng rng;
    // Batch descriptors alternate between two pinned slots; a slot is refilled only after the event behind
    // its previous asynchronous copies has completed, so the host never overwrites data the GPU has not read.
    int *h_idx[2], *h_y[2], *d_idx;
    float* h_t[2];
    cudaEvent_t slot_ev[2];
    int slot;
    float* d_loss;  // per owned block: sum of per-step mean losses since the last read
    cudaEvent_t ev[4];
    PhaseTimes times;
    const BatchOverride* override_batch;
} Trainer;

static void trainer_init(Trainer* tr, Comm* comm, Dataset* data, const DiTConfig* cfg, int global_batch, int micro) {
    memset(tr, 0, sizeof(*tr));
    tr->comm = comm;
    tr->data = data;
    tr->cfg = *cfg;
    config_finalize(&tr->cfg);
    tr->global_batch = global_batch;
    tr->micro = micro;
    int per_replica = global_batch / comm->D;
    if (global_batch % comm->D || micro < 1 || per_replica % micro) {
        fprintf(stderr, "batch %d must split into %d replicas x whole micro-batches of %d\n", global_batch, comm->D, micro);
        exit(EXIT_FAILURE);
    }
    tr->accum = per_replica / micro;
    tr->nblocks = comm->nowned;
    for (int i = 0; i < tr->nblocks; i++) {
        int b = comm->owned[i];
        dit_build(&tr->models[i], &tr->cfg, b, comm->B, LOGNORM_MEAN, LOGNORM_STD, 1);
        dit_init_weights(&tr->models[i], mix_seed(TRAIN_SEED, 7919 + b));  // same weights on every replica of b
    }
    acts_alloc(&tr->acts, &tr->cfg, micro);
    tr->rng.state = mix_seed(TRAIN_SEED, 104729 + comm->world_rank);  // different data on every rank
    for (int j = 0; j < 2; j++) {
        cudaCheck(cudaMallocHost(&tr->h_idx[j], micro * sizeof(int)));
        cudaCheck(cudaMallocHost(&tr->h_y[j], micro * sizeof(int)));
        cudaCheck(cudaMallocHost(&tr->h_t[j], micro * sizeof(float)));
        cudaCheck(cudaEventCreateWithFlags(&tr->slot_ev[j], cudaEventDisableTiming));
    }
    cudaCheck(cudaMalloc(&tr->d_idx, micro * sizeof(int)));
    cudaCheck(cudaMalloc(&tr->d_loss, MAX_BLOCKS * sizeof(float)));
    cudaCheck(cudaMemset(tr->d_loss, 0, MAX_BLOCKS * sizeof(float)));
    for (int i = 0; i < 4; i++) cudaCheck(cudaEventCreate(&tr->ev[i]));
}

static void trainer_free(Trainer* tr) {
    for (int i = 0; i < tr->nblocks; i++) dit_free(&tr->models[i]);
    acts_free(&tr->acts);
    for (int j = 0; j < 2; j++) {
        cudaCheck(cudaFreeHost(tr->h_idx[j]));
        cudaCheck(cudaFreeHost(tr->h_y[j]));
        cudaCheck(cudaFreeHost(tr->h_t[j]));
        cudaCheck(cudaEventDestroy(tr->slot_ev[j]));
    }
    cudaCheck(cudaFree(tr->d_idx));
    cudaCheck(cudaFree(tr->d_loss));
    for (int i = 0; i < 4; i++) cudaCheck(cudaEventDestroy(tr->ev[i]));
}

// Picks micro-batch k for block model m (images, labels, times in the block's interval) and copies it to the
// device. Returns the fixed noise of a BatchOverride, or NULL to generate noise on the GPU.
static const float* trainer_load_batch(Trainer* tr, DiT* m, int k) {
    int n = tr->micro, sl = tr->slot;
    tr->slot ^= 1;
    cudaCheck(cudaEventSynchronize(tr->slot_ev[sl]));
    int *h_idx = tr->h_idx[sl], *h_y = tr->h_y[sl];
    float* h_t = tr->h_t[sl];
    const float* noise = NULL;
    if (tr->override_batch) {
        const BatchOverride* ob = tr->override_batch;
        int start = tr->comm->replica * n * tr->accum + k * n;
        for (int j = 0; j < n; j++) {
            h_idx[j] = ob->idx[start + j];
            h_y[j] = ob->y[start + j];
            h_t[j] = ob->t[start + j];
        }
        noise = ob->d_noise + (long)start * tr->data->C * tr->data->H * tr->data->W;
    } else {
        for (int j = 0; j < n; j++) {
            h_idx[j] = (int)(rng_next(&tr->rng) % (uint64_t)tr->data->N);
            h_y[j] = tr->data->labels[h_idx[j]];
            h_t[j] = sample_block_time(&tr->rng, m->q_lo, m->q_hi, LOGNORM_MEAN, LOGNORM_STD);
        }
    }
    cudaStream_t s = g_ctx.stream;
    cudaCheck(cudaMemcpyAsync(tr->d_idx, h_idx, n * sizeof(int), cudaMemcpyHostToDevice, s));
    cudaCheck(cudaMemcpyAsync(tr->acts.y, h_y, n * sizeof(int), cudaMemcpyHostToDevice, s));
    cudaCheck(cudaMemcpyAsync(tr->acts.t, h_t, n * sizeof(float), cudaMemcpyHostToDevice, s));
    cudaCheck(cudaEventRecord(tr->slot_ev[sl], s));
    return noise;
}

// One optimizer step for every block this rank owns; accumulates per-phase times.
static void trainer_step(Trainer* tr, int step) {
    Acts* a = &tr->acts;
    Comm* c = tr->comm;
    cudaStream_t s = g_ctx.stream;
    const int n = tr->micro;
    const long tok = (long)n * tr->cfg.T * tr->cfg.P;
    double t0 = wall_ms();
    for (int i = 0; i < tr->nblocks; i++) {
        DiT* m = &tr->models[i];
        cudaCheck(cudaEventRecord(tr->ev[0], s));
        dit_zero_grad(m);
        for (int k = 0; k < tr->accum; k++) {
            const float* noise = trainer_load_batch(tr, m, k);
            uint64_t seed = mix_seed(mix_seed(TRAIN_SEED, (uint64_t)m->block_id * 1000003ULL + c->world_rank),
                                     (uint64_t)step * 4099ULL + k);
            LAUNCH(k_make_batch, GRID(tok), BS, 0, s)(a->xp, a->target, tr->data->d_images, tr->d_idx, a->t, noise,
                                                     seed, n, tr->data->C, tr->data->H, tr->data->W, tr->cfg.patch);
            kernelCheck();
            dit_forward(m, a, n);
            dit_loss(m, a, n, 1.0f / tr->accum, 1.0f / tr->accum, tr->d_loss + i);
            dit_backward(m, a, n);
        }
        cudaCheck(cudaEventRecord(tr->ev[1], s));
        comm_allreduce_mean(c, m->grads, m->num_params, s);
        cudaCheck(cudaEventRecord(tr->ev[2], s));
        dit_adamw(m, LR, ADAM_BETA1, ADAM_BETA2, ADAM_EPS, WEIGHT_DECAY);
        dit_ema_update(m, fminf(EMA_MAX_DECAY, (1.0f + m->step) / (10.0f + m->step)));
        cudaCheck(cudaEventRecord(tr->ev[3], s));
        cudaCheck(cudaEventSynchronize(tr->ev[3]));
        float ms[3];
        for (int e = 0; e < 3; e++) cudaCheck(cudaEventElapsedTime(&ms[e], tr->ev[e], tr->ev[e + 1]));
        tr->times.compute += ms[0];
        tr->times.allreduce += ms[1];
        tr->times.optimizer += ms[2];
    }
    tr->times.step_wall += wall_ms() - t0;
    tr->times.steps++;
}

// Mean loss per owned block since the previous call, averaged over the group's replicas.
static void trainer_read_losses(Trainer* tr, int steps_since, double* out) {
    float h[MAX_BLOCKS];
    cudaCheck(cudaMemcpy(h, tr->d_loss, tr->nblocks * sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemset(tr->d_loss, 0, MAX_BLOCKS * sizeof(float)));
    for (int i = 0; i < tr->nblocks; i++) out[i] = h[i] / (steps_since > 0 ? steps_since : 1);
    comm_sum_group(tr->comm, out, tr->nblocks);
    for (int i = 0; i < tr->nblocks; i++) out[i] /= tr->comm->D;
}
