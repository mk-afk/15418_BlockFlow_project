// cuda_emu.h - run the CUDA code paths of BlockFlow on a CPU, for testing without a GPU.
//
// Compile any .cu file of this project with  g++ -x c++ -DBLOCKFLOW_EMU ...  and every
// kernel launch executes on the host.  Each CUDA thread of a block is a user-level fiber
// (ucontext); fibers are switched only at __syncthreads() and warp-level primitives, so
// barrier and shuffle semantics are exact:
//   * a missing __syncthreads() shows up as a wrong answer (threads run in order, not in parallel),
//   * barrier divergence that would hang a GPU is reported as a deadlock,
//   * cudaMalloc'd memory is filled with NaN bytes so reads of uninitialised memory poison results.
// cuBLAS SGEMM/SGEMV calls run as straightforward column-major loops.  Speed is not a goal.
#pragma once
#ifndef BLOCKFLOW_EMU
#error "cuda_emu.h is only for BLOCKFLOW_EMU builds"
#endif

#include <ucontext.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// ------------------------------------------------------------------ qualifiers and built-ins
#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __noinline__
#define __launch_bounds__(...)
#define __shared__ static thread_local
#define __align__(n) alignas(n)

struct dim3 {
    unsigned int x, y, z;
    constexpr dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1) : x(x_), y(y_), z(z_) {}
};
struct uint3 { unsigned int x, y, z; };

namespace emu {
inline thread_local uint3 tl_threadIdx{0, 0, 0};
inline thread_local uint3 tl_blockIdx{0, 0, 0};
inline thread_local dim3 tl_blockDim;
inline thread_local dim3 tl_gridDim;
}  // namespace emu
#define threadIdx (emu::tl_threadIdx)
#define blockIdx (emu::tl_blockIdx)
#define blockDim (emu::tl_blockDim)
#define gridDim (emu::tl_gridDim)

inline float rsqrtf(float x) { return 1.0f / sqrtf(x); }
inline float __expf(float x) { return expf(x); }
inline float __logf(float x) { return logf(x); }
inline float __fdividef(float a, float b) { return a / b; }
template <typename T> inline T __ldg(const T* p) { return *p; }

// ------------------------------------------------------------------ fiber scheduler
namespace emu {

struct Fiber {
    ucontext_t ctx;
    char* stack = nullptr;
    uint3 tid{0, 0, 0};
    int linear = 0;
    bool done = true;
};

struct Runner {
    std::vector<Fiber> fibers;
    size_t stack_bytes = 256 * 1024;
    ucontext_t sched;
    ucontext_t tmpl;
    bool have_tmpl = false;
    int cur = 0, nthreads = 0, alive = 0;
    int bar_arrived = 0;
    unsigned long bar_gen = 0;
    std::vector<int> warp_alive, warp_arrived;
    std::vector<unsigned long> warp_gen;
    std::vector<uint64_t> warp_slots;
    std::vector<unsigned char> dyn_smem;
    const std::function<void()>* body = nullptr;
    unsigned long progress = 0;
    bool in_kernel = false;
    ~Runner() {
        for (auto& f : fibers) free(f.stack);
    }
};

inline Runner& runner() {
    static thread_local Runner r;
    return r;
}

inline void fiber_main() {
    Runner& R = runner();
    (*R.body)();
    R.fibers[R.cur].done = true;
    // returning resumes uc_link == the scheduler context
}

inline void yield_fiber() {
    Runner& R = runner();
    swapcontext(&R.fibers[R.cur].ctx, &R.sched);
}

inline void release_if_complete(Runner& R, int w) {
    if (R.bar_arrived > 0 && R.bar_arrived == R.alive) {
        R.bar_arrived = 0;
        R.bar_gen++;
    }
    if (w >= 0 && R.warp_arrived[w] > 0 && R.warp_arrived[w] == R.warp_alive[w]) {
        R.warp_arrived[w] = 0;
        R.warp_gen[w]++;
    }
}

inline void syncthreads() {
    Runner& R = runner();
    if (!R.in_kernel) { fprintf(stderr, "emu: __syncthreads() outside a kernel\n"); abort(); }
    R.progress++;
    unsigned long gen = R.bar_gen;
    if (++R.bar_arrived == R.alive) {
        R.bar_arrived = 0;
        R.bar_gen++;
        return;
    }
    while (R.bar_gen == gen) yield_fiber();
}

inline void warp_barrier() {
    Runner& R = runner();
    int w = R.fibers[R.cur].linear >> 5;
    R.progress++;
    unsigned long gen = R.warp_gen[w];
    if (++R.warp_arrived[w] == R.warp_alive[w]) {
        R.warp_arrived[w] = 0;
        R.warp_gen[w]++;
        return;
    }
    while (R.warp_gen[w] == gen) yield_fiber();
}

// Exchange a value with lane `src` of the calling warp (src < 0 or out of range: keep own value).
template <typename T>
inline T warp_exchange(T var, int src) {
    static_assert(sizeof(T) <= 8, "warp shuffle of types wider than 8 bytes");
    Runner& R = runner();
    int lin = R.fibers[R.cur].linear, w = lin >> 5, lane = lin & 31;
    uint64_t bits = 0;
    memcpy(&bits, &var, sizeof(T));
    R.warp_slots[w * 32 + lane] = bits;
    warp_barrier();
    T out = var;
    int lanes = R.nthreads - w * 32 < 32 ? R.nthreads - w * 32 : 32;
    if (src >= 0 && src < lanes) {
        uint64_t b = R.warp_slots[w * 32 + src];
        memcpy(&out, &b, sizeof(T));
    }
    warp_barrier();
    return out;
}

inline int lane_of_current() { Runner& R = runner(); return R.fibers[R.cur].linear & 31; }

inline void run_block(Runner& R) {
    for (int i = 0; i < R.nthreads; i++) {
        Fiber& f = R.fibers[i];
        f.done = false;
        f.ctx = R.tmpl;
        f.ctx.uc_stack.ss_sp = f.stack;
        f.ctx.uc_stack.ss_size = R.stack_bytes;
        f.ctx.uc_link = &R.sched;
        makecontext(&f.ctx, (void (*)())fiber_main, 0);
    }
    R.alive = R.nthreads;
    R.bar_arrived = 0;
    int nwarps = (R.nthreads + 31) / 32;
    for (int w = 0; w < nwarps; w++) {
        R.warp_alive[w] = (R.nthreads - w * 32) < 32 ? (R.nthreads - w * 32) : 32;
        R.warp_arrived[w] = 0;
    }
    long idle = 0;
    while (R.alive > 0) {
        unsigned long before = R.progress;
        for (int i = 0; i < R.nthreads; i++) {
            Fiber& f = R.fibers[i];
            if (f.done) continue;
            R.cur = i;
            tl_threadIdx = f.tid;
            swapcontext(&R.sched, &f.ctx);
            if (f.done) {
                R.alive--;
                R.progress++;
                int w = f.linear >> 5;
                R.warp_alive[w]--;
                release_if_complete(R, w);
            }
        }
        if (R.progress == before) {
            if (++idle > 1000) {
                fprintf(stderr, "emu: deadlock in block (%u,%u,%u): barrier not reached by all threads\n",
                        tl_blockIdx.x, tl_blockIdx.y, tl_blockIdx.z);
                abort();
            }
        } else {
            idle = 0;
        }
    }
}

inline void launch(dim3 grid, dim3 block, size_t smem, const std::function<void()>& body, const char* name) {
    Runner& R = runner();
    if (R.in_kernel) { fprintf(stderr, "emu: nested kernel launch (%s)\n", name); abort(); }
    long nt = (long)block.x * block.y * block.z;
    if (nt <= 0 || nt > 1024) { fprintf(stderr, "emu: %s: invalid block size %ld\n", name, nt); abort(); }
    if (grid.x == 0 || grid.y == 0 || grid.z == 0 || grid.y > 65535 || grid.z > 65535) {
        fprintf(stderr, "emu: %s: invalid grid (%u,%u,%u)\n", name, grid.x, grid.y, grid.z);
        abort();
    }
    if (smem > 228 * 1024) { fprintf(stderr, "emu: %s: %zu bytes of shared memory\n", name, smem); abort(); }
    if (!R.have_tmpl) { getcontext(&R.tmpl); R.have_tmpl = true; }
    while ((long)R.fibers.size() < nt) {
        Fiber f;
        f.stack = (char*)malloc(R.stack_bytes);
        R.fibers.push_back(f);
    }
    R.nthreads = (int)nt;
    int nwarps = (int)((nt + 31) / 32);
    R.warp_alive.assign(nwarps, 0);
    R.warp_arrived.assign(nwarps, 0);
    R.warp_gen.assign(nwarps, 0);
    R.warp_slots.assign((size_t)nwarps * 32, 0);
    R.body = &body;
    for (long i = 0; i < nt; i++) {
        Fiber& f = R.fibers[i];
        f.linear = (int)i;
        f.tid = uint3{(unsigned)(i % block.x), (unsigned)((i / block.x) % block.y), (unsigned)(i / ((long)block.x * block.y))};
    }
    tl_blockDim = block;
    tl_gridDim = grid;
    R.in_kernel = true;
    for (unsigned z = 0; z < grid.z; z++)
        for (unsigned y = 0; y < grid.y; y++)
            for (unsigned x = 0; x < grid.x; x++) {
                tl_blockIdx = uint3{x, y, z};
                R.dyn_smem.assign(smem + 16, 0xFF);  // fresh, NaN-filled shared memory per block
                run_block(R);
            }
    R.in_kernel = false;
}

inline void* dyn_smem() { return runner().dyn_smem.data(); }

template <typename... KArgs>
struct LaunchObj {
    void (*fn)(KArgs...);
    dim3 grid, block;
    size_t smem;
    const char* name;
    template <typename... Args>
    void operator()(Args&&... args) const {
        static_assert(sizeof...(Args) == sizeof...(KArgs), "kernel argument count mismatch");
        std::tuple<std::decay_t<KArgs>...> tup(static_cast<std::decay_t<KArgs>>(args)...);
        auto f = fn;
        std::function<void()> body = [&tup, f]() { std::apply(f, tup); };
        launch(grid, block, smem, body, name);
    }
};
template <typename Stream, typename... KArgs>
LaunchObj<KArgs...> make_launch(void (*fn)(KArgs...), dim3 g, dim3 b, size_t s, const char* name, Stream) {
    return LaunchObj<KArgs...>{fn, g, b, s, name};  // streams are ignored: everything runs in order
}

}  // namespace emu

#define __syncthreads() emu::syncthreads()
#define __syncwarp(...) emu::warp_barrier()
#define DYNAMIC_SMEM(T, name) T* name = reinterpret_cast<T*>(emu::dyn_smem())
#define LAUNCH(kernel, grid, block, smem, stream) \
    emu::make_launch(kernel, dim3(grid), dim3(block), (size_t)(smem), #kernel, (stream))

template <typename T>
inline T __shfl_xor_sync(unsigned, T var, int laneMask, int width = 32) {
    int lane = emu::lane_of_current();
    int src = (lane & ~(width - 1)) | ((lane & (width - 1)) ^ laneMask);
    return emu::warp_exchange(var, src);
}
template <typename T>
inline T __shfl_down_sync(unsigned, T var, unsigned delta, int width = 32) {
    int lane = emu::lane_of_current();
    int seg = lane & ~(width - 1);
    int src = lane + (int)delta;
    if (src >= seg + width) src = lane;
    return emu::warp_exchange(var, src);
}
template <typename T>
inline T __shfl_sync(unsigned, T var, int srcLane, int width = 32) {
    int lane = emu::lane_of_current();
    return emu::warp_exchange(var, (lane & ~(width - 1)) + (srcLane & (width - 1)));
}

// Fibers are cooperative and blocks run one at a time, so plain read-modify-write is atomic.
inline float atomicAdd(float* a, float v) { float o = *a; *a = o + v; return o; }
inline double atomicAdd(double* a, double v) { double o = *a; *a = o + v; return o; }
inline int atomicAdd(int* a, int v) { int o = *a; *a = o + v; return o; }
inline unsigned long long atomicAdd(unsigned long long* a, unsigned long long v) { auto o = *a; *a = o + v; return o; }

// ------------------------------------------------------------------ CUDA runtime API (host memory)
typedef int cudaError_t;
#define cudaSuccess 0
#define cudaErrorMemoryAllocation 2
typedef struct emu_stream_s* cudaStream_t;
struct emu_event_s { double t_ms; };
typedef emu_event_s* cudaEvent_t;
enum cudaMemcpyKind { cudaMemcpyHostToHost = 0, cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2,
                      cudaMemcpyDeviceToDevice = 3, cudaMemcpyDefault = 4 };
#define cudaStreamNonBlocking 1
#define cudaEventDefault 0
#define cudaEventDisableTiming 2
#define cudaFuncAttributeMaxDynamicSharedMemorySize 8

struct cudaDeviceProp {
    char name[256];
    size_t totalGlobalMem;
    int multiProcessorCount, major, minor, maxThreadsPerBlock;
    size_t sharedMemPerBlock;
    int pciBusID, pciDeviceID, pciDomainID;
};

namespace emu {
inline double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace emu

inline cudaError_t cudaMalloc(void** p, size_t n) {
    *p = aligned_alloc(256, ((n + 255) / 256) * 256 + 256);
    if (!*p) return cudaErrorMemoryAllocation;
    memset(*p, 0xFF, n);  // NaN pattern: uninitialised reads poison results
    return cudaSuccess;
}
template <typename T> inline cudaError_t cudaMalloc(T** p, size_t n) { return cudaMalloc((void**)p, n); }
inline cudaError_t cudaMallocHost(void** p, size_t n) { *p = malloc(n); return *p ? cudaSuccess : cudaErrorMemoryAllocation; }
template <typename T> inline cudaError_t cudaMallocHost(T** p, size_t n) { return cudaMallocHost((void**)p, n); }
inline cudaError_t cudaFree(void* p) { free(p); return cudaSuccess; }
inline cudaError_t cudaFreeHost(void* p) { free(p); return cudaSuccess; }
inline cudaError_t cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind) { if (n) memmove(d, s, n); return cudaSuccess; }
inline cudaError_t cudaMemcpyAsync(void* d, const void* s, size_t n, cudaMemcpyKind k, cudaStream_t = nullptr) { return cudaMemcpy(d, s, n, k); }
inline cudaError_t cudaMemset(void* p, int v, size_t n) { memset(p, v, n); return cudaSuccess; }
inline cudaError_t cudaMemsetAsync(void* p, int v, size_t n, cudaStream_t = nullptr) { return cudaMemset(p, v, n); }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaPeekAtLastError() { return cudaSuccess; }
inline const char* cudaGetErrorString(cudaError_t) { return "emulated CUDA error"; }
inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
inline cudaError_t cudaGetDevice(int* d) { *d = 0; return cudaSuccess; }
inline cudaError_t cudaGetDeviceCount(int* n) { *n = 1; return cudaSuccess; }
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int) {
    memset(p, 0, sizeof(*p));
    snprintf(p->name, sizeof(p->name), "CPU emulation");
    p->totalGlobalMem = (size_t)8 << 30;
    p->multiProcessorCount = 1; p->major = 8; p->minor = 0; p->maxThreadsPerBlock = 1024;
    p->sharedMemPerBlock = 48 * 1024;
    return cudaSuccess;
}
inline cudaError_t cudaMemGetInfo(size_t* fr, size_t* tot) { *fr = (size_t)8 << 30; *tot = (size_t)8 << 30; return cudaSuccess; }
inline cudaError_t cudaStreamCreate(cudaStream_t* s) { *s = nullptr; return cudaSuccess; }
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned) { *s = nullptr; return cudaSuccess; }
inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaEventCreate(cudaEvent_t* e) { *e = new emu_event_s{0.0}; return cudaSuccess; }
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned) { return cudaEventCreate(e); }
inline cudaError_t cudaEventDestroy(cudaEvent_t e) { delete e; return cudaSuccess; }
inline cudaError_t cudaEventRecord(cudaEvent_t e, cudaStream_t = nullptr) { e->t_ms = emu::now_ms(); return cudaSuccess; }
inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return cudaSuccess; }
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t a, cudaEvent_t b) { *ms = (float)(b->t_ms - a->t_ms); return cudaSuccess; }
inline cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned = 0) { return cudaSuccess; }
template <typename F> inline cudaError_t cudaFuncSetAttribute(F, int, int) { return cudaSuccess; }

// ------------------------------------------------------------------ cuBLAS (column-major, like the real library)
typedef int cublasStatus_t;
#define CUBLAS_STATUS_SUCCESS 0
typedef enum { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1, CUBLAS_OP_C = 2 } cublasOperation_t;
typedef enum { CUBLAS_DEFAULT_MATH = 0, CUBLAS_PEDANTIC_MATH = 2, CUBLAS_TF32_TENSOR_OP_MATH = 3 } cublasMath_t;
struct emu_cublas_s { int unused; };
typedef emu_cublas_s* cublasHandle_t;
inline cublasStatus_t cublasCreate(cublasHandle_t* h) { *h = new emu_cublas_s{0}; return CUBLAS_STATUS_SUCCESS; }
inline cublasStatus_t cublasDestroy(cublasHandle_t h) { delete h; return CUBLAS_STATUS_SUCCESS; }
inline cublasStatus_t cublasSetStream(cublasHandle_t, cudaStream_t) { return CUBLAS_STATUS_SUCCESS; }
inline cublasStatus_t cublasSetMathMode(cublasHandle_t, cublasMath_t) { return CUBLAS_STATUS_SUCCESS; }
inline const char* cublasGetStatusString(cublasStatus_t) { return "emulated cuBLAS error"; }

namespace emu {
// C = alpha * op(A) op(B) + beta * C, all column-major.  beta == 0 never reads C (as in cuBLAS).
inline void sgemm(int ta, int tb, int m, int n, int k, float alpha, const float* A, int lda, const float* B, int ldb,
                  float beta, float* C, int ldc) {
    std::vector<float> At((size_t)m * k), Bt((size_t)n * k);
    for (int i = 0; i < m; i++)
        for (int l = 0; l < k; l++) At[(size_t)i * k + l] = ta ? A[l + (size_t)i * lda] : A[i + (size_t)l * lda];
    for (int j = 0; j < n; j++)
        for (int l = 0; l < k; l++) Bt[(size_t)j * k + l] = tb ? B[j + (size_t)l * ldb] : B[l + (size_t)j * ldb];
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++) {
            const float* a = &At[(size_t)i * k];
            const float* b = &Bt[(size_t)j * k];
            float acc[4] = {0.f, 0.f, 0.f, 0.f};
            int l = 0;
            for (; l + 4 <= k; l += 4)
                for (int u = 0; u < 4; u++) acc[u] += a[l + u] * b[l + u];
            for (; l < k; l++) acc[0] += a[l] * b[l];
            float s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
            float* c = &C[i + (size_t)j * ldc];
            *c = (beta == 0.0f) ? alpha * s : alpha * s + beta * *c;
        }
}
}  // namespace emu

inline cublasStatus_t cublasSgemm(cublasHandle_t, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                                  const float* alpha, const float* A, int lda, const float* B, int ldb,
                                  const float* beta, float* C, int ldc) {
    emu::sgemm(ta != CUBLAS_OP_N, tb != CUBLAS_OP_N, m, n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
    return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t, cublasOperation_t ta, cublasOperation_t tb, int m, int n,
                                                int k, const float* alpha, const float* A, int lda, long long sA,
                                                const float* B, int ldb, long long sB, const float* beta, float* C,
                                                int ldc, long long sC, int batch) {
    for (int i = 0; i < batch; i++)
        emu::sgemm(ta != CUBLAS_OP_N, tb != CUBLAS_OP_N, m, n, k, *alpha, A + i * sA, lda, B + i * sB, ldb, *beta,
                   C + i * sC, ldc);
    return CUBLAS_STATUS_SUCCESS;
}
inline cublasStatus_t cublasSgemv(cublasHandle_t, cublasOperation_t trans, int m, int n, const float* alpha,
                                  const float* A, int lda, const float* x, int incx, const float* beta, float* y,
                                  int incy) {
    int rows = (trans == CUBLAS_OP_N) ? m : n, cols = (trans == CUBLAS_OP_N) ? n : m;
    for (int i = 0; i < rows; i++) {
        float s = 0.0f;
        for (int j = 0; j < cols; j++) {
            float a = (trans == CUBLAS_OP_N) ? A[i + (size_t)j * lda] : A[j + (size_t)i * lda];
            s += a * x[(size_t)j * incx];
        }
        float* o = &y[(size_t)i * incy];
        *o = (*beta == 0.0f) ? *alpha * s : *alpha * s + *beta * *o;
    }
    return CUBLAS_STATUS_SUCCESS;
}
