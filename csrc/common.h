// common.h - build-mode switch (GPU vs CPU emulation), error checks and host utilities.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#ifdef BLOCKFLOW_EMU
#include "cuda_emu.h"
#else
#include <cublas_v2.h>
#include <cuda_runtime.h>
#define LAUNCH(kernel, grid, block, smem, stream) kernel<<<(grid), (block), (smem), (stream)>>>
#define DYNAMIC_SMEM(T, name)                                   \
    extern __shared__ __align__(16) unsigned char _dsm_##name[]; \
    T* name = reinterpret_cast<T*>(_dsm_##name)
#endif

#define cudaCheck(call)                                                                                      \
    do {                                                                                                     \
        cudaError_t e_ = (call);                                                                             \
        if (e_ != cudaSuccess) {                                                                             \
            fprintf(stderr, "[CUDA ERROR] %s at %s:%d: %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(e_)); \
            exit(EXIT_FAILURE);                                                                              \
        }                                                                                                    \
    } while (0)
#define cublasCheck(call)                                                                                 \
    do {                                                                                                  \
        cublasStatus_t s_ = (call);                                                                       \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                                                \
            fprintf(stderr, "[cuBLAS ERROR] %d (%s) at %s:%d\n", (int)s_, #call, __FILE__, __LINE__);    \
            exit(EXIT_FAILURE);                                                                           \
        }                                                                                                 \
    } while (0)
#define kernelCheck() cudaCheck(cudaGetLastError())

static inline long ceil_div(long a, long b) { return (a + b - 1) / b; }

// ------------------------------------------------------------------ host file helpers
static inline FILE* fopen_check(const char* path, const char* mode) {
    FILE* f = fopen(path, mode);
    if (!f) {
        fprintf(stderr, "error: cannot open '%s' (mode %s)\n", path, mode);
        exit(EXIT_FAILURE);
    }
    return f;
}
static inline void fread_check(void* dst, size_t size, size_t n, FILE* f, const char* what) {
    if (fread(dst, size, n, f) != n) {
        fprintf(stderr, "error: short read of %s\n", what);
        exit(EXIT_FAILURE);
    }
}
static inline void fwrite_check(const void* src, size_t size, size_t n, FILE* f) {
    if (fwrite(src, size, n, f) != n) {
        fprintf(stderr, "error: short write\n");
        exit(EXIT_FAILURE);
    }
}
static inline void* malloc_check(size_t n) {
    void* p = malloc(n ? n : 1);
    if (!p) {
        fprintf(stderr, "error: host malloc of %zu bytes failed\n", n);
        exit(EXIT_FAILURE);
    }
    return p;
}
static inline double wall_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

// ------------------------------------------------------------------ RNG
// splitmix64 is used both on the host (batch sampling) and on the device (counter-based noise).
__host__ __device__ inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
typedef struct {
    uint64_t state;
} Rng;
static inline uint64_t rng_next(Rng* r) {
    r->state += 0x9E3779B97F4A7C15ULL;
    return splitmix64(r->state);
}
static inline double rng_uniform(Rng* r) { return (double)(rng_next(r) >> 11) * (1.0 / 9007199254740992.0); }
static inline double rng_normal(Rng* r) {
    double u1 = (rng_next(r) >> 11) * (1.0 / 9007199254740992.0) + 1e-300, u2 = rng_uniform(r);
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}
static inline uint64_t mix_seed(uint64_t a, uint64_t b) { return splitmix64(a ^ splitmix64(b + 0x632BE59BD9B4E019ULL)); }

// ------------------------------------------------------------------ time partition (see python/blockflow/partition.py)
// Inverse standard-normal CDF, Wichura's AS241 (PPND16) - the algorithm of Python's statistics.NormalDist.
static inline double normal_ppf(double p) {
    double q = p - 0.5, r, num, den, x;
    if (fabs(q) <= 0.425) {
        r = 0.180625 - q * q;
        num = (((((((2.5090809287301226727e+3 * r + 3.3430575583588128105e+4) * r + 6.7265770927008700853e+4) * r +
                   4.5921953931549871457e+4) * r + 1.3731693765509461125e+4) * r + 1.9715909503065514427e+3) * r +
                1.3314166789178437745e+2) * r + 3.3871328727963666080e+0) * q;
        den = (((((((5.2264952788528545610e+3 * r + 2.8729085735721942674e+4) * r + 3.9307895800092710610e+4) * r +
                   2.1213794301586595867e+4) * r + 5.3941960214247511077e+3) * r + 6.8718700749205790830e+2) * r +
                4.2313330701600911252e+1) * r + 1.0);
        return num / den;
    }
    r = (q <= 0.0) ? p : 1.0 - p;
    r = sqrt(-log(r));
    if (r <= 5.0) {
        r -= 1.6;
        num = (((((((7.74545014278341407640e-4 * r + 2.27238449892691845833e-2) * r + 2.41780725177450611770e-1) * r +
                   1.27045825245236838258e+0) * r + 3.64784832476320460504e+0) * r + 5.76949722146069140550e+0) * r +
                4.63033784615654529590e+0) * r + 1.42343711074968357734e+0);
        den = (((((((1.05075007164441684324e-9 * r + 5.47593808499534494600e-4) * r + 1.51986665636164571966e-2) * r +
                   1.48103976427480074590e-1) * r + 6.89767334985100004550e-1) * r + 1.67638483018380384940e+0) * r +
                2.05319162663775882187e+0) * r + 1.0);
    } else {
        r -= 5.0;
        num = (((((((2.01033439929228813265e-7 * r + 2.71155556874348757815e-5) * r + 1.24266094738807843860e-3) * r +
                   2.65321895265761230930e-2) * r + 2.96560571828504891230e-1) * r + 1.78482653991729133580e+0) * r +
                5.46378491116411436990e+0) * r + 6.65790464350110377720e+0);
        den = (((((((2.04426310338993978564e-15 * r + 1.42151175831644588870e-7) * r + 1.84631831751005468180e-5) * r +
                   7.86869131145613259100e-4) * r + 1.48753612908506148525e-2) * r + 1.36929880922735805310e-1) * r +
                5.99832206555887937690e-1) * r + 1.0);
    }
    x = num / den;
    return (q < 0.0) ? -x : x;
}
static inline double logit_normal_ppf(double q, double mean, double stdv) {
    if (q <= 0.0) return 0.0;
    if (q >= 1.0) return 1.0;
    return 1.0 / (1.0 + exp(-(mean + stdv * normal_ppf(q))));
}
// edges[0..B]: block b owns t in [edges[b], edges[b+1]); equal probability under logit-normal(mean, std)
static inline void partition_edges(int B, double mean, double stdv, double* edges) {
    for (int b = 0; b <= B; b++) edges[b] = logit_normal_ppf((double)b / B, mean, stdv);
}
static inline int block_of(double t, const double* edges, int B) {
    for (int b = B - 1; b > 0; b--)
        if (t >= edges[b]) return b;
    return 0;
}
// training time for a block: inverse-CDF sample restricted to the quantile range [q_lo, q_hi]
static inline float sample_block_time(Rng* r, double q_lo, double q_hi, double mean, double stdv) {
    double u = q_lo + (q_hi - q_lo) * rng_uniform(r);
    if (u < 1e-12) u = 1e-12;
    if (u > 1.0 - 1e-12) u = 1.0 - 1e-12;
    return (float)(1.0 / (1.0 + exp(-(mean + stdv * normal_ppf(u)))));
}
