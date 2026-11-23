// kernels.cuh - CUDA kernels for the BlockFlow DiT trainer (fp32).
//
// Layout conventions (row-major, as in llm.c):
//   tokens  x      (N, T, C)        T = (img/patch)^2 tokens, C = hidden
//   patches xp     (N, T, P)        P = patch*patch*in_ch, feature order (c, pi, pj)
//   adaLN   mod    (N, L*6C)        layer l, chunk k at column l*6C + k*C (k: shift1 scale1 gate1 shift2 scale2 gate2)
//   heads   q,k,v  (N, NH, T, hs)
// Matrix multiplies are not here: they go through cuBLAS (dit.cuh).
#pragma once
#include "common.h"

#define BS 256                         // default threads per block (multiple of 32: warp-per-row kernels rely on it)
#define LN_EPS 1e-6f
#define GELU_BETA 0.7978845608028654f  // sqrt(2/pi)
#define GELU_KAPPA 0.044715f
#define NEG_LOG_MAX_PERIOD (-9.210340371976184f)  // -ln(10000)

// ------------------------------------------------------------------ warp reductions
__device__ __forceinline__ float warp_reduce_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_reduce_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// Block-wide sum; every thread of the block must call it. Result valid in all threads.
__device__ inline float block_reduce_sum(float v) {
    __shared__ float sh[32];
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    v = warp_reduce_sum(v);
    __syncthreads();  // protect sh from a previous call
    if (lane == 0) sh[w] = v;
    __syncthreads();
    v = (lane < nw) ? sh[lane] : 0.0f;
    return warp_reduce_sum(v);
}

// Counter-based Gaussian noise: one independent N(0,1) sample per (seed, counter).
__device__ inline float gaussian_noise(uint64_t seed, uint64_t ctr) {
    uint64_t z = splitmix64(seed ^ splitmix64(ctr));
    float u1 = ((float)(z >> 40) + 0.5f) * (1.0f / 16777216.0f);        // (0, 1)
    float u2 = (float)((z >> 16) & 0xFFFFFFULL) * (1.0f / 16777216.0f);  // [0, 1)
    return sqrtf(-2.0f * logf(u1)) * cosf(6.283185307179586f * u2);
}

// ------------------------------------------------------------------ training batch
// One thread per output element (n, token, feature): gathers image idx[n], x0 = u8/127.5 - 1,
// eps = noise (given, NCHW) or generated, x_t = (1-t) x0 + t eps, v = eps - x0.
__global__ void k_make_batch(float* xp, float* target, const unsigned char* images, const int* idx, const float* t,
                             const float* noise, uint64_t seed, int N, int Cin, int H, int W, int p) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    int gw = W / p, T = (H / p) * gw, P = Cin * p * p;
    if (i >= (long)N * T * P) return;
    int pd = (int)(i % P);
    long r = i / P;
    int tok = (int)(r % T), n = (int)(r / T);
    int c = pd / (p * p), rem = pd % (p * p), pi = rem / p, pj = rem % p;
    int yy = (tok / gw) * p + pi, xx = (tok % gw) * p + pj;
    float x0 = (float)images[(((long)idx[n] * Cin + c) * H + yy) * W + xx] / 127.5f - 1.0f;
    long nidx = (((long)n * Cin + c) * H + yy) * W + xx;
    float eps = noise ? noise[nidx] : gaussian_noise(seed, (uint64_t)nidx);
    float tt = t[n];
    xp[i] = (1.0f - tt) * x0 + tt * eps;
    target[i] = eps - x0;
}

// x[i] = N(0,1) noise (sampler initial state)
__global__ void k_gaussian_fill(float* x, long n, uint64_t seed) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = gaussian_noise(seed, (uint64_t)i);
}

// image (N, Cin, H, W) <-> patches (N, T, P)
__global__ void k_patchify(float* xp, const float* img, int N, int Cin, int H, int W, int p) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    int gw = W / p, T = (H / p) * gw, P = Cin * p * p;
    if (i >= (long)N * T * P) return;
    int pd = (int)(i % P);
    long r = i / P;
    int tok = (int)(r % T), n = (int)(r / T);
    int c = pd / (p * p), rem = pd % (p * p);
    int yy = (tok / gw) * p + rem / p, xx = (tok % gw) * p + rem % p;
    xp[i] = img[(((long)n * Cin + c) * H + yy) * W + xx];
}
__global__ void k_unpatchify(float* img, const float* xp, int N, int Cin, int H, int W, int p) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    int gw = W / p, T = (H / p) * gw, P = Cin * p * p;
    if (i >= (long)N * T * P) return;
    int pd = (int)(i % P);
    long r = i / P;
    int tok = (int)(r % T), n = (int)(r / T);
    int c = pd / (p * p), rem = pd % (p * p);
    int yy = (tok / gw) * p + rem / p, xx = (tok % gw) * p + rem % p;
    img[(((long)n * Cin + c) * H + yy) * W + xx] = xp[i];
}

// ------------------------------------------------------------------ conditioning
// DiT timestep embedding of t*t_scale: [cos(a_i), sin(a_i)], a_i = t*t_scale*exp(-ln(10000) i/half)
__global__ void k_timestep_embed(float* out, const float* t, int N, int Fd, float t_scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = Fd / 2;
    if (idx >= N * half) return;
    int n = idx / half, i = idx % half;
    float freq = expf((NEG_LOG_MAX_PERIOD * (float)i) / (float)half);
    float arg = (t[n] * t_scale) * freq;
    out[n * Fd + i] = cosf(arg);
    out[n * Fd + half + i] = sinf(arg);
}
__global__ void k_silu(float* out, const float* inp, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = inp[i];
        out[i] = x / (1.0f + expf(-x));
    }
}
__global__ void k_silu_backward(float* dinp, const float* dout, const float* inp, long n) {  // dinp may alias dout
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = inp[i], s = 1.0f / (1.0f + expf(-x));
        dinp[i] = dout[i] * s * (1.0f + x * (1.0f - s));
    }
}
__global__ void k_add_class_emb(float* c, const float* table, const int* y, int N, int C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N * C) c[i] += table[(long)y[i / C] * C + (i % C)];
}
// Deterministic scatter-add: thread per (row, channel) sums over the batch.
__global__ void k_class_emb_backward(float* dtable, const float* dc, const int* y, int N, int C, int rows) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * C) return;
    int k = i / C, j = i % C;
    float acc = 0.0f;
    bool any = false;
    for (int n = 0; n < N; n++)
        if (y[n] == k) {
            acc += dc[(long)n * C + j];
            any = true;
        }
    if (any) dtable[i] += acc;
}
__global__ void k_add_bias(float* out, const float* bias, long rows, int OC) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * OC) out[i] += bias[i % OC];
}
__global__ void k_add_pos(float* x, const float* pos, int N, long TC) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N * TC) x[i] += pos[i % TC];
}

// ------------------------------------------------------------------ LayerNorm without affine (warp per row)
__global__ void k_layernorm_forward(float* out, float* rstd, const float* inp, long rows, int C, float eps) {
    long row = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (row >= rows) return;  // whole warp exits together
    const float* x = inp + row * C;
    float s = 0.0f;
    for (int i = lane; i < C; i += 32) s += x[i];
    float mean = warp_reduce_sum(s) / C;
    float v = 0.0f;
    for (int i = lane; i < C; i += 32) {
        float d = x[i] - mean;
        v += d * d;
    }
    float r = rsqrtf(warp_reduce_sum(v) / C + eps);
    float* o = out + row * C;
    for (int i = lane; i < C; i += 32) o[i] = (x[i] - mean) * r;
    if (lane == 0) rstd[row] = r;
}
// dinp += rstd * (dout - mean(dout) - xhat * mean(dout * xhat))
__global__ void k_layernorm_backward(float* dinp, const float* dout, const float* xhat, const float* rstd, long rows,
                                     int C) {
    long row = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (row >= rows) return;
    const float* g = dout + row * C;
    const float* xh = xhat + row * C;
    float a = 0.0f, b = 0.0f;
    for (int i = lane; i < C; i += 32) {
        a += g[i];
        b += g[i] * xh[i];
    }
    a = warp_reduce_sum(a) / C;
    b = warp_reduce_sum(b) / C;
    float r = rstd[row];
    float* d = dinp + row * C;
    for (int i = lane; i < C; i += 32) d[i] += r * (g[i] - a - xh[i] * b);
}

// ------------------------------------------------------------------ adaLN modulation and gating
// out[n,t,j] = xhat[n,t,j] * (1 + scale[n*ms+j]) + shift[n*ms+j]
__global__ void k_modulate(float* out, const float* xhat, const float* shift, const float* scale, int N, int T, int C,
                           int ms) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)N * T * C) return;
    int j = (int)(i % C);
    long n = i / ((long)T * C);
    out[i] = xhat[i] * (1.0f + scale[n * ms + j]) + shift[n * ms + j];
}
// Thread per (n, j): dxhat = dout*(1+scale); dshift = sum_t dout; dscale = sum_t dout*xhat.
__global__ void k_modulate_backward(float* dxhat, float* dshift, float* dscale, const float* dout, const float* xhat,
                                    const float* scale, int N, int T, int C, int ms) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * C) return;
    int n = i / C, j = i % C;
    float s = 1.0f + scale[(long)n * ms + j];
    float ash = 0.0f, asc = 0.0f;
    for (int t = 0; t < T; t++) {
        long k = ((long)n * T + t) * C + j;
        float g = dout[k];
        dxhat[k] = g * s;
        ash += g;
        asc += g * xhat[k];
    }
    dshift[(long)n * ms + j] = ash;
    dscale[(long)n * ms + j] = asc;
}
// x += gate * branch
__global__ void k_gate_residual(float* x, const float* branch, const float* gate, int N, int T, int C, int ms) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)N * T * C) return;
    int j = (int)(i % C);
    long n = i / ((long)T * C);
    x[i] += gate[n * ms + j] * branch[i];
}
// Thread per (n, j): dbranch = dx * gate; dgate = sum_t dx * branch.
__global__ void k_gate_backward(float* dbranch, float* dgate, const float* dx, const float* branch, const float* gate,
                                int N, int T, int C, int ms) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N * C) return;
    int n = i / C, j = i % C;
    float g = gate[(long)n * ms + j], acc = 0.0f;
    for (int t = 0; t < T; t++) {
        long k = ((long)n * T + t) * C + j;
        float d = dx[k];
        dbranch[k] = d * g;
        acc += d * branch[k];
    }
    dgate[(long)n * ms + j] = acc;
}

// ------------------------------------------------------------------ GELU (tanh approximation, as PyTorch)
__global__ void k_gelu(float* out, const float* inp, long n) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = inp[i];
        out[i] = 0.5f * x * (1.0f + tanhf(GELU_BETA * (x + GELU_KAPPA * x * x * x)));
    }
}
__global__ void k_gelu_backward(float* dinp, const float* inp, const float* dout, long n) {  // dinp may alias dout
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = inp[i], x2 = x * x;
        float th = tanhf(GELU_BETA * (x + GELU_KAPPA * x2 * x));
        float dleft = 0.5f * (1.0f + th);
        float dright = 0.5f * x * (1.0f - th * th) * GELU_BETA * (1.0f + 3.0f * GELU_KAPPA * x2);
        dinp[i] = dout[i] * (dleft + dright);
    }
}

// ------------------------------------------------------------------ attention layout + softmax
// qkv (N, T, 3, NH, hs) -> q, k, v (N, NH, T, hs)
__global__ void k_permute_qkv(float* q, float* k, float* v, const float* qkv, int N, int T, int NH, int hs) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)N * NH * T * hs) return;
    int d = (int)(i % hs);
    long r = i / hs;
    int t = (int)(r % T);
    r /= T;
    int h = (int)(r % NH);
    long n = r / NH;
    long src = ((n * T + t) * 3 * NH + h) * hs + d;
    q[i] = qkv[src];
    k[i] = qkv[src + (long)NH * hs];
    v[i] = qkv[src + 2L * NH * hs];
}
__global__ void k_permute_qkv_backward(float* dqkv, const float* dq, const float* dk, const float* dv, int N, int T,
                                       int NH, int hs) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)N * NH * T * hs) return;
    int d = (int)(i % hs);
    long r = i / hs;
    int t = (int)(r % T);
    r /= T;
    int h = (int)(r % NH);
    long n = r / NH;
    long dst = ((n * T + t) * 3 * NH + h) * hs + d;
    dqkv[dst] = dq[i];
    dqkv[dst + (long)NH * hs] = dk[i];
    dqkv[dst + 2L * NH * hs] = dv[i];
}
// y (N, NH, T, hs) -> out (N, T, NH*hs)
__global__ void k_unpermute(float* out, const float* y, int N, int T, int NH, int hs) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)N * NH * T * hs) return;
    int d = (int)(i % hs);
    long r = i / hs;
    int t = (int)(r % T);
    r /= T;
    int h = (int)(r % NH);
    long n = r / NH;
    out[(n * T + t) * NH * hs + (long)h * hs + d] = y[i];
}
__global__ void k_unpermute_backward(float* dy, const float* dout, int N, int T, int NH, int hs) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)N * NH * T * hs) return;
    int d = (int)(i % hs);
    long r = i / hs;
    int t = (int)(r % T);
    r /= T;
    int h = (int)(r % NH);
    long n = r / NH;
    dy[i] = dout[(n * T + t) * NH * hs + (long)h * hs + d];
}
// In-place row softmax over T keys (non-causal); warp per row.
__global__ void k_softmax_rows(float* att, long rows, int T) {
    long row = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (row >= rows) return;
    float* a = att + row * T;
    float m = -INFINITY;
    for (int i = lane; i < T; i += 32) m = fmaxf(m, a[i]);
    m = warp_reduce_max(m);
    float s = 0.0f;
    for (int i = lane; i < T; i += 32) {
        float e = expf(a[i] - m);
        a[i] = e;
        s += e;
    }
    float inv = 1.0f / warp_reduce_sum(s);
    for (int i = lane; i < T; i += 32) a[i] *= inv;
}
// In place: datt <- att * (datt - sum_j att_j datt_j)
__global__ void k_softmax_backward_rows(float* datt, const float* att, long rows, int T) {
    long row = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (row >= rows) return;
    float* d = datt + row * T;
    const float* a = att + row * T;
    float s = 0.0f;
    for (int i = lane; i < T; i += 32) s += a[i] * d[i];
    s = warp_reduce_sum(s);
    for (int i = lane; i < T; i += 32) d[i] = a[i] * (d[i] - s);
}

// ------------------------------------------------------------------ flow-matching loss
// dout = gscale * 2 (out - target); partial[block] = sum of squared errors of this block.
__global__ void k_mse_loss(float* dout, float* partial, const float* out, const float* target, long n, float gscale) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    float e = 0.0f;
    if (i < n) {
        float d = out[i] - target[i];
        dout[i] = gscale * 2.0f * d;
        e = d * d;
    }
    e = block_reduce_sum(e);  // all threads participate (no early return)
    if (threadIdx.x == 0) partial[blockIdx.x] = e;
}
// Single block: acc[0] += scale * sum(partial)
__global__ void k_reduce_sum_into(float* acc, const float* partial, int nparts, float scale) {
    float s = 0.0f;
    for (int i = threadIdx.x; i < nparts; i += blockDim.x) s += partial[i];
    s = block_reduce_sum(s);
    if (threadIdx.x == 0) acc[0] += scale * s;
}

// ------------------------------------------------------------------ optimizer, EMA, sampler helpers
// AdamW exactly as torch.optim.AdamW (decoupled weight decay, bias-corrected moments).
__global__ void k_adamw(float* params, float* m, float* v, const float* grads, long n, float lr, float beta1,
                        float beta2, float eps, float wd, float step_size, float bc2_sqrt) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = grads[i], p = params[i];
    if (wd != 0.0f) p = p * (1.0f - lr * wd);
    float mi = m[i] + (1.0f - beta1) * (g - m[i]);
    float vi = v[i] * beta2 + (1.0f - beta2) * g * g;
    m[i] = mi;
    v[i] = vi;
    params[i] = p - step_size * (mi / (sqrtf(vi) / bc2_sqrt + eps));
}
__global__ void k_ema(float* ema, const float* p, long n, float decay) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) ema[i] = decay * ema[i] + (1.0f - decay) * p[i];
}
__global__ void k_fill(float* x, long n, float val) {
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = val;
}
__global__ void k_axpy(float* x, const float* v, float a, long n) {  // x += a * v  (Euler step)
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += a * v[i];
}
