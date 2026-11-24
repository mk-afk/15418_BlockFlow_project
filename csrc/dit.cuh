// dit.cuh - one DiT flow-matching specialist in C/CUDA (fp32, cuBLAS for all matrix multiplies).
//
// A blockwise model is B independent DiT structs (depth L/B each); the end-to-end
// model is one DiT with depth L.  The parameter order is the canonical flat order of
// python/blockflow/model.py::param_specs, so one cudaMemcpy moves a whole model and a
// single collective all-reduces a whole gradient.
#pragma once
#include "kernels.cuh"

// ------------------------------------------------------------------ global context
typedef struct {
    cublasHandle_t blas;
    cudaStream_t stream;
    float* ones;  // all-ones vector for bias gradients (gemv)
    long ones_len;
} Context;
static Context g_ctx;

static void ctx_init() {
    cudaCheck(cudaStreamCreate(&g_ctx.stream));
    cublasCheck(cublasCreate(&g_ctx.blas));
    cublasCheck(cublasSetStream(g_ctx.blas, g_ctx.stream));
    g_ctx.ones = NULL;
    g_ctx.ones_len = 0;
}
static void ctx_ensure_ones(long n) {
    if (n <= g_ctx.ones_len) return;
    if (g_ctx.ones) cudaCheck(cudaFree(g_ctx.ones));
    cudaCheck(cudaMalloc(&g_ctx.ones, n * sizeof(float)));
    LAUNCH(k_fill, ceil_div(n, BS), BS, 0, g_ctx.stream)(g_ctx.ones, n, 1.0f);
    kernelCheck();
    g_ctx.ones_len = n;
}
static void ctx_destroy() {
    if (g_ctx.ones) cudaCheck(cudaFree(g_ctx.ones));
    cublasCheck(cublasDestroy(g_ctx.blas));
    cudaCheck(cudaStreamDestroy(g_ctx.stream));
}

// ------------------------------------------------------------------ config and parameters
typedef struct {
    int img_size, in_ch, patch, hidden, depth, heads, mlp_ratio, num_classes, freq_dim;
    float t_scale;
    int grid, T, P;  // derived
} DiTConfig;

static void config_finalize(DiTConfig* c) {
    if (c->img_size % c->patch || c->hidden % c->heads || c->hidden % 4 || c->freq_dim % 2 || c->depth < 1) {
        fprintf(stderr, "invalid DiT config (img %d patch %d hidden %d heads %d freq %d depth %d)\n", c->img_size,
                c->patch, c->hidden, c->heads, c->freq_dim, c->depth);
        exit(EXIT_FAILURE);
    }
    c->grid = c->img_size / c->patch;
    c->T = c->grid * c->grid;
    c->P = c->patch * c->patch * c->in_ch;
}

#define NUM_PARAM_TENSORS 21
__attribute__((unused)) static const char* PARAM_NAMES[NUM_PARAM_TENSORS] = {
    "x_emb_w", "x_emb_b", "t_w1",   "t_b1",   "t_w2",  "t_b2",  "y_emb",   "ada_w",   "ada_b",   "qkv_w", "qkv_b",
    "proj_w",  "proj_b",  "fc1_w",  "fc1_b",  "fc2_w", "fc2_b", "f_ada_w", "f_ada_b", "f_out_w", "f_out_b"};

typedef struct {  // pointers into one flat buffer, canonical order
    float *x_emb_w, *x_emb_b, *t_w1, *t_b1, *t_w2, *t_b2, *y_emb, *ada_w, *ada_b, *qkv_w, *qkv_b, *proj_w, *proj_b,
        *fc1_w, *fc1_b, *fc2_w, *fc2_b, *f_ada_w, *f_ada_b, *f_out_w, *f_out_b;
} ParamTensors;

static void fill_param_sizes(size_t* s, const DiTConfig* c) {
    size_t C = c->hidden, P = c->P, F = c->freq_dim, L = c->depth, NC = c->num_classes, H = (size_t)c->mlp_ratio * C;
    size_t v[NUM_PARAM_TENSORS] = {C * P, C,      C * F,     C,         C * C,     C,     NC * C,
                                   L * 6 * C * C, L * 6 * C, L * 3 * C * C, L * 3 * C, L * C * C, L * C,
                                   L * H * C, L * H, L * C * H, L * C, 2 * C * C, 2 * C, P * C, P};
    memcpy(s, v, sizeof(v));
}
static size_t count_params(const DiTConfig* c) {
    size_t s[NUM_PARAM_TENSORS], n = 0;
    fill_param_sizes(s, c);
    for (int i = 0; i < NUM_PARAM_TENSORS; i++) n += s[i];
    return n;
}
static void point_params(ParamTensors* t, float* base, const size_t* sizes) {
    float** ptrs = (float**)t;  // ParamTensors is NUM_PARAM_TENSORS consecutive float* fields
    for (int i = 0; i < NUM_PARAM_TENSORS; i++) {
        ptrs[i] = base;
        base += sizes[i];
    }
}

typedef struct {
    DiTConfig cfg;
    int block_id, num_blocks;
    double q_lo, q_hi, t_lo, t_hi, ln_mean, ln_std;  // time interval this specialist is trained on
    size_t sizes[NUM_PARAM_TENSORS];
    size_t num_params;
    float *params, *grads, *m, *v, *ema;  // device, num_params each (m, v, ema may be NULL)
    ParamTensors p, g;
    float* pos;  // (T, C) fixed sin-cos position embedding
    int step;    // optimizer steps taken
} DiT;

static void pos_embed_host(float* out, int C, int gh, int gw) {
    int D = C / 2, half = D / 2;  // per-axis width and its sin/cos halves
    for (int r = 0; r < gh; r++)
        for (int c = 0; c < gw; c++) {
            float* o = out + (long)(r * gw + c) * C;
            for (int k = 0; k < half; k++) {
                double omega = 1.0 / pow(10000.0, (double)k / (D / 2.0));
                o[k] = (float)sin(c * omega);
                o[half + k] = (float)cos(c * omega);
                o[D + k] = (float)sin(r * omega);
                o[D + half + k] = (float)cos(r * omega);
            }
        }
}

// Allocates parameters (+ gradients, Adam moments and EMA when train != 0) on the device.
static void dit_build(DiT* m, const DiTConfig* cfg, int block_id, int num_blocks, double ln_mean, double ln_std,
                      int train) {
    memset(m, 0, sizeof(*m));
    m->cfg = *cfg;
    config_finalize(&m->cfg);
    m->block_id = block_id;
    m->num_blocks = num_blocks;
    m->ln_mean = ln_mean;
    m->ln_std = ln_std;
    m->q_lo = (double)block_id / num_blocks;
    m->q_hi = (double)(block_id + 1) / num_blocks;
    m->t_lo = logit_normal_ppf(m->q_lo, ln_mean, ln_std);
    m->t_hi = logit_normal_ppf(m->q_hi, ln_mean, ln_std);
    fill_param_sizes(m->sizes, &m->cfg);
    m->num_params = count_params(&m->cfg);
    size_t bytes = m->num_params * sizeof(float);
    cudaCheck(cudaMalloc(&m->params, bytes));
    point_params(&m->p, m->params, m->sizes);
    if (train) {
        cudaCheck(cudaMalloc(&m->grads, bytes));
        cudaCheck(cudaMalloc(&m->m, bytes));
        cudaCheck(cudaMalloc(&m->v, bytes));
        cudaCheck(cudaMalloc(&m->ema, bytes));
        cudaCheck(cudaMemset(m->grads, 0, bytes));
        cudaCheck(cudaMemset(m->m, 0, bytes));
        cudaCheck(cudaMemset(m->v, 0, bytes));
        point_params(&m->g, m->grads, m->sizes);
    }
    int T = m->cfg.T, C = m->cfg.hidden;
    float* hpos = (float*)malloc_check((size_t)T * C * sizeof(float));
    pos_embed_host(hpos, C, m->cfg.grid, m->cfg.grid);
    cudaCheck(cudaMalloc(&m->pos, (size_t)T * C * sizeof(float)));
    cudaCheck(cudaMemcpy(m->pos, hpos, (size_t)T * C * sizeof(float), cudaMemcpyHostToDevice));
    free(hpos);
}
static void dit_free(DiT* m) {
    cudaCheck(cudaFree(m->params));
    if (m->grads) cudaCheck(cudaFree(m->grads));
    if (m->m) cudaCheck(cudaFree(m->m));
    if (m->v) cudaCheck(cudaFree(m->v));
    if (m->ema) cudaCheck(cudaFree(m->ema));
    cudaCheck(cudaFree(m->pos));
}

// DiT (adaLN-Zero) initialisation, computed on the host then uploaded.  EMA starts equal to the weights.
static void dit_init_weights(DiT* m, uint64_t seed) {
    const DiTConfig* c = &m->cfg;
    float* h = (float*)malloc_check(m->num_params * sizeof(float));
    memset(h, 0, m->num_params * sizeof(float));
    ParamTensors p;
    point_params(&p, h, m->sizes);
    Rng r = {seed};
    auto xavier = [&](float* w, long fan_out, long fan_in) {
        double bound = sqrt(6.0 / (double)(fan_in + fan_out));
        for (long i = 0; i < fan_out * fan_in; i++) w[i] = (float)((2.0 * rng_uniform(&r) - 1.0) * bound);
    };
    auto normal = [&](float* w, long n, double std) {
        for (long i = 0; i < n; i++) w[i] = (float)(std * rng_normal(&r));
    };
    long C = c->hidden, P = c->P, F = c->freq_dim, H = (long)c->mlp_ratio * C;
    xavier(p.x_emb_w, C, P);
    normal(p.t_w1, C * F, 0.02);
    normal(p.t_w2, C * C, 0.02);
    normal(p.y_emb, (long)c->num_classes * C, 0.02);
    for (int l = 0; l < c->depth; l++) {
        xavier(p.qkv_w + l * 3 * C * C, 3 * C, C);
        xavier(p.proj_w + l * C * C, C, C);
        xavier(p.fc1_w + l * H * C, H, C);
        xavier(p.fc2_w + l * C * H, C, H);
    }
    // ada_w/ada_b, f_ada_w/f_ada_b, f_out_w/f_out_b stay zero (adaLN-Zero)
    cudaCheck(cudaMemcpy(m->params, h, m->num_params * sizeof(float), cudaMemcpyHostToDevice));
    if (m->ema) cudaCheck(cudaMemcpy(m->ema, h, m->num_params * sizeof(float), cudaMemcpyHostToDevice));
    free(h);
    m->step = 0;
}

// ------------------------------------------------------------------ activations (shared by specialists of equal depth)
typedef struct {
    int N, L;  // capacity (micro-batch, depth)
    float *xp, *target, *t;
    int* y;
    float *tfreq, *t_h, *t_a, *c, *sc, *mod, *fmod;
    float *ln1, *rstd1, *qkvr, *att, *atty, *ao, *ln2, *rstd2, *h, *fo;
    float *x, *lnf, *rstdf, *out;
    float *buf_m, *buf_qkv, *buf_y, *buf_g;
    float *dres, *dbranch, *dm, *dln, *datty, *datt, *dqkvr, *dh, *dout, *dmod, *dfmod, *dsc, *dc, *dt_a;
    float *loss_partial, *loss_acc;
    float* mem;
    size_t mem_floats;
} Acts;

static void acts_alloc(Acts* a, const DiTConfig* cfg_in, int N) {
    DiTConfig cfg = *cfg_in;
    config_finalize(&cfg);
    memset(a, 0, sizeof(*a));
    a->N = N;
    a->L = cfg.depth;
    size_t T = cfg.T, C = cfg.hidden, P = cfg.P, F = cfg.freq_dim, L = cfg.depth, NH = cfg.heads;
    size_t H = (size_t)cfg.mlp_ratio * C, n = N, NT = n * T, NTC = NT * C;
    size_t loss_parts = ceil_div((long)(NT * P), BS);
    float** ptrs[] = {&a->xp,    &a->target, &a->t,     &a->tfreq, &a->t_h,    &a->t_a,     &a->c,      &a->sc,
                      &a->mod,   &a->fmod,   &a->ln1,   &a->rstd1, &a->qkvr,   &a->att,     &a->atty,   &a->ao,
                      &a->ln2,   &a->rstd2,  &a->h,     &a->fo,    &a->x,      &a->lnf,     &a->rstdf,  &a->out,
                      &a->buf_m, &a->buf_qkv, &a->buf_y, &a->buf_g, &a->dres,    &a->dbranch, &a->dm,
                      &a->dln,   &a->datty,  &a->datt,  &a->dqkvr, &a->dh,     &a->dout,    &a->dmod,   &a->dfmod,
                      &a->dsc,   &a->dc,     &a->dt_a,  &a->loss_partial, &a->loss_acc};
    size_t sizes[] = {NT * P,  NT * P,  n,       n * F,   n * C,     n * C,          n * C,   n * C,
                      n * L * 6 * C, n * 2 * C, L * NTC, L * NT, L * 3 * NTC, L * n * NH * T * T, L * NTC, L * NTC,
                      L * NTC, L * NT, L * NT * H, L * NTC, NTC,       NTC,            NT,      NT * P,
                      NTC,     3 * NTC, NTC,     NT * H,  NTC,            NTC,     NTC,
                      NTC,     NTC,     n * NH * T * T, 3 * NTC, NT * H, NT * P,   n * L * 6 * C, n * 2 * C,
                      n * C,   n * C,   n * C,   loss_parts, 1};
    const int nbuf = (int)(sizeof(sizes) / sizeof(sizes[0]));
    static_assert(sizeof(ptrs) / sizeof(ptrs[0]) == sizeof(sizes) / sizeof(sizes[0]), "acts table mismatch");
    size_t total = 0;
    for (int i = 0; i < nbuf; i++) total += (sizes[i] + 63) & ~(size_t)63;  // 256-byte aligned sub-buffers
    if (cudaMalloc(&a->mem, total * sizeof(float)) != cudaSuccess) {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        fprintf(stderr, "a batch of %d needs %.2f GB of activations but the GPU has %.2f of %.2f GB free: use a smaller --micro\n", N,
                total * sizeof(float) / 1e9, free_b / 1e9, total_b / 1e9);
        exit(EXIT_FAILURE);
    }
    a->mem_floats = total;
    float* p = a->mem;
    for (int i = 0; i < nbuf; i++) {
        *ptrs[i] = p;
        p += (sizes[i] + 63) & ~(size_t)63;
    }
    cudaCheck(cudaMalloc(&a->y, n * sizeof(int)));
    cudaCheck(cudaMemset(a->loss_acc, 0, sizeof(float)));
    ctx_ensure_ones((long)NT);
}
static void acts_free(Acts* a) {
    cudaCheck(cudaFree(a->mem));
    cudaCheck(cudaFree(a->y));
}

// ------------------------------------------------------------------ cuBLAS matmuls (row-major via the transpose trick)
// out (rows, OC) = inp (rows, IC) @ W^T + b,  W is (OC, IC) like nn.Linear.weight
static void matmul_forward(float* out, const float* inp, const float* W, const float* b, long rows, int IC, int OC) {
    const float one = 1.0f, zero = 0.0f;
    cublasCheck(cublasSgemm(g_ctx.blas, CUBLAS_OP_T, CUBLAS_OP_N, OC, (int)rows, IC, &one, W, IC, inp, IC, &zero, out, OC));
    if (b) {
        LAUNCH(k_add_bias, ceil_div(rows * OC, BS), BS, 0, g_ctx.stream)(out, b, rows, OC);
        kernelCheck();
    }
}
// dinp (rows, IC) = (or +=) dout @ W;   dW (OC, IC) += dout^T @ inp;   db (OC) += column sums of dout
static void matmul_backward(float* dinp, float* dW, float* db, const float* dout, const float* inp, const float* W,
                            long rows, int IC, int OC, int accumulate_dinp) {
    const float one = 1.0f, zero = 0.0f;
    if (dinp)
        cublasCheck(cublasSgemm(g_ctx.blas, CUBLAS_OP_N, CUBLAS_OP_N, IC, (int)rows, OC, &one, W, IC, dout, OC,
                                accumulate_dinp ? &one : &zero, dinp, IC));
    cublasCheck(cublasSgemm(g_ctx.blas, CUBLAS_OP_N, CUBLAS_OP_T, IC, OC, (int)rows, &one, inp, IC, dout, OC, &one, dW, IC));
    if (db) {
        if (rows > g_ctx.ones_len) ctx_ensure_ones(rows);
        cublasCheck(cublasSgemv(g_ctx.blas, CUBLAS_OP_N, OC, (int)rows, &one, dout, OC, g_ctx.ones, 1, &one, db, 1));
    }
}

// ------------------------------------------------------------------ attention (non-causal, cuBLAS batched GEMMs)
// qkv (N,T,3C) -> qkvr = [q|k|v] each (N,NH,T,hs); att (N,NH,T,T) = softmax(q k^T / sqrt(hs)); atty (N,T,C)
static void attention_forward(float* atty, float* qkvr, float* att, float* buf_y, const float* qkv, int N, int T, int C,
                              int NH) {
    int hs = C / NH;
    long NTC = (long)N * T * C;
    float *q = qkvr, *k = qkvr + NTC, *v = qkvr + 2 * NTC;
    cudaStream_t s = g_ctx.stream;
    LAUNCH(k_permute_qkv, ceil_div(NTC, BS), BS, 0, s)(q, k, v, qkv, N, T, NH, hs);
    kernelCheck();
    const float scale = 1.0f / sqrtf((float)hs), zero = 0.0f, one = 1.0f;
    cublasCheck(cublasSgemmStridedBatched(g_ctx.blas, CUBLAS_OP_T, CUBLAS_OP_N, T, T, hs, &scale, k, hs, (long long)T * hs,
                                          q, hs, (long long)T * hs, &zero, att, T, (long long)T * T, N * NH));
    long rows = (long)N * NH * T;
    LAUNCH(k_softmax_rows, ceil_div(rows * 32, BS), BS, 0, s)(att, rows, T);
    kernelCheck();
    cublasCheck(cublasSgemmStridedBatched(g_ctx.blas, CUBLAS_OP_N, CUBLAS_OP_N, hs, T, T, &one, v, hs, (long long)T * hs,
                                          att, T, (long long)T * T, &zero, buf_y, hs, (long long)T * hs, N * NH));
    LAUNCH(k_unpermute, ceil_div(NTC, BS), BS, 0, s)(atty, buf_y, N, T, NH, hs);
    kernelCheck();
}
// datty (N,T,C) -> dqkv (N,T,3C).  Scratch: dqkvr (3NTC), datt (N NH T T), dbuf_y (NTC).
static void attention_backward(float* dqkv, float* dqkvr, float* datt, float* dbuf_y, const float* datty,
                               const float* qkvr, const float* att, int N, int T, int C, int NH) {
    int hs = C / NH;
    long NTC = (long)N * T * C;
    const float *q = qkvr, *k = qkvr + NTC, *v = qkvr + 2 * NTC;
    float *dq = dqkvr, *dk = dqkvr + NTC, *dv = dqkvr + 2 * NTC;
    cudaStream_t s = g_ctx.stream;
    const float scale = 1.0f / sqrtf((float)hs), zero = 0.0f, one = 1.0f;
    long long sT = (long long)T * hs, sA = (long long)T * T;
    LAUNCH(k_unpermute_backward, ceil_div(NTC, BS), BS, 0, s)(dbuf_y, datty, N, T, NH, hs);
    kernelCheck();
    // datt = dy @ v^T ; dv = att^T @ dy
    cublasCheck(cublasSgemmStridedBatched(g_ctx.blas, CUBLAS_OP_T, CUBLAS_OP_N, T, T, hs, &one, v, hs, sT, dbuf_y, hs, sT,
                                          &zero, datt, T, sA, N * NH));
    cublasCheck(cublasSgemmStridedBatched(g_ctx.blas, CUBLAS_OP_N, CUBLAS_OP_T, hs, T, T, &one, dbuf_y, hs, sT, att, T, sA,
                                          &zero, dv, hs, sT, N * NH));
    long rows = (long)N * NH * T;
    LAUNCH(k_softmax_backward_rows, ceil_div(rows * 32, BS), BS, 0, s)(datt, att, rows, T);
    kernelCheck();
    // dq = scale * dpre @ k ; dk = scale * dpre^T @ q
    cublasCheck(cublasSgemmStridedBatched(g_ctx.blas, CUBLAS_OP_N, CUBLAS_OP_N, hs, T, T, &scale, k, hs, sT, datt, T, sA,
                                          &zero, dq, hs, sT, N * NH));
    cublasCheck(cublasSgemmStridedBatched(g_ctx.blas, CUBLAS_OP_N, CUBLAS_OP_T, hs, T, T, &scale, q, hs, sT, datt, T, sA,
                                          &zero, dk, hs, sT, N * NH));
    LAUNCH(k_permute_qkv_backward, ceil_div(NTC, BS), BS, 0, s)(dqkv, dq, dk, dv, N, T, NH, hs);
    kernelCheck();
}

// ------------------------------------------------------------------ forward / loss / backward
#define GRID(n) ((unsigned)ceil_div((long)(n), BS))
#define WARP_GRID(rows) ((unsigned)ceil_div((long)(rows) * 32, BS))

// Inputs a->xp (n,T,P), a->t (n), a->y (n) must be on the device.  Output: a->out (n,T,P).
static void dit_forward(DiT* m, Acts* a, int n) {
    const DiTConfig* c = &m->cfg;
    const int C = c->hidden, T = c->T, P = c->P, F = c->freq_dim, L = c->depth, NH = c->heads, H = c->mlp_ratio * C;
    const long NT = (long)n * T, NTC = NT * C;
    const int ms = L * 6 * C;
    if (n > a->N || L > a->L) {
        fprintf(stderr, "dit_forward: batch %d / depth %d exceed activation capacity %d / %d\n", n, L, a->N, a->L);
        exit(EXIT_FAILURE);
    }
    ParamTensors* p = &m->p;
    cudaStream_t s = g_ctx.stream;

    LAUNCH(k_timestep_embed, GRID(n * (F / 2)), BS, 0, s)(a->tfreq, a->t, n, F, c->t_scale);
    matmul_forward(a->t_h, a->tfreq, p->t_w1, p->t_b1, n, F, C);
    LAUNCH(k_silu, GRID(n * C), BS, 0, s)(a->t_a, a->t_h, (long)n * C);
    matmul_forward(a->c, a->t_a, p->t_w2, p->t_b2, n, C, C);
    LAUNCH(k_add_class_emb, GRID(n * C), BS, 0, s)(a->c, p->y_emb, a->y, n, C);
    LAUNCH(k_silu, GRID(n * C), BS, 0, s)(a->sc, a->c, (long)n * C);
    matmul_forward(a->mod, a->sc, p->ada_w, p->ada_b, n, C, ms);  // all layers' adaLN in one GEMM
    matmul_forward(a->fmod, a->sc, p->f_ada_w, p->f_ada_b, n, C, 2 * C);
    kernelCheck();

    matmul_forward(a->x, a->xp, p->x_emb_w, p->x_emb_b, NT, P, C);
    LAUNCH(k_add_pos, GRID(NTC), BS, 0, s)(a->x, m->pos, n, (long)T * C);
    for (int l = 0; l < L; l++) {
        const float* mod = a->mod + (long)l * 6 * C;  // chunk k of layer l: mod + k*C, row stride ms
        float* ln1 = a->ln1 + l * NTC;
        float* rstd1 = a->rstd1 + l * NT;
        float* qkvr = a->qkvr + l * 3 * NTC;
        float* att = a->att + (long)l * n * NH * T * T;
        float* atty = a->atty + l * NTC;
        float* ao = a->ao + l * NTC;
        float* ln2 = a->ln2 + l * NTC;
        float* rstd2 = a->rstd2 + l * NT;
        float* h = a->h + l * NT * H;
        float* fo = a->fo + l * NTC;

        LAUNCH(k_layernorm_forward, WARP_GRID(NT), BS, 0, s)(ln1, rstd1, a->x, NT, C, LN_EPS);
        LAUNCH(k_modulate, GRID(NTC), BS, 0, s)(a->buf_m, ln1, mod + 0 * C, mod + 1 * C, n, T, C, ms);
        matmul_forward(a->buf_qkv, a->buf_m, p->qkv_w + (long)l * 3 * C * C, p->qkv_b + (long)l * 3 * C, NT, C, 3 * C);
        attention_forward(atty, qkvr, att, a->buf_y, a->buf_qkv, n, T, C, NH);
        matmul_forward(ao, atty, p->proj_w + (long)l * C * C, p->proj_b + (long)l * C, NT, C, C);
        LAUNCH(k_gate_residual, GRID(NTC), BS, 0, s)(a->x, ao, mod + 2 * C, n, T, C, ms);

        LAUNCH(k_layernorm_forward, WARP_GRID(NT), BS, 0, s)(ln2, rstd2, a->x, NT, C, LN_EPS);
        LAUNCH(k_modulate, GRID(NTC), BS, 0, s)(a->buf_m, ln2, mod + 3 * C, mod + 4 * C, n, T, C, ms);
        matmul_forward(h, a->buf_m, p->fc1_w + (long)l * H * C, p->fc1_b + (long)l * H, NT, C, H);
        LAUNCH(k_gelu, GRID(NT * H), BS, 0, s)(a->buf_g, h, NT * H);
        matmul_forward(fo, a->buf_g, p->fc2_w + (long)l * C * H, p->fc2_b + (long)l * C, NT, H, C);
        LAUNCH(k_gate_residual, GRID(NTC), BS, 0, s)(a->x, fo, mod + 5 * C, n, T, C, ms);
        kernelCheck();
    }
    LAUNCH(k_layernorm_forward, WARP_GRID(NT), BS, 0, s)(a->lnf, a->rstdf, a->x, NT, C, LN_EPS);
    LAUNCH(k_modulate, GRID(NTC), BS, 0, s)(a->buf_m, a->lnf, a->fmod, a->fmod + C, n, T, C, 2 * C);
    matmul_forward(a->out, a->buf_m, p->f_out_w, p->f_out_b, NT, C, P);
    kernelCheck();
}

// MSE flow-matching loss.  Writes a->dout = grad_scale * d(mean loss)/d(out) and adds
// loss_scale * mean loss to acc[0] (default a->loss_acc; no host sync).
static void dit_loss(DiT* m, Acts* a, int n, float grad_scale, float loss_scale, float* acc = NULL) {
    if (!acc) acc = a->loss_acc;
    long tot = (long)n * m->cfg.T * m->cfg.P;
    int parts = (int)ceil_div(tot, BS);
    LAUNCH(k_mse_loss, parts, BS, 0, g_ctx.stream)(a->dout, a->loss_partial, a->out, a->target, tot, grad_scale / (float)tot);
    LAUNCH(k_reduce_sum_into, 1, 1024, 0, g_ctx.stream)(acc, a->loss_partial, parts, loss_scale / (float)tot);
    kernelCheck();
}

// Accumulates parameter gradients (+=) into m->grads.  Requires dit_forward + dit_loss on the same Acts.
static void dit_backward(DiT* m, Acts* a, int n) {
    const DiTConfig* c = &m->cfg;
    const int C = c->hidden, T = c->T, P = c->P, F = c->freq_dim, L = c->depth, NH = c->heads, H = c->mlp_ratio * C;
    const int NC = c->num_classes;
    const long NT = (long)n * T, NTC = NT * C;
    const int ms = L * 6 * C;
    ParamTensors *p = &m->p, *g = &m->g;
    cudaStream_t s = g_ctx.stream;

    // final layer: out = Linear(modulate(LN(x), shift_f, scale_f))
    LAUNCH(k_modulate, GRID(NTC), BS, 0, s)(a->buf_m, a->lnf, a->fmod, a->fmod + C, n, T, C, 2 * C);  // recompute
    matmul_backward(a->dm, g->f_out_w, g->f_out_b, a->dout, a->buf_m, p->f_out_w, NT, C, P, 0);
    LAUNCH(k_modulate_backward, GRID(n * C), BS, 0, s)(a->dln, a->dfmod, a->dfmod + C, a->dm, a->lnf, a->fmod + C, n, T, C, 2 * C);
    cudaCheck(cudaMemsetAsync(a->dres, 0, NTC * sizeof(float), s));
    LAUNCH(k_layernorm_backward, WARP_GRID(NT), BS, 0, s)(a->dres, a->dln, a->lnf, a->rstdf, NT, C);
    kernelCheck();

    for (int l = L - 1; l >= 0; l--) {
        const float* mod = a->mod + (long)l * 6 * C;
        float* dmod = a->dmod + (long)l * 6 * C;
        const float* ln1 = a->ln1 + l * NTC;
        const float* rstd1 = a->rstd1 + l * NT;
        const float* qkvr = a->qkvr + l * 3 * NTC;
        const float* att = a->att + (long)l * n * NH * T * T;
        const float* atty = a->atty + l * NTC;
        const float* ao = a->ao + l * NTC;
        const float* ln2 = a->ln2 + l * NTC;
        const float* rstd2 = a->rstd2 + l * NT;
        const float* h = a->h + l * NT * H;
        const float* fo = a->fo + l * NTC;

        // MLP branch: x_out = x_mid + gate2 * fc2(gelu(fc1(modulate(LN(x_mid)))))
        LAUNCH(k_gate_backward, GRID(n * C), BS, 0, s)(a->dbranch, dmod + 5 * C, a->dres, fo, mod + 5 * C, n, T, C, ms);
        LAUNCH(k_gelu, GRID(NT * H), BS, 0, s)(a->buf_g, h, NT * H);  // recompute
        matmul_backward(a->dh, g->fc2_w + (long)l * C * H, g->fc2_b + (long)l * C, a->dbranch, a->buf_g,
                        p->fc2_w + (long)l * C * H, NT, H, C, 0);
        LAUNCH(k_gelu_backward, GRID(NT * H), BS, 0, s)(a->dh, h, a->dh, NT * H);
        LAUNCH(k_modulate, GRID(NTC), BS, 0, s)(a->buf_m, ln2, mod + 3 * C, mod + 4 * C, n, T, C, ms);  // recompute
        matmul_backward(a->dm, g->fc1_w + (long)l * H * C, g->fc1_b + (long)l * H, a->dh, a->buf_m,
                        p->fc1_w + (long)l * H * C, NT, C, H, 0);
        LAUNCH(k_modulate_backward, GRID(n * C), BS, 0, s)(a->dln, dmod + 3 * C, dmod + 4 * C, a->dm, ln2, mod + 4 * C, n, T, C, ms);
        LAUNCH(k_layernorm_backward, WARP_GRID(NT), BS, 0, s)(a->dres, a->dln, ln2, rstd2, NT, C);
        kernelCheck();

        // attention branch: x_mid = x_in + gate1 * proj(attn(modulate(LN(x_in))))
        LAUNCH(k_gate_backward, GRID(n * C), BS, 0, s)(a->dbranch, dmod + 2 * C, a->dres, ao, mod + 2 * C, n, T, C, ms);
        matmul_backward(a->datty, g->proj_w + (long)l * C * C, g->proj_b + (long)l * C, a->dbranch, atty,
                        p->proj_w + (long)l * C * C, NT, C, C, 0);
        attention_backward(a->buf_qkv, a->dqkvr, a->datt, a->buf_y, a->datty, qkvr, att, n, T, C, NH);
        LAUNCH(k_modulate, GRID(NTC), BS, 0, s)(a->buf_m, ln1, mod + 0 * C, mod + 1 * C, n, T, C, ms);  // recompute
        matmul_backward(a->dm, g->qkv_w + (long)l * 3 * C * C, g->qkv_b + (long)l * 3 * C, a->buf_qkv, a->buf_m,
                        p->qkv_w + (long)l * 3 * C * C, NT, C, 3 * C, 0);
        LAUNCH(k_modulate_backward, GRID(n * C), BS, 0, s)(a->dln, dmod + 0 * C, dmod + 1 * C, a->dm, ln1, mod + 1 * C, n, T, C, ms);
        LAUNCH(k_layernorm_backward, WARP_GRID(NT), BS, 0, s)(a->dres, a->dln, ln1, rstd1, NT, C);
        kernelCheck();
    }
    // patch embedding (input is data: no input gradient)
    matmul_backward(NULL, g->x_emb_w, g->x_emb_b, a->dres, a->xp, p->x_emb_w, NT, P, C, 0);
    // conditioning: every adaLN linear reads sc = SiLU(c)
    matmul_backward(a->dsc, g->ada_w, g->ada_b, a->dmod, a->sc, p->ada_w, n, C, ms, 0);
    matmul_backward(a->dsc, g->f_ada_w, g->f_ada_b, a->dfmod, a->sc, p->f_ada_w, n, C, 2 * C, 1);
    LAUNCH(k_silu_backward, GRID(n * C), BS, 0, s)(a->dc, a->dsc, a->c, (long)n * C);
    LAUNCH(k_class_emb_backward, GRID(NC * C), BS, 0, s)(g->y_emb, a->dc, a->y, n, C, NC);
    matmul_backward(a->dt_a, g->t_w2, g->t_b2, a->dc, a->t_a, p->t_w2, n, C, C, 0);
    LAUNCH(k_silu_backward, GRID(n * C), BS, 0, s)(a->dt_a, a->dt_a, a->t_h, (long)n * C);
    matmul_backward(NULL, g->t_w1, g->t_b1, a->dt_a, a->tfreq, p->t_w1, n, F, C, 0);
    kernelCheck();
}

static void dit_zero_grad(DiT* m) { cudaCheck(cudaMemsetAsync(m->grads, 0, m->num_params * sizeof(float), g_ctx.stream)); }

static void dit_adamw(DiT* m, float lr, float beta1, float beta2, float eps, float wd) {
    m->step++;
    double bc1 = 1.0 - pow((double)beta1, m->step), bc2 = 1.0 - pow((double)beta2, m->step);
    long n = (long)m->num_params;
    LAUNCH(k_adamw, GRID(n), BS, 0, g_ctx.stream)(m->params, m->m, m->v, m->grads, n, lr, beta1, beta2, eps, wd,
                                                  (float)(lr / bc1), (float)sqrt(bc2));
    kernelCheck();
}
static void dit_ema_update(DiT* m, float decay) {
    long n = (long)m->num_params;
    LAUNCH(k_ema, GRID(n), BS, 0, g_ctx.stream)(m->ema, m->params, n, decay);
    kernelCheck();
}
// ------------------------------------------------------------------ checkpoints (format: python/blockflow/ckpt.py)
#define CKPT_MAGIC 4180418
#define CKPT_VERSION 2
#define CKPT_FLAG_EMA 1

// Writes the weights and, for a training model, the EMA copy.
static void dit_save(const DiT* m, const char* path) {
    int hdr[64] = {0};
    double dh[16] = {0};
    const DiTConfig* c = &m->cfg;
    int vals[] = {CKPT_MAGIC, CKPT_VERSION, c->img_size, c->in_ch,  c->patch,     c->hidden,   c->depth,
                  c->heads,   c->mlp_ratio, c->num_classes, c->freq_dim, m->block_id, m->num_blocks, m->step,
                  m->ema ? CKPT_FLAG_EMA : 0, (int)(m->num_params & 0x7FFFFFFF), (int)(m->num_params >> 31)};
    memcpy(hdr, vals, sizeof(vals));
    double dv[] = {m->t_lo, m->t_hi, m->q_lo, m->q_hi, m->ln_mean, m->ln_std, (double)c->t_scale};
    memcpy(dh, dv, sizeof(dv));
    size_t bytes = m->num_params * sizeof(float);
    float* buf = (float*)malloc_check(bytes);
    FILE* f = fopen_check(path, "wb");
    fwrite_check(hdr, sizeof(int), 64, f);
    fwrite_check(dh, sizeof(double), 16, f);
    const float* srcs[2] = {m->params, m->ema};
    for (int i = 0; i < 2; i++) {
        if (!srcs[i]) continue;
        cudaCheck(cudaMemcpy(buf, srcs[i], bytes, cudaMemcpyDeviceToHost));
        fwrite_check(buf, 1, bytes, f);
    }
    fclose(f);
    free(buf);
}

// Reads a checkpoint header: model config, block id, number of blocks, logit-normal partition.
static void ckpt_read_header(const char* path, DiTConfig* c, int* block_id, int* num_blocks, double* ln_mean,
                             double* ln_std) {
    int hdr[64];
    double dh[16];
    FILE* f = fopen_check(path, "rb");
    fread_check(hdr, sizeof(int), 64, f, "checkpoint header");
    fread_check(dh, sizeof(double), 16, f, "checkpoint header");
    fclose(f);
    if (hdr[0] != CKPT_MAGIC || hdr[1] != CKPT_VERSION) {
        fprintf(stderr, "%s: not a BlockFlow v%d checkpoint\n", path, CKPT_VERSION);
        exit(EXIT_FAILURE);
    }
    memset(c, 0, sizeof(*c));
    c->img_size = hdr[2]; c->in_ch = hdr[3]; c->patch = hdr[4]; c->hidden = hdr[5]; c->depth = hdr[6];
    c->heads = hdr[7]; c->mlp_ratio = hdr[8]; c->num_classes = hdr[9]; c->freq_dim = hdr[10];
    c->t_scale = (float)dh[6];
    config_finalize(c);
    *block_id = hdr[11];
    *num_blocks = hdr[12];
    *ln_mean = dh[4];
    *ln_std = dh[5];
}

// Loads the EMA weights (or the raw weights if there is no EMA copy) into a model with a matching config.
static void dit_load(DiT* m, const char* path) {
    int hdr[64];
    double dh[16];
    FILE* f = fopen_check(path, "rb");
    fread_check(hdr, sizeof(int), 64, f, "checkpoint header");
    fread_check(dh, sizeof(double), 16, f, "checkpoint header");
    size_t n = (size_t)(unsigned)hdr[15] | ((size_t)(unsigned)hdr[16] << 31);
    if (hdr[0] != CKPT_MAGIC || n != m->num_params) {
        fprintf(stderr, "%s: checkpoint does not match the model (%zu vs %zu params)\n", path, n, m->num_params);
        exit(EXIT_FAILURE);
    }
    size_t bytes = n * sizeof(float);
    float* buf = (float*)malloc_check(bytes);
    fread_check(buf, 1, bytes, f, "params");
    if (hdr[14] & CKPT_FLAG_EMA) fread_check(buf, 1, bytes, f, "ema");
    cudaCheck(cudaMemcpy(m->params, buf, bytes, cudaMemcpyHostToDevice));
    fclose(f);
    free(buf);
}

// ------------------------------------------------------------------ block-dispatched Euler sampler
// x: (n, T, P) on the device, noise at t = 1 on entry and the sample at t = 0 on exit.
// Step k evaluates only models[block_of(t_k)].
static void euler_sample(DiT** models, int B, const double* edges, Acts* a, float* x, const int* y_host, int n,
                         int steps, int* block_trace) {
    const DiTConfig* c = &models[0]->cfg;
    long tot = (long)n * c->T * c->P;
    float* th = (float*)malloc_check(n * sizeof(float));
    cudaStream_t s = g_ctx.stream;
    cudaCheck(cudaMemcpyAsync(a->y, y_host, n * sizeof(int), cudaMemcpyHostToDevice, s));
    for (int k = 0; k < steps; k++) {
        double t = 1.0 - (double)k / steps, tn = 1.0 - (double)(k + 1) / steps;
        int b = block_of(t, edges, B);
        if (block_trace) block_trace[k] = b;
        for (int i = 0; i < n; i++) th[i] = (float)t;
        cudaCheck(cudaMemcpyAsync(a->t, th, n * sizeof(float), cudaMemcpyHostToDevice, s));
        cudaCheck(cudaMemcpyAsync(a->xp, x, tot * sizeof(float), cudaMemcpyDeviceToDevice, s));
        dit_forward(models[b], a, n);
        LAUNCH(k_axpy, GRID(tot), BS, 0, s)(x, a->out, (float)(tn - t), tot);
        kernelCheck();
        cudaCheck(cudaStreamSynchronize(s));  // th is reused next step
    }
    free(th);
}
