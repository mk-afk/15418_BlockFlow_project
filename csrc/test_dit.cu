// test_dit.cu - checks the C/CUDA DiT against PyTorch (python/make_test_case.py).
//
//   ./test_dit tests/data/tiny.bin
//
// Checks the training batch, the forward output, the loss, every parameter gradient, a K-step AdamW loss
// curve, and two-block Euler sampling. Error per tensor: max|got - ref| / max|ref| and ||got - ref|| / ||ref||,
// both at most 1e-4. Exit code 0 iff everything passes.
#include "dit.cuh"

#define TOL 1e-4

static int g_checks = 0, g_failed = 0;
static double g_worst = 0.0;

static void check(const char* name, const float* got, const float* ref, long n) {
    double maxd = 0, maxr = 0, d2 = 0, r2 = 0;
    bool finite = true;
    for (long i = 0; i < n; i++) {
        finite &= std::isfinite(got[i]);
        double d = fabs((double)got[i] - ref[i]);
        maxd = fmax(maxd, d);
        maxr = fmax(maxr, fabs((double)ref[i]));
        d2 += d * d;
        r2 += (double)ref[i] * ref[i];
    }
    double rel = maxd / fmax(maxr, 1e-30), rel2 = sqrt(d2) / fmax(sqrt(r2), 1e-30);
    bool ok = finite && rel <= TOL && rel2 <= TOL;
    g_checks++;
    g_failed += !ok;
    if (finite) g_worst = fmax(g_worst, rel);
    printf("  %-22s rel %.2e  relL2 %.2e  %s\n", name, rel, rel2, !finite ? "NON-FINITE" : (ok ? "ok" : "FAIL"));
}

static void download(float* host, const float* dev, long n) {
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaMemcpy(host, dev, n * sizeof(float), cudaMemcpyDeviceToHost));
}
static void upload(void* dev, const void* host, size_t bytes) { cudaCheck(cudaMemcpy(dev, host, bytes, cudaMemcpyHostToDevice)); }

template <typename T>
static std::vector<T> read_vec(FILE* f, long n) {
    std::vector<T> v(n);
    fread_check(v.data(), sizeof(T), n, f, "test case");
    return v;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s testcase.bin\n", argv[0]);
        return 2;
    }
    FILE* f = fopen_check(argv[1], "rb");
    std::vector<int> h = read_vec<int>(f, 32);
    std::vector<float> fh = read_vec<float>(f, 16);
    if (h[0] != 4180777 || h[1] != 2) {
        fprintf(stderr, "%s: not a v2 test case (run make testdata)\n", argv[1]);
        return 2;
    }
    DiTConfig cfg = {};
    cfg.img_size = h[2]; cfg.in_ch = h[3]; cfg.patch = h[4]; cfg.hidden = h[5]; cfg.depth = h[6]; cfg.heads = h[7];
    cfg.mlp_ratio = h[8]; cfg.num_classes = h[9]; cfg.freq_dim = h[10]; cfg.t_scale = fh[0];
    config_finalize(&cfg);
    const int N = h[11], K = h[13], Bs = h[14], Ns = h[15], Ks = h[16], S = cfg.img_size, Cin = cfg.in_ch;
    const long npar = h[12], img = (long)N * Cin * S * S, tok = (long)N * cfg.T * cfg.P;
    printf("test_dit: %s (%dx%dx%d images, %d tokens, hidden %d, %d heads, depth %d, batch %d)\n", argv[1], S, S, Cin,
           cfg.T, cfg.hidden, cfg.heads, cfg.depth, N);

    std::vector<float> params0 = read_vec<float>(f, npar);
    std::vector<unsigned char> images = read_vec<unsigned char>(f, (img + 3) / 4 * 4);
    std::vector<int> idx = read_vec<int>(f, N), y = read_vec<int>(f, N);
    std::vector<float> t = read_vec<float>(f, N), noise = read_vec<float>(f, img);
    std::vector<float> xp_ref = read_vec<float>(f, tok), target_ref = read_vec<float>(f, tok);
    std::vector<float> out_ref = read_vec<float>(f, tok), loss_ref = read_vec<float>(f, 1);
    std::vector<float> grads_ref = read_vec<float>(f, npar), losses_ref = read_vec<float>(f, K);
    std::vector<float> block_params = read_vec<float>(f, (long)Bs * npar);
    const long s_img = (long)Ns * Cin * S * S, s_tok = (long)Ns * cfg.T * cfg.P;
    std::vector<float> x1 = read_vec<float>(f, s_img);
    std::vector<int> ys = read_vec<int>(f, Ns);
    std::vector<float> xs_ref = read_vec<float>(f, s_img);
    std::vector<int> trace_ref = read_vec<int>(f, Ks);
    fclose(f);

    ctx_init();
    DiT model;
    dit_build(&model, &cfg, 0, 1, 0.0, 1.0, 1);
    if ((long)model.num_params != npar) {
        printf("FAIL: %zu parameters in C, %ld in PyTorch\n", model.num_params, npar);
        return 1;
    }
    upload(model.params, params0.data(), npar * sizeof(float));
    Acts a;
    acts_alloc(&a, &cfg, N > Ns ? N : Ns);
    std::vector<float> buf(npar > tok ? npar : tok);

    printf("[1] training batch\n");
    unsigned char* d_images;
    int* d_idx;
    float* d_noise;
    cudaCheck(cudaMalloc(&d_images, images.size()));
    cudaCheck(cudaMalloc(&d_idx, N * sizeof(int)));
    cudaCheck(cudaMalloc(&d_noise, img * sizeof(float)));
    upload(d_images, images.data(), images.size());
    upload(d_idx, idx.data(), N * sizeof(int));
    upload(d_noise, noise.data(), img * sizeof(float));
    upload(a.t, t.data(), N * sizeof(float));
    LAUNCH(k_make_batch, GRID(tok), BS, 0, g_ctx.stream)(a.xp, a.target, d_images, d_idx, a.t, d_noise, 0ULL, N, Cin,
                                                        S, S, cfg.patch);
    download(buf.data(), a.xp, tok);
    check("x_t", buf.data(), xp_ref.data(), tok);
    download(buf.data(), a.target, tok);
    check("v", buf.data(), target_ref.data(), tok);

    printf("[2] forward, loss, backward\n");
    upload(a.xp, xp_ref.data(), tok * sizeof(float));
    upload(a.target, target_ref.data(), tok * sizeof(float));
    upload(a.y, y.data(), N * sizeof(int));
    dit_forward(&model, &a, N);
    download(buf.data(), a.out, tok);
    check("velocity output", buf.data(), out_ref.data(), tok);
    cudaCheck(cudaMemset(a.loss_acc, 0, sizeof(float)));
    dit_loss(&model, &a, N, 1.0f, 1.0f);
    float loss;
    download(&loss, a.loss_acc, 1);
    check("loss", &loss, loss_ref.data(), 1);
    dit_zero_grad(&model);
    dit_backward(&model, &a, N);
    download(buf.data(), model.grads, npar);
    const float *g = buf.data(), *r = grads_ref.data();
    for (int i = 0; i < NUM_PARAM_TENSORS; i++) {
        char name[64];
        snprintf(name, sizeof(name), "d%s", PARAM_NAMES[i]);
        check(name, g, r, (long)model.sizes[i]);
        g += model.sizes[i];
        r += model.sizes[i];
    }

    printf("[3] %d AdamW steps\n", K);
    upload(model.params, params0.data(), npar * sizeof(float));
    std::vector<float> losses(K);
    for (int k = 0; k < K; k++) {
        dit_zero_grad(&model);
        cudaCheck(cudaMemset(a.loss_acc, 0, sizeof(float)));
        dit_forward(&model, &a, N);
        dit_loss(&model, &a, N, 1.0f, 1.0f);
        dit_backward(&model, &a, N);
        dit_adamw(&model, fh[1], fh[2], fh[3], fh[4], fh[5]);
        download(&losses[k], a.loss_acc, 1);
    }
    check("loss curve", losses.data(), losses_ref.data(), K);

    printf("[4] Euler sampling with %d blocks, %d steps\n", Bs, Ks);
    std::vector<DiT> blocks(Bs);
    std::vector<DiT*> bp(Bs);
    for (int b = 0; b < Bs; b++) {
        dit_build(&blocks[b], &cfg, b, Bs, 0.0, 1.0, 0);
        upload(blocks[b].params, block_params.data() + (long)b * npar, npar * sizeof(float));
        bp[b] = &blocks[b];
    }
    std::vector<double> edges(Bs + 1);
    partition_edges(Bs, 0.0, 1.0, edges.data());
    float *d_img, *d_x;
    cudaCheck(cudaMalloc(&d_img, s_img * sizeof(float)));
    cudaCheck(cudaMalloc(&d_x, s_tok * sizeof(float)));
    upload(d_img, x1.data(), s_img * sizeof(float));
    LAUNCH(k_patchify, GRID(s_tok), BS, 0, g_ctx.stream)(d_x, d_img, Ns, Cin, S, S, cfg.patch);
    std::vector<int> trace(Ks);
    euler_sample(bp.data(), Bs, edges.data(), &a, d_x, ys.data(), Ns, Ks, trace.data());
    LAUNCH(k_unpatchify, GRID(s_tok), BS, 0, g_ctx.stream)(d_img, d_x, Ns, Cin, S, S, cfg.patch);
    std::vector<float> xs(s_img);
    download(xs.data(), d_img, s_img);
    check("sample", xs.data(), xs_ref.data(), s_img);
    g_checks++;
    g_failed += trace != trace_ref;
    printf("  %-22s %s\n", "block per step", trace == trace_ref ? "ok" : "FAIL");

    for (int b = 0; b < Bs; b++) dit_free(&blocks[b]);
    cudaCheck(cudaFree(d_img));
    cudaCheck(cudaFree(d_x));
    cudaCheck(cudaFree(d_images));
    cudaCheck(cudaFree(d_idx));
    cudaCheck(cudaFree(d_noise));
    acts_free(&a);
    dit_free(&model);
    ctx_destroy();
    printf("%s: %d/%d checks passed, worst error %.2e (tolerance %.0e)\n", g_failed ? "FAILED" : "PASSED",
           g_checks - g_failed, g_checks, g_worst, TOL);
    return g_failed ? 1 : 0;
}
