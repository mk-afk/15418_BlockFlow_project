// sample.cu - generates images from block checkpoints with the block-dispatched Euler sampler.
//
//   ./sample --ckpt runs/b2 --n 10000 --steps 50 --out runs/b2/samples.npy
//
// Uses the EMA weights of <ckpt>/block<b>.bin; labels cycle through the classes. Writes a uint8 .npy of
// shape (N, H, W, C) for python/fid_eval.py and prints which block runs each Euler step.
#include "dit.cuh"

#define SAMPLE_BATCH 64  // the activation buffers hold a full training batch: 64 DiT-S/2 images need about 5.5 GB
#define SAMPLE_SEED 0ULL

static void write_npy_u8(const char* path, const unsigned char* data, int N, int H, int W, int C) {
    char header[256];
    int len = snprintf(header, sizeof(header), "{'descr': '|u1', 'fortran_order': False, 'shape': (%d, %d, %d, %d), }",
                       N, H, W, C);
    int pad = (64 - (10 + len + 1) % 64) % 64;
    unsigned short hl = (unsigned short)(len + pad + 1);
    FILE* f = fopen_check(path, "wb");
    fwrite_check("\x93NUMPY\x01\x00", 1, 8, f);
    fwrite_check(&hl, 2, 1, f);
    fwrite_check(header, 1, len, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f);
    fwrite_check(data, 1, (size_t)N * H * W * C, f);
    fclose(f);
}

int main(int argc, char** argv) {
    const char *ckpt = NULL, *out = "samples.npy";
    int n_total = 10000, steps = 50;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--ckpt")) ckpt = argv[i + 1];
        else if (!strcmp(argv[i], "--out")) out = argv[i + 1];
        else if (!strcmp(argv[i], "--n")) n_total = atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--steps")) steps = atoi(argv[i + 1]);
        else {
            fprintf(stderr, "unknown option %s\n", argv[i]);
            return 1;
        }
    }
    if (!ckpt) {
        fprintf(stderr, "usage: sample --ckpt DIR [--n 10000 --steps 50 --out samples.npy]\n");
        return 1;
    }
    ctx_init();

    char path[1024];
    snprintf(path, sizeof(path), "%s/block0.bin", ckpt);
    DiTConfig cfg;
    int bid, B;
    double ln_mean, ln_std;
    ckpt_read_header(path, &cfg, &bid, &B, &ln_mean, &ln_std);
    std::vector<DiT> models(B);
    std::vector<DiT*> mp(B);
    for (int b = 0; b < B; b++) {
        snprintf(path, sizeof(path), "%s/block%d.bin", ckpt, b);
        dit_build(&models[b], &cfg, b, B, ln_mean, ln_std, 0);
        dit_load(&models[b], path);
        mp[b] = &models[b];
    }
    std::vector<double> edges(B + 1);
    partition_edges(B, ln_mean, ln_std, edges.data());

    const int batch = n_total < SAMPLE_BATCH ? n_total : SAMPLE_BATCH, S = cfg.img_size, Cin = cfg.in_ch;
    Acts a;
    acts_alloc(&a, &cfg, batch);
    float *d_x, *d_img;
    cudaCheck(cudaMalloc(&d_x, (size_t)batch * cfg.T * cfg.P * sizeof(float)));
    cudaCheck(cudaMalloc(&d_img, (size_t)batch * Cin * S * S * sizeof(float)));
    std::vector<float> h_img((size_t)batch * Cin * S * S);
    std::vector<unsigned char> u8((size_t)n_total * S * S * Cin);
    std::vector<int> labels(batch), trace(steps);
    double t0 = wall_ms();
    for (int start = 0; start < n_total; start += batch) {
        int n = n_total - start < batch ? n_total - start : batch;
        for (int j = 0; j < n; j++) labels[j] = (start + j) % cfg.num_classes;
        long ntok = (long)n * cfg.T * cfg.P;
        LAUNCH(k_gaussian_fill, GRID(ntok), BS, 0, g_ctx.stream)(d_x, ntok, mix_seed(SAMPLE_SEED, (uint64_t)start));
        euler_sample(mp.data(), B, edges.data(), &a, d_x, labels.data(), n, steps, trace.data());
        if (start == 0) {
            printf("%d blocks, edges", B);
            for (int b = 0; b <= B; b++) printf(" %.4f", edges[b]);
            printf(" | block per Euler step:");
            for (int k = 0; k < steps; k++) printf(" %d", trace[k]);
            printf("\n");
        }
        LAUNCH(k_unpatchify, GRID(ntok), BS, 0, g_ctx.stream)(d_img, d_x, n, Cin, S, S, cfg.patch);
        cudaCheck(cudaMemcpy(h_img.data(), d_img, (size_t)n * Cin * S * S * sizeof(float), cudaMemcpyDeviceToHost));
        for (int j = 0; j < n; j++)
            for (int c = 0; c < Cin; c++)
                for (int p = 0; p < S * S; p++) {
                    float v = (h_img[((size_t)j * Cin + c) * S * S + p] + 1.0f) * 127.5f + 0.5f;
                    u8[((size_t)(start + j) * S * S + p) * Cin + c] = (unsigned char)(v < 0 ? 0 : (v > 255 ? 255 : v));
                }
    }
    double secs = (wall_ms() - t0) / 1000.0;
    write_npy_u8(out, u8.data(), n_total, S, S, Cin);
    printf("wrote %s: %d images in %.1f s\n", out, n_total, secs);
    cudaCheck(cudaFree(d_x));
    cudaCheck(cudaFree(d_img));
    acts_free(&a);
    for (int b = 0; b < B; b++) dit_free(&models[b]);
    ctx_destroy();
    return 0;
}
