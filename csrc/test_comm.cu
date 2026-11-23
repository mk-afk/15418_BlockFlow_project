// test_comm.cu - data-parallel equivalence of the training step (run with several ranks):
//
//   mpirun -np 4 ./test_comm                  4 GPUs on one node, or 4 ranks sharing one GPU
//   mpirun -np 2 --host a,b ./test_comm       one GPU on each of two machines
//
// For each layout, D replicas train shards of one fixed global batch and are compared with a single
// process that trains the whole batch with two accumulated micro-batches: gradients after the all-reduce,
// loss, and weights after 3 steps. Replicas of a block must stay bitwise identical and different blocks
// must differ. Exit code 0 iff every rank passes.
#include "trainer.cuh"

static int g_fail = 0;

static double rel_err(const std::vector<float>& a, const std::vector<float>& b) {
    double md = 0, mr = 0;
    for (size_t i = 0; i < a.size(); i++) {
        if (!std::isfinite(a[i])) return INFINITY;
        md = fmax(md, fabs((double)a[i] - b[i]));
        mr = fmax(mr, fabs((double)b[i]));
    }
    return md / fmax(mr, 1e-30);
}
static uint64_t fnv1a(const void* p, size_t n) {
    const unsigned char* c = (const unsigned char*)p;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) h = (h ^ c[i]) * 1099511628211ULL;
    return h;
}
static std::vector<float> fetch(const float* dev, size_t n) {
    std::vector<float> v(n);
    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaMemcpy(v.data(), dev, n * sizeof(float), cudaMemcpyDeviceToHost));
    return v;
}

static void test_layout(Comm* base, int B, int D, Dataset* data) {
    Comm c = *base;
    if (comm_layout(&c, B, D)) {
        g_fail++;
        return;
    }
    DiTConfig cfg = {};
    cfg.img_size = data->H; cfg.in_ch = data->C; cfg.patch = 2; cfg.hidden = 32; cfg.depth = 2; cfg.heads = 2;
    cfg.mlp_ratio = 4; cfg.num_classes = data->num_classes; cfg.freq_dim = 16; cfg.t_scale = 1000.0f;
    const int GB = (8 % c.world_size == 0) ? 8 : 2 * c.world_size, STEPS = 3, block = c.owned[0];
    Trainer dp, ref;
    trainer_init(&dp, &c, data, &cfg, GB, GB / D);  // one micro-batch per replica
    Comm local;
    comm_local(&local, B, block);
    trainer_init(&ref, &local, data, &cfg, GB, GB / 2);  // whole batch in one process, 2 micro-batches

    // one fixed global batch for this block, identical on every rank
    std::vector<int> idx(GB), y(GB);
    std::vector<float> t(GB);
    Rng rng = {mix_seed(5, block)};
    for (int j = 0; j < GB; j++) {
        idx[j] = (int)(rng_next(&rng) % data->N);
        y[j] = data->labels[idx[j]];
        t[j] = sample_block_time(&rng, dp.models[0].q_lo, dp.models[0].q_hi, 0.0, 1.0);
    }
    long nimg = (long)GB * data->C * data->H * data->W;
    float* d_noise;
    cudaCheck(cudaMalloc(&d_noise, nimg * sizeof(float)));
    LAUNCH(k_gaussian_fill, ceil_div(nimg, 256), 256, 0, g_ctx.stream)(d_noise, nimg, mix_seed(9, block));
    BatchOverride ob = {idx.data(), y.data(), t.data(), d_noise};
    dp.override_batch = ref.override_batch = &ob;

    size_t np = dp.models[0].num_params;
    trainer_step(&dp, 0);
    trainer_step(&ref, 0);
    double eg = rel_err(fetch(dp.models[0].grads, np), fetch(ref.models[0].grads, np));
    double ld[MAX_BLOCKS], lr[MAX_BLOCKS];
    trainer_read_losses(&dp, 1, ld);
    trainer_read_losses(&ref, 1, lr);
    double el = fabs(ld[0] - lr[0]) / fabs(lr[0]);
    for (int s = 1; s < STEPS; s++) {
        trainer_step(&dp, s);
        trainer_step(&ref, s);
    }
    std::vector<float> w = fetch(dp.models[0].params, np);
    double ew = rel_err(w, fetch(ref.models[0].params, np));
    bool ok = eg <= 1e-5 && el <= 1e-5 && ew <= 1e-5;

    // replicas of a block identical, different blocks different
    uint64_t hsh = fnv1a(w.data(), np * sizeof(float));
    int r = c.world_rank, P = c.world_size;
    std::vector<double> all(3 * P, 0.0);
    all[3 * r] = (double)(hsh >> 32);
    all[3 * r + 1] = (double)(hsh & 0xFFFFFFFFULL);
    all[3 * r + 2] = block;
    comm_sum_world(&c, all.data(), 3 * P);
    bool same = true, differ = true;
    for (int q = 0; q < P; q++) {
        bool eq = all[3 * q] == all[3 * r] && all[3 * q + 1] == all[3 * r + 1];
        if ((int)all[3 * q + 2] == block) same &= eq;
        else differ &= !eq;
    }
    ok &= same && differ;
    g_fail += !ok;
    double worst[3] = {eg, el, ew}, nfail[1] = {ok ? 0.0 : 1.0};
    comm_max_world(&c, worst, 3);
    comm_sum_world(&c, nfail, 1);
    if (r == 0)
        printf("  B=%d D=%d: gradient %.1e | loss %.1e | weights after %d steps %.1e | replicas identical, blocks differ: %s\n",
               B, D, worst[0], worst[1], STEPS, worst[2], nfail[0] == 0 ? "ok" : "FAIL");
    cudaCheck(cudaFree(d_noise));
    trainer_free(&dp);
    trainer_free(&ref);
    comm_layout_free(&c);
}

int main(int argc, char** argv) {
    Comm base;
    comm_init(&base, &argc, &argv);
    ctx_init();
    int P = base.world_size;
    if (base.world_rank == 0)
        printf("test_comm: %d ranks on %d GPU(s) of %d machine(s), all-reduce: %s\n", P, base.gpus, base.hosts, COMM_BACKEND);

    // 64 random 8x8x3 images, the same on every rank
    Dataset data = {64, 3, 8, 8, 10, NULL, NULL};
    std::vector<unsigned char> pix(64 * 3 * 8 * 8);
    Rng rng = {4242};
    for (auto& v : pix) v = (unsigned char)(rng_next(&rng) & 255);
    data.labels = (unsigned char*)malloc_check(64);
    for (int i = 0; i < 64; i++) data.labels[i] = (unsigned char)(rng_next(&rng) % 10);
    cudaCheck(cudaMalloc(&data.d_images, pix.size()));
    cudaCheck(cudaMemcpy(data.d_images, pix.data(), pix.size(), cudaMemcpyHostToDevice));

    test_layout(&base, 1, P, &data);                          // pure data parallelism
    if (P % 2 == 0 && P >= 4) test_layout(&base, 2, P / 2, &data);  // hybrid
    if (P >= 2) test_layout(&base, P, 1, &data);              // blocks only

    dataset_free(&data);
    double f[1] = {(double)g_fail};
    comm_sum_world(&base, f, 1);
    if (base.world_rank == 0) printf("%s\n", f[0] == 0 ? "PASSED" : "FAILED");
    ctx_destroy();
    comm_finalize(&base);
    return f[0] == 0 ? 0 : 1;
}
