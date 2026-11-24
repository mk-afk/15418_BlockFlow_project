// train.cu - hybrid block/data-parallel flow-matching training of DiT-S/2.
//
//   ./train --data data/cifar10_train.bin                                        1 GPU, end-to-end
//   ./train --data data/cifar10_train.bin --blocks 2                             1 GPU, 2 blocks in turn
//   mpirun -np 2 --host a,b ./train --data data/cifar10_train.bin --blocks 2     2 machines, one block on each
//   mpirun -np 4 ./train --data data/cifar10_train.bin --blocks 2 --replicas 2   4 GPUs: 2 blocks x 2 replicas
//
// Options: --blocks B (1), --replicas D (ranks / B), --batch images per block per step (512),
//          --micro images per GPU per micro-batch (32: the full model then uses 3.8 GB of an 8 GB GPU),
//          --steps (1000), --out DIR (runs/default).
// Writes <out>/block<b>.bin at the end and <out>/step<S>/block<b>.bin every SNAPSHOT_EVERY steps
// (readable by python/blockflow/ckpt.py), <out>/log.jsonl, and <out>/summary.json with the layout,
// parameter count, all-reduce volume and per-phase time per step.
#include "trainer.cuh"

// DiT-S/2: each of the B blocks gets MODEL_DEPTH / B layers.
#define MODEL_PATCH 2
#define MODEL_HIDDEN 384
#define MODEL_DEPTH 12
#define MODEL_HEADS 6
#define LOG_EVERY 50
#define SNAPSHOT_EVERY 5000
#define TIMING_WARMUP 5  // steps left out of summary.json

int main(int argc, char** argv) {
    Comm comm;
    comm_init(&comm, &argc, &argv);
    const int root = comm.world_rank == 0;
    const char *data_path = NULL, *out = "runs/default";
    int B = 1, D = -1, batch = 512, micro = 32, steps = 1000;
    for (int i = 1; i < argc; i += 2) {
        const char* k = argv[i];
        const char* v = i + 1 < argc ? argv[i + 1] : "";
        if (!strcmp(k, "--data")) data_path = v;
        else if (!strcmp(k, "--out")) out = v;
        else if (!strcmp(k, "--blocks")) B = atoi(v);
        else if (!strcmp(k, "--replicas")) D = atoi(v);
        else if (!strcmp(k, "--batch")) batch = atoi(v);
        else if (!strcmp(k, "--micro")) micro = atoi(v);
        else if (!strcmp(k, "--steps")) steps = atoi(v);
        else {
            if (root) fprintf(stderr, "unknown option %s\n", k);
            return 1;
        }
    }
    if (!data_path) {
        if (root) fprintf(stderr, "usage: train --data FILE [--blocks 1 --replicas P/B --batch 512 --micro 32 --steps 1000 --out DIR]\n");
        return 1;
    }
    if (D < 0) D = comm.world_size >= B ? comm.world_size / B : 1;
    if (MODEL_DEPTH % B) {
        if (root) fprintf(stderr, "%d layers cannot be split into %d blocks\n", MODEL_DEPTH, B);
        return 1;
    }
    if (comm_layout(&comm, B, D)) return 1;
    ctx_init();

    Dataset data;
    dataset_load(&data, data_path);
    DiTConfig cfg = {};
    cfg.img_size = data.H; cfg.in_ch = data.C; cfg.num_classes = data.num_classes;
    cfg.patch = MODEL_PATCH; cfg.hidden = MODEL_HIDDEN; cfg.depth = MODEL_DEPTH / B; cfg.heads = MODEL_HEADS;
    cfg.mlp_ratio = 4; cfg.freq_dim = 256; cfg.t_scale = 1000.0f;
    Trainer tr;
    trainer_init(&tr, &comm, &data, &cfg, batch, micro);

    const char* backend = D == 1 ? "none" : COMM_BACKEND;
    const size_t np_block = tr.models[0].num_params;
    const double ar_bytes = ring_allreduce_bytes(np_block, D);
    cudaDeviceProp prop;
    cudaCheck(cudaGetDeviceProperties(&prop, comm.device));
    if (root) {
        printf("BlockFlow train | %d ranks on %d GPU(s) of %d machine(s), %s | B=%d blocks x D=%d replicas (%d groups) | all-reduce: %s\n",
               comm.world_size, comm.gpus, comm.hosts, prop.name, B, D, comm.G, backend);
        if (comm.gpus < comm.world_size)
            printf("note: ranks share a GPU, so the step times do not measure a %d-GPU run\n", comm.world_size);
        printf("DiT-S/2 on %dx%dx%d images: %d layers per block, %zu params per block, %.1f MB all-reduce per rank per block step\n",
               cfg.img_size, cfg.img_size, cfg.in_ch, cfg.depth, np_block, ar_bytes / 1e6);
        printf("batch %d per block = %d replicas x %d micro x %d accumulation, %d steps\n", batch, D, micro, tr.accum, steps);
    }
    char path[1024];
    snprintf(path, sizeof(path), "mkdir -p '%s'", out);
    if (root && system(path) != 0) fprintf(stderr, "warning: could not create %s\n", out);
    comm_barrier_world(&comm);
    FILE* log = NULL;
    if (root) {
        snprintf(path, sizeof(path), "%s/log.jsonl", out);
        log = fopen_check(path, "w");
    }

    double t_start = wall_ms(), t_log = t_start;
    int last_log = 0;
    PhaseTimes warm = {};
    for (int step = 0; step < steps; step++) {
        if (step == TIMING_WARMUP) warm = tr.times;
        trainer_step(&tr, step);
        if ((step + 1) % LOG_EVERY == 0 || step + 1 == steps) {
            double loss[MAX_BLOCKS] = {0}, mine[MAX_BLOCKS];
            trainer_read_losses(&tr, step + 1 - last_log, mine);
            if (comm.replica == 0)
                for (int i = 0; i < tr.nblocks; i++) loss[tr.models[i].block_id] = mine[i];
            comm_sum_world(&comm, loss, B);
            double now = wall_ms(), ms = (now - t_log) / (step + 1 - last_log);
            if (root) {
                printf("step %6d | %8.1f ms/step | loss", step + 1, ms);
                fprintf(log, "{\"step\": %d, \"time_s\": %.3f, \"ms_per_step\": %.3f, \"loss\": [", step + 1,
                        (now - t_start) / 1000.0, ms);
                for (int b = 0; b < B; b++) {
                    printf(" b%d %.4f", b, loss[b]);
                    fprintf(log, "%s%.6f", b ? ", " : "", loss[b]);
                }
                printf("\n");
                fprintf(log, "]}\n");
                fflush(log);
                fflush(stdout);
            }
            t_log = now;
            last_log = step + 1;
        }
        if ((step + 1) % SNAPSHOT_EVERY == 0 || step + 1 == steps) {
            int final = step + 1 == steps;
            if (!final) {
                snprintf(path, sizeof(path), "mkdir -p '%s/step%d'", out, step + 1);
                if (root && system(path) != 0) fprintf(stderr, "warning: could not create a snapshot directory\n");
                comm_barrier_world(&comm);
            }
            if (comm.replica == 0)
                for (int i = 0; i < tr.nblocks; i++) {
                    if (final) snprintf(path, sizeof(path), "%s/block%d.bin", out, tr.models[i].block_id);
                    else snprintf(path, sizeof(path), "%s/step%d/block%d.bin", out, step + 1, tr.models[i].block_id);
                    dit_save(&tr.models[i], path);
                }
        }
    }
    cudaCheck(cudaDeviceSynchronize());

    // summary: ms per step after the warm-up, mean and max over ranks (the slowest rank sets the pace)
    PhaseTimes t = tr.times;
    long timed = t.steps - warm.steps;
    double ph[4] = {0, 0, 0, 0};
    if (timed > 0) {
        ph[0] = (t.compute - warm.compute) / timed;
        ph[1] = (t.allreduce - warm.allreduce) / timed;
        ph[2] = (t.optimizer - warm.optimizer) / timed;
        ph[3] = (t.step_wall - warm.step_wall) / timed;
    }
    double mean[4], mx[4];
    memcpy(mean, ph, sizeof(ph));
    memcpy(mx, ph, sizeof(ph));
    comm_sum_world(&comm, mean, 4);
    comm_max_world(&comm, mx, 4);
    size_t free_b = 0, total_b = 0;
    cudaCheck(cudaMemGetInfo(&free_b, &total_b));
    double mem[1] = {(double)(total_b - free_b) / 1e9};
    comm_max_world(&comm, mem, 1);
    if (root) {
        for (int i = 0; i < 4; i++) mean[i] /= comm.world_size;
        double ips = mx[3] > 0 ? 1000.0 * batch * B / mx[3] : 0.0;
        printf("ms per step over %ld steps (mean / max over ranks): compute %.1f / %.1f, all-reduce %.1f / %.1f, "
               "optimizer %.1f / %.1f, step %.1f / %.1f | %.0f images/s | %.2f GB in use\n",
               timed, mean[0], mx[0], mean[1], mx[1], mean[2], mx[2], mean[3], mx[3], ips, mem[0]);
        snprintf(path, sizeof(path), "%s/summary.json", out);
        FILE* f = fopen_check(path, "w");
        fprintf(f,
                "{\"device\": \"%s\", \"world_size\": %d, \"gpus\": %d, \"hosts\": %d, \"blocks\": %d, \"replicas\": %d, \"groups\": %d,\n"
                " \"backend\": \"%s\", \"params_per_block\": %zu, \"allreduce_bytes_per_rank_per_block_step\": %.0f,\n"
                " \"batch_per_block\": %d, \"micro_batch\": %d, \"accumulation\": %d, \"steps\": %d, \"timed_steps\": %ld,\n"
                " \"ms_mean\": {\"compute\": %.4f, \"allreduce\": %.4f, \"optimizer\": %.4f, \"step\": %.4f},\n"
                " \"ms_max\": {\"compute\": %.4f, \"allreduce\": %.4f, \"optimizer\": %.4f, \"step\": %.4f},\n"
                " \"images_per_second\": %.3f, \"device_memory_gb_max\": %.3f}\n",
                prop.name, comm.world_size, comm.gpus, comm.hosts, B, D, comm.G, backend, np_block, ar_bytes, batch, micro, tr.accum, steps, timed,
                mean[0], mean[1], mean[2], mean[3], mx[0], mx[1], mx[2], mx[3], ips, mem[0]);
        fclose(f);
        fclose(log);
    }
    trainer_free(&tr);
    dataset_free(&data);
    ctx_destroy();
    comm_finalize(&comm);
    return 0;
}
