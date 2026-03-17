/*
 * bench_kv64_mixed.c — True mixed workload benchmark for simd_map64.h
 *
 * Interleaved GET/INSERT/DELETE in a single pipeline with per-op
 * prefetch dispatch. Sweeps N=1,2,4 × PF for 4 workload profiles.
 *
 * Build & run:
 *   cc -O3 -march=native -std=gnu11 -o bench_kv64_mix bench_kv64_mixed.c
 *   taskset -c 0 ./bench_kv64_mix > results.tsv
 */

#define SIMD_MAP_NAME          kv64_n1
#define SIMD_MAP64_VAL_WORDS    1
#define SIMD_MAP64_BLOCK_STRIDE 1
#include "simd_map64.h"

#define SIMD_MAP_NAME          kv64_n2
#define SIMD_MAP64_VAL_WORDS    1
#define SIMD_MAP64_BLOCK_STRIDE 2
#include "simd_map64.h"

#define SIMD_MAP_NAME          kv64_n4
#define SIMD_MAP64_VAL_WORDS    1
#define SIMD_MAP64_BLOCK_STRIDE 4
#include "simd_map64.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sys/mman.h>

#define OP_GET    0
#define OP_INSERT 1
#define OP_DELETE 2

/* ================================================================
 * xoshiro256** PRNG
 * ================================================================ */

static uint64_t rng_s[4];

static inline uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t xoshiro256ss(void) {
    uint64_t result = rotl(rng_s[1] * 5, 7) * 9;
    uint64_t t = rng_s[1] << 17;
    rng_s[2] ^= rng_s[0];
    rng_s[3] ^= rng_s[1];
    rng_s[1] ^= rng_s[2];
    rng_s[0] ^= rng_s[3];
    rng_s[2] ^= t;
    rng_s[3] = rotl(rng_s[3], 45);
    return result;
}

static void seed_rng(void) {
    uint64_t s = 0xdeadbeefcafe1234ULL;
    for (int i = 0; i < 4; i++) {
        s += 0x9e3779b97f4a7c15ULL;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        rng_s[i] = z ^ (z >> 31);
    }
}

/* ================================================================
 * Hugepage allocator
 * ================================================================ */

#define HP_ROUND(sz) (((sz) + (2u<<20) - 1) & ~((size_t)(2u<<20) - 1))

static void *hp_alloc(size_t bytes) {
    size_t total = HP_ROUND(bytes);
    void *p = mmap(NULL, total, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE,
                   -1, 0);
    if (p == MAP_FAILED) {
        p = mmap(NULL, total, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
        if (p != MAP_FAILED) madvise(p, total, MADV_HUGEPAGE);
    }
    return p;
}

static void hp_free(void *p, size_t bytes) {
    if (p) munmap(p, HP_ROUND(bytes));
}

/* ================================================================
 * Timing
 * ================================================================ */

static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ================================================================
 * Workload
 * ================================================================ */

struct work {
    uint64_t *keys;
    uint64_t *vals;
    uint32_t *op_idx;
    uint8_t  *op_type;
    uint64_t  pool_size;
    uint64_t  n_init;
    uint64_t  n_ops;
};

static struct work gen_insert(uint64_t n) {
    struct work w;
    w.pool_size = n;
    w.n_init = 0;
    w.n_ops = n;
    w.keys    = hp_alloc(n * sizeof(uint64_t));
    w.vals    = hp_alloc(n * sizeof(uint64_t));
    w.op_idx  = hp_alloc(n * sizeof(uint32_t));
    w.op_type = hp_alloc(n * sizeof(uint8_t));
    for (uint64_t i = 0; i < n; i++) {
        uint64_t k;
        do { k = xoshiro256ss(); } while (k == 0);
        w.keys[i] = k;
        w.vals[i] = xoshiro256ss();
        w.op_idx[i]  = (uint32_t)i;
        w.op_type[i] = OP_INSERT;
    }
    return w;
}

static struct work gen_mixed(uint64_t n_init, uint64_t n_ops,
                             int pct_r, int pct_i, int pct_d) {
    struct work w;
    uint64_t pool_size = n_init + n_ops;
    w.pool_size = pool_size;
    w.n_init    = n_init;
    w.n_ops     = n_ops;
    w.keys    = hp_alloc(pool_size * sizeof(uint64_t));
    w.vals    = hp_alloc(pool_size * sizeof(uint64_t));
    w.op_idx  = hp_alloc(n_ops * sizeof(uint32_t));
    w.op_type = hp_alloc(n_ops * sizeof(uint8_t));

    for (uint64_t i = 0; i < pool_size; i++) {
        uint64_t k;
        do { k = xoshiro256ss(); } while (k == 0);
        w.keys[i] = k;
        w.vals[i] = xoshiro256ss();
    }

    uint32_t *live_list = malloc(pool_size * sizeof(uint32_t));
    uint32_t live_count = (uint32_t)n_init;
    uint64_t fresh_cursor = n_init;
    for (uint32_t i = 0; i < (uint32_t)n_init; i++)
        live_list[i] = i;

    int thresh_i = pct_r + pct_i;
    (void)pct_d;

    for (uint64_t i = 0; i < n_ops; i++) {
        int roll = (int)(xoshiro256ss() % 100);
        int op;
        if (roll < pct_r) op = OP_GET;
        else if (roll < thresh_i) op = OP_INSERT;
        else op = OP_DELETE;

        if (op == OP_GET && live_count == 0)
            op = (fresh_cursor < pool_size) ? OP_INSERT : OP_GET;
        if (op == OP_INSERT && fresh_cursor >= pool_size)
            op = (live_count > 0) ? OP_GET : OP_GET;
        if (op == OP_DELETE && live_count == 0)
            op = (fresh_cursor < pool_size) ? OP_INSERT : OP_GET;

        switch (op) {
        case OP_GET:
            if (live_count > 0)
                w.op_idx[i] = live_list[(uint32_t)(xoshiro256ss() % live_count)];
            else
                w.op_idx[i] = (uint32_t)(xoshiro256ss() % pool_size);
            w.op_type[i] = OP_GET;
            break;
        case OP_INSERT: {
            uint32_t idx = (uint32_t)fresh_cursor++;
            w.op_idx[i]  = idx;
            w.op_type[i] = OP_INSERT;
            live_list[live_count++] = idx;
            break;
        }
        case OP_DELETE: {
            uint32_t li = (uint32_t)(xoshiro256ss() % live_count);
            w.op_idx[i]  = live_list[li];
            w.op_type[i] = OP_DELETE;
            live_count--;
            live_list[li] = live_list[live_count];
            break;
        }
        }
    }
    free(live_list);
    return w;
}

static void free_work(struct work *w) {
    hp_free(w->keys, w->pool_size * sizeof(uint64_t));
    hp_free(w->vals, w->pool_size * sizeof(uint64_t));
    hp_free(w->op_idx, w->n_ops * sizeof(uint32_t));
    hp_free(w->op_type, w->n_ops * sizeof(uint8_t));
}

/* ================================================================
 * Benchmark macro — true mixed dispatch
 *
 * GET → prefetch (key + value lines)
 * INSERT/DELETE → prefetch_insert (key line only)
 * ================================================================ */

#define BENCH_INSERT(NAME, LABEL, w, pf) do {                                \
    struct NAME m;                                                           \
    NAME##_init_cap(&m, (uint32_t)(w)->n_ops);                               \
    double t0 = now_sec();                                                   \
    for (uint64_t i = 0; i < (w)->n_ops; i++) {                              \
        if (i + (pf) < (w)->n_ops)                                           \
            NAME##_prefetch_insert(&m, (w)->keys[(w)->op_idx[i + (pf)]]);    \
        NAME##_insert_unique(&m, (w)->keys[(w)->op_idx[i]],                  \
                             &(w)->vals[(w)->op_idx[i]]);                    \
    }                                                                        \
    double ns = (now_sec() - t0) * 1e9 / (double)(w)->n_ops;                 \
    printf("%s\t%d\tinsert_only\t%.2f\t%u\n",                                \
           LABEL, pf, ns, m.count);                                          \
    NAME##_destroy(&m);                                                      \
} while (0)

#define BENCH_MIXED(NAME, LABEL, w, pf, wl_name) do {                       \
    struct NAME m;                                                           \
    NAME##_init_cap(&m, (uint32_t)((w)->n_init + (w)->n_ops));               \
    for (uint64_t i = 0; i < (w)->n_init; i++) {                             \
        if (i + (pf) < (w)->n_init)                                          \
            NAME##_prefetch_insert(&m, (w)->keys[i + (pf)]);                 \
        NAME##_insert_unique(&m, (w)->keys[i], &(w)->vals[i]);               \
    }                                                                        \
    double t0 = now_sec();                                                   \
    for (uint64_t i = 0; i < (w)->n_ops; i++) {                              \
        if (i + (pf) < (w)->n_ops) {                                         \
            uint64_t pk = (w)->keys[(w)->op_idx[i + (pf)]];                  \
            if ((w)->op_type[i + (pf)] == OP_GET)                            \
                NAME##_prefetch(&m, pk);                                     \
            else                                                             \
                NAME##_prefetch_insert(&m, pk);                              \
        }                                                                    \
        uint32_t idx = (w)->op_idx[i];                                       \
        uint64_t k = (w)->keys[idx];                                         \
        switch ((w)->op_type[i]) {                                           \
        case OP_GET:    NAME##_get(&m, k); break;                            \
        case OP_INSERT: NAME##_insert(&m, k, &(w)->vals[idx]); break;        \
        case OP_DELETE: NAME##_delete(&m, k); break;                         \
        }                                                                    \
    }                                                                        \
    double ns = (now_sec() - t0) * 1e9 / (double)(w)->n_ops;                 \
    printf("%s\t%d\t%s\t%.2f\t%u\n",                                         \
           LABEL, pf, wl_name, ns, m.count);                                 \
    NAME##_destroy(&m);                                                      \
} while (0)

/* ================================================================
 * Main
 * ================================================================ */

#define N_INSERT 2000000
#define N_INIT   1000000
#define N_OPS    2000000

static const int pfs[] = {4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 64};
#define NPF ((int)(sizeof(pfs) / sizeof(pfs[0])))

struct wl_def {
    const char *name;
    int pct_r, pct_i, pct_d;
};

static const struct wl_def workloads[] = {
    {"churn_80_10_10", 80, 10, 10},
    {"churn_50_25_25", 50, 25, 25},
    {"churn_33_33_34", 33, 33, 34},
    {"churn_20_40_40", 20, 40, 40},
};
#define NWL ((int)(sizeof(workloads) / sizeof(workloads[0])))

#define RUN_STRIDE(NAME, LABEL) do {                                         \
    /* insert_only */                                                        \
    for (int pi = 0; pi < NPF; pi++) {                                       \
        seed_rng();                                                          \
        struct work w = gen_insert(N_INSERT);                                \
        BENCH_INSERT(NAME, LABEL, &w, pfs[pi]);                              \
        free_work(&w);                                                       \
    }                                                                        \
    /* mixed workloads */                                                    \
    for (int wi = 0; wi < NWL; wi++) {                                       \
        for (int pi = 0; pi < NPF; pi++) {                                   \
            seed_rng();                                                      \
            struct work w = gen_mixed(N_INIT, N_OPS,                         \
                workloads[wi].pct_r, workloads[wi].pct_i,                    \
                workloads[wi].pct_d);                                        \
            BENCH_MIXED(NAME, LABEL, &w, pfs[pi], workloads[wi].name);       \
            free_work(&w);                                                   \
        }                                                                    \
    }                                                                        \
} while (0)

int main(void) {
    printf("# stride\tpf\tworkload\tns_per_op\tfinal_count\n");

    RUN_STRIDE(kv64_n1, "N=1");
    RUN_STRIDE(kv64_n2, "N=2");
    RUN_STRIDE(kv64_n4, "N=4");

    return 0;
}
