/*
 * bench_kv_pf_tuning.c — PF distance sweep + delete prefetch mode A/B
 *
 * Sweeps PF distance for insert-only and churn workloads, and compares
 * metadata-only vs full 5-line prefetch for delete operations.
 *
 * Sentinel inline only (KW=2, VW=1, layout 1). Sentinel and bitstealing
 * are at parity — no need to double the sweep.
 *
 * Build & run:
 *   cc -O3 -march=native -std=gnu11 -o bench_pf bench_kv_pf_tuning.c
 *   taskset -c 0 ./bench_pf > results.tsv
 */

#define SIMD_MAP_NAME      pf_sent
#define SIMD_MAP_KEY_WORDS 2
#define SIMD_MAP_VAL_WORDS 1
#define SIMD_MAP_LAYOUT     1
#include "simd_sentinel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sys/mman.h>

#define KW 2
#define VW 1
#define N_INSERT  2000000
#define N_INIT    1000000
#define N_OPS     2000000

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
 * Workload types
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

    w.keys    = hp_alloc(n * KW * sizeof(uint64_t));
    w.vals    = hp_alloc(n * VW * sizeof(uint64_t));
    w.op_idx  = hp_alloc(n * sizeof(uint32_t));
    w.op_type = hp_alloc(n * sizeof(uint8_t));

    for (uint64_t i = 0; i < n * KW; i++)
        w.keys[i] = xoshiro256ss();
    for (uint64_t i = 0; i < n * VW; i++)
        w.vals[i] = xoshiro256ss();
    for (uint64_t i = 0; i < n; i++) {
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

    w.keys    = hp_alloc(pool_size * KW * sizeof(uint64_t));
    w.vals    = hp_alloc(pool_size * VW * sizeof(uint64_t));
    w.op_idx  = hp_alloc(n_ops * sizeof(uint32_t));
    w.op_type = hp_alloc(n_ops * sizeof(uint8_t));

    for (uint64_t i = 0; i < pool_size * KW; i++)
        w.keys[i] = xoshiro256ss();
    for (uint64_t i = 0; i < pool_size * VW; i++)
        w.vals[i] = xoshiro256ss();

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

        if (roll < pct_r)
            op = OP_GET;
        else if (roll < thresh_i)
            op = OP_INSERT;
        else
            op = OP_DELETE;

        if (op == OP_GET && live_count == 0)
            op = (fresh_cursor < pool_size) ? OP_INSERT : OP_GET;
        if (op == OP_INSERT && fresh_cursor >= pool_size)
            op = (live_count > 0) ? OP_GET : OP_GET;
        if (op == OP_DELETE && live_count == 0)
            op = (fresh_cursor < pool_size) ? OP_INSERT : OP_GET;

        switch (op) {
        case OP_GET:
            if (live_count > 0) {
                uint32_t li = (uint32_t)(xoshiro256ss() % live_count);
                w.op_idx[i] = live_list[li];
            } else {
                w.op_idx[i] = (uint32_t)(xoshiro256ss() % pool_size);
            }
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
    hp_free(w->keys,    w->pool_size * KW * sizeof(uint64_t));
    hp_free(w->vals,    w->pool_size * VW * sizeof(uint64_t));
    hp_free(w->op_idx,  w->n_ops * sizeof(uint32_t));
    hp_free(w->op_type, w->n_ops * sizeof(uint8_t));
}

/* ================================================================
 * Benchmark: insert-only
 * ================================================================ */

static void bench_insert(const struct work *w, int pf,
                          double *ns_out, uint64_t *count_out) {
    struct pf_sent m;
    pf_sent_init_cap(&m, (uint32_t)w->n_ops);

    double t0 = now_sec();
    for (uint64_t i = 0; i < w->n_ops; i++) {
        if (i + pf < w->n_ops)
            pf_sent_prefetch_insert(&m, &w->keys[w->op_idx[i + pf] * KW]);
        pf_sent_insert_unique(&m, &w->keys[w->op_idx[i] * KW],
                              &w->vals[w->op_idx[i] * VW]);
    }
    *ns_out = (now_sec() - t0) * 1e9 / (double)w->n_ops;
    *count_out = m.count;
    pf_sent_destroy(&m);
}

/* ================================================================
 * Benchmark: mixed (churn)
 *
 * del_full=0 (current): GET=full, INSERT+DELETE=meta
 * del_full=1 (fix):     GET+DELETE=full, INSERT=meta
 * ================================================================ */

static void bench_mixed(const struct work *w, int pf, int del_full,
                         double *ns_out, uint64_t *count_out) {
    struct pf_sent m;
    pf_sent_init_cap(&m, (uint32_t)(w->n_init + w->n_ops));

    /* pre-insert n_init keys */
    for (uint64_t i = 0; i < w->n_init; i++) {
        if (i + pf < w->n_init)
            pf_sent_prefetch_insert(&m, &w->keys[(i + pf) * KW]);
        pf_sent_insert_unique(&m, &w->keys[i * KW], &w->vals[i * VW]);
    }

    /* timed phase */
    double t0 = now_sec();
    for (uint64_t i = 0; i < w->n_ops; i++) {
        if (i + pf < w->n_ops) {
            const uint64_t *pk = &w->keys[w->op_idx[i + pf] * KW];
            if (w->op_type[i + pf] == OP_INSERT)
                pf_sent_prefetch_insert(&m, pk);
            else if (w->op_type[i + pf] == OP_GET || del_full)
                pf_sent_prefetch(&m, pk);
            else
                pf_sent_prefetch_insert(&m, pk);
        }
        uint32_t idx = w->op_idx[i];
        const uint64_t *k = &w->keys[idx * KW];
        switch (w->op_type[i]) {
        case OP_GET:    pf_sent_get(&m, k); break;
        case OP_INSERT: pf_sent_insert(&m, k, &w->vals[idx * VW]); break;
        case OP_DELETE: pf_sent_delete(&m, k); break;
        }
    }
    *ns_out = (now_sec() - t0) * 1e9 / (double)w->n_ops;
    *count_out = m.count;
    pf_sent_destroy(&m);
}

/* ================================================================
 * Main
 * ================================================================ */

int main(void) {
    static const int pfs[] = {4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 64};
    int npf = (int)(sizeof(pfs) / sizeof(pfs[0]));

    printf("# pf\tdel_mode\tworkload\tns_per_op\tfinal_count\n");

    /* insert_only: sweep PF */
    for (int pi = 0; pi < npf; pi++) {
        int pf = pfs[pi];
        seed_rng();
        struct work w = gen_insert(N_INSERT);
        double ns;
        uint64_t count;
        bench_insert(&w, pf, &ns, &count);
        printf("%d\tmeta\tinsert_only\t%.2f\t%lu\n",
               pf, ns, (unsigned long)count);
        free_work(&w);
        fflush(stdout);
    }

    /* churn_50_25_25: sweep PF x delete mode */
    for (int pi = 0; pi < npf; pi++) {
        int pf = pfs[pi];
        for (int del_full = 0; del_full <= 1; del_full++) {
            seed_rng();
            struct work w = gen_mixed(N_INIT, N_OPS, 50, 25, 25);
            double ns;
            uint64_t count;
            bench_mixed(&w, pf, del_full, &ns, &count);
            printf("%d\t%s\tchurn_50_25_25\t%.2f\t%lu\n",
                   pf, del_full ? "full" : "meta", ns, (unsigned long)count);
            free_work(&w);
            fflush(stdout);
        }
    }

    return 0;
}
