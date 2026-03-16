/*
 * bench_kv_vs_boost.c — C benchmark functions for KV sentinel/bitstealing
 * vs boost::unordered_flat_map.
 *
 * Compiled as C, linked from bench_kv_vs_boost_main.cpp.
 *
 * Build:
 *   cc -O3 -march=native -std=gnu11 -c bench_kv_vs_boost.c
 *   c++ -O3 -march=native -std=c++17 -c bench_kv_vs_boost_main.cpp
 *   c++ -O3 -o bench_kv_vs_boost bench_kv_vs_boost.o bench_kv_vs_boost_main.o
 */

/* --- Sentinel inline (strategy 1) --- */
#define SIMD_MAP_NAME      kv_sent
#define SIMD_MAP_KEY_WORDS 2
#define SIMD_MAP_VAL_WORDS 1
#define SIMD_MAP_LAYOUT     1
#include "simd_sentinel.h"

/* --- Bitstealing inline (strategy 1) --- */
#define SIMD_MAP_NAME      kv_bs
#define SIMD_MAP_KEY_WORDS 2
#define SIMD_MAP_VAL_WORDS 1
#define SIMD_MAP_LAYOUT     1
#include "simd_bitstealing.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sys/mman.h>

/* ================================================================
 * Shared types (must match declarations in .cpp)
 * ================================================================ */

#define KW 2
#define VW 1
#define OP_GET    0
#define OP_INSERT 1
#define OP_DELETE 2

struct kv_work {
    uint64_t *keys;      /* pool_size * KW */
    uint64_t *vals;      /* pool_size * VW */
    uint32_t *op_idx;    /* n_ops: key pool index per op */
    uint8_t  *op_type;   /* n_ops: OP_GET / OP_INSERT / OP_DELETE */
    uint64_t  pool_size;
    uint64_t  n_init;    /* keys pre-inserted before timed phase */
    uint64_t  n_ops;
};

struct kv_result {
    double ns_per_op;
    uint64_t final_count;
};

/* ================================================================
 * Hugepage mmap allocator
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
 * xoshiro256** PRNG (fixed seed for reproducibility)
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

void kv_seed_rng(void) {
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
 * Timing
 * ================================================================ */

static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ================================================================
 * Workload generators
 * ================================================================ */

struct kv_work kv_gen_insert(uint64_t n) {
    struct kv_work w;
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

struct kv_work kv_gen_mixed(uint64_t n_init, uint64_t n_ops,
                            int pct_r, int pct_i, int pct_d) {
    struct kv_work w;
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

    /* simulate map state to build valid op sequence */
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

        /* fallback if chosen op is impossible */
        if (op == OP_GET && live_count == 0) {
            op = (fresh_cursor < pool_size) ? OP_INSERT : OP_GET;
        }
        if (op == OP_INSERT && fresh_cursor >= pool_size) {
            op = (live_count > 0) ? OP_GET : OP_GET;
        }
        if (op == OP_DELETE && live_count == 0) {
            op = (fresh_cursor < pool_size) ? OP_INSERT : OP_GET;
        }

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

void kv_free_work(struct kv_work *w) {
    hp_free(w->keys,    w->pool_size * KW * sizeof(uint64_t));
    hp_free(w->vals,    w->pool_size * VW * sizeof(uint64_t));
    hp_free(w->op_idx,  w->n_ops * sizeof(uint32_t));
    hp_free(w->op_type, w->n_ops * sizeof(uint8_t));
}

/* ================================================================
 * Sentinel benchmark
 * ================================================================ */

struct kv_result bench_sent(const struct kv_work *w, int pf) {
    struct kv_result r;
    struct kv_sent m;

    if (w->n_init == 0) {
        /* insert-only: init_cap + insert_unique, metadata-only prefetch */
        kv_sent_init_cap(&m, (uint32_t)w->n_ops);
        double t0 = now_sec();
        for (uint64_t i = 0; i < w->n_ops; i++) {
            if (i + pf < w->n_ops)
                kv_sent_prefetch_insert(&m, &w->keys[w->op_idx[i + pf] * KW]);
            kv_sent_insert_unique(&m, &w->keys[w->op_idx[i] * KW],
                                  &w->vals[w->op_idx[i] * VW]);
        }
        r.ns_per_op = (now_sec() - t0) * 1e9 / (double)w->n_ops;
        r.final_count = m.count;
        kv_sent_destroy(&m);
        return r;
    }

    /* mixed: init_cap, pre-insert, timed phase */
    kv_sent_init_cap(&m, (uint32_t)(w->n_init + w->n_ops));

    for (uint64_t i = 0; i < w->n_init; i++) {
        if (i + pf < w->n_init)
            kv_sent_prefetch_insert(&m, &w->keys[(i + pf) * KW]);
        kv_sent_insert_unique(&m, &w->keys[i * KW], &w->vals[i * VW]);
    }

    double t0 = now_sec();
    for (uint64_t i = 0; i < w->n_ops; i++) {
        if (i + pf < w->n_ops) {
            const uint64_t *pk = &w->keys[w->op_idx[i + pf] * KW];
            if (w->op_type[i + pf] == OP_GET)
                kv_sent_prefetch(&m, pk);
            else
                kv_sent_prefetch_insert(&m, pk);
        }
        uint32_t idx = w->op_idx[i];
        const uint64_t *k = &w->keys[idx * KW];
        switch (w->op_type[i]) {
        case OP_GET:    kv_sent_get(&m, k); break;
        case OP_INSERT: kv_sent_insert(&m, k, &w->vals[idx * VW]); break;
        case OP_DELETE: kv_sent_delete(&m, k); break;
        }
    }
    r.ns_per_op = (now_sec() - t0) * 1e9 / (double)w->n_ops;
    r.final_count = m.count;
    kv_sent_destroy(&m);
    return r;
}

/* ================================================================
 * Bitstealing benchmark
 * ================================================================ */

struct kv_result bench_bs(const struct kv_work *w, int pf) {
    struct kv_result r;
    struct kv_bs m;

    if (w->n_init == 0) {
        kv_bs_init_cap(&m, (uint32_t)w->n_ops);
        double t0 = now_sec();
        for (uint64_t i = 0; i < w->n_ops; i++) {
            if (i + pf < w->n_ops)
                kv_bs_prefetch_insert(&m, &w->keys[w->op_idx[i + pf] * KW]);
            kv_bs_insert_unique(&m, &w->keys[w->op_idx[i] * KW],
                                &w->vals[w->op_idx[i] * VW]);
        }
        r.ns_per_op = (now_sec() - t0) * 1e9 / (double)w->n_ops;
        r.final_count = m.count;
        kv_bs_destroy(&m);
        return r;
    }

    kv_bs_init_cap(&m, (uint32_t)(w->n_init + w->n_ops));

    for (uint64_t i = 0; i < w->n_init; i++) {
        if (i + pf < w->n_init)
            kv_bs_prefetch_insert(&m, &w->keys[(i + pf) * KW]);
        kv_bs_insert_unique(&m, &w->keys[i * KW], &w->vals[i * VW]);
    }

    double t0 = now_sec();
    for (uint64_t i = 0; i < w->n_ops; i++) {
        if (i + pf < w->n_ops) {
            const uint64_t *pk = &w->keys[w->op_idx[i + pf] * KW];
            if (w->op_type[i + pf] == OP_GET)
                kv_bs_prefetch(&m, pk);
            else
                kv_bs_prefetch_insert(&m, pk);
        }
        uint32_t idx = w->op_idx[i];
        const uint64_t *k = &w->keys[idx * KW];
        switch (w->op_type[i]) {
        case OP_GET:    kv_bs_get(&m, k); break;
        case OP_INSERT: kv_bs_insert(&m, k, &w->vals[idx * VW]); break;
        case OP_DELETE: kv_bs_delete(&m, k); break;
        }
    }
    r.ns_per_op = (now_sec() - t0) * 1e9 / (double)w->n_ops;
    r.final_count = m.count;
    kv_bs_destroy(&m);
    return r;
}
