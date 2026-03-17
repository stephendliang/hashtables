/*
 * bench_map48_arch.c — Benchmark: 5 split48 architecture variants
 *
 * 1. split 1CL/10  (baseline, simd_map48_split.h)
 * 2. 3CL/31        (simd_map48_3cl.h)
 * 3. 2CL/20        (simd_map48_2cl.h)
 * 4. backshift     (simd_map48_bs.h)
 * 5. lemire        (simd_map48_lemire.h)
 *
 * Build:
 *   cc -O3 -march=native -std=gnu11 -o bench_map48_arch bench_map48_arch.c
 *   taskset -c 0 ./bench_map48_arch
 */

/* 1. baseline */
#define SIMD_MAP_NAME b_base
#include "simd_map48_split.h"

/* 2. 3CL/31 */
#define SIMD_MAP_NAME b_3cl
#include "simd_map48_3cl.h"

/* 3. 2CL/20 */
#define SIMD_MAP_NAME b_2cl
#include "simd_map48_2cl.h"

/* 4. backshift */
#define SIMD_MAP_NAME b_bs
#include "simd_map48_bs.h"

/* 5. lemire */
#define SIMD_MAP_NAME b_lem
#include "simd_map48_lemire.h"

/* 6. lemire + backshift */
#define SIMD_MAP_NAME b_lembs
#include "simd_map48_lembs.h"

#include <stdio.h>
#include <time.h>

#define N_INSERT  2000000
#define N_CHURN   2000000
#define PF        24

/* xoshiro256** */
static uint64_t xs[4] = {
    0x180ec6d33cfd0abaULL, 0xd5a61266f0c9392cULL,
    0xa9582618e03fc9aaULL, 0x39abdc4529b1661cULL
};

static inline uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t xoshiro(void) {
    uint64_t r = rotl(xs[1] * 5, 7) * 9;
    uint64_t t = xs[1] << 17;
    xs[2] ^= xs[0]; xs[3] ^= xs[1];
    xs[1] ^= xs[2]; xs[0] ^= xs[3];
    xs[2] ^= t; xs[3] = rotl(xs[3], 45);
    return r;
}

static inline uint64_t make48(void) {
    uint64_t k;
    do { k = xoshiro() & 0x0000FFFFFFFFFFFFULL; } while (k == 0);
    return k;
}

static inline uint64_t make48_bs(void) {
    uint64_t k;
    do {
        k = xoshiro() & 0x0000FFFFFFFFFFFFULL;
    } while (k == 0 || (k >> 16) == 0xFFFFFFFF);
    return k;
}

static inline double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

static void *hp_alloc(size_t sz) {
    sz = (sz + (2u << 20) - 1) & ~((size_t)(2u << 20) - 1);
    void *p = mmap(NULL, sz, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE,
                   -1, 0);
    if (p == MAP_FAILED) {
        p = mmap(NULL, sz, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
        if (p != MAP_FAILED) madvise(p, sz, MADV_HUGEPAGE);
    }
    return p;
}

static void hp_free(void *p, size_t sz) {
    sz = (sz + (2u << 20) - 1) & ~((size_t)(2u << 20) - 1);
    munmap(p, sz);
}

static uint64_t *gen_keys(int n, uint64_t (*mkfn)(void)) {
    uint64_t *k = hp_alloc((size_t)n * sizeof(uint64_t));
    for (int i = 0; i < n; i++) k[i] = mkfn();
    return k;
}

enum { OP_GET, OP_INS, OP_DEL };

struct mixed_work {
    uint64_t *keys;
    uint8_t  *ops;
    int       n;
};

static struct mixed_work gen_mixed(int pool, int ops, uint64_t (*mkfn)(void)) {
    struct mixed_work w;
    w.keys = hp_alloc((size_t)(pool + ops) * sizeof(uint64_t));
    w.ops  = hp_alloc((size_t)ops);
    w.n    = ops;
    for (int i = 0; i < pool; i++) w.keys[i] = mkfn();
    for (int i = 0; i < ops; i++) {
        uint64_t r = xoshiro();
        int pct = (int)(r % 100);
        if (pct < 50) {
            w.ops[i] = OP_GET;
            w.keys[pool + i] = w.keys[r % (uint64_t)pool];
        } else if (pct < 75) {
            w.ops[i] = OP_INS;
            w.keys[pool + i] = mkfn();
        } else {
            w.ops[i] = OP_DEL;
            w.keys[pool + i] = w.keys[r % (uint64_t)pool];
        }
    }
    return w;
}

static void free_mixed(struct mixed_work *w, int pool) {
    hp_free(w->keys, (size_t)(pool + w->n) * sizeof(uint64_t));
    hp_free(w->ops, (size_t)w->n);
}

int main(void) {
    printf("# Map48 arch benchmark (N=%d, PF=%d)\n", N_INSERT, PF);
    printf("# variant\tworkload\tns_per_op\tcount\talloc_mib\n");

    uint64_t seed[4] = {0x180ec6d33cfd0abaULL, 0xd5a61266f0c9392cULL,
                        0xa9582618e03fc9aaULL, 0x39abdc4529b1661cULL};

    /* === INSERT-ONLY === */
    {
        /* baseline */
        {
            memcpy(xs, seed, sizeof(xs));
            uint64_t *keys = gen_keys(N_INSERT, make48);
            struct b_base m;
            b_base_init_cap(&m, N_INSERT);
            double alloc = (double)b_base_mapsize(m.ng) / (1024.0 * 1024.0);
            double t0 = now_ns();
            for (int i = 0; i < N_INSERT; i++) {
                if (i + PF < N_INSERT)
                    b_base_prefetch_insert(&m, keys[i + PF]);
                b_base_insert_unique(&m, keys[i]);
            }
            double dt = now_ns() - t0;
            printf("split\tinsert_only\t%.1f\t%u\t%.1f\n",
                   dt / N_INSERT, m.count, alloc);
            b_base_destroy(&m);
            hp_free(keys, (size_t)N_INSERT * 8);
        }

        /* 3CL */
        {
            memcpy(xs, seed, sizeof(xs));
            uint64_t *keys = gen_keys(N_INSERT, make48);
            struct b_3cl m;
            b_3cl_init_cap(&m, N_INSERT);
            double alloc = (double)b_3cl_mapsize(m.ng) / (1024.0 * 1024.0);
            double t0 = now_ns();
            for (int i = 0; i < N_INSERT; i++) {
                if (i + PF < N_INSERT)
                    b_3cl_prefetch_insert(&m, keys[i + PF]);
                b_3cl_insert_unique(&m, keys[i]);
            }
            double dt = now_ns() - t0;
            printf("3cl\tinsert_only\t%.1f\t%u\t%.1f\n",
                   dt / N_INSERT, m.count, alloc);
            b_3cl_destroy(&m);
            hp_free(keys, (size_t)N_INSERT * 8);
        }

        /* 2CL */
        {
            memcpy(xs, seed, sizeof(xs));
            uint64_t *keys = gen_keys(N_INSERT, make48);
            struct b_2cl m;
            b_2cl_init_cap(&m, N_INSERT);
            double alloc = (double)b_2cl_mapsize(m.ng) / (1024.0 * 1024.0);
            double t0 = now_ns();
            for (int i = 0; i < N_INSERT; i++) {
                if (i + PF < N_INSERT)
                    b_2cl_prefetch_insert(&m, keys[i + PF]);
                b_2cl_insert_unique(&m, keys[i]);
            }
            double dt = now_ns() - t0;
            printf("2cl\tinsert_only\t%.1f\t%u\t%.1f\n",
                   dt / N_INSERT, m.count, alloc);
            b_2cl_destroy(&m);
            hp_free(keys, (size_t)N_INSERT * 8);
        }

        /* backshift */
        {
            memcpy(xs, seed, sizeof(xs));
            uint64_t *keys = gen_keys(N_INSERT, make48_bs);
            struct b_bs m;
            b_bs_init_cap(&m, N_INSERT);
            double alloc = (double)b_bs_mapsize(m.ng) / (1024.0 * 1024.0);
            double t0 = now_ns();
            for (int i = 0; i < N_INSERT; i++) {
                if (i + PF < N_INSERT)
                    b_bs_prefetch_insert(&m, keys[i + PF]);
                b_bs_insert_unique(&m, keys[i]);
            }
            double dt = now_ns() - t0;
            printf("backshift\tinsert_only\t%.1f\t%u\t%.1f\n",
                   dt / N_INSERT, m.count, alloc);
            b_bs_destroy(&m);
            hp_free(keys, (size_t)N_INSERT * 8);
        }

        /* lemire */
        {
            memcpy(xs, seed, sizeof(xs));
            uint64_t *keys = gen_keys(N_INSERT, make48);
            struct b_lem m;
            b_lem_init_cap(&m, N_INSERT);
            double alloc = (double)b_lem_mapsize(m.ng) / (1024.0 * 1024.0);
            double t0 = now_ns();
            for (int i = 0; i < N_INSERT; i++) {
                if (i + PF < N_INSERT)
                    b_lem_prefetch_insert(&m, keys[i + PF]);
                b_lem_insert_unique(&m, keys[i]);
            }
            double dt = now_ns() - t0;
            printf("lemire\tinsert_only\t%.1f\t%u\t%.1f\n",
                   dt / N_INSERT, m.count, alloc);
            b_lem_destroy(&m);
            hp_free(keys, (size_t)N_INSERT * 8);
        }

        /* lemire+backshift */
        {
            memcpy(xs, seed, sizeof(xs));
            uint64_t *keys = gen_keys(N_INSERT, make48_bs);
            struct b_lembs m;
            b_lembs_init_cap(&m, N_INSERT);
            double alloc = (double)b_lembs_mapsize(m.ng) / (1024.0 * 1024.0);
            double t0 = now_ns();
            for (int i = 0; i < N_INSERT; i++) {
                if (i + PF < N_INSERT)
                    b_lembs_prefetch_insert(&m, keys[i + PF]);
                b_lembs_insert_unique(&m, keys[i]);
            }
            double dt = now_ns() - t0;
            printf("lem+bs\tinsert_only\t%.1f\t%u\t%.1f\n",
                   dt / N_INSERT, m.count, alloc);
            b_lembs_destroy(&m);
            hp_free(keys, (size_t)N_INSERT * 8);
        }
    }

    /* === CONTAINS HIT === */
    {
        /* baseline */
        {
            memcpy(xs, seed, sizeof(xs));
            uint64_t *keys = gen_keys(N_INSERT, make48);
            struct b_base m;
            b_base_init_cap(&m, N_INSERT);
            for (int i = 0; i < N_INSERT; i++) b_base_insert_unique(&m, keys[i]);
            double alloc = (double)b_base_mapsize(m.ng) / (1024.0 * 1024.0);
            double t0 = now_ns();
            int hits = 0;
            for (int i = 0; i < N_INSERT; i++) {
                if (i + PF < N_INSERT)
                    b_base_prefetch(&m, keys[i + PF]);
                hits += b_base_contains(&m, keys[i]);
            }
            double dt = now_ns() - t0;
            printf("split\tcontains_hit\t%.1f\t%d\t%.1f\n",
                   dt / N_INSERT, hits, alloc);
            b_base_destroy(&m);
            hp_free(keys, (size_t)N_INSERT * 8);
        }

        /* 3CL */
        {
            memcpy(xs, seed, sizeof(xs));
            uint64_t *keys = gen_keys(N_INSERT, make48);
            struct b_3cl m;
            b_3cl_init_cap(&m, N_INSERT);
            for (int i = 0; i < N_INSERT; i++) b_3cl_insert_unique(&m, keys[i]);
            double alloc = (double)b_3cl_mapsize(m.ng) / (1024.0 * 1024.0);
            double t0 = now_ns();
            int hits = 0;
            for (int i = 0; i < N_INSERT; i++) {
                if (i + PF < N_INSERT)
                    b_3cl_prefetch(&m, keys[i + PF]);
                hits += b_3cl_contains(&m, keys[i]);
            }
            double dt = now_ns() - t0;
            printf("3cl\tcontains_hit\t%.1f\t%d\t%.1f\n",
                   dt / N_INSERT, hits, alloc);
            b_3cl_destroy(&m);
            hp_free(keys, (size_t)N_INSERT * 8);
        }

        /* 2CL */
        {
            memcpy(xs, seed, sizeof(xs));
            uint64_t *keys = gen_keys(N_INSERT, make48);
            struct b_2cl m;
            b_2cl_init_cap(&m, N_INSERT);
            for (int i = 0; i < N_INSERT; i++) b_2cl_insert_unique(&m, keys[i]);
            double alloc = (double)b_2cl_mapsize(m.ng) / (1024.0 * 1024.0);
            double t0 = now_ns();
            int hits = 0;
            for (int i = 0; i < N_INSERT; i++) {
                if (i + PF < N_INSERT)
                    b_2cl_prefetch(&m, keys[i + PF]);
                hits += b_2cl_contains(&m, keys[i]);
            }
            double dt = now_ns() - t0;
            printf("2cl\tcontains_hit\t%.1f\t%d\t%.1f\n",
                   dt / N_INSERT, hits, alloc);
            b_2cl_destroy(&m);
            hp_free(keys, (size_t)N_INSERT * 8);
        }

        /* backshift */
        {
            memcpy(xs, seed, sizeof(xs));
            uint64_t *keys = gen_keys(N_INSERT, make48_bs);
            struct b_bs m;
            b_bs_init_cap(&m, N_INSERT);
            for (int i = 0; i < N_INSERT; i++) b_bs_insert_unique(&m, keys[i]);
            double alloc = (double)b_bs_mapsize(m.ng) / (1024.0 * 1024.0);
            double t0 = now_ns();
            int hits = 0;
            for (int i = 0; i < N_INSERT; i++) {
                if (i + PF < N_INSERT)
                    b_bs_prefetch(&m, keys[i + PF]);
                hits += b_bs_contains(&m, keys[i]);
            }
            double dt = now_ns() - t0;
            printf("backshift\tcontains_hit\t%.1f\t%d\t%.1f\n",
                   dt / N_INSERT, hits, alloc);
            b_bs_destroy(&m);
            hp_free(keys, (size_t)N_INSERT * 8);
        }

        /* lemire */
        {
            memcpy(xs, seed, sizeof(xs));
            uint64_t *keys = gen_keys(N_INSERT, make48);
            struct b_lem m;
            b_lem_init_cap(&m, N_INSERT);
            for (int i = 0; i < N_INSERT; i++) b_lem_insert_unique(&m, keys[i]);
            double alloc = (double)b_lem_mapsize(m.ng) / (1024.0 * 1024.0);
            double t0 = now_ns();
            int hits = 0;
            for (int i = 0; i < N_INSERT; i++) {
                if (i + PF < N_INSERT)
                    b_lem_prefetch(&m, keys[i + PF]);
                hits += b_lem_contains(&m, keys[i]);
            }
            double dt = now_ns() - t0;
            printf("lemire\tcontains_hit\t%.1f\t%d\t%.1f\n",
                   dt / N_INSERT, hits, alloc);
            b_lem_destroy(&m);
            hp_free(keys, (size_t)N_INSERT * 8);
        }

        /* lemire+backshift */
        {
            memcpy(xs, seed, sizeof(xs));
            uint64_t *keys = gen_keys(N_INSERT, make48_bs);
            struct b_lembs m;
            b_lembs_init_cap(&m, N_INSERT);
            for (int i = 0; i < N_INSERT; i++) b_lembs_insert_unique(&m, keys[i]);
            double alloc = (double)b_lembs_mapsize(m.ng) / (1024.0 * 1024.0);
            double t0 = now_ns();
            int hits = 0;
            for (int i = 0; i < N_INSERT; i++) {
                if (i + PF < N_INSERT)
                    b_lembs_prefetch(&m, keys[i + PF]);
                hits += b_lembs_contains(&m, keys[i]);
            }
            double dt = now_ns() - t0;
            printf("lem+bs\tcontains_hit\t%.1f\t%d\t%.1f\n",
                   dt / N_INSERT, hits, alloc);
            b_lembs_destroy(&m);
            hp_free(keys, (size_t)N_INSERT * 8);
        }
    }

    /* === MIXED 50/25/25 === */
    {
        uint64_t mix_seed[4] = {0xCAFEBABE12345678ULL, 0xDEADBEEF87654321ULL,
                                0x0123456789ABCDEFULL, 0xFEDCBA9876543210ULL};
        int pool = N_INSERT;

        /* baseline */
        {
            memcpy(xs, mix_seed, sizeof(xs));
            uint64_t *pool_keys = gen_keys(pool, make48);
            struct mixed_work w = gen_mixed(pool, N_CHURN, make48);
            struct b_base m;
            b_base_init_cap(&m, pool * 2);
            double alloc = (double)b_base_mapsize(m.ng) / (1024.0 * 1024.0);
            for (int i = 0; i < pool; i++) b_base_insert_unique(&m, pool_keys[i]);
            double t0 = now_ns();
            for (int i = 0; i < w.n; i++) {
                uint64_t k = w.keys[pool + i];
                if (i + PF < w.n) {
                    uint64_t pk = w.keys[pool + i + PF];
                    if (w.ops[i + PF] == OP_INS)
                        b_base_prefetch_insert(&m, pk);
                    else
                        b_base_prefetch(&m, pk);
                }
                switch (w.ops[i]) {
                    case OP_GET: b_base_contains(&m, k); break;
                    case OP_INS: b_base_insert(&m, k); break;
                    case OP_DEL: b_base_delete(&m, k); break;
                }
            }
            double dt = now_ns() - t0;
            printf("split\tmixed_50_25_25\t%.1f\t%u\t%.1f\n",
                   dt / w.n, m.count, alloc);
            b_base_destroy(&m);
            hp_free(pool_keys, (size_t)pool * 8);
            free_mixed(&w, pool);
        }

        /* 3CL */
        {
            memcpy(xs, mix_seed, sizeof(xs));
            uint64_t *pool_keys = gen_keys(pool, make48);
            struct mixed_work w = gen_mixed(pool, N_CHURN, make48);
            struct b_3cl m;
            b_3cl_init_cap(&m, pool * 2);
            double alloc = (double)b_3cl_mapsize(m.ng) / (1024.0 * 1024.0);
            for (int i = 0; i < pool; i++) b_3cl_insert_unique(&m, pool_keys[i]);
            double t0 = now_ns();
            for (int i = 0; i < w.n; i++) {
                uint64_t k = w.keys[pool + i];
                if (i + PF < w.n) {
                    uint64_t pk = w.keys[pool + i + PF];
                    if (w.ops[i + PF] == OP_INS)
                        b_3cl_prefetch_insert(&m, pk);
                    else
                        b_3cl_prefetch(&m, pk);
                }
                switch (w.ops[i]) {
                    case OP_GET: b_3cl_contains(&m, k); break;
                    case OP_INS: b_3cl_insert(&m, k); break;
                    case OP_DEL: b_3cl_delete(&m, k); break;
                }
            }
            double dt = now_ns() - t0;
            printf("3cl\tmixed_50_25_25\t%.1f\t%u\t%.1f\n",
                   dt / w.n, m.count, alloc);
            b_3cl_destroy(&m);
            hp_free(pool_keys, (size_t)pool * 8);
            free_mixed(&w, pool);
        }

        /* 2CL */
        {
            memcpy(xs, mix_seed, sizeof(xs));
            uint64_t *pool_keys = gen_keys(pool, make48);
            struct mixed_work w = gen_mixed(pool, N_CHURN, make48);
            struct b_2cl m;
            b_2cl_init_cap(&m, pool * 2);
            double alloc = (double)b_2cl_mapsize(m.ng) / (1024.0 * 1024.0);
            for (int i = 0; i < pool; i++) b_2cl_insert_unique(&m, pool_keys[i]);
            double t0 = now_ns();
            for (int i = 0; i < w.n; i++) {
                uint64_t k = w.keys[pool + i];
                if (i + PF < w.n) {
                    uint64_t pk = w.keys[pool + i + PF];
                    if (w.ops[i + PF] == OP_INS)
                        b_2cl_prefetch_insert(&m, pk);
                    else
                        b_2cl_prefetch(&m, pk);
                }
                switch (w.ops[i]) {
                    case OP_GET: b_2cl_contains(&m, k); break;
                    case OP_INS: b_2cl_insert(&m, k); break;
                    case OP_DEL: b_2cl_delete(&m, k); break;
                }
            }
            double dt = now_ns() - t0;
            printf("2cl\tmixed_50_25_25\t%.1f\t%u\t%.1f\n",
                   dt / w.n, m.count, alloc);
            b_2cl_destroy(&m);
            hp_free(pool_keys, (size_t)pool * 8);
            free_mixed(&w, pool);
        }

        /* backshift */
        {
            memcpy(xs, mix_seed, sizeof(xs));
            uint64_t *pool_keys = gen_keys(pool, make48_bs);
            struct mixed_work w = gen_mixed(pool, N_CHURN, make48_bs);
            struct b_bs m;
            b_bs_init_cap(&m, pool * 2);
            double alloc = (double)b_bs_mapsize(m.ng) / (1024.0 * 1024.0);
            for (int i = 0; i < pool; i++) b_bs_insert_unique(&m, pool_keys[i]);
            double t0 = now_ns();
            for (int i = 0; i < w.n; i++) {
                uint64_t k = w.keys[pool + i];
                if (i + PF < w.n) {
                    uint64_t pk = w.keys[pool + i + PF];
                    if (w.ops[i + PF] == OP_INS)
                        b_bs_prefetch_insert(&m, pk);
                    else
                        b_bs_prefetch(&m, pk);
                }
                switch (w.ops[i]) {
                    case OP_GET: b_bs_contains(&m, k); break;
                    case OP_INS: b_bs_insert(&m, k); break;
                    case OP_DEL: b_bs_delete(&m, k); break;
                }
            }
            double dt = now_ns() - t0;
            printf("backshift\tmixed_50_25_25\t%.1f\t%u\t%.1f\n",
                   dt / w.n, m.count, alloc);
            b_bs_destroy(&m);
            hp_free(pool_keys, (size_t)pool * 8);
            free_mixed(&w, pool);
        }

        /* lemire */
        {
            memcpy(xs, mix_seed, sizeof(xs));
            uint64_t *pool_keys = gen_keys(pool, make48);
            struct mixed_work w = gen_mixed(pool, N_CHURN, make48);
            struct b_lem m;
            b_lem_init_cap(&m, pool * 2);
            double alloc = (double)b_lem_mapsize(m.ng) / (1024.0 * 1024.0);
            for (int i = 0; i < pool; i++) b_lem_insert_unique(&m, pool_keys[i]);
            double t0 = now_ns();
            for (int i = 0; i < w.n; i++) {
                uint64_t k = w.keys[pool + i];
                if (i + PF < w.n) {
                    uint64_t pk = w.keys[pool + i + PF];
                    if (w.ops[i + PF] == OP_INS)
                        b_lem_prefetch_insert(&m, pk);
                    else
                        b_lem_prefetch(&m, pk);
                }
                switch (w.ops[i]) {
                    case OP_GET: b_lem_contains(&m, k); break;
                    case OP_INS: b_lem_insert(&m, k); break;
                    case OP_DEL: b_lem_delete(&m, k); break;
                }
            }
            double dt = now_ns() - t0;
            printf("lemire\tmixed_50_25_25\t%.1f\t%u\t%.1f\n",
                   dt / w.n, m.count, alloc);
            b_lem_destroy(&m);
            hp_free(pool_keys, (size_t)pool * 8);
            free_mixed(&w, pool);
        }

        /* lemire+backshift */
        {
            memcpy(xs, mix_seed, sizeof(xs));
            uint64_t *pool_keys = gen_keys(pool, make48_bs);
            struct mixed_work w = gen_mixed(pool, N_CHURN, make48_bs);
            struct b_lembs m;
            b_lembs_init_cap(&m, pool * 2);
            double alloc = (double)b_lembs_mapsize(m.ng) / (1024.0 * 1024.0);
            for (int i = 0; i < pool; i++) b_lembs_insert_unique(&m, pool_keys[i]);
            double t0 = now_ns();
            for (int i = 0; i < w.n; i++) {
                uint64_t k = w.keys[pool + i];
                if (i + PF < w.n) {
                    uint64_t pk = w.keys[pool + i + PF];
                    if (w.ops[i + PF] == OP_INS)
                        b_lembs_prefetch_insert(&m, pk);
                    else
                        b_lembs_prefetch(&m, pk);
                }
                switch (w.ops[i]) {
                    case OP_GET: b_lembs_contains(&m, k); break;
                    case OP_INS: b_lembs_insert(&m, k); break;
                    case OP_DEL: b_lembs_delete(&m, k); break;
                }
            }
            double dt = now_ns() - t0;
            printf("lem+bs\tmixed_50_25_25\t%.1f\t%u\t%.1f\n",
                   dt / w.n, m.count, alloc);
            b_lembs_destroy(&m);
            hp_free(pool_keys, (size_t)pool * 8);
            free_mixed(&w, pool);
        }
    }

    return 0;
}
