/*
 * bench_map48_direct.c — Benchmark: packed vs split vs map48 vs map64
 *
 * All four store 48-bit keys in set mode.
 * packed:   3×u16 interleaved, 10 keys/CL, direct compare
 * split:    hi32+lo16, 10 keys/CL, direct compare
 * map48:    sentinel metadata, 31 keys, 4 cache lines
 * map64:    direct compare, 8 keys/CL
 *
 * Build:
 *   cc -O3 -march=native -std=gnu11 -o bench_map48_direct bench_map48_direct.c
 */

/* packed 3×u16 */
#define SIMD_MAP_NAME bpacked
#include "simd_map48_packed.h"

/* split hi32+lo16 */
#define SIMD_MAP_NAME bsplit
#include "simd_map48_split.h"

/* map48 sentinel */
#define SIMD_MAP_NAME b48
#include "simd_map48.h"

/* map64 */
#define SIMD_MAP_NAME b64
#include "simd_map64.h"

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

static uint64_t *gen_keys(int n) {
    uint64_t *k = hp_alloc((size_t)n * sizeof(uint64_t));
    for (int i = 0; i < n; i++) k[i] = make48();
    return k;
}

enum { OP_GET, OP_INS, OP_DEL };

struct mixed_work {
    uint64_t *keys;
    uint8_t  *ops;
    int       n;
};

static struct mixed_work gen_mixed(int pool, int ops) {
    struct mixed_work w;
    w.keys = hp_alloc((size_t)(pool + ops) * sizeof(uint64_t));
    w.ops  = hp_alloc((size_t)ops);
    w.n    = ops;
    for (int i = 0; i < pool; i++) w.keys[i] = make48();
    int next_new = pool;
    (void)next_new;
    for (int i = 0; i < ops; i++) {
        uint64_t r = xoshiro();
        int pct = (int)(r % 100);
        if (pct < 50) {
            w.ops[i] = OP_GET;
            w.keys[pool + i] = w.keys[r % (uint64_t)pool];
        } else if (pct < 75) {
            w.ops[i] = OP_INS;
            w.keys[pool + i] = make48();
        } else {
            w.ops[i] = OP_DEL;
            w.keys[pool + i] = w.keys[r % (uint64_t)pool];
        }
    }
    return w;
}

int main(void) {
    printf("# Map48 direct-compare benchmark (N=%d, PF=%d)\n", N_INSERT, PF);
    printf("# variant\tworkload\tns_per_op\tcount\n");

    uint64_t seed[4] = {0x180ec6d33cfd0abaULL, 0xd5a61266f0c9392cULL,
                        0xa9582618e03fc9aaULL, 0x39abdc4529b1661cULL};

    /* === INSERT-ONLY === */
    {
        memcpy(xs, seed, sizeof(xs));
        uint64_t *keys = gen_keys(N_INSERT);

        /* packed */
        {
            struct bpacked m;
            bpacked_init_cap(&m, N_INSERT);
            double t0 = now_ns();
            for (int i = 0; i < N_INSERT; i++) {
                if (i + PF < N_INSERT)
                    bpacked_prefetch_insert(&m, keys[i + PF]);
                bpacked_insert_unique(&m, keys[i]);
            }
            double dt = now_ns() - t0;
            printf("packed\tinsert_only\t%.1f\t%u\n", dt / N_INSERT, m.count);
            bpacked_destroy(&m);
        }

        /* split */
        {
            struct bsplit m;
            bsplit_init_cap(&m, N_INSERT);
            double t0 = now_ns();
            for (int i = 0; i < N_INSERT; i++) {
                if (i + PF < N_INSERT)
                    bsplit_prefetch_insert(&m, keys[i + PF]);
                bsplit_insert_unique(&m, keys[i]);
            }
            double dt = now_ns() - t0;
            printf("split\tinsert_only\t%.1f\t%u\n", dt / N_INSERT, m.count);
            bsplit_destroy(&m);
        }

        /* map48 */
        {
            struct b48 m;
            b48_init_cap(&m, N_INSERT);
            double t0 = now_ns();
            for (int i = 0; i < N_INSERT; i++) {
                if (i + PF < N_INSERT)
                    b48_prefetch_insert(&m, keys[i + PF]);
                b48_insert_unique(&m, keys[i]);
            }
            double dt = now_ns() - t0;
            printf("map48\tinsert_only\t%.1f\t%u\n", dt / N_INSERT, m.count);
            b48_destroy(&m);
        }

        /* map64 */
        {
            struct b64 m;
            b64_init_cap(&m, N_INSERT);
            double t0 = now_ns();
            for (int i = 0; i < N_INSERT; i++) {
                if (i + PF < N_INSERT)
                    b64_prefetch_insert(&m, keys[i + PF]);
                b64_insert_unique(&m, keys[i]);
            }
            double dt = now_ns() - t0;
            printf("map64\tinsert_only\t%.1f\t%u\n", dt / N_INSERT, m.count);
            b64_destroy(&m);
        }

        munmap(keys, ((size_t)N_INSERT * 8 + (2u<<20)-1) & ~((size_t)(2u<<20)-1));
    }

    /* === CONTAINS HIT === */
    {
        memcpy(xs, seed, sizeof(xs));
        uint64_t *keys = gen_keys(N_INSERT);

        /* packed */
        {
            struct bpacked m;
            bpacked_init_cap(&m, N_INSERT);
            for (int i = 0; i < N_INSERT; i++) bpacked_insert_unique(&m, keys[i]);
            double t0 = now_ns();
            int hits = 0;
            for (int i = 0; i < N_INSERT; i++) {
                if (i + PF < N_INSERT)
                    bpacked_prefetch(&m, keys[i + PF]);
                hits += bpacked_contains(&m, keys[i]);
            }
            double dt = now_ns() - t0;
            printf("packed\tcontains_hit\t%.1f\t%d\n", dt / N_INSERT, hits);
            bpacked_destroy(&m);
        }

        /* split */
        {
            struct bsplit m;
            bsplit_init_cap(&m, N_INSERT);
            for (int i = 0; i < N_INSERT; i++) bsplit_insert_unique(&m, keys[i]);
            double t0 = now_ns();
            int hits = 0;
            for (int i = 0; i < N_INSERT; i++) {
                if (i + PF < N_INSERT)
                    bsplit_prefetch(&m, keys[i + PF]);
                hits += bsplit_contains(&m, keys[i]);
            }
            double dt = now_ns() - t0;
            printf("split\tcontains_hit\t%.1f\t%d\n", dt / N_INSERT, hits);
            bsplit_destroy(&m);
        }

        /* map48 */
        {
            struct b48 m;
            b48_init_cap(&m, N_INSERT);
            for (int i = 0; i < N_INSERT; i++) b48_insert_unique(&m, keys[i]);
            double t0 = now_ns();
            int hits = 0;
            for (int i = 0; i < N_INSERT; i++) {
                if (i + PF < N_INSERT)
                    b48_prefetch(&m, keys[i + PF]);
                hits += b48_contains(&m, keys[i]);
            }
            double dt = now_ns() - t0;
            printf("map48\tcontains_hit\t%.1f\t%d\n", dt / N_INSERT, hits);
            b48_destroy(&m);
        }

        /* map64 */
        {
            struct b64 m;
            b64_init_cap(&m, N_INSERT);
            for (int i = 0; i < N_INSERT; i++) b64_insert_unique(&m, keys[i]);
            double t0 = now_ns();
            int hits = 0;
            for (int i = 0; i < N_INSERT; i++) {
                if (i + PF < N_INSERT)
                    b64_prefetch(&m, keys[i + PF]);
                hits += b64_contains(&m, keys[i]);
            }
            double dt = now_ns() - t0;
            printf("map64\tcontains_hit\t%.1f\t%d\n", dt / N_INSERT, hits);
            b64_destroy(&m);
        }

        munmap(keys, ((size_t)N_INSERT * 8 + (2u<<20)-1) & ~((size_t)(2u<<20)-1));
    }

    /* === MIXED 50/25/25 === */
    {
        uint64_t mix_seed[4] = {0xCAFEBABE12345678ULL, 0xDEADBEEF87654321ULL,
                                0x0123456789ABCDEFULL, 0xFEDCBA9876543210ULL};
        memcpy(xs, mix_seed, sizeof(xs));

        int pool = N_INSERT;
        uint64_t *pool_keys = gen_keys(pool);
        struct mixed_work w = gen_mixed(pool, N_CHURN);

        /* packed */
        {
            struct bpacked m;
            bpacked_init_cap(&m, pool * 2);
            for (int i = 0; i < pool; i++) bpacked_insert_unique(&m, pool_keys[i]);
            double t0 = now_ns();
            for (int i = 0; i < w.n; i++) {
                uint64_t k = w.keys[pool + i];
                if (i + PF < w.n) {
                    uint64_t pk = w.keys[pool + i + PF];
                    if (w.ops[i + PF] == OP_INS)
                        bpacked_prefetch_insert(&m, pk);
                    else
                        bpacked_prefetch(&m, pk);
                }
                switch (w.ops[i]) {
                    case OP_GET: bpacked_contains(&m, k); break;
                    case OP_INS: bpacked_insert(&m, k); break;
                    case OP_DEL: bpacked_delete(&m, k); break;
                }
            }
            double dt = now_ns() - t0;
            printf("packed\tmixed_50_25_25\t%.1f\t%u\n", dt / w.n, m.count);
            bpacked_destroy(&m);
        }

        /* split */
        {
            struct bsplit m;
            bsplit_init_cap(&m, pool * 2);
            for (int i = 0; i < pool; i++) bsplit_insert_unique(&m, pool_keys[i]);
            double t0 = now_ns();
            for (int i = 0; i < w.n; i++) {
                uint64_t k = w.keys[pool + i];
                if (i + PF < w.n) {
                    uint64_t pk = w.keys[pool + i + PF];
                    if (w.ops[i + PF] == OP_INS)
                        bsplit_prefetch_insert(&m, pk);
                    else
                        bsplit_prefetch(&m, pk);
                }
                switch (w.ops[i]) {
                    case OP_GET: bsplit_contains(&m, k); break;
                    case OP_INS: bsplit_insert(&m, k); break;
                    case OP_DEL: bsplit_delete(&m, k); break;
                }
            }
            double dt = now_ns() - t0;
            printf("split\tmixed_50_25_25\t%.1f\t%u\n", dt / w.n, m.count);
            bsplit_destroy(&m);
        }

        /* map48 */
        {
            struct b48 m;
            b48_init_cap(&m, pool * 2);
            for (int i = 0; i < pool; i++) b48_insert_unique(&m, pool_keys[i]);
            double t0 = now_ns();
            for (int i = 0; i < w.n; i++) {
                uint64_t k = w.keys[pool + i];
                if (i + PF < w.n) {
                    uint64_t pk = w.keys[pool + i + PF];
                    if (w.ops[i + PF] == OP_INS)
                        b48_prefetch_insert(&m, pk);
                    else
                        b48_prefetch(&m, pk);
                }
                switch (w.ops[i]) {
                    case OP_GET: b48_contains(&m, k); break;
                    case OP_INS: b48_insert(&m, k); break;
                    case OP_DEL: b48_delete(&m, k); break;
                }
            }
            double dt = now_ns() - t0;
            printf("map48\tmixed_50_25_25\t%.1f\t%u\n", dt / w.n, m.count);
            b48_destroy(&m);
        }

        /* map64 */
        {
            struct b64 m;
            b64_init_cap(&m, pool * 2);
            for (int i = 0; i < pool; i++) b64_insert_unique(&m, pool_keys[i]);
            double t0 = now_ns();
            for (int i = 0; i < w.n; i++) {
                uint64_t k = w.keys[pool + i];
                if (i + PF < w.n) {
                    uint64_t pk = w.keys[pool + i + PF];
                    if (w.ops[i + PF] == OP_INS)
                        b64_prefetch_insert(&m, pk);
                    else
                        b64_prefetch(&m, pk);
                }
                switch (w.ops[i]) {
                    case OP_GET: b64_contains(&m, k); break;
                    case OP_INS: b64_insert(&m, k); break;
                    case OP_DEL: b64_delete(&m, k); break;
                }
            }
            double dt = now_ns() - t0;
            printf("map64\tmixed_50_25_25\t%.1f\t%u\n", dt / w.n, m.count);
            b64_destroy(&m);
        }
    }

    return 0;
}
