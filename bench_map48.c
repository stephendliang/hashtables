/*
 * bench_map48.c — Benchmark: map48 vs sentinel(KW=1) vs map64
 *
 * All three store 48-bit keys (lower 48 bits of uint64_t).
 * map48: packed 6-byte keys, 256B groups (4 cache lines)
 * sentinel: 8-byte keys, 320B groups (5 cache lines)
 * map64: 8-byte keys, 64B groups (1 cache line)
 *
 * Build:
 *   cc -O3 -march=native -std=gnu11 -o bench_map48 bench_map48.c
 */

/* map48 set mode */
#define SIMD_MAP_NAME b48
#include "simd_map48.h"

/* sentinel KW=1 set mode */
#define SIMD_MAP_NAME bsent
#define SIMD_MAP_KEY_WORDS 1
#include "simd_sentinel.h"

/* map64 set mode */
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

/* Allocate with hugepages */
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

/* Pre-generate keys */
static uint64_t *gen_keys(int n) {
    uint64_t *k = hp_alloc((size_t)n * sizeof(uint64_t));
    for (int i = 0; i < n; i++) k[i] = make48();
    return k;
}

/* Mixed ops: 50% get, 25% insert, 25% delete */
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

    /* fill pool */
    for (int i = 0; i < pool; i++) w.keys[i] = make48();

    /* generate ops referencing pool keys + new keys */
    int next_new = pool;
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
    printf("# Map48 benchmark (N=%d, PF=%d)\n", N_INSERT, PF);
    printf("# variant\tworkload\tns_per_op\tcount\tbytes_per_entry\n");

    /* === INSERT-ONLY === */
    {
        /* Reset RNG to same state for each variant */
        uint64_t seed[4] = {0x180ec6d33cfd0abaULL, 0xd5a61266f0c9392cULL,
                            0xa9582618e03fc9aaULL, 0x39abdc4529b1661cULL};

        /* map48 */
        memcpy(xs, seed, sizeof(xs));
        uint64_t *keys = gen_keys(N_INSERT);
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
            size_t bytes = b48_mapsize(m.cap);
            printf("map48\tinsert_only\t%.1f\t%u\t%.1f\n",
                   dt / N_INSERT, m.count,
                   (double)bytes / m.count);
            b48_destroy(&m);
        }

        /* sentinel KW=1 */
        {
            struct bsent m;
            bsent_init_cap(&m, N_INSERT);
            double t0 = now_ns();
            for (int i = 0; i < N_INSERT; i++) {
                if (i + PF < N_INSERT)
                    bsent_prefetch_insert(&m, &keys[i + PF]);
                bsent_insert_unique(&m, &keys[i]);
            }
            double dt = now_ns() - t0;
            size_t bytes = bsent_mapsize(m.cap);
            printf("sentinel\tinsert_only\t%.1f\t%u\t%.1f\n",
                   dt / N_INSERT, m.count,
                   (double)bytes / m.count);
            bsent_destroy(&m);
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
            size_t bytes = b64_mapsize(m.cap);
            printf("map64\tinsert_only\t%.1f\t%u\t%.1f\n",
                   dt / N_INSERT, m.count,
                   (double)bytes / m.count);
            b64_destroy(&m);
        }

        munmap(keys, ((size_t)N_INSERT * 8 + (2u<<20)-1) & ~((size_t)(2u<<20)-1));
    }

    /* === CONTAINS HIT === */
    {
        uint64_t seed[4] = {0x180ec6d33cfd0abaULL, 0xd5a61266f0c9392cULL,
                            0xa9582618e03fc9aaULL, 0x39abdc4529b1661cULL};

        memcpy(xs, seed, sizeof(xs));
        uint64_t *keys = gen_keys(N_INSERT);

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
            printf("map48\tcontains_hit\t%.1f\t%d\t-\n",
                   dt / N_INSERT, hits);
            b48_destroy(&m);
        }

        /* sentinel */
        {
            struct bsent m;
            bsent_init_cap(&m, N_INSERT);
            for (int i = 0; i < N_INSERT; i++) bsent_insert_unique(&m, &keys[i]);

            double t0 = now_ns();
            int hits = 0;
            for (int i = 0; i < N_INSERT; i++) {
                if (i + PF < N_INSERT)
                    bsent_prefetch(&m, &keys[i + PF]);
                hits += bsent_contains(&m, &keys[i]);
            }
            double dt = now_ns() - t0;
            printf("sentinel\tcontains_hit\t%.1f\t%d\t-\n",
                   dt / N_INSERT, hits);
            bsent_destroy(&m);
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
            printf("map64\tcontains_hit\t%.1f\t%d\t-\n",
                   dt / N_INSERT, hits);
            b64_destroy(&m);
        }

        munmap(keys, ((size_t)N_INSERT * 8 + (2u<<20)-1) & ~((size_t)(2u<<20)-1));
    }

    /* === MIXED 50/25/25 === */
    {
        uint64_t seed[4] = {0xCAFEBABE12345678ULL, 0xDEADBEEF87654321ULL,
                            0x0123456789ABCDEFULL, 0xFEDCBA9876543210ULL};
        memcpy(xs, seed, sizeof(xs));

        int pool = N_INSERT;
        uint64_t *pool_keys = gen_keys(pool);
        struct mixed_work w = gen_mixed(pool, N_CHURN);

        /* Pre-populate all three with the pool, then run mixed ops */

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
            printf("map48\tmixed_50_25_25\t%.1f\t%u\t-\n",
                   dt / w.n, m.count);
            b48_destroy(&m);
        }

        /* sentinel */
        {
            struct bsent m;
            bsent_init_cap(&m, pool * 2);
            for (int i = 0; i < pool; i++) bsent_insert_unique(&m, &pool_keys[i]);

            double t0 = now_ns();
            for (int i = 0; i < w.n; i++) {
                uint64_t k = w.keys[pool + i];
                if (i + PF < w.n) {
                    uint64_t pk = w.keys[pool + i + PF];
                    if (w.ops[i + PF] == OP_INS)
                        bsent_prefetch_insert(&m, &pk);
                    else
                        bsent_prefetch(&m, &pk);
                }
                switch (w.ops[i]) {
                    case OP_GET: bsent_contains(&m, &k); break;
                    case OP_INS: bsent_insert(&m, &k); break;
                    case OP_DEL: bsent_delete(&m, &k); break;
                }
            }
            double dt = now_ns() - t0;
            printf("sentinel\tmixed_50_25_25\t%.1f\t%u\t-\n",
                   dt / w.n, m.count);
            bsent_destroy(&m);
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
            printf("map64\tmixed_50_25_25\t%.1f\t%u\t-\n",
                   dt / w.n, m.count);
            b64_destroy(&m);
        }
    }

    return 0;
}
