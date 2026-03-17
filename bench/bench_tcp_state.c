/*
 * bench_tcp_state.c — Map-mode benchmark: map64 vs split vs lem+bs at VW=1,2,4
 *
 * TCP connection state: 48-bit key (IPv4+port), variable-size value.
 *   VW=1: 8B  (counter/timestamp)
 *   VW=2: 16B (timestamp + flags)
 *   VW=4: 32B (full connection state)
 *
 * Workloads: 90/5/5 (packet processing), 50/25/25 (high churn)
 *
 * Build:
 *   cc -O3 -march=native -std=gnu11 -o bench_tcp_state bench_tcp_state.c
 *   taskset -c 0 ./bench_tcp_state
 */

/* map64 VW=1 */
#define SIMD_MAP_NAME           m64v1
#define SIMD_MAP64_VAL_WORDS    1
#include "simd_map64.h"

/* map64 VW=2 */
#define SIMD_MAP_NAME           m64v2
#define SIMD_MAP64_VAL_WORDS    2
#include "simd_map64.h"

/* map64 VW=4 */
#define SIMD_MAP_NAME           m64v4
#define SIMD_MAP64_VAL_WORDS    4
#include "simd_map64.h"

/* split VW=1 */
#define SIMD_MAP_NAME           sv1
#define SIMD_MAP48S_VAL_WORDS   1
#include "simd_map48_split.h"

/* split VW=2 */
#define SIMD_MAP_NAME           sv2
#define SIMD_MAP48S_VAL_WORDS   2
#include "simd_map48_split.h"

/* split VW=4 */
#define SIMD_MAP_NAME           sv4
#define SIMD_MAP48S_VAL_WORDS   4
#include "simd_map48_split.h"

/* lem+bs VW=1 */
#define SIMD_MAP_NAME           lb1
#define SIMD_MAP48LB_VAL_WORDS  1
#include "simd_map48_lembs.h"

/* lem+bs VW=2 */
#define SIMD_MAP_NAME           lb2
#define SIMD_MAP48LB_VAL_WORDS  2
#include "simd_map48_lembs.h"

/* lem+bs VW=4 */
#define SIMD_MAP_NAME           lb4
#define SIMD_MAP48LB_VAL_WORDS  4
#include "simd_map48_lembs.h"

#include <stdio.h>
#include <time.h>

#define N_CONN  2000000
#define N_OPS   4000000
#define PF      24

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
    do { k = xoshiro() & 0x0000FFFFFFFFFFFFULL; } while (k == 0 || (k >> 16) == 0xFFFFFFFF);
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

enum { OP_GET, OP_INS, OP_DEL };

struct work {
    uint64_t *keys;
    uint8_t  *ops;
    int       n;
};

static struct work gen_work(int pool, int ops, uint64_t *pool_keys,
                            int pct_get, int pct_ins) {
    struct work w;
    w.keys = hp_alloc((size_t)(pool + ops) * sizeof(uint64_t));
    w.ops  = hp_alloc((size_t)ops);
    w.n    = ops;
    for (int i = 0; i < pool; i++) w.keys[i] = pool_keys[i];
    for (int i = 0; i < ops; i++) {
        int pct = (int)(xoshiro() % 100);
        if (pct < pct_get) {
            w.ops[i] = OP_GET;
            w.keys[pool + i] = pool_keys[xoshiro() % (uint64_t)pool];
        } else if (pct < pct_get + pct_ins) {
            w.ops[i] = OP_INS;
            w.keys[pool + i] = make48();
        } else {
            w.ops[i] = OP_DEL;
            w.keys[pool + i] = pool_keys[xoshiro() % (uint64_t)pool];
        }
    }
    return w;
}

/* --- Runner macros --- */

#define RUN_MAP64(NAME, VW, LABEL) do {                                     \
    struct NAME m;                                                          \
    NAME##_init_cap(&m, N_CONN * 2);                                        \
    uint64_t val[VW];                                                       \
    for (int v = 0; v < VW; v++) val[v] = 42;                              \
    for (int i = 0; i < N_CONN; i++)                                        \
        NAME##_insert_unique(&m, pool_keys[i], val);                        \
    size_t bytes = NAME##_mapsize(m.cap);                                   \
    double t0 = now_ns();                                                   \
    for (int i = 0; i < w.n; i++) {                                         \
        uint64_t k = w.keys[N_CONN + i];                                    \
        if (i + PF < w.n) {                                                 \
            uint64_t pk = w.keys[N_CONN + i + PF];                          \
            if (w.ops[i + PF] == OP_INS)                                    \
                NAME##_prefetch_insert(&m, pk);                             \
            else                                                            \
                NAME##_prefetch(&m, pk);                                    \
        }                                                                   \
        switch (w.ops[i]) {                                                 \
            case OP_GET: NAME##_get(&m, k); break;                          \
            case OP_INS: val[0] = k; NAME##_insert(&m, k, val); break;     \
            case OP_DEL: NAME##_delete(&m, k); break;                       \
        }                                                                   \
    }                                                                       \
    double dt = now_ns() - t0;                                              \
    printf(LABEL "\t%s\t%.1f\t%.1f\t%.1f\n", wname, dt / w.n,              \
           (double)bytes / (1 << 20), (double)bytes / N_CONN);              \
    NAME##_destroy(&m);                                                     \
} while (0)

#define RUN_SPLIT(NAME, VW, LABEL) do {                                     \
    struct NAME m;                                                          \
    NAME##_init_cap(&m, N_CONN * 2);                                        \
    uint64_t val[VW];                                                       \
    for (int v = 0; v < VW; v++) val[v] = 42;                              \
    for (int i = 0; i < N_CONN; i++)                                        \
        NAME##_insert_unique(&m, pool_keys[i], val);                        \
    size_t bytes = NAME##_mapsize(m.ng);                                    \
    double t0 = now_ns();                                                   \
    for (int i = 0; i < w.n; i++) {                                         \
        uint64_t k = w.keys[N_CONN + i];                                    \
        if (i + PF < w.n) {                                                 \
            uint64_t pk = w.keys[N_CONN + i + PF];                          \
            if (w.ops[i + PF] == OP_INS)                                    \
                NAME##_prefetch_insert(&m, pk);                             \
            else                                                            \
                NAME##_prefetch(&m, pk);                                    \
        }                                                                   \
        switch (w.ops[i]) {                                                 \
            case OP_GET: NAME##_get(&m, k); break;                          \
            case OP_INS: val[0] = k; NAME##_insert(&m, k, val); break;     \
            case OP_DEL: NAME##_delete(&m, k); break;                       \
        }                                                                   \
    }                                                                       \
    double dt = now_ns() - t0;                                              \
    printf(LABEL "\t%s\t%.1f\t%.1f\t%.1f\n", wname, dt / w.n,              \
           (double)bytes / (1 << 20), (double)bytes / N_CONN);              \
    NAME##_destroy(&m);                                                     \
} while (0)

#define RUN_LEMBS(NAME, VW, LABEL) do {                                     \
    struct NAME m;                                                          \
    NAME##_init_cap(&m, N_CONN * 2);                                        \
    uint64_t val[VW];                                                       \
    for (int v = 0; v < VW; v++) val[v] = 42;                              \
    for (int i = 0; i < N_CONN; i++)                                        \
        NAME##_insert_unique(&m, pool_keys[i], val);                        \
    size_t bytes = NAME##_mapsize(m.ng);                                    \
    double t0 = now_ns();                                                   \
    for (int i = 0; i < w.n; i++) {                                         \
        uint64_t k = w.keys[N_CONN + i];                                    \
        if (i + PF < w.n) {                                                 \
            uint64_t pk = w.keys[N_CONN + i + PF];                          \
            if (w.ops[i + PF] == OP_INS)                                    \
                NAME##_prefetch_insert(&m, pk);                             \
            else                                                            \
                NAME##_prefetch(&m, pk);                                    \
        }                                                                   \
        switch (w.ops[i]) {                                                 \
            case OP_GET: NAME##_get(&m, k); break;                          \
            case OP_INS: val[0] = k; NAME##_insert(&m, k, val); break;     \
            case OP_DEL: NAME##_delete(&m, k); break;                       \
        }                                                                   \
    }                                                                       \
    double dt = now_ns() - t0;                                              \
    printf(LABEL "\t%s\t%.1f\t%.1f\t%.1f\n", wname, dt / w.n,              \
           (double)bytes / (1 << 20), (double)bytes / N_CONN);              \
    NAME##_destroy(&m);                                                     \
} while (0)

int main(void) {
    printf("# TCP state benchmark (N=%d, %d ops, PF=%d)\n", N_CONN, N_OPS, PF);
    printf("# variant\tworkload\tns_per_op\talloc_mib\tbytes_per_conn\n");

    struct { int g, i, d; const char *name; } profiles[] = {
        {90,  5,  5, "90_5_5"},
        {50, 25, 25, "50_25_25"},
    };

    for (int p = 0; p < 2; p++) {
        uint64_t seed[4] = {0xCAFEBABE12345678ULL + (uint64_t)p,
                            0xDEADBEEF87654321ULL,
                            0x0123456789ABCDEFULL,
                            0xFEDCBA9876543210ULL};
        memcpy(xs, seed, sizeof(xs));

        uint64_t *pool_keys = hp_alloc((size_t)N_CONN * sizeof(uint64_t));
        for (int i = 0; i < N_CONN; i++) pool_keys[i] = make48();

        struct work w = gen_work(N_CONN, N_OPS, pool_keys,
                                 profiles[p].g, profiles[p].i);
        const char *wname = profiles[p].name;

        RUN_MAP64(m64v1, 1, "map64_v1");
        RUN_MAP64(m64v2, 2, "map64_v2");
        RUN_MAP64(m64v4, 4, "map64_v4");
        RUN_SPLIT(sv1, 1, "split_v1");
        RUN_SPLIT(sv2, 2, "split_v2");
        RUN_SPLIT(sv4, 4, "split_v4");
        RUN_LEMBS(lb1, 1, "lembs_v1");
        RUN_LEMBS(lb2, 2, "lembs_v2");
        RUN_LEMBS(lb4, 4, "lembs_v4");

        hp_free(pool_keys, (size_t)N_CONN * 8);
        hp_free(w.keys, (size_t)(N_CONN + N_OPS) * 8);
        hp_free(w.ops, (size_t)N_OPS);
    }

    return 0;
}
