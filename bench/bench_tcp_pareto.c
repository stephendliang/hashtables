/*
 * bench_tcp_pareto.c — Pareto frontier: churn speed vs memory for TCP map
 *
 * Use case: IPv4(32) + src_port(16) = 48-bit key → connection state (VW=1)
 * Profiles: 90/5/5 (packet processing), 50/25/25 (high churn)
 *
 * Direct-compare variants only (no h2 metadata — pointless for 48-bit keys):
 *   map48      — native 48-bit keys, h2 metadata, 256B key groups
 *   map64      — 64-bit keys (waste 16 bits), direct compare, 64B groups
 *   split      — split hi32+lo16, direct compare, 64B groups (N=1 inline)
 *   split_n2   — split hi32+lo16, superblock N=2
 *   packed     — packed 3×u16, direct compare, 64B groups (N=1 inline)
 *   packed_n2  — packed 3×u16, superblock N=2
 *
 * All variants use map mode (VW=1).
 *
 * Build:
 *   cc -O3 -march=native -std=gnu11 -o bench_tcp bench_tcp_pareto.c
 */

/* --- map48 VW=1 --- */
#define SIMD_MAP_NAME         tcp_m48
#define SIMD_MAP48_VAL_WORDS  1
#include "simd_map48.h"

/* --- map64 VW=1, N=1 --- */
#define SIMD_MAP_NAME         tcp_m64
#define SIMD_MAP64_VAL_WORDS  1
#include "simd_map64.h"

/* --- split VW=1 N=1 (inline values) --- */
#define SIMD_MAP_NAME         tcp_split
#define SIMD_MAP48S_VAL_WORDS 1
#include "simd_map48_split.h"

/* --- split VW=1 N=2 (superblock) --- */
#define SIMD_MAP_NAME            tcp_split_n2
#define SIMD_MAP48S_VAL_WORDS    1
#define SIMD_MAP48S_BLOCK_STRIDE 2
#include "simd_map48_split.h"

/* --- packed VW=1 N=1 (inline values) --- */
#define SIMD_MAP_NAME         tcp_packed
#define SIMD_MAP48P_VAL_WORDS 1
#include "simd_map48_packed.h"

/* --- packed VW=1 N=2 (superblock) --- */
#define SIMD_MAP_NAME            tcp_packed_n2
#define SIMD_MAP48P_VAL_WORDS    1
#define SIMD_MAP48P_BLOCK_STRIDE 2
#include "simd_map48_packed.h"

#include <stdio.h>
#include <time.h>

#define N_CONN    2000000   /* concurrent connections */
#define N_OPS     4000000   /* ops per churn run */

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

enum { OP_GET, OP_INS, OP_DEL };

struct workload {
    uint64_t *keys;
    uint8_t  *ops;
    int       n;
    int       pct_get, pct_ins, pct_del;
};

static struct workload gen_work(int pool, int ops,
                                int pct_get, int pct_ins, int pct_del) {
    struct workload w;
    w.keys = hp_alloc((size_t)(pool + ops) * sizeof(uint64_t));
    w.ops  = hp_alloc((size_t)ops);
    w.n    = ops;
    w.pct_get = pct_get;
    w.pct_ins = pct_ins;
    w.pct_del = pct_del;

    for (int i = 0; i < pool; i++) w.keys[i] = make48();

    for (int i = 0; i < ops; i++) {
        int pct = (int)(xoshiro() % 100);
        if (pct < pct_get) {
            w.ops[i] = OP_GET;
            w.keys[pool + i] = w.keys[xoshiro() % (uint64_t)pool];
        } else if (pct < pct_get + pct_ins) {
            w.ops[i] = OP_INS;
            w.keys[pool + i] = make48();
        } else {
            w.ops[i] = OP_DEL;
            w.keys[pool + i] = w.keys[xoshiro() % (uint64_t)pool];
        }
    }
    return w;
}

/*
 * Macro-generated churn runners for each variant.
 * Each does: pre-populate pool, then run mixed ops with prefetch.
 */

static double run_m48(struct workload *w, int pool, uint64_t *pool_keys,
                      int pf, size_t *out_bytes) {
    struct tcp_m48 m;
    tcp_m48_init_cap(&m, pool * 2);

    uint64_t val[1] = {42};
    for (int i = 0; i < pool; i++)
        tcp_m48_insert_unique(&m, pool_keys[i], val);

    if (out_bytes) *out_bytes = tcp_m48_mapsize(m.cap);

    double t0 = now_ns();
    for (int i = 0; i < w->n; i++) {
        uint64_t k = w->keys[pool + i];
        if (i + pf < w->n) {
            uint64_t pk = w->keys[pool + i + pf];
            if (w->ops[i + pf] == OP_INS)
                tcp_m48_prefetch_insert(&m, pk);
            else
                tcp_m48_prefetch(&m, pk);
        }
        switch (w->ops[i]) {
            case OP_GET: tcp_m48_get(&m, k); break;
            case OP_INS: val[0] = k; tcp_m48_insert(&m, k, val); break;
            case OP_DEL: tcp_m48_delete(&m, k); break;
        }
    }
    double dt = now_ns() - t0;
    tcp_m48_destroy(&m);
    return dt / w->n;
}

static double run_m64(struct workload *w, int pool, uint64_t *pool_keys,
                      int pf, size_t *out_bytes) {
    struct tcp_m64 m;
    tcp_m64_init_cap(&m, pool * 2);

    uint64_t val[1] = {42};
    for (int i = 0; i < pool; i++)
        tcp_m64_insert_unique(&m, pool_keys[i], val);

    if (out_bytes) *out_bytes = tcp_m64_mapsize(m.cap);

    double t0 = now_ns();
    for (int i = 0; i < w->n; i++) {
        uint64_t k = w->keys[pool + i];
        if (i + pf < w->n) {
            uint64_t pk = w->keys[pool + i + pf];
            if (w->ops[i + pf] == OP_INS)
                tcp_m64_prefetch_insert(&m, pk);
            else
                tcp_m64_prefetch(&m, pk);
        }
        switch (w->ops[i]) {
            case OP_GET: tcp_m64_get(&m, k); break;
            case OP_INS: val[0] = k; tcp_m64_insert(&m, k, val); break;
            case OP_DEL: tcp_m64_delete(&m, k); break;
        }
    }
    double dt = now_ns() - t0;
    tcp_m64_destroy(&m);
    return dt / w->n;
}

static double run_split_n2(struct workload *w, int pool, uint64_t *pool_keys,
                           int pf, size_t *out_bytes) {
    struct tcp_split_n2 m;
    tcp_split_n2_init_cap(&m, pool * 2);

    uint64_t val[1] = {42};
    for (int i = 0; i < pool; i++)
        tcp_split_n2_insert_unique(&m, pool_keys[i], val);

    if (out_bytes) *out_bytes = tcp_split_n2_mapsize(m.ng);

    double t0 = now_ns();
    for (int i = 0; i < w->n; i++) {
        uint64_t k = w->keys[pool + i];
        if (i + pf < w->n) {
            uint64_t pk = w->keys[pool + i + pf];
            if (w->ops[i + pf] == OP_INS)
                tcp_split_n2_prefetch_insert(&m, pk);
            else
                tcp_split_n2_prefetch(&m, pk);
        }
        switch (w->ops[i]) {
            case OP_GET: tcp_split_n2_get(&m, k); break;
            case OP_INS: val[0] = k; tcp_split_n2_insert(&m, k, val); break;
            case OP_DEL: tcp_split_n2_delete(&m, k); break;
        }
    }
    double dt = now_ns() - t0;
    tcp_split_n2_destroy(&m);
    return dt / w->n;
}

static double run_packed_n2(struct workload *w, int pool, uint64_t *pool_keys,
                            int pf, size_t *out_bytes) {
    struct tcp_packed_n2 m;
    tcp_packed_n2_init_cap(&m, pool * 2);

    uint64_t val[1] = {42};
    for (int i = 0; i < pool; i++)
        tcp_packed_n2_insert_unique(&m, pool_keys[i], val);

    if (out_bytes) *out_bytes = tcp_packed_n2_mapsize(m.ng);

    double t0 = now_ns();
    for (int i = 0; i < w->n; i++) {
        uint64_t k = w->keys[pool + i];
        if (i + pf < w->n) {
            uint64_t pk = w->keys[pool + i + pf];
            if (w->ops[i + pf] == OP_INS)
                tcp_packed_n2_prefetch_insert(&m, pk);
            else
                tcp_packed_n2_prefetch(&m, pk);
        }
        switch (w->ops[i]) {
            case OP_GET: tcp_packed_n2_get(&m, k); break;
            case OP_INS: val[0] = k; tcp_packed_n2_insert(&m, k, val); break;
            case OP_DEL: tcp_packed_n2_delete(&m, k); break;
        }
    }
    double dt = now_ns() - t0;
    tcp_packed_n2_destroy(&m);
    return dt / w->n;
}

static double run_split(struct workload *w, int pool, uint64_t *pool_keys,
                        int pf, size_t *out_bytes) {
    struct tcp_split m;
    tcp_split_init_cap(&m, pool * 2);

    uint64_t val[1] = {42};
    for (int i = 0; i < pool; i++)
        tcp_split_insert_unique(&m, pool_keys[i], val);

    if (out_bytes) *out_bytes = tcp_split_mapsize(m.ng);

    double t0 = now_ns();
    for (int i = 0; i < w->n; i++) {
        uint64_t k = w->keys[pool + i];
        if (i + pf < w->n) {
            uint64_t pk = w->keys[pool + i + pf];
            if (w->ops[i + pf] == OP_INS)
                tcp_split_prefetch_insert(&m, pk);
            else
                tcp_split_prefetch(&m, pk);
        }
        switch (w->ops[i]) {
            case OP_GET: tcp_split_get(&m, k); break;
            case OP_INS: val[0] = k; tcp_split_insert(&m, k, val); break;
            case OP_DEL: tcp_split_delete(&m, k); break;
        }
    }
    double dt = now_ns() - t0;
    tcp_split_destroy(&m);
    return dt / w->n;
}

static double run_packed(struct workload *w, int pool, uint64_t *pool_keys,
                         int pf, size_t *out_bytes) {
    struct tcp_packed m;
    tcp_packed_init_cap(&m, pool * 2);

    uint64_t val[1] = {42};
    for (int i = 0; i < pool; i++)
        tcp_packed_insert_unique(&m, pool_keys[i], val);

    if (out_bytes) *out_bytes = tcp_packed_mapsize(m.ng);

    double t0 = now_ns();
    for (int i = 0; i < w->n; i++) {
        uint64_t k = w->keys[pool + i];
        if (i + pf < w->n) {
            uint64_t pk = w->keys[pool + i + pf];
            if (w->ops[i + pf] == OP_INS)
                tcp_packed_prefetch_insert(&m, pk);
            else
                tcp_packed_prefetch(&m, pk);
        }
        switch (w->ops[i]) {
            case OP_GET: tcp_packed_get(&m, k); break;
            case OP_INS: val[0] = k; tcp_packed_insert(&m, k, val); break;
            case OP_DEL: tcp_packed_delete(&m, k); break;
        }
    }
    double dt = now_ns() - t0;
    tcp_packed_destroy(&m);
    return dt / w->n;
}

int main(void) {
    printf("# TCP Pareto benchmark (N=%d connections, %d ops per workload)\n",
           N_CONN, N_OPS);
    printf("# variant\tmode\tworkload\tpf\tns_per_op\talloc_mib\tbytes_per_entry\n");

    static const int pf_vals[] = {16, 24, 32, 40, 48, 64};
    static const int n_pf = sizeof(pf_vals) / sizeof(pf_vals[0]);

    struct { int g, i, d; const char *name; } profiles[] = {
        {90,  5,  5, "90_5_5"},
        {50, 25, 25, "50_25_25"},
    };
    int n_prof = sizeof(profiles) / sizeof(profiles[0]);

    for (int p = 0; p < n_prof; p++) {
        /* Reset RNG per profile for reproducibility */
        uint64_t seed[4] = {0xCAFEBABE12345678ULL + p, 0xDEADBEEF87654321ULL,
                            0x0123456789ABCDEFULL, 0xFEDCBA9876543210ULL};
        memcpy(xs, seed, sizeof(xs));

        uint64_t *pool_keys = hp_alloc((size_t)N_CONN * sizeof(uint64_t));
        for (int i = 0; i < N_CONN; i++) pool_keys[i] = make48();

        struct workload w = gen_work(N_CONN, N_OPS,
                                     profiles[p].g, profiles[p].i, profiles[p].d);

        for (int pi = 0; pi < n_pf; pi++) {
            int pf = pf_vals[pi];
            size_t bytes;
            double ns;

            /* Map-mode variants */
            ns = run_m48(&w, N_CONN, pool_keys, pf, &bytes);
            printf("map48\tmap\t%s\t%d\t%.1f\t%.1f\t%.1f\n",
                   profiles[p].name, pf, ns,
                   (double)bytes / (1 << 20), (double)bytes / N_CONN);

            ns = run_m64(&w, N_CONN, pool_keys, pf, &bytes);
            printf("map64\tmap\t%s\t%d\t%.1f\t%.1f\t%.1f\n",
                   profiles[p].name, pf, ns,
                   (double)bytes / (1 << 20), (double)bytes / N_CONN);

            ns = run_split(&w, N_CONN, pool_keys, pf, &bytes);
            printf("split\tmap\t%s\t%d\t%.1f\t%.1f\t%.1f\n",
                   profiles[p].name, pf, ns,
                   (double)bytes / (1 << 20), (double)bytes / N_CONN);

            ns = run_split_n2(&w, N_CONN, pool_keys, pf, &bytes);
            printf("split_n2\tmap\t%s\t%d\t%.1f\t%.1f\t%.1f\n",
                   profiles[p].name, pf, ns,
                   (double)bytes / (1 << 20), (double)bytes / N_CONN);

            ns = run_packed(&w, N_CONN, pool_keys, pf, &bytes);
            printf("packed\tmap\t%s\t%d\t%.1f\t%.1f\t%.1f\n",
                   profiles[p].name, pf, ns,
                   (double)bytes / (1 << 20), (double)bytes / N_CONN);

            ns = run_packed_n2(&w, N_CONN, pool_keys, pf, &bytes);
            printf("packed_n2\tmap\t%s\t%d\t%.1f\t%.1f\t%.1f\n",
                   profiles[p].name, pf, ns,
                   (double)bytes / (1 << 20), (double)bytes / N_CONN);
        }

        /* free workload */
        size_t ksz = ((size_t)(N_CONN + N_OPS) * 8 + (2u<<20)-1) & ~((size_t)(2u<<20)-1);
        size_t osz = ((size_t)N_OPS + (2u<<20)-1) & ~((size_t)(2u<<20)-1);
        munmap(w.keys, ksz);
        munmap(w.ops, osz);

        size_t psz = ((size_t)N_CONN * 8 + (2u<<20)-1) & ~((size_t)(2u<<20)-1);
        munmap(pool_keys, psz);
    }

    return 0;
}
