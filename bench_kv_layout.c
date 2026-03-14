/*
 * bench_kv_layout.c — Grid search benchmark for KV map layout strategies.
 *
 * Instantiates all layout × overflow combinations and sweeps PF values.
 * Output is TSV for easy parsing.
 *
 * Build & run:
 *   cc -O3 -march=native -std=gnu11 -o bench_kv bench_kv_layout.c
 *   taskset -c 4 ./bench_kv > results.tsv
 */

/* --- Sentinel instantiations --- */
#define SIMD_MAP_NAME      s1_sent
#define SIMD_MAP_KEY_WORDS 2
#define SIMD_MAP_VAL_WORDS 1
#define SIMD_KV_LAYOUT     1
#include "simd_kv_sentinel.h"

#define SIMD_MAP_NAME      s2_sent
#define SIMD_MAP_KEY_WORDS 2
#define SIMD_MAP_VAL_WORDS 1
#define SIMD_KV_LAYOUT     2
#include "simd_kv_sentinel.h"

#define SIMD_MAP_NAME      s3n1_sent
#define SIMD_MAP_KEY_WORDS 2
#define SIMD_MAP_VAL_WORDS 1
#define SIMD_KV_LAYOUT     3
#define SIMD_KV_BLOCK_STRIDE 1
#include "simd_kv_sentinel.h"

#define SIMD_MAP_NAME      s3n2_sent
#define SIMD_MAP_KEY_WORDS 2
#define SIMD_MAP_VAL_WORDS 1
#define SIMD_KV_LAYOUT     3
#define SIMD_KV_BLOCK_STRIDE 2
#include "simd_kv_sentinel.h"

#define SIMD_MAP_NAME      s3n4_sent
#define SIMD_MAP_KEY_WORDS 2
#define SIMD_MAP_VAL_WORDS 1
#define SIMD_KV_LAYOUT     3
#define SIMD_KV_BLOCK_STRIDE 4
#include "simd_kv_sentinel.h"

#define SIMD_MAP_NAME      s3n8_sent
#define SIMD_MAP_KEY_WORDS 2
#define SIMD_MAP_VAL_WORDS 1
#define SIMD_KV_LAYOUT     3
#define SIMD_KV_BLOCK_STRIDE 8
#include "simd_kv_sentinel.h"

#define SIMD_MAP_NAME      s3n16_sent
#define SIMD_MAP_KEY_WORDS 2
#define SIMD_MAP_VAL_WORDS 1
#define SIMD_KV_LAYOUT     3
#define SIMD_KV_BLOCK_STRIDE 16
#include "simd_kv_sentinel.h"

#define SIMD_MAP_NAME      s3n32_sent
#define SIMD_MAP_KEY_WORDS 2
#define SIMD_MAP_VAL_WORDS 1
#define SIMD_KV_LAYOUT     3
#define SIMD_KV_BLOCK_STRIDE 32
#include "simd_kv_sentinel.h"

/* --- Bitstealing instantiations --- */
#define SIMD_MAP_NAME      s1_bs
#define SIMD_MAP_KEY_WORDS 2
#define SIMD_MAP_VAL_WORDS 1
#define SIMD_KV_LAYOUT     1
#include "simd_kv_bitstealing.h"

#define SIMD_MAP_NAME      s2_bs
#define SIMD_MAP_KEY_WORDS 2
#define SIMD_MAP_VAL_WORDS 1
#define SIMD_KV_LAYOUT     2
#include "simd_kv_bitstealing.h"

#define SIMD_MAP_NAME      s3n1_bs
#define SIMD_MAP_KEY_WORDS 2
#define SIMD_MAP_VAL_WORDS 1
#define SIMD_KV_LAYOUT     3
#define SIMD_KV_BLOCK_STRIDE 1
#include "simd_kv_bitstealing.h"

#define SIMD_MAP_NAME      s3n2_bs
#define SIMD_MAP_KEY_WORDS 2
#define SIMD_MAP_VAL_WORDS 1
#define SIMD_KV_LAYOUT     3
#define SIMD_KV_BLOCK_STRIDE 2
#include "simd_kv_bitstealing.h"

#define SIMD_MAP_NAME      s3n4_bs
#define SIMD_MAP_KEY_WORDS 2
#define SIMD_MAP_VAL_WORDS 1
#define SIMD_KV_LAYOUT     3
#define SIMD_KV_BLOCK_STRIDE 4
#include "simd_kv_bitstealing.h"

#define SIMD_MAP_NAME      s3n8_bs
#define SIMD_MAP_KEY_WORDS 2
#define SIMD_MAP_VAL_WORDS 1
#define SIMD_KV_LAYOUT     3
#define SIMD_KV_BLOCK_STRIDE 8
#include "simd_kv_bitstealing.h"

#define SIMD_MAP_NAME      s3n16_bs
#define SIMD_MAP_KEY_WORDS 2
#define SIMD_MAP_VAL_WORDS 1
#define SIMD_KV_LAYOUT     3
#define SIMD_KV_BLOCK_STRIDE 16
#include "simd_kv_bitstealing.h"

#define SIMD_MAP_NAME      s3n32_bs
#define SIMD_MAP_KEY_WORDS 2
#define SIMD_MAP_VAL_WORDS 1
#define SIMD_KV_LAYOUT     3
#define SIMD_KV_BLOCK_STRIDE 32
#include "simd_kv_bitstealing.h"

#include <stdio.h>
#include <time.h>

#define N       2000000
#define KW      2
#define VW      1
#define CHURN   200000
#define ROUNDS  10

static inline uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static double elapsed_ns(struct timespec t0, struct timespec t1, int ops) {
    return ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec)) / ops;
}

static uint64_t *g_keys, *g_vals, *g_miss, *g_exkeys, *g_exvals;

static void gen_data(void) {
    g_keys   = malloc((size_t)N * KW * sizeof(uint64_t));
    g_vals   = malloc((size_t)N * VW * sizeof(uint64_t));
    g_miss   = malloc((size_t)N * KW * sizeof(uint64_t));
    int extra_n = CHURN * ROUNDS;
    g_exkeys = malloc((size_t)extra_n * KW * sizeof(uint64_t));
    g_exvals = malloc((size_t)extra_n * VW * sizeof(uint64_t));

    uint64_t s1 = 0xdeadbeefcafe1234ULL;
    for (int i = 0; i < N * KW; i++) g_keys[i] = splitmix64(&s1);
    uint64_t s2 = 0x1111222233334444ULL;
    for (int i = 0; i < N * VW; i++) g_vals[i] = splitmix64(&s2);
    uint64_t s3 = 0xFEDCBA9876543210ULL;
    for (int i = 0; i < N * KW; i++) g_miss[i] = splitmix64(&s3);
    uint64_t s4 = 0xAAAABBBBCCCCDDDDULL;
    for (int i = 0; i < extra_n * KW; i++) g_exkeys[i] = splitmix64(&s4);
    uint64_t s5 = 0x5555666677778888ULL;
    for (int i = 0; i < extra_n * VW; i++) g_exvals[i] = splitmix64(&s5);
}

/*
 * BENCH macro: for each PF value, run insert/get_hit/get_miss/delete.
 * Then run churn at the best PF.
 */
#define BENCH(NAME, LAYOUT, OVERFLOW, BLOCKN) do {                           \
    static const int pfs[] = {16, 20, 24, 28, 32, 36, 40};                  \
    int npf = (int)(sizeof(pfs) / sizeof(pfs[0]));                           \
    struct timespec t0, t1;                                                  \
                                                                             \
    double best_get_ns = 1e9;                                                \
    int best_pf = 24;                                                        \
                                                                             \
    for (int pi = 0; pi < npf; pi++) {                                       \
        int pf = pfs[pi];                                                    \
                                                                             \
        /* Insert (init_cap + insert_unique, pipelined) */                   \
        struct NAME m;                                                       \
        NAME##_init_cap(&m, N);                                              \
        clock_gettime(CLOCK_MONOTONIC, &t0);                                 \
        for (int i = 0; i < N; i++) {                                        \
            if (i + pf < N)                                                  \
                NAME##_prefetch(&m, &g_keys[(i + pf) * KW]);                \
            NAME##_insert_unique(&m, &g_keys[i * KW], &g_vals[i * VW]);     \
        }                                                                    \
        clock_gettime(CLOCK_MONOTONIC, &t1);                                 \
        double ins_ns = elapsed_ns(t0, t1, N);                               \
        printf("%s\t%s\t%s\t%d\tinsert\t%.1f\n",                            \
               LAYOUT, OVERFLOW, BLOCKN, pf, ins_ns);                        \
                                                                             \
        /* Get hit */                                                        \
        clock_gettime(CLOCK_MONOTONIC, &t0);                                 \
        int found = 0;                                                       \
        for (int i = 0; i < N; i++) {                                        \
            if (i + pf < N)                                                  \
                NAME##_prefetch(&m, &g_keys[(i + pf) * KW]);                \
            found += (NAME##_get(&m, &g_keys[i * KW]) != NULL);             \
        }                                                                    \
        clock_gettime(CLOCK_MONOTONIC, &t1);                                 \
        double hit_ns = elapsed_ns(t0, t1, N);                               \
        printf("%s\t%s\t%s\t%d\tget_hit\t%.1f\n",                           \
               LAYOUT, OVERFLOW, BLOCKN, pf, hit_ns);                        \
        (void)found;                                                         \
                                                                             \
        if (hit_ns < best_get_ns) { best_get_ns = hit_ns; best_pf = pf; }   \
                                                                             \
        /* Get miss */                                                       \
        clock_gettime(CLOCK_MONOTONIC, &t0);                                 \
        for (int i = 0; i < N; i++) {                                        \
            if (i + pf < N)                                                  \
                NAME##_prefetch(&m, &g_miss[(i + pf) * KW]);                \
            NAME##_get(&m, &g_miss[i * KW]);                                \
        }                                                                    \
        clock_gettime(CLOCK_MONOTONIC, &t1);                                 \
        printf("%s\t%s\t%s\t%d\tget_miss\t%.1f\n",                          \
               LAYOUT, OVERFLOW, BLOCKN, pf, elapsed_ns(t0, t1, N));        \
                                                                             \
        /* Delete all */                                                     \
        clock_gettime(CLOCK_MONOTONIC, &t0);                                 \
        for (int i = 0; i < N; i++) {                                        \
            if (i + pf < N)                                                  \
                NAME##_prefetch(&m, &g_keys[(i + pf) * KW]);                \
            NAME##_delete(&m, &g_keys[i * KW]);                             \
        }                                                                    \
        clock_gettime(CLOCK_MONOTONIC, &t1);                                 \
        printf("%s\t%s\t%s\t%d\tdelete\t%.1f\n",                            \
               LAYOUT, OVERFLOW, BLOCKN, pf, elapsed_ns(t0, t1, N));        \
                                                                             \
        NAME##_destroy(&m);                                                  \
    }                                                                        \
                                                                             \
    /* Churn test at best PF */                                              \
    {                                                                        \
        struct NAME m;                                                       \
        NAME##_init_cap(&m, N);                                              \
        for (int i = 0; i < N; i++)                                          \
            NAME##_insert_unique(&m, &g_keys[i * KW], &g_vals[i * VW]);     \
                                                                             \
        int pf = best_pf;                                                    \
        int extra_n = CHURN * ROUNDS;                                        \
        double churn_del = 0, churn_ins = 0;                                 \
        int del_cursor = N - 1;                                              \
                                                                             \
        for (int r = 0; r < ROUNDS; r++) {                                   \
            /* Delete CHURN from initial keys (tail) */                      \
            clock_gettime(CLOCK_MONOTONIC, &t0);                             \
            for (int i = 0; i < CHURN; i++) {                                \
                int idx = del_cursor - (r * CHURN + i);                      \
                if (idx >= 0) {                                              \
                    int pidx = del_cursor - (r * CHURN + i + pf);            \
                    if (pidx >= 0 && i + pf < CHURN)                         \
                        NAME##_prefetch(&m, &g_keys[pidx * KW]);            \
                    NAME##_delete(&m, &g_keys[idx * KW]);                   \
                }                                                            \
            }                                                                \
            clock_gettime(CLOCK_MONOTONIC, &t1);                             \
            churn_del += (t1.tv_sec-t0.tv_sec)*1e9+(t1.tv_nsec-t0.tv_nsec); \
                                                                             \
            /* Insert CHURN new keys */                                      \
            clock_gettime(CLOCK_MONOTONIC, &t0);                             \
            for (int i = 0; i < CHURN; i++) {                                \
                int eidx = r * CHURN + i;                                    \
                if (eidx + pf < extra_n && i + pf < CHURN)                   \
                    NAME##_prefetch(&m, &g_exkeys[(eidx + pf) * KW]);       \
                NAME##_insert(&m, &g_exkeys[eidx * KW],                     \
                              &g_exvals[eidx * VW]);                         \
            }                                                                \
            clock_gettime(CLOCK_MONOTONIC, &t1);                             \
            churn_ins += (t1.tv_sec-t0.tv_sec)*1e9+(t1.tv_nsec-t0.tv_nsec); \
        }                                                                    \
        int churn_ops = CHURN * ROUNDS;                                      \
        printf("%s\t%s\t%s\t%d\tchurn_del\t%.1f\n",                         \
               LAYOUT, OVERFLOW, BLOCKN, pf, churn_del / churn_ops);        \
        printf("%s\t%s\t%s\t%d\tchurn_ins\t%.1f\n",                         \
               LAYOUT, OVERFLOW, BLOCKN, pf, churn_ins / churn_ops);        \
                                                                             \
        /* Post-churn get hit on churned-in keys */                          \
        clock_gettime(CLOCK_MONOTONIC, &t0);                                 \
        int post_hits = 0;                                                   \
        for (int i = 0; i < churn_ops; i++) {                                \
            if (i + pf < churn_ops)                                          \
                NAME##_prefetch(&m, &g_exkeys[(i + pf) * KW]);             \
            post_hits += (NAME##_get(&m, &g_exkeys[i * KW]) != NULL);      \
        }                                                                    \
        clock_gettime(CLOCK_MONOTONIC, &t1);                                 \
        printf("%s\t%s\t%s\t%d\tpost_churn_hit\t%.1f\n",                    \
               LAYOUT, OVERFLOW, BLOCKN, pf,                                \
               elapsed_ns(t0, t1, churn_ops));                               \
        (void)post_hits;                                                     \
                                                                             \
        NAME##_destroy(&m);                                                  \
    }                                                                        \
    fflush(stdout);                                                          \
} while (0)

int main(void) {
    gen_data();

    printf("# layout\toverflow\tblockN\tPF\toperation\tns_per_op\n");

    BENCH(s1_sent,    "inline",   "sentinel",    "-");
    BENCH(s2_sent,    "separate", "sentinel",    "-");
    BENCH(s3n1_sent,  "hybrid",   "sentinel",    "1");
    BENCH(s3n2_sent,  "hybrid",   "sentinel",    "2");
    BENCH(s3n4_sent,  "hybrid",   "sentinel",    "4");
    BENCH(s3n8_sent,  "hybrid",   "sentinel",    "8");
    BENCH(s3n16_sent, "hybrid",   "sentinel",    "16");
    BENCH(s3n32_sent, "hybrid",   "sentinel",    "32");

    BENCH(s1_bs,      "inline",   "bitstealing", "-");
    BENCH(s2_bs,      "separate", "bitstealing", "-");
    BENCH(s3n1_bs,    "hybrid",   "bitstealing", "1");
    BENCH(s3n2_bs,    "hybrid",   "bitstealing", "2");
    BENCH(s3n4_bs,    "hybrid",   "bitstealing", "4");
    BENCH(s3n8_bs,    "hybrid",   "bitstealing", "8");
    BENCH(s3n16_bs,   "hybrid",   "bitstealing", "16");
    BENCH(s3n32_bs,   "hybrid",   "bitstealing", "32");

    free(g_keys); free(g_vals); free(g_miss);
    free(g_exkeys); free(g_exvals);
    return 0;
}
