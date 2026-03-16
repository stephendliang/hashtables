/*
 * Mixed-workload benchmark: sentinel vs bitstealing simd_set128.
 *
 * Compile twice — once per header variant:
 *   cc -O3 -march=native -std=gnu11 -o /tmp/bench_s  bench_map128_throughput.c
 *   cc -O3 -march=native -std=gnu11 -DBITSTEALING -o /tmp/bench_bs bench_map128_throughput.c
 */
#ifdef BITSTEALING
#include "simd_set128_bitstealing.h"
#else
#include "simd_set128_sentinel.h"
#endif

#include <stdio.h>
#include <time.h>

static inline uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static double elapsed_ns(struct timespec t0, struct timespec t1, int ops) {
    return ((t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec)) / ops;
}

/* Pipelined insert: prefetch PF slots ahead */
static void bench_insert_pf(struct simd_set128 *m,
                             uint64_t *klo, uint64_t *khi,
                             int n, int pf) {
    for (int i = 0; i < n; i++) {
        if (i + pf < n)
            simd_set128_prefetch(m, klo[i + pf], khi[i + pf]);
        simd_set128_insert_unique(m, klo[i], khi[i]);
    }
}

/* Pipelined contains */
static int bench_contains_pf(struct simd_set128 *m,
                              uint64_t *klo, uint64_t *khi,
                              int n, int pf) {
    int found = 0;
    for (int i = 0; i < n; i++) {
        if (i + pf < n)
            simd_set128_prefetch(m, klo[i + pf], khi[i + pf]);
        found += simd_set128_contains(m, klo[i], khi[i]);
    }
    return found;
}

/* Pipelined delete */
static int bench_delete_pf(struct simd_set128 *m,
                            uint64_t *klo, uint64_t *khi,
                            int n, int pf) {
    int deleted = 0;
    for (int i = 0; i < n; i++) {
        if (i + pf < n)
            simd_set128_prefetch(m, klo[i + pf], khi[i + pf]);
        deleted += simd_set128_delete(m, klo[i], khi[i]);
    }
    return deleted;
}

#define N       2000000
#define PF      24
#define CHURN   200000   /* keys replaced per churn round */
#define ROUNDS  10       /* churn rounds */

int main(void) {
    uint64_t seed_lo = 0xdeadbeefcafe1234ULL;
    uint64_t seed_hi = 0x0123456789abcdefULL;
    uint64_t *klo = malloc(N * sizeof(uint64_t));
    uint64_t *khi = malloc(N * sizeof(uint64_t));
    for (int i = 0; i < N; i++) {
        klo[i] = splitmix64(&seed_lo);
        khi[i] = splitmix64(&seed_hi);
    }

    /* Extra keys for churn — not in initial set */
    int extra_n = CHURN * ROUNDS;
    uint64_t *exlo = malloc(extra_n * sizeof(uint64_t));
    uint64_t *exhi = malloc(extra_n * sizeof(uint64_t));
    uint64_t ex_seed_lo = 0xAAAABBBBCCCCDDDDULL;
    uint64_t ex_seed_hi = 0x1111222233334444ULL;
    for (int i = 0; i < extra_n; i++) {
        exlo[i] = splitmix64(&ex_seed_lo);
        exhi[i] = splitmix64(&ex_seed_hi);
    }

    /* Miss keys — guaranteed not in map */
    uint64_t *mlo = malloc(N * sizeof(uint64_t));
    uint64_t *mhi = malloc(N * sizeof(uint64_t));
    uint64_t m_seed_lo = 0xFEDCBA9876543210ULL;
    uint64_t m_seed_hi = 0x0F1E2D3C4B5A6978ULL;
    for (int i = 0; i < N; i++) {
        mlo[i] = splitmix64(&m_seed_lo);
        mhi[i] = splitmix64(&m_seed_hi);
    }

    struct timespec t0, t1;

#ifdef BITSTEALING
    printf("=== simd_set128 BITSTEALING (N=%d, PF=%d) ===\n", N, PF);
#else
    printf("=== simd_set128 SENTINEL (N=%d, PF=%d) ===\n", N, PF);
#endif

    /* --- 1. Bulk insert (pipelined, unique) --- */
    struct simd_set128 m;
    simd_set128_init_cap(&m, N);

    clock_gettime(CLOCK_MONOTONIC, &t0);
    bench_insert_pf(&m, klo, khi, N, PF);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("  bulk insert (unique, PF):  %6.1f ns/op\n", elapsed_ns(t0, t1, N));

    /* --- 2. Contains hit (pipelined) --- */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int hits = bench_contains_pf(&m, klo, khi, N, PF);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("  contains hit (PF):         %6.1f ns/op  (found=%d)\n",
           elapsed_ns(t0, t1, N), hits);

    /* --- 3. Contains miss (pipelined) --- */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int misses = bench_contains_pf(&m, mlo, mhi, N, PF);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("  contains miss (PF):        %6.1f ns/op  (found=%d)\n",
           elapsed_ns(t0, t1, N), misses);

    /* --- 4. Mixed churn: delete CHURN old + insert CHURN new, ROUNDS times --- */
    /*    Simulates a running system with steady-state turnover.               */
    /*    After this, ghost overflow bits have accumulated (bitstealing).       */
    double churn_del_total = 0, churn_ins_total = 0;
    int churn_ops = CHURN * ROUNDS;

    /* We delete from the tail of the initial keyspace and insert from extras */
    int del_cursor = N - 1;
    int ins_cursor = 0;

    for (int r = 0; r < ROUNDS; r++) {
        /* Delete CHURN keys (backwards from end of initial set) */
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int i = 0; i < CHURN; i++) {
            int idx = del_cursor - (r * CHURN + i);
            if (idx >= 0) {
                if (idx + PF - i < N && i + PF < CHURN) {
                    int pidx = del_cursor - (r * CHURN + i + PF);
                    if (pidx >= 0)
                        simd_set128_prefetch(&m, klo[pidx], khi[pidx]);
                }
                simd_set128_delete(&m, klo[idx], khi[idx]);
            }
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        churn_del_total += (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);

        /* Insert CHURN new keys */
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int i = 0; i < CHURN; i++) {
            int eidx = ins_cursor + r * CHURN + i;
            if (eidx + PF < extra_n && i + PF < CHURN)
                simd_set128_prefetch(&m, exlo[eidx + PF], exhi[eidx + PF]);
            simd_set128_insert(&m, exlo[eidx], exhi[eidx]);
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        churn_ins_total += (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
    }
    printf("  churn delete (PF):         %6.1f ns/op  (%d rounds × %d)\n",
           churn_del_total / churn_ops, ROUNDS, CHURN);
    printf("  churn insert (PF):         %6.1f ns/op  (%d rounds × %d)\n",
           churn_ins_total / churn_ops, ROUNDS, CHURN);
    printf("  count after churn:         %u\n", m.count);

    /* --- 5. Post-churn contains hit (tests ghost-bit overhead) --- */
    /*    Look up the extra keys that were inserted during churn.     */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int post_hits = 0;
    for (int i = 0; i < churn_ops; i++) {
        if (i + PF < churn_ops)
            simd_set128_prefetch(&m, exlo[i + PF], exhi[i + PF]);
        post_hits += simd_set128_contains(&m, exlo[i], exhi[i]);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("  post-churn hit (PF):       %6.1f ns/op  (found=%d/%d)\n",
           elapsed_ns(t0, t1, churn_ops), post_hits, churn_ops);

    /* --- 6. Post-churn contains miss (ghost-bit impact on misses) --- */
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int post_miss = bench_contains_pf(&m, mlo, mhi, N, PF);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("  post-churn miss (PF):      %6.1f ns/op  (found=%d)\n",
           elapsed_ns(t0, t1, N), post_miss);

    /* --- 7. Delete all remaining --- */
    /* First, delete the surviving initial keys */
    int surviving_start = 0;
    int surviving_end = N - CHURN * ROUNDS;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = surviving_start; i < surviving_end; i++) {
        if (i + PF < surviving_end)
            simd_set128_prefetch(&m, klo[i + PF], khi[i + PF]);
        simd_set128_delete(&m, klo[i], khi[i]);
    }
    /* Then delete the churn-inserted keys */
    for (int i = 0; i < churn_ops; i++) {
        if (i + PF < churn_ops)
            simd_set128_prefetch(&m, exlo[i + PF], exhi[i + PF]);
        simd_set128_delete(&m, exlo[i], exhi[i]);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    int total_del = surviving_end - surviving_start + churn_ops;
    printf("  delete all (PF):           %6.1f ns/op  (count=%u)\n",
           elapsed_ns(t0, t1, total_del), m.count);

    simd_set128_destroy(&m);
    free(klo); free(khi);
    free(exlo); free(exhi);
    free(mlo); free(mhi);
    return m.count != 0;
}
