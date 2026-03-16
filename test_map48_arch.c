/*
 * test_map48_arch.c — Correctness tests for 5 split48 architecture variants
 *
 * Build:
 *   cc -O3 -march=native -std=gnu11 -o test_map48_arch test_map48_arch.c
 *   cc -O3 -march=native -mno-avx2 -mno-avx512f -std=gnu11 -o test_map48_arch_scalar test_map48_arch.c
 */

/* 1. Baseline: split 1CL/10 */
#define SIMD_MAP_NAME t_base
#include "simd_map48_split.h"

/* 2. 3CL/31 fully vectorized */
#define SIMD_MAP_NAME t_3cl
#include "simd_map48_3cl.h"

/* 3. 2CL/20 ACP */
#define SIMD_MAP_NAME t_2cl
#include "simd_map48_2cl.h"

/* 4. 1CL/10 backshift */
#define SIMD_MAP_NAME t_bs
#include "simd_map48_bs.h"

/* 5. 1CL/10 Lemire */
#define SIMD_MAP_NAME t_lem
#include "simd_map48_lemire.h"

#include <stdio.h>

#define N 2000000

static inline uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline uint64_t make48(uint64_t *state) {
    uint64_t k;
    do { k = splitmix64(state) & 0x0000FFFFFFFFFFFFULL; } while (k == 0);
    return k;
}

/* For backshift: ensure (key >> 16) != 0xFFFFFFFF */
static inline uint64_t make48_bs(uint64_t *state) {
    uint64_t k;
    do {
        k = splitmix64(state) & 0x0000FFFFFFFFFFFFULL;
    } while (k == 0 || (k >> 16) == 0xFFFFFFFF);
    return k;
}

#define TEST_SET(NAME, LABEL, KEYFN) do {                                    \
    uint64_t *keys = malloc((size_t)N * sizeof(uint64_t));                   \
    uint64_t s = 0xdeadbeefcafe1234ULL;                                      \
    for (int i = 0; i < N; i++) keys[i] = KEYFN(&s);                        \
                                                                             \
    struct NAME m;                                                           \
    NAME##_init(&m);                                                         \
    int err = 0;                                                             \
                                                                             \
    /* insert all */                                                         \
    for (int i = 0; i < N; i++) {                                            \
        int r = NAME##_insert(&m, keys[i]);                                  \
        if (r != 1) { printf("    insert fail at %d\n", i); err++; break; }  \
    }                                                                        \
    if (m.count != (uint32_t)N) {                                            \
        printf("    count=%u expected=%d\n", m.count, N); err++;             \
    }                                                                        \
                                                                             \
    /* dup insert */                                                         \
    for (int i = 0; i < 1000; i++) {                                         \
        if (NAME##_insert(&m, keys[i]) != 0) {                              \
            printf("    dup insert fail at %d\n", i); err++; break;          \
        }                                                                    \
    }                                                                        \
                                                                             \
    /* contains hit */                                                       \
    for (int i = 0; i < N; i++) {                                            \
        if (!NAME##_contains(&m, keys[i])) {                                \
            printf("    contains miss at %d key=%lx\n", i,                   \
                   (unsigned long)keys[i]);                                   \
            err++; break;                                                    \
        }                                                                    \
    }                                                                        \
                                                                             \
    /* contains miss */                                                      \
    uint64_t s2 = 0xFEDCBA9876543210ULL;                                     \
    for (int i = 0; i < 10000; i++) {                                        \
        uint64_t mk = KEYFN(&s2);                                           \
        (void)NAME##_contains(&m, mk);                                       \
    }                                                                        \
                                                                             \
    /* delete half */                                                        \
    for (int i = 0; i < N / 2; i++) {                                        \
        if (NAME##_delete(&m, keys[i]) != 1) {                              \
            printf("    delete fail at %d\n", i); err++; break;              \
        }                                                                    \
    }                                                                        \
    if (m.count != (uint32_t)(N - N / 2)) {                                  \
        printf("    after del count=%u expected=%d\n",                       \
               m.count, N - N / 2); err++;                                   \
    }                                                                        \
                                                                             \
    /* verify deleted are gone */                                            \
    for (int i = 0; i < N / 2; i++) {                                        \
        if (NAME##_contains(&m, keys[i])) {                                 \
            printf("    deleted still present at %d\n", i);                  \
            err++; break;                                                    \
        }                                                                    \
    }                                                                        \
    /* verify remaining are present */                                       \
    for (int i = N / 2; i < N; i++) {                                        \
        if (!NAME##_contains(&m, keys[i])) {                                \
            printf("    remaining missing at %d\n", i);                      \
            err++; break;                                                    \
        }                                                                    \
    }                                                                        \
                                                                             \
    /* re-insert deleted */                                                  \
    for (int i = 0; i < N / 2; i++) {                                        \
        if (NAME##_insert(&m, keys[i]) != 1) {                              \
            printf("    re-insert fail at %d\n", i); err++; break;           \
        }                                                                    \
    }                                                                        \
    if (m.count != (uint32_t)N) {                                            \
        printf("    after re-insert count=%u expected=%d\n",                 \
               m.count, N); err++;                                           \
    }                                                                        \
                                                                             \
    /* init_cap + insert_unique */                                           \
    NAME##_destroy(&m);                                                      \
    NAME##_init_cap(&m, N);                                                  \
    for (int i = 0; i < N; i++) NAME##_insert_unique(&m, keys[i]);           \
    if (m.count != (uint32_t)N) {                                            \
        printf("    init_cap count=%u expected=%d\n", m.count, N); err++;    \
    }                                                                        \
    for (int i = 0; i < 1000; i++) {                                         \
        if (!NAME##_contains(&m, keys[i])) {                                \
            printf("    init_cap contains fail at %d\n", i);                 \
            err++; break;                                                    \
        }                                                                    \
    }                                                                        \
                                                                             \
    NAME##_destroy(&m);                                                      \
    free(keys);                                                              \
    printf("  %-20s %s\n", LABEL, err ? "FAIL" : "PASS");                   \
    if (err) fail = 1;                                                       \
} while (0)

int main(void) {
    int fail = 0;
    printf("Map48 architecture variants correctness (N=%d):\n", N);

    TEST_SET(t_base, "split 1CL/10:",    make48);
    TEST_SET(t_3cl,  "3CL/31:",          make48);
    TEST_SET(t_2cl,  "2CL/20:",          make48);
    TEST_SET(t_bs,   "backshift 1CL/10:", make48_bs);
    TEST_SET(t_lem,  "lemire 1CL/10:",   make48);

    return fail;
}
