/*
 * test_map48_direct.c — Correctness tests for simd_map48_packed.h and simd_map48_split.h
 *
 * Tests set mode for both direct-compare 48-bit variants.
 *
 * Build:
 *   cc -O3 -march=native -std=gnu11 -o test_map48_direct test_map48_direct.c
 *   cc -O3 -march=native -mno-avx2 -mno-avx512f -std=gnu11 -o test_map48_direct_scalar test_map48_direct.c
 */

/* Packed 3×u16 interleaved */
#define SIMD_MAP_NAME packed48
#include "simd_map48_packed.h"

/* Split hi32+lo16 */
#define SIMD_MAP_NAME split48
#include "simd_map48_split.h"

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

#define TEST_SET(NAME, LABEL) do {                                           \
    uint64_t *keys = malloc((size_t)N * sizeof(uint64_t));                   \
    uint64_t s = 0xdeadbeefcafe1234ULL;                                      \
    for (int i = 0; i < N; i++) keys[i] = make48(&s);                       \
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
        uint64_t mk = make48(&s2);                                           \
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
    printf("Map48 direct-compare correctness (N=%d):\n", N);

    TEST_SET(packed48, "packed 3xu16:");
    TEST_SET(split48,  "split hi32+lo16:");

    return fail;
}
