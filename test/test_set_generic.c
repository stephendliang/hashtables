#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

/* Instantiate sentinel at 128-bit and 256-bit */
#define SIMD_MAP_NAME simd_set128s
#define SIMD_MAP_KEY_WORDS 2
#include "simd_sentinel.h"

#define SIMD_MAP_NAME simd_set256s
#define SIMD_MAP_KEY_WORDS 4
#include "simd_sentinel.h"

/* Instantiate bitstealing at 128-bit and 256-bit */
#define SIMD_MAP_NAME simd_set128b
#define SIMD_MAP_KEY_WORDS 2
#include "simd_bitstealing.h"

#define SIMD_MAP_NAME simd_set256b
#define SIMD_MAP_KEY_WORDS 4
#include "simd_bitstealing.h"

#define N 2000000

static inline uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

/* Run the full test suite for a given set instantiation.
 * WORDS = number of uint64_t per key. */
#define TEST(NAME, WORDS) do {                                              \
    struct NAME m;                                                          \
    NAME##_init(&m);                                                        \
    int ok = 1;                                                             \
                                                                            \
    /* Insert N keys */                                                     \
    for (int i = 0; i < N; i++) {                                           \
        uint64_t key[WORDS];                                                \
        for (int w = 0; w < WORDS; w++)                                     \
            key[w] = keys[i * WORDS + w];                                   \
        if (NAME##_insert(&m, key) != 1) { ok = 0; break; }                \
    }                                                                       \
    if (m.count != (uint32_t)N) ok = 0;                                     \
                                                                            \
    /* Duplicate rejection */                                               \
    for (int i = 0; i < 1000 && ok; i++) {                                  \
        uint64_t key[WORDS];                                                \
        for (int w = 0; w < WORDS; w++)                                     \
            key[w] = keys[i * WORDS + w];                                   \
        if (NAME##_insert(&m, key) != 0) ok = 0;                            \
    }                                                                       \
                                                                            \
    /* Contains hit */                                                      \
    for (int i = 0; i < N && ok; i++) {                                     \
        uint64_t key[WORDS];                                                \
        for (int w = 0; w < WORDS; w++)                                     \
            key[w] = keys[i * WORDS + w];                                   \
        if (!NAME##_contains(&m, key)) ok = 0;                              \
    }                                                                       \
                                                                            \
    /* Contains miss */                                                     \
    for (int i = 0; i < N && ok; i++) {                                     \
        uint64_t key[WORDS];                                                \
        for (int w = 0; w < WORDS; w++)                                     \
            key[w] = miss[i * WORDS + w];                                   \
        if (NAME##_contains(&m, key)) ok = 0;                               \
    }                                                                       \
                                                                            \
    /* Delete first half */                                                 \
    for (int i = 0; i < N / 2 && ok; i++) {                                 \
        uint64_t key[WORDS];                                                \
        for (int w = 0; w < WORDS; w++)                                     \
            key[w] = keys[i * WORDS + w];                                   \
        if (NAME##_delete(&m, key) != 1) ok = 0;                            \
    }                                                                       \
    if (m.count != (uint32_t)(N - N / 2)) ok = 0;                           \
                                                                            \
    /* Deleted keys miss, remaining keys hit */                              \
    for (int i = 0; i < N / 2 && ok; i++) {                                 \
        uint64_t key[WORDS];                                                \
        for (int w = 0; w < WORDS; w++)                                     \
            key[w] = keys[i * WORDS + w];                                   \
        if (NAME##_contains(&m, key)) ok = 0;                               \
    }                                                                       \
    for (int i = N / 2; i < N && ok; i++) {                                  \
        uint64_t key[WORDS];                                                \
        for (int w = 0; w < WORDS; w++)                                     \
            key[w] = keys[i * WORDS + w];                                   \
        if (!NAME##_contains(&m, key)) ok = 0;                              \
    }                                                                       \
                                                                            \
    /* Re-insert deleted keys */                                            \
    for (int i = 0; i < N / 2 && ok; i++) {                                 \
        uint64_t key[WORDS];                                                \
        for (int w = 0; w < WORDS; w++)                                     \
            key[w] = keys[i * WORDS + w];                                   \
        if (NAME##_insert(&m, key) != 1) ok = 0;                            \
    }                                                                       \
    if (m.count != (uint32_t)N) ok = 0;                                     \
                                                                            \
    /* Delete all */                                                        \
    for (int i = 0; i < N && ok; i++) {                                      \
        uint64_t key[WORDS];                                                \
        for (int w = 0; w < WORDS; w++)                                     \
            key[w] = keys[i * WORDS + w];                                   \
        if (NAME##_delete(&m, key) != 1) ok = 0;                            \
    }                                                                       \
    if (m.count != 0) ok = 0;                                               \
                                                                            \
    /* init_cap + insert_unique */                                          \
    NAME##_destroy(&m);                                                     \
    struct NAME m2;                                                         \
    NAME##_init_cap(&m2, N);                                                \
    uint32_t cap_before = m2.cap;                                           \
    for (int i = 0; i < N; i++) {                                            \
        uint64_t key[WORDS];                                                \
        for (int w = 0; w < WORDS; w++)                                     \
            key[w] = keys[i * WORDS + w];                                   \
        NAME##_insert_unique(&m2, key);                                     \
    }                                                                       \
    if (m2.cap != cap_before || m2.count != (uint32_t)N) ok = 0;            \
    NAME##_destroy(&m2);                                                    \
                                                                            \
    printf("  %-16s %s\n", #NAME ":", ok ? "PASS" : "FAIL");               \
    all_ok = all_ok && ok;                                                  \
} while (0)

int main(void) {
    /* Generate keys — max 4 words per key, N keys */
    int max_words = 4;
    uint64_t *keys = malloc(N * max_words * sizeof(uint64_t));
    uint64_t *miss = malloc(N * max_words * sizeof(uint64_t));
    uint64_t seed = 0xdeadbeefcafe1234ULL;
    for (int i = 0; i < N * max_words; i++)
        keys[i] = splitmix64(&seed);
    uint64_t mseed = 0xAAAABBBBCCCCDDDDULL;
    for (int i = 0; i < N * max_words; i++)
        miss[i] = splitmix64(&mseed);

    int all_ok = 1;
    printf("generic set correctness (N=%d):\n", N);
    TEST(simd_set128s, 2);
    TEST(simd_set256s, 4);
    TEST(simd_set128b, 2);
    TEST(simd_set256b, 4);

    free(keys);
    free(miss);
    return all_ok ? 0 : 1;
}
