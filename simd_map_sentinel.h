/*
 * simd_map_sentinel.h — size-agnostic SIMD hash set with overflow sentinel
 *
 * Macro-generated via X-include pattern. Define parameters before including:
 *
 *   #define SIMD_MAP_NAME  simd_map192
 *   #define SIMD_MAP_WORDS 3
 *   #include "simd_map_sentinel.h"
 *
 * Generates: struct simd_map192, simd_map192_init(), simd_map192_insert(), etc.
 * Key: uint64_t[WORDS]. All key parameters are const uint64_t *.
 * Can be included multiple times with different parameters.
 *
 * 31 data slots + 1 overflow sentinel per group. 15-bit h2.
 * See simd_map128_sentinel.h for detailed design notes.
 */

#ifndef SIMD_MAP_NAME
#error "Define SIMD_MAP_NAME before including simd_map_sentinel.h"
#endif
#ifndef SIMD_MAP_WORDS
#error "Define SIMD_MAP_WORDS before including simd_map_sentinel.h"
#endif

/* --- Common (once) --- */
#ifndef SIMD_MAP_COMMON_H_
#define SIMD_MAP_COMMON_H_
#if defined(__AVX512F__) || defined(__AVX2__) || defined(__SSE4_2__)
#include <immintrin.h>
#endif
#if defined(__AVX2__)
#include <x86intrin.h>
#endif
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#define SMCAT_(a, b) a##b
#define SMCAT(a, b)  SMCAT_(a, b)
#endif

/* --- Per-instantiation --- */
#undef SM_
#define SM_(s) SMCAT(SIMD_MAP_NAME, s)

#define SM_GRP_  (64u + 32u * (SIMD_MAP_WORDS) * 8u)
#define SM_DMSK_ 0x7FFFFFFFu

/* --- Types --- */

struct SM_(_kv) { uint64_t w[SIMD_MAP_WORDS]; };

struct SIMD_MAP_NAME {
    char    *data;
    uint32_t count;
    uint32_t cap;
    uint32_t mask;
};

/* --- Hash --- */

struct SM_(_h) { uint32_t lo, hi; };

#if defined(__SSE4_2__)
static inline struct SM_(_h) SM_(_hash)(const uint64_t *key) {
    uint32_t a = 0;
    for (int i = 0; i < SIMD_MAP_WORDS; i++)
        a = (uint32_t)_mm_crc32_u64(a, key[i]);
    uint32_t b = (uint32_t)_mm_crc32_u64(a, key[SIMD_MAP_WORDS - 1]);
    return (struct SM_(_h)){a, b};
}
#else
static inline struct SM_(_h) SM_(_hash)(const uint64_t *key) {
    uint64_t h = key[0];
    for (int i = 1; i < SIMD_MAP_WORDS; i++)
        h ^= key[i] + 0x9e3779b97f4a7c15ULL + (key[i] << 6) + (key[i] >> 2);
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return (struct SM_(_h)){(uint32_t)h, (uint32_t)(h >> 32)};
}
#endif

static inline uint16_t SM_(_h2)(uint32_t lo) {
    return (uint16_t)((lo >> 17) | 0x8000);
}

static inline uint16_t SM_(_overflow_bit)(uint32_t hi) {
    return (uint16_t)(1u << (hi & 15));
}

/* --- Helpers --- */

static inline char *SM_(_group)(const struct SIMD_MAP_NAME *m, uint32_t gi) {
    return m->data + (size_t)gi * SM_GRP_;
}

static inline int SM_(_key_eq)(const struct SM_(_kv) *slot, const uint64_t *key) {
    for (int i = 0; i < SIMD_MAP_WORDS; i++)
        if (slot->w[i] != key[i]) return 0;
    return 1;
}

static inline void SM_(_key_copy)(struct SM_(_kv) *dst, const uint64_t *key) {
    for (int i = 0; i < SIMD_MAP_WORDS; i++)
        dst->w[i] = key[i];
}

/* --- Prefetch --- */

static inline void SM_(_prefetch)(const struct SIMD_MAP_NAME *m,
                                   const uint64_t *key) {
#if defined(__SSE4_2__)
    uint32_t a = 0;
    for (int i = 0; i < SIMD_MAP_WORDS; i++)
        a = (uint32_t)_mm_crc32_u64(a, key[i]);
    uint32_t gi = a & m->mask;
    const char *grp = SM_(_group)(m, gi);
    _mm_prefetch(grp, _MM_HINT_T0);
    _mm_prefetch(grp + 64, _MM_HINT_T0);
    _mm_prefetch(grp + 128, _MM_HINT_T0);
    _mm_prefetch(grp + 192, _MM_HINT_T0);
    _mm_prefetch(grp + 256, _MM_HINT_T0);
#else
    struct SM_(_h) h = SM_(_hash)(key);
    uint32_t gi = h.lo & m->mask;
    const char *grp = SM_(_group)(m, gi);
    __builtin_prefetch(grp, 0, 3);
    __builtin_prefetch(grp + 64, 0, 3);
    __builtin_prefetch(grp + 128, 0, 3);
    __builtin_prefetch(grp + 192, 0, 3);
    __builtin_prefetch(grp + 256, 0, 3);
#endif
}

/* Lightweight prefetch for insert/delete paths: metadata line only.
 * Key writes use write-allocate through the store buffer. */
static inline void SM_(_prefetch_insert)(const struct SIMD_MAP_NAME *m,
                                          const uint64_t *key) {
#if defined(__SSE4_2__)
    uint32_t a = 0;
    for (int i = 0; i < SIMD_MAP_WORDS; i++)
        a = (uint32_t)_mm_crc32_u64(a, key[i]);
    uint32_t gi = a & m->mask;
    _mm_prefetch(SM_(_group)(m, gi), _MM_HINT_T0);
#else
    struct SM_(_h) h = SM_(_hash)(key);
    uint32_t gi = h.lo & m->mask;
    __builtin_prefetch(SM_(_group)(m, gi), 0, 3);
#endif
}

/* --- SIMD backends --- */

#if defined(__AVX512F__)

static inline uint32_t SM_(_match)(const uint16_t *meta, uint16_t h2) {
    __m512i group  = _mm512_load_si512((const __m512i *)meta);
    __m512i needle = _mm512_set1_epi16((short)h2);
    return _mm512_cmpeq_epi16_mask(group, needle) & SM_DMSK_;
}

static inline uint32_t SM_(_empty)(const uint16_t *meta) {
    __m512i group = _mm512_load_si512((const __m512i *)meta);
    return _mm512_testn_epi16_mask(group, group) & SM_DMSK_;
}

#elif defined(__AVX2__)

static inline uint32_t SM_(_movemask16)(__m256i lo, __m256i hi) {
    uint32_t mlo = _pext_u32((uint32_t)_mm256_movemask_epi8(lo), 0xAAAAAAAAu);
    uint32_t mhi = _pext_u32((uint32_t)_mm256_movemask_epi8(hi), 0xAAAAAAAAu);
    return mlo | (mhi << 16);
}

static inline uint32_t SM_(_match)(const uint16_t *meta, uint16_t h2) {
    __m256i needle = _mm256_set1_epi16((short)h2);
    __m256i lo = _mm256_load_si256((const __m256i *)meta);
    __m256i hi = _mm256_load_si256((const __m256i *)(meta + 16));
    return SM_(_movemask16)(_mm256_cmpeq_epi16(lo, needle),
                             _mm256_cmpeq_epi16(hi, needle))
           & SM_DMSK_;
}

static inline uint32_t SM_(_empty)(const uint16_t *meta) {
    __m256i z  = _mm256_setzero_si256();
    __m256i lo = _mm256_load_si256((const __m256i *)meta);
    __m256i hi = _mm256_load_si256((const __m256i *)(meta + 16));
    return SM_(_movemask16)(_mm256_cmpeq_epi16(lo, z),
                             _mm256_cmpeq_epi16(hi, z))
           & SM_DMSK_;
}

#else /* scalar SWAR */

static inline uint32_t SM_(_pack4)(uint64_t z) {
    return (uint32_t)((z * 0x0000200040008001ULL) >> 60);
}

static inline uint32_t SM_(_match)(const uint16_t *meta, uint16_t h2) {
    const uint64_t *w = (const uint64_t *)meta;
    uint64_t needle = (uint64_t)h2 * 0x0001000100010001ULL;
    uint64_t msb    = 0x8000800080008000ULL;
    uint32_t result = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t v  = w[i] ^ needle;
        uint64_t hi = (v | msb) - 0x0001000100010001ULL;
        uint64_t z  = ~hi & ~v & msb;
        result |= SM_(_pack4)(z) << (i * 4);
    }
    return result & SM_DMSK_;
}

static inline uint32_t SM_(_empty)(const uint16_t *meta) {
    const uint64_t *w = (const uint64_t *)meta;
    uint64_t msb = 0x8000800080008000ULL;
    uint32_t result = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t z = ~w[i] & msb;
        result |= SM_(_pack4)(z) << (i * 4);
    }
    return result & SM_DMSK_;
}

#endif

/* --- Alloc / grow --- */

static size_t SM_(_mapsize)(uint32_t cap) {
    size_t raw = (size_t)(cap >> 5) * SM_GRP_;
    return (raw + (2u << 20) - 1) & ~((size_t)(2u << 20) - 1);
}

static void SM_(_alloc)(struct SIMD_MAP_NAME *m, uint32_t cap) {
    size_t total = SM_(_mapsize)(cap);
    m->data = (char *)mmap(NULL, total, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB
                           | MAP_POPULATE, -1, 0);
    if (m->data == MAP_FAILED) {
        m->data = (char *)mmap(NULL, total, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS
                               | MAP_POPULATE, -1, 0);
        if (m->data != MAP_FAILED)
            madvise(m->data, total, MADV_HUGEPAGE);
    }
    m->cap   = cap;
    m->mask  = (cap >> 5) - 1;
    m->count = 0;
}

static void SM_(_grow)(struct SIMD_MAP_NAME *m) {
    uint32_t old_cap  = m->cap;
    char    *old_data = m->data;
    uint32_t old_ng   = old_cap >> 5;

    SM_(_alloc)(m, old_cap * 2);
    uint32_t mask = m->mask;

    for (uint32_t g = 0; g < old_ng; g++) {
        const char     *old_grp = old_data + (size_t)g * SM_GRP_;
        const uint16_t *om      = (const uint16_t *)old_grp;
        const struct SM_(_kv) *ok = (const struct SM_(_kv) *)(old_grp + 64);
        for (int s = 0; s < 31; s++) {
            if (!(om[s] & 0x8000)) continue;
            struct SM_(_h) h = SM_(_hash)(ok[s].w);
            uint16_t h2  = SM_(_h2)(h.lo);
            uint32_t gi  = h.lo & mask;
            for (;;) {
                char     *grp  = SM_(_group)(m, gi);
                uint16_t *base = (uint16_t *)grp;
                struct SM_(_kv) *kp = (struct SM_(_kv) *)(grp + 64);
                uint32_t em = SM_(_empty)(base);
                if (em) {
                    int pos = __builtin_ctz(em);
                    base[pos] = h2;
                    SM_(_key_copy)(&kp[pos], ok[s].w);
                    m->count++;
                    break;
                }
                base[31] |= SM_(_overflow_bit)(h.hi);
                gi = (gi + 1) & mask;
            }
        }
    }
    munmap(old_data, SM_(_mapsize)(old_cap));
}

/* --- Public API --- */

static inline void SM_(_init)(struct SIMD_MAP_NAME *m) {
    memset(m, 0, sizeof(*m));
}

static inline void SM_(_init_cap)(struct SIMD_MAP_NAME *m, uint32_t n) {
    memset(m, 0, sizeof(*m));
    uint64_t need = (uint64_t)n * 8 / 7 + 1;
    uint32_t cap = 32;
    while (cap < need) cap *= 2;
    SM_(_alloc)(m, cap);
}

static inline void SM_(_destroy)(struct SIMD_MAP_NAME *m) {
    if (m->data) munmap(m->data, SM_(_mapsize)(m->cap));
}

static inline int SM_(_insert)(struct SIMD_MAP_NAME *m, const uint64_t *key) {
    if (m->cap == 0) SM_(_alloc)(m, 32);
    if (m->count * 8 >= m->cap * 7)
        SM_(_grow)(m);

    struct SM_(_h) h = SM_(_hash)(key);
    uint16_t h2 = SM_(_h2)(h.lo);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        char     *grp  = SM_(_group)(m, gi);
        uint16_t *base = (uint16_t *)grp;
        struct SM_(_kv) *kp = (struct SM_(_kv) *)(grp + 64);

        uint32_t mm = SM_(_match)(base, h2);
        while (mm) {
            int pos = __builtin_ctz(mm);
            if (SM_(_key_eq)(&kp[pos], key)) return 0;
            mm &= mm - 1;
        }

        uint32_t em = SM_(_empty)(base);
        if (em) {
            int pos = __builtin_ctz(em);
            base[pos] = h2;
            SM_(_key_copy)(&kp[pos], key);
            m->count++;
            return 1;
        }
        base[31] |= SM_(_overflow_bit)(h.hi);
        gi = (gi + 1) & m->mask;
    }
}

static inline void SM_(_insert_unique)(struct SIMD_MAP_NAME *m,
                                        const uint64_t *key) {
    if (m->cap == 0) SM_(_alloc)(m, 32);
    if (m->count * 8 >= m->cap * 7)
        SM_(_grow)(m);

    struct SM_(_h) h = SM_(_hash)(key);
    uint16_t h2 = SM_(_h2)(h.lo);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        char     *grp  = SM_(_group)(m, gi);
        uint16_t *base = (uint16_t *)grp;
        struct SM_(_kv) *kp = (struct SM_(_kv) *)(grp + 64);

        uint32_t em = SM_(_empty)(base);
        if (em) {
            int pos = __builtin_ctz(em);
            base[pos] = h2;
            SM_(_key_copy)(&kp[pos], key);
            m->count++;
            return;
        }
        base[31] |= SM_(_overflow_bit)(h.hi);
        gi = (gi + 1) & m->mask;
    }
}

static inline int SM_(_delete)(struct SIMD_MAP_NAME *m, const uint64_t *key) {
    if (__builtin_expect(m->cap == 0, 0)) return 0;

    struct SM_(_h) h = SM_(_hash)(key);
    uint16_t h2 = SM_(_h2)(h.lo);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        char *grp = SM_(_group)(m, gi);
        uint16_t *base = (uint16_t *)grp;
        struct SM_(_kv) *kp = (struct SM_(_kv) *)(grp + 64);

        uint32_t mm = SM_(_match)(base, h2);
        while (mm) {
            int pos = __builtin_ctz(mm);
            if (SM_(_key_eq)(&kp[pos], key)) {
                base[pos] = 0;
                m->count--;
                return 1;
            }
            mm &= mm - 1;
        }
        if (!((base[31] >> (h.hi & 15)) & 1)) return 0;
        gi = (gi + 1) & m->mask;
    }
}

static inline int SM_(_contains)(struct SIMD_MAP_NAME *m, const uint64_t *key) {
    if (__builtin_expect(m->cap == 0, 0)) return 0;

    struct SM_(_h) h = SM_(_hash)(key);
    uint16_t h2 = SM_(_h2)(h.lo);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        const char     *grp  = SM_(_group)(m, gi);
        const uint16_t *base = (const uint16_t *)grp;
        const struct SM_(_kv) *kp = (const struct SM_(_kv) *)(grp + 64);

        uint32_t mm = SM_(_match)(base, h2);
        while (mm) {
            int pos = __builtin_ctz(mm);
            if (SM_(_key_eq)(&kp[pos], key)) return 1;
            mm &= mm - 1;
        }
        if (!((base[31] >> (h.hi & 15)) & 1)) return 0;
        gi = (gi + 1) & m->mask;
    }
}

/* --- Cleanup --- */
#undef SM_
#undef SM_GRP_
#undef SM_DMSK_
#undef SIMD_MAP_NAME
#undef SIMD_MAP_WORDS
