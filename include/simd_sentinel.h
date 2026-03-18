/*
 * simd_sentinel.h — unified SIMD hash set/map with overflow sentinel
 *
 * Set mode (VAL_WORDS omitted or 0):
 *   #define SIMD_MAP_NAME  my_set
 *   #define SIMD_MAP_KEY_WORDS 2
 *   #include "simd_sentinel.h"
 *
 * Map mode (VAL_WORDS >= 1):
 *   #define SIMD_MAP_NAME       my_map
 *   #define SIMD_MAP_KEY_WORDS  2
 *   #define SIMD_MAP_VAL_WORDS  1
 *   #define SIMD_MAP_LAYOUT     1
 *   #include "simd_sentinel.h"
 *
 * Key: uint64_t[KEY_WORDS]. Value: uint64_t[VAL_WORDS] (map mode only).
 * All key/value parameters are const uint64_t *.
 * Can be included multiple times with different parameters.
 *
 * 31 data slots + 1 overflow sentinel per group. 15-bit h2.
 * Layout strategies (map mode): 1=inline values in group, 2=separate flat
 * array, 3=hybrid (value blocks every BLOCK_STRIDE key groups).
 */

#ifndef SIMD_MAP_NAME
#error "Define SIMD_MAP_NAME before including simd_sentinel.h"
#endif
#ifndef SIMD_MAP_KEY_WORDS
#error "Define SIMD_MAP_KEY_WORDS before including simd_sentinel.h"
#endif

/* Default VAL_WORDS to 0 (set mode) */
#ifndef SIMD_MAP_VAL_WORDS
#define SIMD_MAP_VAL_WORDS 0
#endif

/* Validate map-mode requirements */
#if SIMD_MAP_VAL_WORDS > 0
  #ifndef SIMD_MAP_LAYOUT
  #error "Map mode (VAL_WORDS >= 1) requires SIMD_MAP_LAYOUT (1, 2, or 3)"
  #endif
  #if SIMD_MAP_LAYOUT == 3 && !defined(SIMD_MAP_BLOCK_STRIDE)
  #error "Strategy 3 requires SIMD_MAP_BLOCK_STRIDE (power of 2)"
  #endif
#endif

/* --- Common (once) --- */
#ifndef SIMD_SENTINEL_COMMON_H_
#define SIMD_SENTINEL_COMMON_H_
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
#include "simd_compat.h"
#define SMCAT_(a, b) a##b
#define SMCAT(a, b)  SMCAT_(a, b)
#endif

/* --- Per-instantiation --- */
#undef SM_
#define SM_(s) SMCAT(SIMD_MAP_NAME, s)

#define SM_VW_       SIMD_MAP_VAL_WORDS
#define SM_KEY_GRP_  (64u + 32u * (SIMD_MAP_KEY_WORDS) * 8u)
#define SM_DMSK_     0x7FFFFFFFu

#if SM_VW_ > 0
  #define SM_VAL_SZ_ ((SIMD_MAP_VAL_WORDS) * 8u)
  #if SIMD_MAP_LAYOUT == 1
    #define SM_GRP_  (SM_KEY_GRP_ + 32u * SM_VAL_SZ_)
  #elif SIMD_MAP_LAYOUT == 2
    #define SM_GRP_  SM_KEY_GRP_
  #elif SIMD_MAP_LAYOUT == 3
    #define SM_GRP_       SM_KEY_GRP_
    #define SM_BLK_SHIFT_ ((unsigned)__builtin_ctz(SIMD_MAP_BLOCK_STRIDE))
    #define SM_BLK_MASK_  ((unsigned)(SIMD_MAP_BLOCK_STRIDE) - 1u)
    #define SM_SUPER_     ((size_t)(SIMD_MAP_BLOCK_STRIDE) * SM_KEY_GRP_ + \
                           (size_t)(SIMD_MAP_BLOCK_STRIDE) * 32u * SM_VAL_SZ_)
  #endif
#else
  #define SM_GRP_  SM_KEY_GRP_
#endif

/* --- Types --- */

struct SM_(_key) { uint64_t w[SIMD_MAP_KEY_WORDS]; };

#if SM_VW_ > 0
struct SM_(_val) { uint64_t w[SIMD_MAP_VAL_WORDS]; };
#endif

struct SIMD_MAP_NAME {
    char    *data;
#if SM_VW_ > 0 && SIMD_MAP_LAYOUT == 2
    char    *val_data;
#endif
    uint32_t count;
    uint32_t cap;
    uint32_t mask;
};

/* --- Hash --- */

struct SM_(_h) { uint32_t lo, hi; };

#if defined(__SSE4_2__) || defined(__ARM_FEATURE_CRC32)
static inline struct SM_(_h) SM_(_hash)(const uint64_t *key) {
    uint32_t a = 0;
    for (int i = 0; i < SIMD_MAP_KEY_WORDS; i++)
        a = (uint32_t)_mm_crc32_u64(a, key[i]);
    uint32_t b = (uint32_t)_mm_crc32_u64(a, key[SIMD_MAP_KEY_WORDS - 1]);
    return (struct SM_(_h)){a, b};
}
#else
static inline struct SM_(_h) SM_(_hash)(const uint64_t *key) {
    uint64_t h = key[0];
    for (int i = 1; i < SIMD_MAP_KEY_WORDS; i++)
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
#if SM_VW_ > 0 && SIMD_MAP_LAYOUT == 3
    uint32_t super = gi >> SM_BLK_SHIFT_;
    uint32_t local = gi & SM_BLK_MASK_;
    return m->data + (size_t)super * SM_SUPER_ + (size_t)local * SM_KEY_GRP_;
#else
    return m->data + (size_t)gi * SM_GRP_;
#endif
}

static inline int SM_(_key_eq)(const struct SM_(_key) *slot,
                                const uint64_t *key) {
    for (int i = 0; i < SIMD_MAP_KEY_WORDS; i++)
        if (slot->w[i] != key[i]) return 0;
    return 1;
}

static inline void SM_(_key_copy)(struct SM_(_key) *dst, const uint64_t *key) {
    for (int i = 0; i < SIMD_MAP_KEY_WORDS; i++)
        dst->w[i] = key[i];
}

#if SM_VW_ > 0
static inline struct SM_(_val) *SM_(_val_at)(const struct SIMD_MAP_NAME *m,
                                              uint32_t gi, int slot) {
#if SIMD_MAP_LAYOUT == 1
    char *grp = m->data + (size_t)gi * SM_GRP_;
    return (struct SM_(_val) *)(grp + SM_KEY_GRP_) + slot;
#elif SIMD_MAP_LAYOUT == 2
    return (struct SM_(_val) *)(m->val_data) + (size_t)gi * 32 + slot;
#elif SIMD_MAP_LAYOUT == 3
    uint32_t super = gi >> SM_BLK_SHIFT_;
    uint32_t local = gi & SM_BLK_MASK_;
    char *sb = m->data + (size_t)super * SM_SUPER_;
    char *vb = sb + (size_t)SIMD_MAP_BLOCK_STRIDE * SM_KEY_GRP_;
    return (struct SM_(_val) *)(vb) + (size_t)local * 32 + slot;
#endif
}

static inline void SM_(_val_copy)(struct SM_(_val) *dst, const uint64_t *val) {
    for (int i = 0; i < SIMD_MAP_VAL_WORDS; i++)
        dst->w[i] = val[i];
}
#endif /* SM_VW_ > 0 */

/* --- Prefetch --- */

static inline void SM_(_prefetch)(const struct SIMD_MAP_NAME *m,
                                   const uint64_t *key) {
#if defined(__SSE4_2__) || defined(__ARM_FEATURE_CRC32)
    uint32_t a = 0;
    for (int i = 0; i < SIMD_MAP_KEY_WORDS; i++)
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
 * Key/value writes use write-allocate through the store buffer. */
static inline void SM_(_prefetch_insert)(const struct SIMD_MAP_NAME *m,
                                          const uint64_t *key) {
#if defined(__SSE4_2__) || defined(__ARM_FEATURE_CRC32)
    uint32_t a = 0;
    for (int i = 0; i < SIMD_MAP_KEY_WORDS; i++)
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

#elif defined(__ARM_NEON)

static inline uint32_t SM_(_match)(const uint16_t *meta, uint16_t h2) {
    uint16x8_t needle = vdupq_n_u16(h2);
    uint32_t result = 0;
    result |= neon_movemask_u16(vceqq_u16(vld1q_u16(meta + 0), needle));
    result |= neon_movemask_u16(vceqq_u16(vld1q_u16(meta + 8), needle)) << 8;
    result |= neon_movemask_u16(vceqq_u16(vld1q_u16(meta + 16), needle)) << 16;
    result |= neon_movemask_u16(vceqq_u16(vld1q_u16(meta + 24), needle)) << 24;
    return result & SM_DMSK_;
}

static inline uint32_t SM_(_empty)(const uint16_t *meta) {
    uint16x8_t z = vdupq_n_u16(0);
    uint32_t result = 0;
    result |= neon_movemask_u16(vceqq_u16(vld1q_u16(meta + 0), z));
    result |= neon_movemask_u16(vceqq_u16(vld1q_u16(meta + 8), z)) << 8;
    result |= neon_movemask_u16(vceqq_u16(vld1q_u16(meta + 16), z)) << 16;
    result |= neon_movemask_u16(vceqq_u16(vld1q_u16(meta + 24), z)) << 24;
    return result & SM_DMSK_;
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
#if SM_VW_ > 0 && SIMD_MAP_LAYOUT == 3
    size_t raw = (size_t)((cap >> 5) >> SM_BLK_SHIFT_) * SM_SUPER_;
#else
    size_t raw = (size_t)(cap >> 5) * SM_GRP_;
#endif
    return (raw + (2u << 20) - 1) & ~((size_t)(2u << 20) - 1);
}

#if SM_VW_ > 0 && SIMD_MAP_LAYOUT == 2
static size_t SM_(_val_mapsize)(uint32_t cap) {
    size_t raw = (size_t)cap * SM_VAL_SZ_;
    return (raw + (2u << 20) - 1) & ~((size_t)(2u << 20) - 1);
}
#endif

static void SM_(_alloc)(struct SIMD_MAP_NAME *m, uint32_t cap) {
#if SM_VW_ > 0 && SIMD_MAP_LAYOUT == 3
    if (cap < 32u * SIMD_MAP_BLOCK_STRIDE)
        cap = 32u * SIMD_MAP_BLOCK_STRIDE;
#endif
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
#if SM_VW_ > 0 && SIMD_MAP_LAYOUT == 2
    size_t vtotal = SM_(_val_mapsize)(cap);
    m->val_data = (char *)mmap(NULL, vtotal, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB
                               | MAP_POPULATE, -1, 0);
    if (m->val_data == MAP_FAILED) {
        m->val_data = (char *)mmap(NULL, vtotal, PROT_READ | PROT_WRITE,
                                   MAP_PRIVATE | MAP_ANONYMOUS
                                   | MAP_POPULATE, -1, 0);
        if (m->val_data != MAP_FAILED)
            madvise(m->val_data, vtotal, MADV_HUGEPAGE);
    }
#endif
    m->cap   = cap;
    m->mask  = (cap >> 5) - 1;
    m->count = 0;
}

static void SM_(_grow)(struct SIMD_MAP_NAME *m) {
    uint32_t old_cap  = m->cap;
    char    *old_data = m->data;
#if SM_VW_ > 0 && SIMD_MAP_LAYOUT == 2
    char    *old_vd   = m->val_data;
#endif
    uint32_t old_ng   = old_cap >> 5;

    SM_(_alloc)(m, old_cap * 2);
    uint32_t mask = m->mask;

    for (uint32_t g = 0; g < old_ng; g++) {
#if SM_VW_ > 0 && SIMD_MAP_LAYOUT == 3
        uint32_t osup = g >> SM_BLK_SHIFT_;
        uint32_t oloc = g & SM_BLK_MASK_;
        const char *old_grp = old_data + (size_t)osup * SM_SUPER_
                              + (size_t)oloc * SM_KEY_GRP_;
#else
        const char *old_grp = old_data + (size_t)g * SM_GRP_;
#endif
        const uint16_t *om = (const uint16_t *)old_grp;
        const struct SM_(_key) *ok = (const struct SM_(_key) *)(old_grp + 64);

        for (int s = 0; s < 31; s++) {
            if (!(om[s] & 0x8000)) continue;

#if SM_VW_ > 0
            const struct SM_(_val) *ov;
  #if SIMD_MAP_LAYOUT == 1
            ov = (const struct SM_(_val) *)(old_grp + SM_KEY_GRP_) + s;
  #elif SIMD_MAP_LAYOUT == 2
            ov = (const struct SM_(_val) *)old_vd + (size_t)g * 32 + s;
  #elif SIMD_MAP_LAYOUT == 3
            ov = (const struct SM_(_val) *)(old_data
                    + (size_t)osup * SM_SUPER_
                    + (size_t)SIMD_MAP_BLOCK_STRIDE * SM_KEY_GRP_)
                 + (size_t)oloc * 32 + s;
  #endif
#endif /* SM_VW_ > 0 */

            struct SM_(_h) h = SM_(_hash)(ok[s].w);
            uint16_t h2  = SM_(_h2)(h.lo);
            uint32_t gi  = h.lo & mask;
            for (;;) {
                char     *grp  = SM_(_group)(m, gi);
                uint16_t *base = (uint16_t *)grp;
                struct SM_(_key) *kp = (struct SM_(_key) *)(grp + 64);
                uint32_t em = SM_(_empty)(base);
                if (em) {
                    int pos = __builtin_ctz(em);
                    base[pos] = h2;
                    SM_(_key_copy)(&kp[pos], ok[s].w);
#if SM_VW_ > 0
                    *SM_(_val_at)(m, gi, pos) = *ov;
#endif
                    m->count++;
                    break;
                }
                base[31] |= SM_(_overflow_bit)(h.hi);
                gi = (gi + 1) & mask;
            }
        }
    }
    munmap(old_data, SM_(_mapsize)(old_cap));
#if SM_VW_ > 0 && SIMD_MAP_LAYOUT == 2
    munmap(old_vd, SM_(_val_mapsize)(old_cap));
#endif
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
#if SM_VW_ > 0 && SIMD_MAP_LAYOUT == 2
    if (m->val_data) munmap(m->val_data, SM_(_val_mapsize)(m->cap));
#endif
}

#if SM_VW_ > 0

static inline int SM_(_insert)(struct SIMD_MAP_NAME *m,
                                const uint64_t *key, const uint64_t *val) {
    if (m->cap == 0) SM_(_alloc)(m, 32);
    if (m->count * 8 >= m->cap * 7)
        SM_(_grow)(m);

    struct SM_(_h) h = SM_(_hash)(key);
    uint16_t h2 = SM_(_h2)(h.lo);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        char     *grp  = SM_(_group)(m, gi);
        uint16_t *base = (uint16_t *)grp;
        struct SM_(_key) *kp = (struct SM_(_key) *)(grp + 64);

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
            SM_(_val_copy)(SM_(_val_at)(m, gi, pos), val);
            m->count++;
            return 1;
        }
        base[31] |= SM_(_overflow_bit)(h.hi);
        gi = (gi + 1) & m->mask;
    }
}

static inline void SM_(_insert_unique)(struct SIMD_MAP_NAME *m,
                                        const uint64_t *key,
                                        const uint64_t *val) {
    if (m->cap == 0) SM_(_alloc)(m, 32);
    if (m->count * 8 >= m->cap * 7)
        SM_(_grow)(m);

    struct SM_(_h) h = SM_(_hash)(key);
    uint16_t h2 = SM_(_h2)(h.lo);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        char     *grp  = SM_(_group)(m, gi);
        uint16_t *base = (uint16_t *)grp;
        struct SM_(_key) *kp = (struct SM_(_key) *)(grp + 64);

        uint32_t em = SM_(_empty)(base);
        if (em) {
            int pos = __builtin_ctz(em);
            base[pos] = h2;
            SM_(_key_copy)(&kp[pos], key);
            SM_(_val_copy)(SM_(_val_at)(m, gi, pos), val);
            m->count++;
            return;
        }
        base[31] |= SM_(_overflow_bit)(h.hi);
        gi = (gi + 1) & m->mask;
    }
}

static inline uint64_t *SM_(_get)(struct SIMD_MAP_NAME *m,
                                   const uint64_t *key) {
    if (__builtin_expect(m->cap == 0, 0)) return NULL;

    struct SM_(_h) h = SM_(_hash)(key);
    uint16_t h2 = SM_(_h2)(h.lo);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        const char     *grp  = SM_(_group)(m, gi);
        const uint16_t *base = (const uint16_t *)grp;
        const struct SM_(_key) *kp = (const struct SM_(_key) *)(grp + 64);

        uint32_t mm = SM_(_match)(base, h2);
        while (mm) {
            int pos = __builtin_ctz(mm);
            if (SM_(_key_eq)(&kp[pos], key))
                return SM_(_val_at)(m, gi, pos)->w;
            mm &= mm - 1;
        }
        if (!((base[31] >> (h.hi & 15)) & 1)) return NULL;
        gi = (gi + 1) & m->mask;
    }
}

#else /* SM_VW_ == 0: set mode */

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
        struct SM_(_key) *kp = (struct SM_(_key) *)(grp + 64);

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
        struct SM_(_key) *kp = (struct SM_(_key) *)(grp + 64);

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

#endif /* SM_VW_ > 0 */

static inline int SM_(_contains)(struct SIMD_MAP_NAME *m,
                                  const uint64_t *key) {
    if (__builtin_expect(m->cap == 0, 0)) return 0;

    struct SM_(_h) h = SM_(_hash)(key);
    uint16_t h2 = SM_(_h2)(h.lo);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        const char     *grp  = SM_(_group)(m, gi);
        const uint16_t *base = (const uint16_t *)grp;
        const struct SM_(_key) *kp = (const struct SM_(_key) *)(grp + 64);

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

static inline int SM_(_delete)(struct SIMD_MAP_NAME *m, const uint64_t *key) {
    if (__builtin_expect(m->cap == 0, 0)) return 0;

    struct SM_(_h) h = SM_(_hash)(key);
    uint16_t h2 = SM_(_h2)(h.lo);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        char *grp = SM_(_group)(m, gi);
        uint16_t *base = (uint16_t *)grp;
        struct SM_(_key) *kp = (struct SM_(_key) *)(grp + 64);

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

/* --- Cleanup --- */
#undef SM_
#undef SM_VW_
#undef SM_KEY_GRP_
#undef SM_DMSK_
#undef SM_GRP_
#ifdef SM_VAL_SZ_
#undef SM_VAL_SZ_
#endif
#ifdef SM_BLK_SHIFT_
#undef SM_BLK_SHIFT_
#endif
#ifdef SM_BLK_MASK_
#undef SM_BLK_MASK_
#endif
#ifdef SM_SUPER_
#undef SM_SUPER_
#endif
#undef SIMD_MAP_NAME
#undef SIMD_MAP_KEY_WORDS
#undef SIMD_MAP_VAL_WORDS
#ifdef SIMD_MAP_LAYOUT
#undef SIMD_MAP_LAYOUT
#endif
#ifdef SIMD_MAP_BLOCK_STRIDE
#undef SIMD_MAP_BLOCK_STRIDE
#endif
