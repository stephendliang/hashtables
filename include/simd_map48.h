/*
 * simd_map48.h — unified SIMD hash set/map for 48-bit keys
 *
 * Set mode (VAL_WORDS omitted or 0):
 *   #define SIMD_MAP_NAME  my_set48
 *   #include "simd_map48.h"
 *
 * Map mode (VAL_WORDS >= 1):
 *   #define SIMD_MAP_NAME       my_map48
 *   #define SIMD_MAP48_VAL_WORDS 1
 *   #include "simd_map48.h"
 *
 * Key: uint64_t (lower 48 bits, upper 16 must be 0, key=0 reserved).
 * Sentinel-style metadata with 15-bit h2 and 16 overflow partitions.
 * 31 data slots + 1 sentinel per group.
 *
 * Group layout (set mode, 256B = 4 cache lines):
 *   Line 0:   uint16_t meta[32]  (64B: 31 data + 1 sentinel)
 *   Lines 1-3: packed 6-byte keys (192B: 31 keys × 6B + 6B pad)
 * Compare simd_sentinel KW=1: 5 cache lines (320B). This saves 20%.
 *
 * Backends: AVX-512, AVX2, scalar.
 * Load factor: 7/8 (87.5%).
 */

#ifndef SIMD_MAP_NAME
#error "Define SIMD_MAP_NAME before including simd_map48.h"
#endif
#ifndef SIMD_MAP48_VAL_WORDS
#define SIMD_MAP48_VAL_WORDS 0
#endif

/* --- Common (once) --- */
#ifndef SIMD_MAP48_COMMON_H_
#define SIMD_MAP48_COMMON_H_

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

#ifndef SMCAT_
#define SMCAT_(a, b) a##b
#define SMCAT(a, b)  SMCAT_(a, b)
#endif

#define SM48_KEY_MASK_ 0x0000FFFFFFFFFFFFULL

/* Hash: CRC32 on SIMD paths, murmur3 finalizer on scalar */
struct sm48_h { uint32_t lo, hi; };

#if defined(__SSE4_2__) || defined(__ARM_FEATURE_CRC32)
static inline struct sm48_h sm48_hash(uint64_t key) {
    uint32_t a = (uint32_t)_mm_crc32_u64(0, key);
    uint32_t b = (uint32_t)_mm_crc32_u64(a, key);
    return (struct sm48_h){a, b};
}
#else
static inline struct sm48_h sm48_hash(uint64_t key) {
    uint64_t h = key;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return (struct sm48_h){(uint32_t)h, (uint32_t)(h >> 32)};
}
#endif

static inline uint16_t sm48_h2(uint32_t lo) {
    return (uint16_t)((lo >> 17) | 0x8000);
}

static inline uint16_t sm48_overflow_bit(uint32_t hi) {
    return (uint16_t)(1u << (hi & 15));
}

/* --- SIMD backends (identical to sentinel: 16-bit metadata, 31+1 slots) --- */

#define SM48_DMSK_ 0x7FFFFFFFu

#if defined(__AVX512F__)

static inline uint32_t sm48_match(const uint16_t *meta, uint16_t h2) {
    __m512i group  = _mm512_load_si512((const __m512i *)meta);
    __m512i needle = _mm512_set1_epi16((short)h2);
    return _mm512_cmpeq_epi16_mask(group, needle) & SM48_DMSK_;
}

static inline uint32_t sm48_empty(const uint16_t *meta) {
    __m512i group = _mm512_load_si512((const __m512i *)meta);
    return _mm512_testn_epi16_mask(group, group) & SM48_DMSK_;
}

#elif defined(__AVX2__)

static inline uint32_t sm48_movemask16(__m256i lo, __m256i hi) {
    uint32_t mlo = _pext_u32((uint32_t)_mm256_movemask_epi8(lo), 0xAAAAAAAAu);
    uint32_t mhi = _pext_u32((uint32_t)_mm256_movemask_epi8(hi), 0xAAAAAAAAu);
    return mlo | (mhi << 16);
}

static inline uint32_t sm48_match(const uint16_t *meta, uint16_t h2) {
    __m256i needle = _mm256_set1_epi16((short)h2);
    __m256i lo = _mm256_load_si256((const __m256i *)meta);
    __m256i hi = _mm256_load_si256((const __m256i *)(meta + 16));
    return sm48_movemask16(_mm256_cmpeq_epi16(lo, needle),
                           _mm256_cmpeq_epi16(hi, needle))
           & SM48_DMSK_;
}

static inline uint32_t sm48_empty(const uint16_t *meta) {
    __m256i z  = _mm256_setzero_si256();
    __m256i lo = _mm256_load_si256((const __m256i *)meta);
    __m256i hi = _mm256_load_si256((const __m256i *)(meta + 16));
    return sm48_movemask16(_mm256_cmpeq_epi16(lo, z),
                           _mm256_cmpeq_epi16(hi, z))
           & SM48_DMSK_;
}

#else /* scalar SWAR */

static inline uint32_t sm48_pack4(uint64_t z) {
    return (uint32_t)((z * 0x0000200040008001ULL) >> 60);
}

static inline uint32_t sm48_match(const uint16_t *meta, uint16_t h2) {
    const uint64_t *w = (const uint64_t *)meta;
    uint64_t needle = (uint64_t)h2 * 0x0001000100010001ULL;
    uint64_t msb    = 0x8000800080008000ULL;
    uint32_t result = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t v  = w[i] ^ needle;
        uint64_t hi = (v | msb) - 0x0001000100010001ULL;
        uint64_t z  = ~hi & ~v & msb;
        result |= sm48_pack4(z) << (i * 4);
    }
    return result & SM48_DMSK_;
}

static inline uint32_t sm48_empty(const uint16_t *meta) {
    const uint64_t *w = (const uint64_t *)meta;
    uint64_t msb = 0x8000800080008000ULL;
    uint32_t result = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t z = ~w[i] & msb;
        result |= sm48_pack4(z) << (i * 4);
    }
    return result & SM48_DMSK_;
}

#endif

#if defined(__SSE4_2__) || defined(__AVX2__) || defined(__AVX512F__)
#define sm48_prefetch_line(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_T0)
#else
#define sm48_prefetch_line(ptr) __builtin_prefetch((const void *)(ptr), 0, 3)
#endif

#endif /* SIMD_MAP48_COMMON_H_ */

/* --- Per-instantiation --- */
#undef SM_
#define SM_(s) SMCAT(SIMD_MAP_NAME, s)

#define SM_VW_  (SIMD_MAP48_VAL_WORDS)

/* Group sizes:
 * Key group: 64B meta + 192B packed keys = 256B (4 cache lines)
 * Inline val: 32 * VW * 8B (31 used + 1 wasted for alignment) */
#define SM_KEY_GRP_  256u

#if SM_VW_ > 0
  #define SM_VAL_SZ_ ((SIMD_MAP48_VAL_WORDS) * 8u)
  #define SM_GRP_    (SM_KEY_GRP_ + 32u * SM_VAL_SZ_)
#else
  #define SM_GRP_    SM_KEY_GRP_
#endif

/* --- Types --- */

#if SM_VW_ > 0
struct SM_(_val) { uint64_t w[SM_VW_]; };
#endif

struct SIMD_MAP_NAME {
    char    *data;
    uint32_t count;
    uint32_t cap;    /* ng * 32 (slots, not groups) */
    uint32_t mask;   /* (cap >> 5) - 1 */
};

/* --- Helpers --- */

static inline char *SM_(_group)(const struct SIMD_MAP_NAME *m, uint32_t gi) {
    return m->data + (size_t)gi * SM_GRP_;
}

/* Read packed 48-bit key at slot s within group.
 * Keys start at grp+64, each is 6 bytes. We read 8 bytes and mask. */
static inline uint64_t SM_(_read_key)(const char *grp, int s) {
    uint64_t v;
    memcpy(&v, grp + 64 + s * 6, 8);
    return v & SM48_KEY_MASK_;
}

/* Write packed 48-bit key at slot s within group (6 bytes). */
static inline void SM_(_write_key)(char *grp, int s, uint64_t key) {
    char *p = grp + 64 + s * 6;
    uint32_t lo = (uint32_t)key;
    uint16_t hi = (uint16_t)(key >> 32);
    memcpy(p, &lo, 4);
    memcpy(p + 4, &hi, 2);
}

#if SM_VW_ > 0
static inline struct SM_(_val) *SM_(_val_at)(const struct SIMD_MAP_NAME *m,
                                              uint32_t gi, int slot) {
    char *grp = m->data + (size_t)gi * SM_GRP_;
    return (struct SM_(_val) *)(grp + SM_KEY_GRP_) + slot;
}

static inline void SM_(_val_copy)(struct SM_(_val) *dst, const uint64_t *val) {
    for (int i = 0; i < SM_VW_; i++)
        dst->w[i] = val[i];
}
#endif

/* --- Prefetch --- */

/* Read path: metadata + 3 key cache lines (4 total, vs 5 for sentinel KW=1) */
static inline void SM_(_prefetch)(const struct SIMD_MAP_NAME *m,
                                    uint64_t key) {
#if defined(__SSE4_2__) || defined(__ARM_FEATURE_CRC32)
    uint32_t gi = (uint32_t)_mm_crc32_u64(0, key) & m->mask;
    const char *grp = SM_(_group)(m, gi);
    _mm_prefetch(grp, _MM_HINT_T0);
    _mm_prefetch(grp + 64, _MM_HINT_T0);
    _mm_prefetch(grp + 128, _MM_HINT_T0);
    _mm_prefetch(grp + 192, _MM_HINT_T0);
#else
    struct sm48_h h = sm48_hash(key);
    uint32_t gi = h.lo & m->mask;
    const char *grp = SM_(_group)(m, gi);
    __builtin_prefetch(grp, 0, 3);
    __builtin_prefetch(grp + 64, 0, 3);
    __builtin_prefetch(grp + 128, 0, 3);
    __builtin_prefetch(grp + 192, 0, 3);
#endif
}

/* Insert path: metadata line only. */
static inline void SM_(_prefetch_insert)(const struct SIMD_MAP_NAME *m,
                                          uint64_t key) {
#if defined(__SSE4_2__) || defined(__ARM_FEATURE_CRC32)
    uint32_t gi = (uint32_t)_mm_crc32_u64(0, key) & m->mask;
    sm48_prefetch_line(SM_(_group)(m, gi));
#else
    struct sm48_h h = sm48_hash(key);
    uint32_t gi = h.lo & m->mask;
    __builtin_prefetch(SM_(_group)(m, gi), 0, 3);
#endif
}

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
        const char *old_grp = old_data + (size_t)g * SM_GRP_;
        const uint16_t *om = (const uint16_t *)old_grp;

        for (int s = 0; s < 31; s++) {
            if (!(om[s] & 0x8000)) continue;

            uint64_t key = SM_(_read_key)(old_grp, s);
#if SM_VW_ > 0
            const struct SM_(_val) *ov =
                (const struct SM_(_val) *)(old_grp + SM_KEY_GRP_) + s;
#endif
            struct sm48_h h = sm48_hash(key);
            uint16_t h2  = sm48_h2(h.lo);
            uint32_t gi  = h.lo & mask;
            for (;;) {
                char     *grp  = SM_(_group)(m, gi);
                uint16_t *base = (uint16_t *)grp;
                uint32_t em = sm48_empty(base);
                if (em) {
                    int pos = __builtin_ctz(em);
                    base[pos] = h2;
                    SM_(_write_key)(grp, pos, key);
#if SM_VW_ > 0
                    *SM_(_val_at)(m, gi, pos) = *ov;
#endif
                    m->count++;
                    break;
                }
                base[31] |= sm48_overflow_bit(h.hi);
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

#if SM_VW_ > 0

static inline int SM_(_insert)(struct SIMD_MAP_NAME *m,
                                uint64_t key, const uint64_t *val) {
    if (m->cap == 0) SM_(_alloc)(m, 32);
    if (m->count * 8 >= m->cap * 7) SM_(_grow)(m);

    struct sm48_h h = sm48_hash(key);
    uint16_t h2 = sm48_h2(h.lo);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        char     *grp  = SM_(_group)(m, gi);
        uint16_t *base = (uint16_t *)grp;

        uint32_t mm = sm48_match(base, h2);
        while (mm) {
            int pos = __builtin_ctz(mm);
            if (SM_(_read_key)(grp, pos) == key) return 0;
            mm &= mm - 1;
        }

        uint32_t em = sm48_empty(base);
        if (em) {
            int pos = __builtin_ctz(em);
            base[pos] = h2;
            SM_(_write_key)(grp, pos, key);
            SM_(_val_copy)(SM_(_val_at)(m, gi, pos), val);
            m->count++;
            return 1;
        }
        base[31] |= sm48_overflow_bit(h.hi);
        gi = (gi + 1) & m->mask;
    }
}

static inline void SM_(_insert_unique)(struct SIMD_MAP_NAME *m,
                                        uint64_t key, const uint64_t *val) {
    if (m->cap == 0) SM_(_alloc)(m, 32);
    if (m->count * 8 >= m->cap * 7) SM_(_grow)(m);

    struct sm48_h h = sm48_hash(key);
    uint16_t h2 = sm48_h2(h.lo);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        char     *grp  = SM_(_group)(m, gi);
        uint16_t *base = (uint16_t *)grp;

        uint32_t em = sm48_empty(base);
        if (em) {
            int pos = __builtin_ctz(em);
            base[pos] = h2;
            SM_(_write_key)(grp, pos, key);
            SM_(_val_copy)(SM_(_val_at)(m, gi, pos), val);
            m->count++;
            return;
        }
        base[31] |= sm48_overflow_bit(h.hi);
        gi = (gi + 1) & m->mask;
    }
}

static inline uint64_t *SM_(_get)(struct SIMD_MAP_NAME *m, uint64_t key) {
    if (__builtin_expect(m->cap == 0, 0)) return NULL;

    struct sm48_h h = sm48_hash(key);
    uint16_t h2 = sm48_h2(h.lo);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        const char     *grp  = SM_(_group)(m, gi);
        const uint16_t *base = (const uint16_t *)grp;

        uint32_t mm = sm48_match(base, h2);
        while (mm) {
            int pos = __builtin_ctz(mm);
            if (SM_(_read_key)(grp, pos) == key)
                return SM_(_val_at)(m, gi, pos)->w;
            mm &= mm - 1;
        }
        if (!((base[31] >> (h.hi & 15)) & 1)) return NULL;
        gi = (gi + 1) & m->mask;
    }
}

#else /* SM_VW_ == 0: set mode */

static inline int SM_(_insert)(struct SIMD_MAP_NAME *m, uint64_t key) {
    if (m->cap == 0) SM_(_alloc)(m, 32);
    if (m->count * 8 >= m->cap * 7) SM_(_grow)(m);

    struct sm48_h h = sm48_hash(key);
    uint16_t h2 = sm48_h2(h.lo);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        char     *grp  = SM_(_group)(m, gi);
        uint16_t *base = (uint16_t *)grp;

        uint32_t mm = sm48_match(base, h2);
        while (mm) {
            int pos = __builtin_ctz(mm);
            if (SM_(_read_key)(grp, pos) == key) return 0;
            mm &= mm - 1;
        }

        uint32_t em = sm48_empty(base);
        if (em) {
            int pos = __builtin_ctz(em);
            base[pos] = h2;
            SM_(_write_key)(grp, pos, key);
            m->count++;
            return 1;
        }
        base[31] |= sm48_overflow_bit(h.hi);
        gi = (gi + 1) & m->mask;
    }
}

static inline void SM_(_insert_unique)(struct SIMD_MAP_NAME *m,
                                        uint64_t key) {
    if (m->cap == 0) SM_(_alloc)(m, 32);
    if (m->count * 8 >= m->cap * 7) SM_(_grow)(m);

    struct sm48_h h = sm48_hash(key);
    uint16_t h2 = sm48_h2(h.lo);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        char     *grp  = SM_(_group)(m, gi);
        uint16_t *base = (uint16_t *)grp;

        uint32_t em = sm48_empty(base);
        if (em) {
            int pos = __builtin_ctz(em);
            base[pos] = h2;
            SM_(_write_key)(grp, pos, key);
            m->count++;
            return;
        }
        base[31] |= sm48_overflow_bit(h.hi);
        gi = (gi + 1) & m->mask;
    }
}

#endif /* SM_VW_ */

static inline int SM_(_contains)(struct SIMD_MAP_NAME *m, uint64_t key) {
    if (__builtin_expect(m->cap == 0, 0)) return 0;

    struct sm48_h h = sm48_hash(key);
    uint16_t h2 = sm48_h2(h.lo);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        const char     *grp  = SM_(_group)(m, gi);
        const uint16_t *base = (const uint16_t *)grp;

        uint32_t mm = sm48_match(base, h2);
        while (mm) {
            int pos = __builtin_ctz(mm);
            if (SM_(_read_key)(grp, pos) == key) return 1;
            mm &= mm - 1;
        }
        if (!((base[31] >> (h.hi & 15)) & 1)) return 0;
        gi = (gi + 1) & m->mask;
    }
}

static inline int SM_(_delete)(struct SIMD_MAP_NAME *m, uint64_t key) {
    if (__builtin_expect(m->cap == 0, 0)) return 0;

    struct sm48_h h = sm48_hash(key);
    uint16_t h2 = sm48_h2(h.lo);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        char     *grp  = SM_(_group)(m, gi);
        uint16_t *base = (uint16_t *)grp;

        uint32_t mm = sm48_match(base, h2);
        while (mm) {
            int pos = __builtin_ctz(mm);
            if (SM_(_read_key)(grp, pos) == key) {
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
#undef SM_GRP_
#ifdef SM_VAL_SZ_
#undef SM_VAL_SZ_
#endif
#undef SIMD_MAP_NAME
#undef SIMD_MAP48_VAL_WORDS
