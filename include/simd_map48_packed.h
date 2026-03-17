/*
 * simd_map48_packed.h — direct-compare 48-bit set/map, packed 3×u16 interleaved
 *
 * 10 keys per 64B cache line. No metadata — SIMD compares all 3 words of
 * each key simultaneously via broadcast+coalesce+PEXT.
 *
 * Group layout (uint16_t grp[32]):
 *   grp[0]:      ctrl — bits [9:0] occupancy, bits [15:10] reserved
 *   grp[1]:      ovf  — 16-bit overflow partition (same as sentinel)
 *   grp[2..31]:  10 keys × 3 words, interleaved
 *     Key i: words at [2+3i], [2+3i+1], [2+3i+2]
 *
 * Set mode when VAL_WORDS is omitted or 0, map mode when VAL_WORDS >= 1.
 * Map mode: values stored inline after keys (64B keys + 10*VW*8B values).
 * Key: uint64_t (lower 48 bits, upper 16 must be 0, key=0 reserved).
 * Backends: AVX2, scalar. Load factor: 7/8 (87.5%).
 *
 * Set mode:
 *   #define SIMD_MAP_NAME my_set48p
 *   #include "simd_map48_packed.h"
 *
 * Map mode:
 *   #define SIMD_MAP_NAME              my_map48p
 *   #define SIMD_MAP48P_VAL_WORDS      1
 *   #define SIMD_MAP48P_BLOCK_STRIDE   1   // power of 2, default 1
 *   #include "simd_map48_packed.h"
 *
 * Superblock layout (BLOCK_STRIDE=N, map mode):
 *   [N key groups (N×64B)] [N value groups (N×10×VW×8B)]
 * N=1 degenerates to inline: [10 keys | 10 values] per group.
 */

#ifndef SIMD_MAP_NAME
#error "Define SIMD_MAP_NAME before including simd_map48_packed.h"
#endif

/* --- Common (once) --- */
#ifndef SIMD_MAP48P_COMMON_H_
#define SIMD_MAP48P_COMMON_H_

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

#ifndef SMCAT_
#define SMCAT_(a, b) a##b
#define SMCAT(a, b)  SMCAT_(a, b)
#endif

#define SM48P_SLOTS_ 10
#define SM48P_OCC_MASK_ 0x03FFu

/* Hash: CRC32 on SIMD paths, murmur3 finalizer on scalar */
struct sm48p_h { uint32_t lo, hi; };

#if defined(__SSE4_2__)
static inline struct sm48p_h sm48p_hash(uint64_t key) {
    uint32_t a = (uint32_t)_mm_crc32_u64(0, key);
    uint32_t b = (uint32_t)_mm_crc32_u64(a, key);
    return (struct sm48p_h){a, b};
}
#else
static inline struct sm48p_h sm48p_hash(uint64_t key) {
    uint64_t h = key;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return (struct sm48p_h){(uint32_t)h, (uint32_t)(h >> 32)};
}
#endif

/* Read packed key from group */
static inline uint64_t sm48p_read_key(const uint16_t *grp, int slot) {
    int base = 2 + slot * 3;
    return (uint64_t)grp[base]
         | ((uint64_t)grp[base + 1] << 16)
         | ((uint64_t)grp[base + 2] << 32);
}

/* Write packed key to group */
static inline void sm48p_write_key(uint16_t *grp, int slot, uint64_t key) {
    int base = 2 + slot * 3;
    grp[base]     = (uint16_t)key;
    grp[base + 1] = (uint16_t)(key >> 16);
    grp[base + 2] = (uint16_t)(key >> 32);
}

/* --- SIMD match --- */

#if defined(__AVX512BW__)

/* AVX-512: single zmm covers all 32 uint16 positions.
 * 3 cmpeq → 3 k-masks, coalesce+PEXT extracts 10-bit match. */
static inline uint16_t sm48p_match(const uint16_t *grp, uint64_t key) {
    __m512i data = _mm512_loadu_si512(grp);

    __mmask32 m0 = _mm512_cmpeq_epi16_mask(data,
                       _mm512_set1_epi16((short)(uint16_t)key));
    __mmask32 m1 = _mm512_cmpeq_epi16_mask(data,
                       _mm512_set1_epi16((short)(uint16_t)(key >> 16)));
    __mmask32 m2 = _mm512_cmpeq_epi16_mask(data,
                       _mm512_set1_epi16((short)(uint16_t)(key >> 32)));

    uint32_t combined = (uint32_t)(m0 & (m1 >> 1) & (m2 >> 2));
    uint16_t match = (uint16_t)_pext_u32(combined, 0x24924924u);
    return match & (grp[0] & SM48P_OCC_MASK_);
}

static inline uint16_t sm48p_empty(const uint16_t *grp) {
    return (~grp[0]) & SM48P_OCC_MASK_;
}

#define sm48p_prefetch_line(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_T0)

#elif defined(__AVX2__)

/* Broadcast+coalesce match: 3 word compares across 32 uint16 positions,
 * then shift+AND+PEXT to extract per-slot match bits */
static inline uint16_t sm48p_match(const uint16_t *grp, uint64_t key) {
    __m256i lo = _mm256_loadu_si256((const __m256i *)grp);
    __m256i hi = _mm256_loadu_si256((const __m256i *)(grp + 16));

    __m256i k0 = _mm256_set1_epi16((short)(uint16_t)key);
    __m256i k1 = _mm256_set1_epi16((short)(uint16_t)(key >> 16));
    __m256i k2 = _mm256_set1_epi16((short)(uint16_t)(key >> 32));

    uint32_t mlo, mhi;

    mlo = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi16(lo, k0));
    mhi = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi16(hi, k0));
    uint32_t m0 = _pext_u32(mlo, 0xAAAAAAAAu) | (_pext_u32(mhi, 0xAAAAAAAAu) << 16);

    mlo = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi16(lo, k1));
    mhi = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi16(hi, k1));
    uint32_t m1 = _pext_u32(mlo, 0xAAAAAAAAu) | (_pext_u32(mhi, 0xAAAAAAAAu) << 16);

    mlo = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi16(lo, k2));
    mhi = (uint32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi16(hi, k2));
    uint32_t m2 = _pext_u32(mlo, 0xAAAAAAAAu) | (_pext_u32(mhi, 0xAAAAAAAAu) << 16);

    /* Coalesce: shift to align words from same slot, AND */
    uint32_t combined = m0 & (m1 >> 1) & (m2 >> 2);

    /* Extract every-3rd bit starting at bit 2 → 10-bit match */
    uint16_t match = (uint16_t)_pext_u32(combined, 0x24924924u);
    return match & (grp[0] & SM48P_OCC_MASK_);
}

static inline uint16_t sm48p_empty(const uint16_t *grp) {
    return (~grp[0]) & SM48P_OCC_MASK_;
}

#define sm48p_prefetch_line(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_T0)

#else /* scalar */

static inline uint16_t sm48p_match(const uint16_t *grp, uint64_t key) {
    uint16_t occ = grp[0] & SM48P_OCC_MASK_;
    uint16_t result = 0;
    for (int i = 0; i < SM48P_SLOTS_; i++) {
        if ((occ >> i) & 1) {
            if (sm48p_read_key(grp, i) == key)
                result |= (uint16_t)(1u << i);
        }
    }
    return result;
}

static inline uint16_t sm48p_empty(const uint16_t *grp) {
    return (~grp[0]) & SM48P_OCC_MASK_;
}

#if defined(__SSE4_2__)
#define sm48p_prefetch_line(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_T0)
#else
#define sm48p_prefetch_line(ptr) __builtin_prefetch((const void *)(ptr), 0, 3)
#endif

#endif /* AVX2 */

#endif /* SIMD_MAP48P_COMMON_H_ */

/* --- Per-instantiation --- */
#undef SM_
#define SM_(s) SMCAT(SIMD_MAP_NAME, s)

#ifndef SIMD_MAP48P_VAL_WORDS
#define SIMD_MAP48P_VAL_WORDS 0
#endif
#ifndef SIMD_MAP48P_BLOCK_STRIDE
#define SIMD_MAP48P_BLOCK_STRIDE 1
#endif
#if SIMD_MAP48P_BLOCK_STRIDE > 1 && (SIMD_MAP48P_BLOCK_STRIDE & (SIMD_MAP48P_BLOCK_STRIDE - 1))
#error "SIMD_MAP48P_BLOCK_STRIDE must be a power of 2"
#endif

#define SM48P_VW_ (SIMD_MAP48P_VAL_WORDS)

#if SM48P_VW_ > 0
#define SM48P_VAL_SZ_   (SM48P_VW_ * 8u)
#define SM48P_VAL_GRP_  (SM48P_SLOTS_ * SM48P_VAL_SZ_)
#define SM48P_ENTRY_SZ_ (64u + SM48P_VAL_GRP_)

struct SM_(_val) { uint64_t w[SM48P_VW_]; };
#else
#define SM48P_ENTRY_SZ_ 64u
#endif

#if SIMD_MAP48P_BLOCK_STRIDE > 1 && SM48P_VW_ > 0
#define SM48P_BLK_SHIFT_ ((unsigned)__builtin_ctz(SIMD_MAP48P_BLOCK_STRIDE))
#define SM48P_BLK_MASK_  ((unsigned)(SIMD_MAP48P_BLOCK_STRIDE) - 1u)
#define SM48P_SUPER_     ((size_t)(SIMD_MAP48P_BLOCK_STRIDE) * SM48P_ENTRY_SZ_)
#endif

struct SIMD_MAP_NAME {
    char    *data;
    uint32_t count;
    uint32_t ng;     /* number of groups, always power of 2 */
    uint32_t mask;   /* ng - 1 */
};

/* --- Helpers --- */

static inline char *SM_(_group)(const struct SIMD_MAP_NAME *m, uint32_t gi) {
#if SIMD_MAP48P_BLOCK_STRIDE > 1 && SM48P_VW_ > 0
    uint32_t super = gi >> SM48P_BLK_SHIFT_;
    uint32_t local = gi & SM48P_BLK_MASK_;
    return m->data + (size_t)super * SM48P_SUPER_ + (size_t)local * 64;
#else
    return m->data + (size_t)gi * SM48P_ENTRY_SZ_;
#endif
}

#if SM48P_VW_ > 0
static inline uint64_t *SM_(_val_at)(const struct SIMD_MAP_NAME *m,
                                      uint32_t gi, int slot) {
#if SIMD_MAP48P_BLOCK_STRIDE > 1
    uint32_t super = gi >> SM48P_BLK_SHIFT_;
    uint32_t local = gi & SM48P_BLK_MASK_;
    char *vb = m->data + (size_t)super * SM48P_SUPER_
               + (size_t)SIMD_MAP48P_BLOCK_STRIDE * 64;
    return (uint64_t *)(vb + (size_t)local * SM48P_VAL_GRP_
                        + (unsigned)slot * SM48P_VAL_SZ_);
#else
    return (uint64_t *)(SM_(_group)(m, gi) + 64 + (unsigned)slot * SM48P_VAL_SZ_);
#endif
}
#endif

/* --- Prefetch --- */

static inline void SM_(_prefetch)(const struct SIMD_MAP_NAME *m, uint64_t key) {
    uint32_t gi = sm48p_hash(key).lo & m->mask;
    sm48p_prefetch_line(SM_(_group)(m, gi));
#if SM48P_VW_ > 0
    sm48p_prefetch_line((const char *)SM_(_val_at)(m, gi, 0));
#endif
}

static inline void SM_(_prefetch_insert)(const struct SIMD_MAP_NAME *m,
                                          uint64_t key) {
    uint32_t gi = sm48p_hash(key).lo & m->mask;
    sm48p_prefetch_line(SM_(_group)(m, gi));
}

/* --- Alloc / grow --- */

static size_t SM_(_mapsize)(uint32_t ng) {
#if SIMD_MAP48P_BLOCK_STRIDE > 1 && SM48P_VW_ > 0
    size_t raw = (size_t)(ng >> SM48P_BLK_SHIFT_) * SM48P_SUPER_;
#else
    size_t raw = (size_t)ng * SM48P_ENTRY_SZ_;
#endif
    return (raw + (2u << 20) - 1) & ~((size_t)(2u << 20) - 1);
}

static void SM_(_alloc)(struct SIMD_MAP_NAME *m, uint32_t ng) {
#if SIMD_MAP48P_BLOCK_STRIDE > 1
    if (ng < SIMD_MAP48P_BLOCK_STRIDE) ng = SIMD_MAP48P_BLOCK_STRIDE;
#endif
    size_t total = SM_(_mapsize)(ng);
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
    m->ng    = ng;
    m->mask  = ng - 1;
    m->count = 0;
}

static void SM_(_grow)(struct SIMD_MAP_NAME *m) {
    uint32_t old_ng   = m->ng;
    char    *old_data = m->data;

    SM_(_alloc)(m, old_ng * 2);
    uint32_t mask = m->mask;

    for (uint32_t g = 0; g < old_ng; g++) {
#if SIMD_MAP48P_BLOCK_STRIDE > 1 && SM48P_VW_ > 0
        uint32_t osup = g >> SM48P_BLK_SHIFT_;
        uint32_t oloc = g & SM48P_BLK_MASK_;
        const char *old_grp = old_data + (size_t)osup * SM48P_SUPER_
                              + (size_t)oloc * 64;
#else
        const char *old_grp = old_data + (size_t)g * SM48P_ENTRY_SZ_;
#endif
        const uint16_t *old_keys = (const uint16_t *)old_grp;
        uint16_t occ = old_keys[0] & SM48P_OCC_MASK_;

        while (occ) {
            int s = __builtin_ctz(occ);
            occ &= occ - 1;

            uint64_t key = sm48p_read_key(old_keys, s);
            struct sm48p_h h = sm48p_hash(key);
            uint32_t gi = h.lo & mask;

            for (;;) {
                char *grp = SM_(_group)(m, gi);
                uint16_t *keys = (uint16_t *)grp;
                uint16_t em = sm48p_empty(keys);
                if (em) {
                    int pos = __builtin_ctz(em);
                    keys[0] |= (uint16_t)(1u << pos);
                    sm48p_write_key(keys, pos, key);
#if SM48P_VW_ > 0
                    {
#if SIMD_MAP48P_BLOCK_STRIDE > 1
                        const char *old_val = old_data
                            + (size_t)osup * SM48P_SUPER_
                            + (size_t)SIMD_MAP48P_BLOCK_STRIDE * 64
                            + (size_t)oloc * SM48P_VAL_GRP_
                            + (unsigned)s * SM48P_VAL_SZ_;
#else
                        const char *old_val = old_grp + 64
                            + (unsigned)s * SM48P_VAL_SZ_;
#endif
                        memcpy(SM_(_val_at)(m, gi, pos), old_val,
                               SM48P_VAL_SZ_);
                    }
#endif
                    m->count++;
                    break;
                }
                keys[1] |= (uint16_t)(1u << (h.hi & 15));
                gi = (gi + 1) & mask;
            }
        }
    }
    munmap(old_data, SM_(_mapsize)(old_ng));
}

/* --- Public API --- */

static inline void SM_(_init)(struct SIMD_MAP_NAME *m) {
    memset(m, 0, sizeof(*m));
}

static inline void SM_(_init_cap)(struct SIMD_MAP_NAME *m, uint32_t n) {
    memset(m, 0, sizeof(*m));
    /* Load factor 7/8: need ng such that ng*10*7/8 >= n → ng >= n*8/(10*7) */
    uint64_t need = ((uint64_t)n * 8 + 69) / 70;
    uint32_t ng = 1;
    while (ng < need) ng *= 2;
    SM_(_alloc)(m, ng);
}

static inline void SM_(_destroy)(struct SIMD_MAP_NAME *m) {
    if (m->data) munmap(m->data, SM_(_mapsize)(m->ng));
}

#if SM48P_VW_ > 0
static inline int SM_(_insert)(struct SIMD_MAP_NAME *m, uint64_t key,
                               const uint64_t *val) {
#else
static inline int SM_(_insert)(struct SIMD_MAP_NAME *m, uint64_t key) {
#endif
    if (m->ng == 0) SM_(_alloc)(m, 16);
    if (m->count * 8 >= (uint64_t)m->ng * SM48P_SLOTS_ * 7) SM_(_grow)(m);

    struct sm48p_h h = sm48p_hash(key);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        char *grp = SM_(_group)(m, gi);
        uint16_t *keys = (uint16_t *)grp;
        uint16_t mm = sm48p_match(keys, key);
        if (mm) {
#if SM48P_VW_ > 0
            int slot = __builtin_ctz(mm);
            memcpy(SM_(_val_at)(m, gi, slot), val, SM48P_VAL_SZ_);
#endif
            return 0;
        }

        uint16_t em = sm48p_empty(keys);
        if (em) {
            int pos = __builtin_ctz(em);
            keys[0] |= (uint16_t)(1u << pos);
            sm48p_write_key(keys, pos, key);
#if SM48P_VW_ > 0
            memcpy(SM_(_val_at)(m, gi, pos), val, SM48P_VAL_SZ_);
#endif
            m->count++;
            return 1;
        }
        keys[1] |= (uint16_t)(1u << (h.hi & 15));
        gi = (gi + 1) & m->mask;
    }
}

#if SM48P_VW_ > 0
static inline void SM_(_insert_unique)(struct SIMD_MAP_NAME *m, uint64_t key,
                                       const uint64_t *val) {
#else
static inline void SM_(_insert_unique)(struct SIMD_MAP_NAME *m, uint64_t key) {
#endif
    if (m->ng == 0) SM_(_alloc)(m, 16);
    if (m->count * 8 >= (uint64_t)m->ng * SM48P_SLOTS_ * 7) SM_(_grow)(m);

    struct sm48p_h h = sm48p_hash(key);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        char *grp = SM_(_group)(m, gi);
        uint16_t *keys = (uint16_t *)grp;
        uint16_t em = sm48p_empty(keys);
        if (em) {
            int pos = __builtin_ctz(em);
            keys[0] |= (uint16_t)(1u << pos);
            sm48p_write_key(keys, pos, key);
#if SM48P_VW_ > 0
            memcpy(SM_(_val_at)(m, gi, pos), val, SM48P_VAL_SZ_);
#endif
            m->count++;
            return;
        }
        keys[1] |= (uint16_t)(1u << (h.hi & 15));
        gi = (gi + 1) & m->mask;
    }
}

static inline int SM_(_contains)(struct SIMD_MAP_NAME *m, uint64_t key) {
    if (__builtin_expect(m->ng == 0, 0)) return 0;

    struct sm48p_h h = sm48p_hash(key);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        const uint16_t *grp = (const uint16_t *)SM_(_group)(m, gi);
        if (sm48p_match(grp, key)) return 1;
        if (!((grp[1] >> (h.hi & 15)) & 1)) return 0;
        gi = (gi + 1) & m->mask;
    }
}

#if SM48P_VW_ > 0
static inline uint64_t *SM_(_get)(struct SIMD_MAP_NAME *m, uint64_t key) {
    if (__builtin_expect(m->ng == 0, 0)) return NULL;

    struct sm48p_h h = sm48p_hash(key);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        char *grp = SM_(_group)(m, gi);
        uint16_t *keys = (uint16_t *)grp;
        uint16_t mm = sm48p_match(keys, key);
        if (mm) {
            int slot = __builtin_ctz(mm);
            return SM_(_val_at)(m, gi, slot);
        }
        if (!((keys[1] >> (h.hi & 15)) & 1)) return NULL;
        gi = (gi + 1) & m->mask;
    }
}
#endif

static inline int SM_(_delete)(struct SIMD_MAP_NAME *m, uint64_t key) {
    if (__builtin_expect(m->ng == 0, 0)) return 0;

    struct sm48p_h h = sm48p_hash(key);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        uint16_t *keys = (uint16_t *)SM_(_group)(m, gi);
        uint16_t mm = sm48p_match(keys, key);
        if (mm) {
            int slot = __builtin_ctz(mm);
            keys[0] &= ~(uint16_t)(1u << slot);
            m->count--;
            return 1;
        }
        if (!((keys[1] >> (h.hi & 15)) & 1)) return 0;
        gi = (gi + 1) & m->mask;
    }
}

/* --- Cleanup --- */
#undef SM_
#undef SM48P_VW_
#undef SM48P_ENTRY_SZ_
#ifdef SM48P_VAL_SZ_
#undef SM48P_VAL_SZ_
#undef SM48P_VAL_GRP_
#endif
#ifdef SM48P_BLK_SHIFT_
#undef SM48P_BLK_SHIFT_
#undef SM48P_BLK_MASK_
#undef SM48P_SUPER_
#endif
#undef SIMD_MAP_NAME
#undef SIMD_MAP48P_VAL_WORDS
#undef SIMD_MAP48P_BLOCK_STRIDE
