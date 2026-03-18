/*
 * simd_map48_split.h — direct-compare 48-bit set/map, split hi32+lo16
 *
 * 10 keys per 64B cache line. No metadata filtering — SIMD compares
 * hi32 (vpcmpeqd) and lo16 (vpcmpeqw) separately, then AND.
 *
 * Group layout (64B key block):
 *   Offset 0-3:   uint32_t ctrl
 *                    bits [9:0]   = occupancy (10 slots)
 *                    bits [25:10] = overflow (16 partitions)
 *                    bits [31:26] = reserved
 *   Offset 4-43:  uint32_t hi[10]  (upper 32 bits: key >> 16)
 *   Offset 44-63: uint16_t lo[10]  (lower 16 bits: key & 0xFFFF)
 *   Total key block: 4 + 40 + 20 = 64 bytes
 *
 * Set mode when VAL_WORDS is omitted or 0, map mode when VAL_WORDS >= 1.
 * Map mode: values stored inline after keys (64B keys + 10*VW*8B values).
 * Key: uint64_t (lower 48 bits, upper 16 must be 0, key=0 reserved).
 * Backends: AVX2, scalar. Load factor: 7/8 (87.5%).
 *
 * Set mode:
 *   #define SIMD_MAP_NAME my_set48s
 *   #include "simd_map48_split.h"
 *
 * Map mode:
 *   #define SIMD_MAP_NAME              my_map48s
 *   #define SIMD_MAP48S_VAL_WORDS      1
 *   #define SIMD_MAP48S_BLOCK_STRIDE   1   // power of 2, default 1
 *   #include "simd_map48_split.h"
 *
 * Superblock layout (BLOCK_STRIDE=N, map mode):
 *   [N key groups (N×64B)] [N value groups (N×10×VW×8B)]
 * N=1 degenerates to inline: [10 keys | 10 values] per group.
 */

#ifndef SIMD_MAP_NAME
#error "Define SIMD_MAP_NAME before including simd_map48_split.h"
#endif

/* --- Common (once) --- */
#ifndef SIMD_MAP48S_COMMON_H_
#define SIMD_MAP48S_COMMON_H_

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

#define SM48S_SLOTS_    10
#define SM48S_OCC_MASK_ 0x03FFu
#define SM48S_OVF_SHIFT_ 10

/* Hash: CRC32 on SIMD paths, murmur3 finalizer on scalar */
struct sm48s_h { uint32_t lo, hi; };

#if defined(__SSE4_2__) || defined(__ARM_FEATURE_CRC32)
static inline struct sm48s_h sm48s_hash(uint64_t key) {
    uint32_t a = (uint32_t)_mm_crc32_u64(0, key);
    uint32_t b = (uint32_t)_mm_crc32_u64(a, key);
    return (struct sm48s_h){a, b};
}
#else
static inline struct sm48s_h sm48s_hash(uint64_t key) {
    uint64_t h = key;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return (struct sm48s_h){(uint32_t)h, (uint32_t)(h >> 32)};
}
#endif

/* Group access helpers — cast char* to typed pointers */
static inline uint32_t *sm48s_ctrl(void *grp) { return (uint32_t *)grp; }
static inline uint32_t *sm48s_hi(void *grp)   { return (uint32_t *)((char *)grp + 4); }
static inline uint16_t *sm48s_lo(void *grp)   { return (uint16_t *)((char *)grp + 44); }

static inline const uint32_t *sm48s_ctrl_c(const void *grp) { return (const uint32_t *)grp; }
static inline const uint32_t *sm48s_hi_c(const void *grp)   { return (const uint32_t *)((const char *)grp + 4); }
static inline const uint16_t *sm48s_lo_c(const void *grp)   { return (const uint16_t *)((const char *)grp + 44); }

static inline uint64_t sm48s_read_key(const void *grp, int slot) {
    return ((uint64_t)sm48s_hi_c(grp)[slot] << 16) | sm48s_lo_c(grp)[slot];
}

static inline void sm48s_write_key(void *grp, int slot, uint64_t key) {
    sm48s_hi(grp)[slot] = (uint32_t)(key >> 16);
    sm48s_lo(grp)[slot] = (uint16_t)(key);
}

/* --- SIMD match --- */

#if defined(__AVX512BW__)

static inline uint16_t sm48s_match(const void *grp, uint64_t key) {
    uint32_t key_hi = (uint32_t)(key >> 16);
    uint16_t key_lo = (uint16_t)key;

    /* Hi: vpcmpeqd zmm — all 10 hi-words, no scalar tail.
     * Masked load: fault suppression for overread past last group. */
    __m512i h16 = _mm512_maskz_loadu_epi32((__mmask16)SM48S_OCC_MASK_,
                                           sm48s_hi_c(grp));
    __mmask16 m_hi = _mm512_cmpeq_epi32_mask(h16, _mm512_set1_epi32((int)key_hi));

    /* Lo: vpcmpeqw ymm — all 10 lo-words, no scalar tail.
     * Masked load: fault suppression for overread past group. */
    __m256i l16 = _mm256_maskz_loadu_epi16((__mmask16)SM48S_OCC_MASK_,
                                           sm48s_lo_c(grp));
    __mmask16 m_lo = _mm256_cmpeq_epi16_mask(l16, _mm256_set1_epi16((short)key_lo));

    uint32_t ctrl = *sm48s_ctrl_c(grp);
    return (uint16_t)(m_hi & m_lo) & (uint16_t)(ctrl & SM48S_OCC_MASK_);
}

static inline uint16_t sm48s_empty(const void *grp) {
    uint32_t ctrl = *sm48s_ctrl_c(grp);
    return (~ctrl) & SM48S_OCC_MASK_;
}

#define sm48s_prefetch_line(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_T0)

#elif defined(__AVX2__)

static inline uint16_t sm48s_match(const void *grp, uint64_t key) {
    uint32_t key_hi = (uint32_t)(key >> 16);
    uint16_t key_lo = (uint16_t)key;

    /* Hi: vpcmpeqd on hi[0..7], scalar for hi[8..9] */
    __m256i h8 = _mm256_loadu_si256((const __m256i *)sm48s_hi_c(grp));
    __m256i kh = _mm256_set1_epi32((int)key_hi);
    uint32_t m_hi8 = (uint32_t)_mm256_movemask_ps(
        _mm256_castsi256_ps(_mm256_cmpeq_epi32(h8, kh)));
    uint16_t m_hi = (uint16_t)m_hi8;
    const uint32_t *hip = sm48s_hi_c(grp);
    if (hip[8] == key_hi) m_hi |= (1u << 8);
    if (hip[9] == key_hi) m_hi |= (1u << 9);

    /* Lo: vpcmpeqw on lo[0..7], scalar for lo[8..9] */
    __m128i l8 = _mm_loadu_si128((const __m128i *)sm48s_lo_c(grp));
    __m128i kl = _mm_set1_epi16((short)key_lo);
    uint32_t raw = (uint32_t)_mm_movemask_epi8(_mm_cmpeq_epi16(l8, kl));
    uint16_t m_lo = (uint16_t)_pext_u32(raw, 0xAAAAu);
    const uint16_t *lop = sm48s_lo_c(grp);
    if (lop[8] == key_lo) m_lo |= (1u << 8);
    if (lop[9] == key_lo) m_lo |= (1u << 9);

    uint32_t ctrl = *sm48s_ctrl_c(grp);
    return m_hi & m_lo & (uint16_t)(ctrl & SM48S_OCC_MASK_);
}

static inline uint16_t sm48s_empty(const void *grp) {
    uint32_t ctrl = *sm48s_ctrl_c(grp);
    return (~ctrl) & SM48S_OCC_MASK_;
}

#define sm48s_prefetch_line(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_T0)

#elif defined(__ARM_NEON)

static inline uint16_t sm48s_match(const void *grp, uint64_t key) {
    uint32_t key_hi = (uint32_t)(key >> 16);
    uint16_t key_lo = (uint16_t)key;
    const uint32_t *hip = sm48s_hi_c(grp);
    const uint16_t *lop = sm48s_lo_c(grp);

    /* Hi: vceqq on hi[0..3] and hi[4..7], scalar for hi[8..9] */
    uint32x4_t kh = vdupq_n_u32(key_hi);
    uint32_t m_hi = neon_movemask_u32(vceqq_u32(vld1q_u32(hip + 0), kh));
    m_hi |= neon_movemask_u32(vceqq_u32(vld1q_u32(hip + 4), kh)) << 4;
    if (hip[8] == key_hi) m_hi |= (1u << 8);
    if (hip[9] == key_hi) m_hi |= (1u << 9);

    /* Lo: vceqq on lo[0..7], scalar for lo[8..9] */
    uint16x8_t kl = vdupq_n_u16(key_lo);
    uint32_t m_lo = neon_movemask_u16(vceqq_u16(vld1q_u16(lop), kl));
    if (lop[8] == key_lo) m_lo |= (1u << 8);
    if (lop[9] == key_lo) m_lo |= (1u << 9);

    uint32_t ctrl = *sm48s_ctrl_c(grp);
    return (uint16_t)(m_hi & m_lo) & (uint16_t)(ctrl & SM48S_OCC_MASK_);
}

static inline uint16_t sm48s_empty(const void *grp) {
    uint32_t ctrl = *sm48s_ctrl_c(grp);
    return (~ctrl) & SM48S_OCC_MASK_;
}

#define sm48s_prefetch_line(ptr) __builtin_prefetch((const void *)(ptr), 0, 3)

#else /* scalar */

static inline uint16_t sm48s_match(const void *grp, uint64_t key) {
    uint32_t ctrl = *sm48s_ctrl_c(grp);
    uint16_t occ = (uint16_t)(ctrl & SM48S_OCC_MASK_);
    uint32_t key_hi = (uint32_t)(key >> 16);
    uint16_t key_lo = (uint16_t)key;
    const uint32_t *hip = sm48s_hi_c(grp);
    const uint16_t *lop = sm48s_lo_c(grp);
    uint16_t result = 0;
    for (int i = 0; i < SM48S_SLOTS_; i++) {
        if ((occ >> i) & 1) {
            if (hip[i] == key_hi && lop[i] == key_lo)
                result |= (uint16_t)(1u << i);
        }
    }
    return result;
}

static inline uint16_t sm48s_empty(const void *grp) {
    uint32_t ctrl = *sm48s_ctrl_c(grp);
    return (~ctrl) & SM48S_OCC_MASK_;
}

#if defined(__SSE4_2__)
#define sm48s_prefetch_line(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_T0)
#else
#define sm48s_prefetch_line(ptr) __builtin_prefetch((const void *)(ptr), 0, 3)
#endif

#endif /* AVX2 */

#endif /* SIMD_MAP48S_COMMON_H_ */

/* --- Per-instantiation --- */
#undef SM_
#define SM_(s) SMCAT(SIMD_MAP_NAME, s)

#ifndef SIMD_MAP48S_VAL_WORDS
#define SIMD_MAP48S_VAL_WORDS 0
#endif
#ifndef SIMD_MAP48S_BLOCK_STRIDE
#define SIMD_MAP48S_BLOCK_STRIDE 1
#endif
#if SIMD_MAP48S_BLOCK_STRIDE > 1 && (SIMD_MAP48S_BLOCK_STRIDE & (SIMD_MAP48S_BLOCK_STRIDE - 1))
#error "SIMD_MAP48S_BLOCK_STRIDE must be a power of 2"
#endif

#define SM48S_VW_ (SIMD_MAP48S_VAL_WORDS)

#if SM48S_VW_ > 0
#define SM48S_VAL_SZ_   (SM48S_VW_ * 8u)
#define SM48S_VAL_GRP_  (SM48S_SLOTS_ * SM48S_VAL_SZ_)
#define SM48S_ENTRY_SZ_ (64u + SM48S_VAL_GRP_)

struct SM_(_val) { uint64_t w[SM48S_VW_]; };
#else
#define SM48S_ENTRY_SZ_ 64u
#endif

#if SIMD_MAP48S_BLOCK_STRIDE > 1 && SM48S_VW_ > 0
#define SM48S_BLK_SHIFT_ ((unsigned)__builtin_ctz(SIMD_MAP48S_BLOCK_STRIDE))
#define SM48S_BLK_MASK_  ((unsigned)(SIMD_MAP48S_BLOCK_STRIDE) - 1u)
#define SM48S_SUPER_     ((size_t)(SIMD_MAP48S_BLOCK_STRIDE) * SM48S_ENTRY_SZ_)
#endif

struct SIMD_MAP_NAME {
    char    *data;
    uint32_t count;
    uint32_t ng;     /* number of groups, always power of 2 */
    uint32_t mask;   /* ng - 1 */
};

/* --- Helpers --- */

static inline char *SM_(_group)(const struct SIMD_MAP_NAME *m, uint32_t gi) {
#if SIMD_MAP48S_BLOCK_STRIDE > 1 && SM48S_VW_ > 0
    uint32_t super = gi >> SM48S_BLK_SHIFT_;
    uint32_t local = gi & SM48S_BLK_MASK_;
    return m->data + (size_t)super * SM48S_SUPER_ + (size_t)local * 64;
#else
    return m->data + (size_t)gi * SM48S_ENTRY_SZ_;
#endif
}

#if SM48S_VW_ > 0
static inline uint64_t *SM_(_val_at)(const struct SIMD_MAP_NAME *m,
                                      uint32_t gi, int slot) {
#if SIMD_MAP48S_BLOCK_STRIDE > 1
    uint32_t super = gi >> SM48S_BLK_SHIFT_;
    uint32_t local = gi & SM48S_BLK_MASK_;
    char *vb = m->data + (size_t)super * SM48S_SUPER_
               + (size_t)SIMD_MAP48S_BLOCK_STRIDE * 64;
    return (uint64_t *)(vb + (size_t)local * SM48S_VAL_GRP_
                        + (unsigned)slot * SM48S_VAL_SZ_);
#else
    return (uint64_t *)(SM_(_group)(m, gi) + 64 + (unsigned)slot * SM48S_VAL_SZ_);
#endif
}
#endif

/* --- Prefetch --- */

static inline void SM_(_prefetch)(const struct SIMD_MAP_NAME *m, uint64_t key) {
    uint32_t gi = sm48s_hash(key).lo & m->mask;
    sm48s_prefetch_line(SM_(_group)(m, gi));
#if SM48S_VW_ > 0
    sm48s_prefetch_line((const char *)SM_(_val_at)(m, gi, 0));
#endif
}

static inline void SM_(_prefetch_insert)(const struct SIMD_MAP_NAME *m,
                                          uint64_t key) {
    uint32_t gi = sm48s_hash(key).lo & m->mask;
    sm48s_prefetch_line(SM_(_group)(m, gi));
}

/* --- Alloc / grow --- */

static size_t SM_(_mapsize)(uint32_t ng) {
#if SIMD_MAP48S_BLOCK_STRIDE > 1 && SM48S_VW_ > 0
    size_t raw = (size_t)(ng >> SM48S_BLK_SHIFT_) * SM48S_SUPER_;
#else
    size_t raw = (size_t)ng * SM48S_ENTRY_SZ_;
#endif
    return (raw + (2u << 20) - 1) & ~((size_t)(2u << 20) - 1);
}

static void SM_(_alloc)(struct SIMD_MAP_NAME *m, uint32_t ng) {
#if SIMD_MAP48S_BLOCK_STRIDE > 1
    if (ng < SIMD_MAP48S_BLOCK_STRIDE) ng = SIMD_MAP48S_BLOCK_STRIDE;
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
#if SIMD_MAP48S_BLOCK_STRIDE > 1 && SM48S_VW_ > 0
        uint32_t osup = g >> SM48S_BLK_SHIFT_;
        uint32_t oloc = g & SM48S_BLK_MASK_;
        const char *old_grp = old_data + (size_t)osup * SM48S_SUPER_
                              + (size_t)oloc * 64;
#else
        const char *old_grp = old_data + (size_t)g * SM48S_ENTRY_SZ_;
#endif
        uint32_t ctrl = *(const uint32_t *)old_grp;
        uint16_t occ = (uint16_t)(ctrl & SM48S_OCC_MASK_);

        while (occ) {
            int s = __builtin_ctz(occ);
            occ &= occ - 1;

            uint64_t key = sm48s_read_key(old_grp, s);
            struct sm48s_h h = sm48s_hash(key);
            uint32_t gi = h.lo & mask;

            for (;;) {
                char *grp = SM_(_group)(m, gi);
                uint16_t em = sm48s_empty(grp);
                if (em) {
                    int pos = __builtin_ctz(em);
                    uint32_t *cp = sm48s_ctrl(grp);
                    *cp |= (1u << pos);
                    sm48s_write_key(grp, pos, key);
#if SM48S_VW_ > 0
                    {
#if SIMD_MAP48S_BLOCK_STRIDE > 1
                        const char *old_val = old_data
                            + (size_t)osup * SM48S_SUPER_
                            + (size_t)SIMD_MAP48S_BLOCK_STRIDE * 64
                            + (size_t)oloc * SM48S_VAL_GRP_
                            + (unsigned)s * SM48S_VAL_SZ_;
#else
                        const char *old_val = old_grp + 64
                            + (unsigned)s * SM48S_VAL_SZ_;
#endif
                        memcpy(SM_(_val_at)(m, gi, pos), old_val,
                               SM48S_VAL_SZ_);
                    }
#endif
                    m->count++;
                    break;
                }
                uint32_t *cp = sm48s_ctrl(grp);
                *cp |= (1u << (SM48S_OVF_SHIFT_ + (h.hi & 15)));
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
    uint64_t need = ((uint64_t)n * 8 + 69) / 70;
    uint32_t ng = 1;
    while (ng < need) ng *= 2;
    SM_(_alloc)(m, ng);
}

static inline void SM_(_destroy)(struct SIMD_MAP_NAME *m) {
    if (m->data) munmap(m->data, SM_(_mapsize)(m->ng));
}

#if SM48S_VW_ > 0
static inline int SM_(_insert)(struct SIMD_MAP_NAME *m, uint64_t key,
                               const uint64_t *val) {
#else
static inline int SM_(_insert)(struct SIMD_MAP_NAME *m, uint64_t key) {
#endif
    if (m->ng == 0) SM_(_alloc)(m, 16);
    if (m->count * 8 >= (uint64_t)m->ng * SM48S_SLOTS_ * 7) SM_(_grow)(m);

    struct sm48s_h h = sm48s_hash(key);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        char *grp = SM_(_group)(m, gi);
        uint16_t mm = sm48s_match(grp, key);
        if (mm) {
#if SM48S_VW_ > 0
            int slot = __builtin_ctz(mm);
            memcpy(SM_(_val_at)(m, gi, slot), val, SM48S_VAL_SZ_);
#endif
            return 0;
        }

        uint16_t em = sm48s_empty(grp);
        if (em) {
            int pos = __builtin_ctz(em);
            uint32_t *cp = sm48s_ctrl(grp);
            *cp |= (1u << pos);
            sm48s_write_key(grp, pos, key);
#if SM48S_VW_ > 0
            memcpy(SM_(_val_at)(m, gi, pos), val, SM48S_VAL_SZ_);
#endif
            m->count++;
            return 1;
        }
        uint32_t *cp = sm48s_ctrl(grp);
        *cp |= (1u << (SM48S_OVF_SHIFT_ + (h.hi & 15)));
        gi = (gi + 1) & m->mask;
    }
}

#if SM48S_VW_ > 0
static inline void SM_(_insert_unique)(struct SIMD_MAP_NAME *m, uint64_t key,
                                       const uint64_t *val) {
#else
static inline void SM_(_insert_unique)(struct SIMD_MAP_NAME *m, uint64_t key) {
#endif
    if (m->ng == 0) SM_(_alloc)(m, 16);
    if (m->count * 8 >= (uint64_t)m->ng * SM48S_SLOTS_ * 7) SM_(_grow)(m);

    struct sm48s_h h = sm48s_hash(key);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        char *grp = SM_(_group)(m, gi);
        uint16_t em = sm48s_empty(grp);
        if (em) {
            int pos = __builtin_ctz(em);
            uint32_t *cp = sm48s_ctrl(grp);
            *cp |= (1u << pos);
            sm48s_write_key(grp, pos, key);
#if SM48S_VW_ > 0
            memcpy(SM_(_val_at)(m, gi, pos), val, SM48S_VAL_SZ_);
#endif
            m->count++;
            return;
        }
        uint32_t *cp = sm48s_ctrl(grp);
        *cp |= (1u << (SM48S_OVF_SHIFT_ + (h.hi & 15)));
        gi = (gi + 1) & m->mask;
    }
}

static inline int SM_(_contains)(struct SIMD_MAP_NAME *m, uint64_t key) {
    if (__builtin_expect(m->ng == 0, 0)) return 0;

    struct sm48s_h h = sm48s_hash(key);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        const char *grp = SM_(_group)(m, gi);
        if (sm48s_match(grp, key)) return 1;
        uint32_t ctrl = *sm48s_ctrl_c(grp);
        if (!((ctrl >> (SM48S_OVF_SHIFT_ + (h.hi & 15))) & 1)) return 0;
        gi = (gi + 1) & m->mask;
    }
}

#if SM48S_VW_ > 0
static inline uint64_t *SM_(_get)(struct SIMD_MAP_NAME *m, uint64_t key) {
    if (__builtin_expect(m->ng == 0, 0)) return NULL;

    struct sm48s_h h = sm48s_hash(key);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        char *grp = SM_(_group)(m, gi);
        uint16_t mm = sm48s_match(grp, key);
        if (mm) {
            int slot = __builtin_ctz(mm);
            return SM_(_val_at)(m, gi, slot);
        }
        uint32_t ctrl = *sm48s_ctrl_c(grp);
        if (!((ctrl >> (SM48S_OVF_SHIFT_ + (h.hi & 15))) & 1)) return NULL;
        gi = (gi + 1) & m->mask;
    }
}
#endif

static inline int SM_(_delete)(struct SIMD_MAP_NAME *m, uint64_t key) {
    if (__builtin_expect(m->ng == 0, 0)) return 0;

    struct sm48s_h h = sm48s_hash(key);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        char *grp = SM_(_group)(m, gi);
        uint16_t mm = sm48s_match(grp, key);
        if (mm) {
            int slot = __builtin_ctz(mm);
            uint32_t *cp = sm48s_ctrl(grp);
            *cp &= ~(1u << slot);
            m->count--;
            return 1;
        }
        uint32_t ctrl = *sm48s_ctrl_c(grp);
        if (!((ctrl >> (SM48S_OVF_SHIFT_ + (h.hi & 15))) & 1)) return 0;
        gi = (gi + 1) & m->mask;
    }
}

/* --- Cleanup --- */
#undef SM_
#undef SM48S_VW_
#undef SM48S_ENTRY_SZ_
#ifdef SM48S_VAL_SZ_
#undef SM48S_VAL_SZ_
#undef SM48S_VAL_GRP_
#endif
#ifdef SM48S_BLK_SHIFT_
#undef SM48S_BLK_SHIFT_
#undef SM48S_BLK_MASK_
#undef SM48S_SUPER_
#endif
#undef SIMD_MAP_NAME
#undef SIMD_MAP48S_VAL_WORDS
#undef SIMD_MAP48S_BLOCK_STRIDE
