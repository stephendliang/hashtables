/*
 * simd_map64.h — unified uint64_t key set/map, superblock layout
 *
 * Macro-generated via X-include pattern. Define parameters before including:
 *
 * Set mode (VAL_WORDS omitted or 0):
 *   #define SIMD_MAP_NAME  simd_set64
 *   #include "simd_map64.h"
 *
 * Map mode (VAL_WORDS >= 1):
 *   #define SIMD_MAP_NAME           my_map64
 *   #define SIMD_MAP64_VAL_WORDS    1
 *   #define SIMD_MAP64_BLOCK_STRIDE 1   // power of 2, default 1
 *   #include "simd_map64.h"
 *
 * Key: uint64_t (0 reserved). Value: uint64_t[VAL_WORDS] (map mode).
 * Group: 8 key slots (64B, 1 cache line). SIMD compares all 8 directly —
 * zero false positives, no metadata, no scalar verify.
 *
 * Superblock layout (BLOCK_STRIDE=N, map mode):
 *   [N key groups (N×64B)] [N value groups (N×8×VW×8B)]
 * N=1 degenerates to inline: [8 keys | 8 values] per group.
 *
 * Backends: AVX-512 (1 instr/group), AVX2 (5 instr), scalar (portable).
 * Delete: backshift (no tombstones), moves values alongside keys.
 * Load factor: 75%.
 */

#ifndef SIMD_MAP_NAME
#error "Define SIMD_MAP_NAME before including simd_map64.h"
#endif
#ifndef SIMD_MAP64_VAL_WORDS
#define SIMD_MAP64_VAL_WORDS 0
#endif
#ifndef SIMD_MAP64_BLOCK_STRIDE
#define SIMD_MAP64_BLOCK_STRIDE 1
#endif
#if SIMD_MAP64_BLOCK_STRIDE > 1 && (SIMD_MAP64_BLOCK_STRIDE & (SIMD_MAP64_BLOCK_STRIDE - 1))
#error "SIMD_MAP64_BLOCK_STRIDE must be a power of 2"
#endif

/* --- Common (once) --- */
#ifndef SIMD_MAP64_COMMON_H_
#define SIMD_MAP64_COMMON_H_

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

/* Hash: CRC32 on SIMD paths, murmur3 finalizer on scalar */
#if defined(__AVX512F__) || defined(__AVX2__) || defined(__SSE4_2__) || defined(__ARM_FEATURE_CRC32)
#define sm64_hash(key) ((uint32_t)_mm_crc32_u64(0, (key)))
#else
static inline uint32_t sm64_hash(uint64_t key) {
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return (uint32_t)key;
}
#endif

/* SIMD backends: match/empty on 8 × uint64_t key group */
#if defined(__AVX512F__)

static inline uint8_t sm64_match(const uint64_t *grp, uint64_t key) {
    return (uint8_t)_mm512_cmpeq_epi64_mask(
        _mm512_load_si512((const __m512i *)grp),
        _mm512_set1_epi64((long long)key));
}

static inline uint8_t sm64_empty(const uint64_t *grp) {
    return (uint8_t)_mm512_cmpeq_epi64_mask(
        _mm512_load_si512((const __m512i *)grp),
        _mm512_setzero_si512());
}

#define sm64_prefetch_line(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_T0)

#elif defined(__AVX2__)

static inline uint8_t sm64_match(const uint64_t *grp, uint64_t key) {
    __m256i k  = _mm256_set1_epi64x((long long)key);
    __m256i lo = _mm256_load_si256((const __m256i *)grp);
    __m256i hi = _mm256_load_si256((const __m256i *)(grp + 4));
    int mlo = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(lo, k)));
    int mhi = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(hi, k)));
    return (uint8_t)(mlo | (mhi << 4));
}

static inline uint8_t sm64_empty(const uint64_t *grp) {
    __m256i z  = _mm256_setzero_si256();
    __m256i lo = _mm256_load_si256((const __m256i *)grp);
    __m256i hi = _mm256_load_si256((const __m256i *)(grp + 4));
    int mlo = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(lo, z)));
    int mhi = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(hi, z)));
    return (uint8_t)(mlo | (mhi << 4));
}

#define sm64_prefetch_line(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_T0)

#elif defined(__ARM_NEON)

static inline uint8_t sm64_match(const uint64_t *grp, uint64_t key) {
    uint64x2_t k = vdupq_n_u64(key);
    uint32_t m0 = neon_movemask_u64(vceqq_u64(vld1q_u64(grp + 0), k));
    uint32_t m1 = neon_movemask_u64(vceqq_u64(vld1q_u64(grp + 2), k));
    uint32_t m2 = neon_movemask_u64(vceqq_u64(vld1q_u64(grp + 4), k));
    uint32_t m3 = neon_movemask_u64(vceqq_u64(vld1q_u64(grp + 6), k));
    return (uint8_t)(m0 | (m1 << 2) | (m2 << 4) | (m3 << 6));
}

static inline uint8_t sm64_empty(const uint64_t *grp) {
    uint64x2_t z = vdupq_n_u64(0);
    uint32_t m0 = neon_movemask_u64(vceqq_u64(vld1q_u64(grp + 0), z));
    uint32_t m1 = neon_movemask_u64(vceqq_u64(vld1q_u64(grp + 2), z));
    uint32_t m2 = neon_movemask_u64(vceqq_u64(vld1q_u64(grp + 4), z));
    uint32_t m3 = neon_movemask_u64(vceqq_u64(vld1q_u64(grp + 6), z));
    return (uint8_t)(m0 | (m1 << 2) | (m2 << 4) | (m3 << 6));
}

#define sm64_prefetch_line(ptr) __builtin_prefetch((const void *)(ptr), 0, 3)

#else /* Scalar */

static inline uint8_t sm64_match(const uint64_t *grp, uint64_t key) {
    uint8_t m = 0;
    for (int i = 0; i < 8; i++)
        m |= (uint8_t)((grp[i] == key) << i);
    return m;
}

static inline uint8_t sm64_empty(const uint64_t *grp) {
    uint8_t m = 0;
    for (int i = 0; i < 8; i++)
        m |= (uint8_t)((grp[i] == 0) << i);
    return m;
}

#define sm64_prefetch_line(ptr) __builtin_prefetch((const void *)(ptr), 0, 3)

#endif

#endif /* SIMD_MAP64_COMMON_H_ */

/* --- Per-instantiation --- */
#undef SM_
#define SM_(s) SMCAT(SIMD_MAP_NAME, s)

#define SM_VW_        (SIMD_MAP64_VAL_WORDS)
#if SM_VW_ > 0
#define SM_VAL_GRP_   (8u * SM_VW_ * 8u)
#define SM_ENTRY_SZ_  (64u + SM_VAL_GRP_)
#else
#define SM_ENTRY_SZ_  64u
#endif

#if SIMD_MAP64_BLOCK_STRIDE > 1
  #define SM_BLK_SHIFT_ ((unsigned)__builtin_ctz(SIMD_MAP64_BLOCK_STRIDE))
  #define SM_BLK_MASK_  ((unsigned)(SIMD_MAP64_BLOCK_STRIDE) - 1u)
  #define SM_SUPER_     ((size_t)(SIMD_MAP64_BLOCK_STRIDE) * SM_ENTRY_SZ_)
#endif

/* --- Types --- */

#if SM_VW_ > 0
struct SM_(_val) { uint64_t w[SM_VW_]; };
#endif

struct SIMD_MAP_NAME {
#if SM_VW_ > 0
    char    *data;
#else
    uint64_t *data;
#endif
    uint32_t count;
    uint32_t cap;    /* ng * 8 */
    uint32_t mask;   /* (cap >> 3) - 1 */
};

/* --- Helpers --- */

static inline uint64_t *SM_(_key_group)(const struct SIMD_MAP_NAME *m,
                                         uint32_t gi) {
#if SIMD_MAP64_BLOCK_STRIDE > 1
    uint32_t super = gi >> SM_BLK_SHIFT_;
    uint32_t local = gi & SM_BLK_MASK_;
    return (uint64_t *)(m->data + (size_t)super * SM_SUPER_
                        + (size_t)local * 64);
#elif SM_VW_ == 0
    return m->data + (gi << 3);
#else
    return (uint64_t *)(m->data + (size_t)gi * SM_ENTRY_SZ_);
#endif
}

#if SM_VW_ > 0
static inline struct SM_(_val) *SM_(_val_at)(const struct SIMD_MAP_NAME *m,
                                              uint32_t gi, int slot) {
#if SIMD_MAP64_BLOCK_STRIDE > 1
    uint32_t super = gi >> SM_BLK_SHIFT_;
    uint32_t local = gi & SM_BLK_MASK_;
    char *sb = m->data + (size_t)super * SM_SUPER_;
    char *vb = sb + (size_t)SIMD_MAP64_BLOCK_STRIDE * 64;
    return (struct SM_(_val) *)(vb) + (size_t)local * 8 + slot;
#else
    char *grp = m->data + (size_t)gi * SM_ENTRY_SZ_;
    return (struct SM_(_val) *)(grp + 64) + slot;
#endif
}

static inline void SM_(_val_copy)(struct SM_(_val) *dst, const uint64_t *val) {
    for (int i = 0; i < SM_VW_; i++)
        dst->w[i] = val[i];
}
#endif /* SM_VW_ > 0 */

/* --- Prefetch --- */

/* Full prefetch for read paths: key line (+ value lines in map mode) */
static inline void SM_(_prefetch)(const struct SIMD_MAP_NAME *m,
                                    uint64_t key) {
    uint32_t gi = sm64_hash(key) & m->mask;
    sm64_prefetch_line(SM_(_key_group)(m, gi));
#if SM_VW_ > 0
    char *vp = (char *)SM_(_val_at)(m, gi, 0);
    for (unsigned i = 0; i < SM_VAL_GRP_; i += 64)
        sm64_prefetch_line(vp + i);
#endif
}

/* Lightweight prefetch for insert paths: key line only.
 * Value writes use write-allocate through the store buffer. */
static inline void SM_(_prefetch_insert)(const struct SIMD_MAP_NAME *m,
                                          uint64_t key) {
    uint32_t gi = sm64_hash(key) & m->mask;
    sm64_prefetch_line(SM_(_key_group)(m, gi));
}

/* --- Alloc / grow --- */

static size_t SM_(_mapsize)(uint32_t cap) {
#if SIMD_MAP64_BLOCK_STRIDE > 1
    size_t raw = (size_t)((cap >> 3) >> SM_BLK_SHIFT_) * SM_SUPER_;
#else
    size_t raw = (size_t)(cap >> 3) * SM_ENTRY_SZ_;
#endif
    return (raw + (2u << 20) - 1) & ~((size_t)(2u << 20) - 1);
}

static void SM_(_alloc)(struct SIMD_MAP_NAME *m, uint32_t cap) {
#if SIMD_MAP64_BLOCK_STRIDE > 1
    if (cap < 8u * SIMD_MAP64_BLOCK_STRIDE)
        cap = 8u * SIMD_MAP64_BLOCK_STRIDE;
#endif
    size_t total = SM_(_mapsize)(cap);
    m->data = mmap(NULL, total, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB
                   | MAP_POPULATE, -1, 0);
    if (m->data == MAP_FAILED) {
        m->data = mmap(NULL, total, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS
                       | MAP_POPULATE, -1, 0);
        if (m->data != MAP_FAILED)
            madvise(m->data, total, MADV_HUGEPAGE);
    }
    m->cap   = cap;
    m->mask  = (cap >> 3) - 1;
    m->count = 0;
}

static void SM_(_grow)(struct SIMD_MAP_NAME *m) {
    uint32_t old_cap  = m->cap;
    uint32_t old_ng   = old_cap >> 3;
#if SM_VW_ > 0
    char    *old_data = m->data;
#else
    uint64_t *old_data = m->data;
#endif

    SM_(_alloc)(m, old_cap * 2);
    uint32_t mask = m->mask;

    for (uint32_t g = 0; g < old_ng; g++) {
#if SIMD_MAP64_BLOCK_STRIDE > 1
        uint32_t osup = g >> SM_BLK_SHIFT_;
        uint32_t oloc = g & SM_BLK_MASK_;
#if SM_VW_ > 0
        uint64_t *old_keys = (uint64_t *)(old_data + (size_t)osup * SM_SUPER_
                                          + (size_t)oloc * 64);
        struct SM_(_val) *old_vals = (struct SM_(_val) *)(
            old_data + (size_t)osup * SM_SUPER_
            + (size_t)SIMD_MAP64_BLOCK_STRIDE * 64)
            + (size_t)oloc * 8;
#else
        uint64_t *old_keys = old_data
            + (size_t)(osup * SIMD_MAP64_BLOCK_STRIDE + oloc) * 8;
#endif
#else /* BLOCK_STRIDE == 1 */
#if SM_VW_ > 0
        uint64_t *old_keys = (uint64_t *)(old_data + (size_t)g * SM_ENTRY_SZ_);
        struct SM_(_val) *old_vals = (struct SM_(_val) *)(
            old_data + (size_t)g * SM_ENTRY_SZ_ + 64);
#else
        uint64_t *old_keys = old_data + (size_t)g * 8;
#endif
#endif
        for (int s = 0; s < 8; s++) {
            uint64_t key = old_keys[s];
            if (!key) continue;
            uint32_t gi = sm64_hash(key) & mask;
            for (;;) {
                uint64_t *grp = SM_(_key_group)(m, gi);
                uint8_t em = sm64_empty(grp);
                if (em) {
                    int pos = __builtin_ctz(em);
                    grp[pos] = key;
#if SM_VW_ > 0
                    *SM_(_val_at)(m, gi, pos) = old_vals[s];
#endif
                    m->count++;
                    break;
                }
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
    uint64_t need = (uint64_t)n * 4 / 3 + 1;
    uint32_t cap = 8;
    while (cap < need) cap *= 2;
    SM_(_alloc)(m, cap);
}

static inline void SM_(_destroy)(struct SIMD_MAP_NAME *m) {
    if (m->data) munmap(m->data, SM_(_mapsize)(m->cap));
}

#if SM_VW_ > 0
static inline int SM_(_insert)(struct SIMD_MAP_NAME *m,
                                uint64_t key, const uint64_t *val) {
#else
static inline int SM_(_insert)(struct SIMD_MAP_NAME *m, uint64_t key) {
#endif
    if (m->cap == 0) SM_(_alloc)(m, 64);
    if (m->count * 4 >= m->cap * 3) SM_(_grow)(m);

    uint32_t gi = sm64_hash(key) & m->mask;
    for (;;) {
        uint64_t *grp = SM_(_key_group)(m, gi);
        if (sm64_match(grp, key)) return 0;
        uint8_t em = sm64_empty(grp);
        if (em) {
            int pos = __builtin_ctz(em);
            grp[pos] = key;
#if SM_VW_ > 0
            SM_(_val_copy)(SM_(_val_at)(m, gi, pos), val);
#endif
            m->count++;
            return 1;
        }
        gi = (gi + 1) & m->mask;
    }
}

#if SM_VW_ > 0
static inline void SM_(_insert_unique)(struct SIMD_MAP_NAME *m,
                                        uint64_t key, const uint64_t *val) {
#else
static inline void SM_(_insert_unique)(struct SIMD_MAP_NAME *m,
                                        uint64_t key) {
#endif
    if (m->cap == 0) SM_(_alloc)(m, 64);
    if (m->count * 4 >= m->cap * 3) SM_(_grow)(m);

    uint32_t gi = sm64_hash(key) & m->mask;
    for (;;) {
        uint64_t *grp = SM_(_key_group)(m, gi);
        uint8_t em = sm64_empty(grp);
        if (em) {
            int pos = __builtin_ctz(em);
            grp[pos] = key;
#if SM_VW_ > 0
            SM_(_val_copy)(SM_(_val_at)(m, gi, pos), val);
#endif
            m->count++;
            return;
        }
        gi = (gi + 1) & m->mask;
    }
}

#if SM_VW_ > 0
static inline uint64_t *SM_(_get)(struct SIMD_MAP_NAME *m, uint64_t key) {
    if (__builtin_expect(m->cap == 0, 0)) return NULL;
    uint32_t gi = sm64_hash(key) & m->mask;
    for (;;) {
        uint64_t *grp = SM_(_key_group)(m, gi);
        uint8_t mm = sm64_match(grp, key);
        if (mm) return SM_(_val_at)(m, gi, __builtin_ctz(mm))->w;
        if (sm64_empty(grp)) return NULL;
        gi = (gi + 1) & m->mask;
    }
}
#endif

static inline int SM_(_contains)(struct SIMD_MAP_NAME *m, uint64_t key) {
    if (__builtin_expect(m->cap == 0, 0)) return 0;
    uint32_t gi = sm64_hash(key) & m->mask;
    for (;;) {
        uint64_t *grp = SM_(_key_group)(m, gi);
        if (sm64_match(grp, key)) return 1;
        if (sm64_empty(grp))      return 0;
        gi = (gi + 1) & m->mask;
    }
}

/* Backshift: repair probe chain after deletion, moving values alongside keys */
static inline void SM_(_backshift_at)(struct SIMD_MAP_NAME *m,
                                       uint32_t gi, int slot) {
    uint32_t mask = m->mask;
    uint32_t hole_gi = gi;
    int hole_slot = slot;
    uint32_t scan_gi = (gi + 1) & mask;

    for (;;) {
        uint32_t pf_gi = (scan_gi + 2) & mask;
        sm64_prefetch_line(SM_(_key_group)(m, pf_gi));

        uint64_t *scan_grp = SM_(_key_group)(m, scan_gi);
        uint8_t scan_empty = sm64_empty(scan_grp);

        if (scan_empty == 0xFF) return;

        uint64_t cand_keys[8];
        uint32_t cand_homes[8];
        int cand_slots[8];
        int n_cand = 0;

        for (uint8_t todo = (~scan_empty) & 0xFF; todo; todo &= todo - 1) {
            int s = __builtin_ctz(todo);
            cand_keys[n_cand] = scan_grp[s];
            cand_slots[n_cand] = s;
            n_cand++;
        }

        for (int j = 0; j < n_cand; j++)
            cand_homes[j] = sm64_hash(cand_keys[j]) & mask;

        int moved = 0;
        for (int j = 0; j < n_cand; j++) {
            if (((hole_gi - cand_homes[j]) & mask) <
                ((scan_gi - cand_homes[j]) & mask)) {
                uint64_t *hole_grp = SM_(_key_group)(m, hole_gi);
                hole_grp[hole_slot] = cand_keys[j];
#if SM_VW_ > 0
                *SM_(_val_at)(m, hole_gi, hole_slot) =
                    *SM_(_val_at)(m, scan_gi, cand_slots[j]);
#endif
                scan_grp[cand_slots[j]] = 0;
                if (scan_empty) return;
                hole_gi = scan_gi;
                hole_slot = cand_slots[j];
                moved = 1;
                break;
            }
        }

        if (!moved && scan_empty) return;
        scan_gi = (scan_gi + 1) & mask;
    }
}

static inline int SM_(_delete)(struct SIMD_MAP_NAME *m, uint64_t key) {
    if (__builtin_expect(m->cap == 0, 0)) return 0;
    uint32_t gi = sm64_hash(key) & m->mask;
    for (;;) {
        uint64_t *grp = SM_(_key_group)(m, gi);
        uint8_t mm = sm64_match(grp, key);
        uint8_t empty = sm64_empty(grp);
        if (mm) {
            int slot = __builtin_ctz(mm);
            grp[slot] = 0;
            m->count--;
            if (!empty) SM_(_backshift_at)(m, gi, slot);
            return 1;
        }
        if (empty) return 0;
        gi = (gi + 1) & m->mask;
    }
}

/* --- Set-specific convenience (VW == 0 only) --- */
#if SM_VW_ == 0

/* Prefetch home + overflow group */
static inline void SM_(_prefetch2)(const struct SIMD_MAP_NAME *m,
                                     uint64_t key) {
    uint32_t gi = sm64_hash(key) & m->mask;
    sm64_prefetch_line(SM_(_key_group)(m, gi));
    gi = (gi + 1) & m->mask;
    sm64_prefetch_line(SM_(_key_group)(m, gi));
}

/* Unified op: single probe loop. op: 0=contains, 1=insert, 2=delete */
static inline int SM_(_op)(struct SIMD_MAP_NAME *m, uint64_t key, int op) {
    if (__builtin_expect(m->cap == 0, 0)) {
        if (op == 1) SM_(_alloc)(m, 64);
        else return 0;
    }
    if (__builtin_expect(op == 1 && m->count * 4 >= m->cap * 3, 0))
        SM_(_grow)(m);
    uint32_t gi = sm64_hash(key) & m->mask, mask = m->mask;
    for (;;) {
        uint64_t *grp = SM_(_key_group)(m, gi);
        uint8_t mm = sm64_match(grp, key);
        uint8_t empty = sm64_empty(grp);
        if (mm) {
            if (__builtin_expect(op == 2, 0)) {
                int slot = __builtin_ctz(mm);
                grp[slot] = 0;
                m->count--;
                if (!empty) SM_(_backshift_at)(m, gi, slot);
                return 1;
            }
            return op == 0; // 1 for contains, 0 for insert-dup
        }
        if (empty) {
            if (__builtin_expect(op == 1, 0)) {
                grp[__builtin_ctz(empty)] = key;
                m->count++;
                return 1;
            }
            return 0;
        }
        gi = (gi + 1) & mask;
    }
}

#endif /* SM_VW_ == 0 */

/* --- Cleanup --- */
#undef SM_
#undef SM_VW_
#if SIMD_MAP64_VAL_WORDS > 0
#undef SM_VAL_GRP_
#endif
#undef SM_ENTRY_SZ_
#undef SM_BLK_SHIFT_
#undef SM_BLK_MASK_
#undef SM_SUPER_
#undef SIMD_MAP_NAME
#undef SIMD_MAP64_VAL_WORDS
#undef SIMD_MAP64_BLOCK_STRIDE
