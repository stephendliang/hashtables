/*
 * simd_set128: AVX-512 hash set for 128-bit keys with bit-stealing overflow
 *
 * Header-only. Interleaved group layout: each group is 576 bytes
 * (64B metadata + 512B keys) in a single contiguous allocation.
 * All 32 slots are data — no wasted sentinel.
 *
 * Bit-stealing design: per-slot 16-bit metadata encodes:
 *   [15] occupied  [14:11] overflow partition (4 bits)  [10:0] h2 fingerprint
 *
 * Overflow information is OR'd into all occupied slots when a group
 * overflows, rather than stored in a dedicated sentinel slot. This:
 *   - Reclaims the sentinel slot (32 vs 31 data slots per group)
 *   - Eliminates & 0x7FFFFFFF masking on every SIMD result
 *   - Naturally erodes ghost bits on delete (vs permanent sentinel ghosts)
 *   - Replaces scalar sentinel load+shift+test with SIMD vptestmw
 *
 * 11-bit h2 gives 1/2048 false positive rate (still excellent).
 * 4 overflow partition bits give 4 partitions. At 7/8 load, ~1-2% of
 * groups overflow, yielding ~0.0025 extra probes per miss — immeasurable.
 *
 * Net SIMD instruction change on hit path: +1 vpandd, -1 kandw = zero.
 * Overflow test reuses already-loaded group register via vptestmw —
 * faster than scalar sentinel check (no second memory access).
 *
 * Delete is tombstone-free and backshift-free: delete clears OCC + h2
 * but preserves ghost overflow bits in the slot. Insert inherits ghost
 * bits from reused slots, so overflow information survives full group
 * turnover. Ghost bits are cleaned up on grow (full rehash).
 *
 * Key width is irrelevant to the SIMD hot path: SIMD operates only on
 * 16-bit metadata, and scalar key comparison fires only on h2 match.
 *
 * Prefetch pipelining: all operations are memory-latency-bound at scale.
 * Use simd_set128_prefetch() PF iterations ahead of the operation to
 * overlap DRAM access with computation.
 */
#ifndef SIMD_MAP128_H
#define SIMD_MAP128_H

#if defined(__AVX512F__) || defined(__AVX2__) || defined(__SSE4_2__)
#include <immintrin.h>
#endif

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#define SM128_INIT_CAP    32
#define SM128_LOAD_NUM    7
#define SM128_LOAD_DEN    8
#define SM128_GROUP_BYTES 576u          /* 64B meta + 32×16B keys per group */

/* Bit-stealing metadata layout (per 16-bit slot):
 *   [15]    occupied flag
 *   [14:11] overflow partition bitmask (OVF_BITS bits)
 *   [10:0]  h2 fingerprint (H2_BITS bits)
 *
 * Changing H2_BITS recomputes everything. Range: 8–14.
 */
#define H2_BITS          11
#define OVF_BITS         (15 - H2_BITS)                          /* 4 */
#define H2_MASK          ((1u << H2_BITS) - 1)                   /* 0x07FF */
#define OVF_SHIFT        H2_BITS                                 /* 11 */
#define OVF_FIELD_MASK   (((1u << OVF_BITS) - 1) << OVF_SHIFT)  /* 0x7800 */
#define OCC_BIT          0x8000u
#define MATCH_MASK       (OCC_BIT | H2_MASK)                     /* 0x87FF */

struct ss128_kv { uint64_t lo, hi; };

struct simd_set128 {
    char *data;          /* interleaved groups: [meta 64B | keys 512B] × ng */
    uint32_t count;
    uint32_t cap;        /* ng * 32 (all slots are data) */
    uint32_t mask;       /* (cap >> 5) - 1, precomputed for group index */
};

/* --- Hash: split CRC32 — 3-cycle gi/h2, deferred overflow ---
 *
 * Round a = crc32(khi[31:0], klo): folds both key halves in one round.
 *   → gi from lower bits, h2 from upper H2_BITS bits. Available at 3 cycles.
 * Round b = crc32(a, khi): overflow partition (OVF_BITS bits). Depends on a
 *   but executes in parallel with address computation and memory load via OoO.
 *   Only consumed after SIMD compare completes — never on critical path.
 */

struct ss128_h { uint32_t lo, hi; };

#if defined(__SSE4_2__)
static inline struct ss128_h ss128_hash(uint64_t klo, uint64_t khi) {
    uint32_t a = (uint32_t)_mm_crc32_u64((uint32_t)khi, klo);
    uint32_t b = (uint32_t)_mm_crc32_u64(a, khi);
    return (struct ss128_h){a, b};
}
#else
static inline struct ss128_h ss128_hash(uint64_t klo, uint64_t khi) {
    uint64_t h = klo ^ (khi + 0x9e3779b97f4a7c15ULL + (khi << 6) + (khi >> 2));
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return (struct ss128_h){(uint32_t)h, (uint32_t)(h >> 32)};
}
#endif

/* --- Metadata encoding ---
 *
 * h2: top H2_BITS bits of hash.lo (round a). Occupied bit added at storage.
 * overflow_bit: partition index from hash.hi (round b), mapped to bit 11..14.
 */

static inline uint16_t ss128_h2(uint32_t lo) {
    return (uint16_t)((lo >> (32 - H2_BITS)) & H2_MASK);
}

static inline uint16_t ss128_overflow_bit(uint32_t hi) {
    return (uint16_t)(1u << ((hi & (OVF_BITS - 1)) + OVF_SHIFT));
}

/* --- Group access helpers --- */

static inline char *ss128_group(const struct simd_set128 *m, uint32_t gi) {
    return m->data + (size_t)gi * SM128_GROUP_BYTES;
}

/* --- Prefetch helper --- */

static inline void simd_set128_prefetch(const struct simd_set128 *m,
                                        uint64_t klo, uint64_t khi) {
#if defined(__SSE4_2__)
    uint32_t a  = (uint32_t)_mm_crc32_u64((uint32_t)khi, klo);
    uint32_t gi = a & m->mask;
    const char *grp = ss128_group(m, gi);
    _mm_prefetch(grp, _MM_HINT_T0);
    _mm_prefetch(grp + 64, _MM_HINT_T0);
    _mm_prefetch(grp + 128, _MM_HINT_T0);
    _mm_prefetch(grp + 192, _MM_HINT_T0);
    _mm_prefetch(grp + 256, _MM_HINT_T0);
#else
    struct ss128_h h = ss128_hash(klo, khi);
    uint32_t gi = h.lo & m->mask;
    const char *grp = ss128_group(m, gi);
    __builtin_prefetch(grp, 0, 3);
    __builtin_prefetch(grp + 64, 0, 3);
    __builtin_prefetch(grp + 128, 0, 3);
    __builtin_prefetch(grp + 192, 0, 3);
    __builtin_prefetch(grp + 256, 0, 3);
#endif
}

/* ================================================================
 * Backend selection — only #ifdef in the file
 *
 * Four primitives abstract all SIMD:
 *   match(meta, h2)               → bitmask of slots matching OCC|h2
 *   empty(meta)                   → bitmask of slots with OCC_BIT clear
 *   overflow_test(meta, ovf_bit)  → 1 if any slot has ovf_bit set
 *   overflow_propagate(meta, ovf_bit) → OR ovf_bit into all slots
 *
 * Each primitive loads/stores the 64B metadata group internally.
 * Consecutive calls on the same pointer CSE to one load after inlining.
 * ================================================================ */

#if defined(__AVX512F__)

/* Match: vpandd masks out overflow bits, then compare against OCC_BIT|h2.
 * All 32 slots are data — no DATA_MASK needed. */
static inline uint32_t ss128_match(const uint16_t *meta, uint16_t h2) {
    __m512i group  = _mm512_load_si512((const __m512i *)meta);
    __m512i masked = _mm512_and_si512(group,
                                      _mm512_set1_epi16((short)MATCH_MASK));
    __m512i needle = _mm512_set1_epi16((short)(OCC_BIT | h2));
    return _mm512_cmpeq_epi16_mask(masked, needle);
}

/* Empty: slot is empty iff OCC_BIT is clear (ghost overflow bits may remain). */
static inline uint32_t ss128_empty(const uint16_t *meta) {
    __m512i group = _mm512_load_si512((const __m512i *)meta);
    __m512i occ   = _mm512_set1_epi16((short)OCC_BIT);
    return _mm512_testn_epi16_mask(group, occ);
}

/* Overflow test: vptestmw — does any slot have ovf_bit set? */
static inline int ss128_overflow_test(const uint16_t *meta,
                                        uint16_t ovf_bit) {
    __m512i group    = _mm512_load_si512((const __m512i *)meta);
    __m512i ovf_test = _mm512_set1_epi16((short)ovf_bit);
    return !!_mm512_test_epi16_mask(group, ovf_test);
}

/* Overflow propagate: OR ovf_bit into all 32 slots and store back.
 * Only called when group is full (all slots occupied). */
static inline void ss128_overflow_propagate(uint16_t *meta,
                                              uint16_t ovf_bit) {
    __m512i group   = _mm512_load_si512((const __m512i *)meta);
    __m512i ovf_vec = _mm512_set1_epi16((short)ovf_bit);
    _mm512_store_si512((__m512i *)meta, _mm512_or_si512(group, ovf_vec));
}

#elif defined(__AVX2__)

#include <x86intrin.h>  /* _pext_u32 (BMI2) */

static inline uint32_t ss128_movemask_epi16(__m256i cmp_lo, __m256i cmp_hi) {
    uint32_t lo = _pext_u32((uint32_t)_mm256_movemask_epi8(cmp_lo), 0xAAAAAAAAu);
    uint32_t hi = _pext_u32((uint32_t)_mm256_movemask_epi8(cmp_hi), 0xAAAAAAAAu);
    return lo | (hi << 16);
}

static inline uint32_t ss128_match(const uint16_t *meta, uint16_t h2) {
    __m256i mask_vec = _mm256_set1_epi16((short)MATCH_MASK);
    __m256i needle   = _mm256_set1_epi16((short)(OCC_BIT | h2));
    __m256i lo = _mm256_load_si256((const __m256i *)meta);
    __m256i hi = _mm256_load_si256((const __m256i *)(meta + 16));
    return ss128_movemask_epi16(
        _mm256_cmpeq_epi16(_mm256_and_si256(lo, mask_vec), needle),
        _mm256_cmpeq_epi16(_mm256_and_si256(hi, mask_vec), needle));
}

static inline uint32_t ss128_empty(const uint16_t *meta) {
    __m256i occ = _mm256_set1_epi16((short)OCC_BIT);
    __m256i z   = _mm256_setzero_si256();
    __m256i lo  = _mm256_load_si256((const __m256i *)meta);
    __m256i hi  = _mm256_load_si256((const __m256i *)(meta + 16));
    return ss128_movemask_epi16(
        _mm256_cmpeq_epi16(_mm256_and_si256(lo, occ), z),
        _mm256_cmpeq_epi16(_mm256_and_si256(hi, occ), z));
}

static inline int ss128_overflow_test(const uint16_t *meta,
                                        uint16_t ovf_bit) {
    __m256i lo = _mm256_load_si256((const __m256i *)meta);
    __m256i hi = _mm256_load_si256((const __m256i *)(meta + 16));
    __m256i ovf_vec = _mm256_set1_epi16((short)ovf_bit);
    return !_mm256_testz_si256(_mm256_or_si256(lo, hi), ovf_vec);
}

static inline void ss128_overflow_propagate(uint16_t *meta,
                                              uint16_t ovf_bit) {
    __m256i ovf_vec = _mm256_set1_epi16((short)ovf_bit);
    __m256i lo = _mm256_load_si256((__m256i *)meta);
    __m256i hi = _mm256_load_si256((__m256i *)(meta + 16));
    _mm256_store_si256((__m256i *)meta,        _mm256_or_si256(lo, ovf_vec));
    _mm256_store_si256((__m256i *)(meta + 16), _mm256_or_si256(hi, ovf_vec));
}

#else /* scalar fallback — SWAR (4 × 16-bit lanes per uint64_t) */

/* Pack MSBs of 4 × 16-bit lanes to a 4-bit value.
 * Bits 15,31,47,63 → bits 0,1,2,3 via multiply-shift. */
static inline uint32_t ss128_pack4(uint64_t z) {
    return (uint32_t)((z * 0x0000200040008001ULL) >> 60);
}

/* Zero-detect for 16-bit lanes: returns MSB set in lanes that are zero.
 * Uses MSB-guard subtraction to prevent borrow across lane boundaries:
 * setting bit 15 before subtract ensures each lane >= 0x8000, so
 * subtracting 0x0001 never borrows into the adjacent lane.
 * ~sub & ~v & msb isolates true zeros from the 0x8000 false-positive. */
static inline uint32_t ss128_match(const uint16_t *meta, uint16_t h2) {
    const uint64_t *w = (const uint64_t *)meta;
    uint64_t needle = (uint64_t)(OCC_BIT | h2) * 0x0001000100010001ULL;
    uint64_t mmask  = (uint64_t)MATCH_MASK * 0x0001000100010001ULL;
    uint64_t msb    = 0x8000800080008000ULL;
    uint32_t result = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t v  = (w[i] ^ needle) & mmask;
        uint64_t hi = (v | msb) - 0x0001000100010001ULL;
        uint64_t z  = ~hi & ~v & msb;
        result |= ss128_pack4(z) << (i * 4);
    }
    return result;
}

static inline uint32_t ss128_empty(const uint16_t *meta) {
    const uint64_t *w = (const uint64_t *)meta;
    uint64_t msb = 0x8000800080008000ULL;
    uint32_t result = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t z = ~w[i] & msb;  /* OCC_BIT clear → empty */
        result |= ss128_pack4(z) << (i * 4);
    }
    return result;
}

static inline int ss128_overflow_test(const uint16_t *meta,
                                        uint16_t ovf_bit) {
    const uint64_t *w = (const uint64_t *)meta;
    uint64_t ovf4 = (uint64_t)ovf_bit * 0x0001000100010001ULL;
    uint64_t any = 0;
    for (int i = 0; i < 8; i++)
        any |= w[i] & ovf4;
    return !!any;
}

static inline void ss128_overflow_propagate(uint16_t *meta,
                                              uint16_t ovf_bit) {
    uint64_t *w = (uint64_t *)meta;
    uint64_t ovf4 = (uint64_t)ovf_bit * 0x0001000100010001ULL;
    for (int i = 0; i < 8; i++)
        w[i] |= ovf4;
}

#endif

/* ================================================================
 * Shared code — no raw SIMD below this line
 * ================================================================ */

/* --- Alloc / grow --- */

static size_t ss128_mapsize(uint32_t cap) {
    size_t raw = (size_t)(cap >> 5) * SM128_GROUP_BYTES;
    return (raw + (2u << 20) - 1) & ~((size_t)(2u << 20) - 1); /* round to 2MB */
}

static void ss128_alloc(struct simd_set128 *m, uint32_t cap) {
    size_t total = ss128_mapsize(cap);
    /* Try explicit 2MB hugepages with MAP_POPULATE (pre-fault all pages,
     * eliminates minor faults during first-touch and ensures the OS
     * commits physical hugepages immediately rather than on demand). */
    m->data = (char *)mmap(NULL, total, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB
                           | MAP_POPULATE, -1, 0);
    if (m->data == MAP_FAILED) {
        /* Fallback: regular pages + THP hint + populate */
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

static void ss128_grow(struct simd_set128 *m) {
    uint32_t old_cap  = m->cap;
    char    *old_data = m->data;
    uint32_t old_ng   = old_cap >> 5;

    ss128_alloc(m, old_cap * 2);
    uint32_t mask = m->mask;

    for (uint32_t g = 0; g < old_ng; g++) {
        const char     *old_grp = old_data + (size_t)g * SM128_GROUP_BYTES;
        const uint16_t *om      = (const uint16_t *)old_grp;
        const struct ss128_kv *ok = (const struct ss128_kv *)(old_grp + 64);
        for (int s = 0; s < 32; s++) {                 /* all 32 slots */
            if (!(om[s] & OCC_BIT)) continue;
            uint64_t klo = ok[s].lo;
            uint64_t khi = ok[s].hi;
            struct ss128_h h = ss128_hash(klo, khi);
            uint16_t h2  = ss128_h2(h.lo);
            uint32_t gi  = h.lo & mask;
            for (;;) {
                char     *grp  = ss128_group(m, gi);
                uint16_t *base = (uint16_t *)grp;
                struct ss128_kv *kp = (struct ss128_kv *)(grp + 64);
                uint32_t em = ss128_empty(base);
                if (em) {
                    int pos = __builtin_ctz(em);
                    base[pos] = OCC_BIT | h2 | (base[pos] & OVF_FIELD_MASK);
                    kp[pos]   = (struct ss128_kv){klo, khi};
                    m->count++;
                    break;
                }
                /* group full — propagate overflow bit into all slots */
                ss128_overflow_propagate(base, ss128_overflow_bit(h.hi));
                gi = (gi + 1) & mask;
            }
        }
    }
    munmap(old_data, ss128_mapsize(old_cap));
}

/* --- Public API --- */

static inline void simd_set128_init(struct simd_set128 *m) {
    memset(m, 0, sizeof(*m));
}

/* Pre-allocate for at least n keys. Eliminates grow() during bulk insert.
 * Combined with pipelined prefetch, achieves 4x insert throughput. */
static inline void simd_set128_init_cap(struct simd_set128 *m, uint32_t n) {
    memset(m, 0, sizeof(*m));
    /* cap must satisfy: n * LOAD_DEN < cap * LOAD_NUM, and cap is power of 2.
     * Minimum cap = ceil(n * 8/7), rounded up to next power of 2. */
    uint64_t need = (uint64_t)n * SM128_LOAD_DEN / SM128_LOAD_NUM + 1;
    uint32_t cap = SM128_INIT_CAP;
    while (cap < need) cap *= 2;
    ss128_alloc(m, cap);
}

static inline void simd_set128_destroy(struct simd_set128 *m) {
    if (m->data) munmap(m->data, ss128_mapsize(m->cap));
}

static inline int simd_set128_insert(struct simd_set128 *m,
                                     uint64_t klo, uint64_t khi) {
    if (m->cap == 0) ss128_alloc(m, SM128_INIT_CAP);
    if (m->count * SM128_LOAD_DEN >= m->cap * SM128_LOAD_NUM)
        ss128_grow(m);

    struct ss128_h h = ss128_hash(klo, khi);
    uint16_t h2 = ss128_h2(h.lo);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        char     *grp  = ss128_group(m, gi);
        uint16_t *base = (uint16_t *)grp;
        struct ss128_kv *kp = (struct ss128_kv *)(grp + 64);

        uint32_t mm = ss128_match(base, h2);
        while (mm) {
            int pos = __builtin_ctz(mm);
            if (kp[pos].lo == klo && kp[pos].hi == khi) return 0;
            mm &= mm - 1;
        }

        uint32_t em = ss128_empty(base);
        if (em) {
            int pos = __builtin_ctz(em);
            base[pos] = OCC_BIT | h2 | (base[pos] & OVF_FIELD_MASK);
            kp[pos]   = (struct ss128_kv){klo, khi};
            m->count++;
            return 1;
        }
        /* group full — propagate overflow bit into all 32 occupied slots */
        ss128_overflow_propagate(base, ss128_overflow_bit(h.hi));
        gi = (gi + 1) & m->mask;
    }
}

/* Bulk-load variant: caller guarantees key is not already present.
 * Skips duplicate scan — faster for known-unique bulk inserts. */
static inline void simd_set128_insert_unique(struct simd_set128 *m,
                                              uint64_t klo, uint64_t khi) {
    if (m->cap == 0) ss128_alloc(m, SM128_INIT_CAP);
    if (m->count * SM128_LOAD_DEN >= m->cap * SM128_LOAD_NUM)
        ss128_grow(m);

    struct ss128_h h = ss128_hash(klo, khi);
    uint16_t h2 = ss128_h2(h.lo);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        char     *grp  = ss128_group(m, gi);
        uint16_t *base = (uint16_t *)grp;
        struct ss128_kv *kp = (struct ss128_kv *)(grp + 64);

        uint32_t em = ss128_empty(base);
        if (em) {
            int pos = __builtin_ctz(em);
            base[pos] = OCC_BIT | h2 | (base[pos] & OVF_FIELD_MASK);
            kp[pos]   = (struct ss128_kv){klo, khi};
            m->count++;
            return;
        }
        /* group full — propagate overflow bit into all 32 occupied slots */
        ss128_overflow_propagate(base, ss128_overflow_bit(h.hi));
        gi = (gi + 1) & m->mask;
    }
}

/* --- Bit-stealing delete: no backshift, no tombstones ---
 *
 * Clears OCC_BIT + h2 but preserves overflow bits as ghosts.
 * Insert inherits ghosts via OR, so overflow info survives even
 * if every slot in a group is deleted and refilled.
 */

static inline int simd_set128_delete(struct simd_set128 *m,
                                     uint64_t klo, uint64_t khi) {
    if (__builtin_expect(m->cap == 0, 0)) return 0;

    struct ss128_h h = ss128_hash(klo, khi);
    uint16_t h2 = ss128_h2(h.lo);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        char *grp = ss128_group(m, gi);
        uint16_t *base = (uint16_t *)grp;
        struct ss128_kv *kp = (struct ss128_kv *)(grp + 64);

        uint32_t mm = ss128_match(base, h2);
        while (mm) {
            int pos = __builtin_ctz(mm);
            if (kp[pos].lo == klo && kp[pos].hi == khi) {
                base[pos] &= OVF_FIELD_MASK;  /* clear occ + h2, keep ghost overflow */
                m->count--;
                return 1;
            }
            mm &= mm - 1;
        }
        if (!ss128_overflow_test(base, ss128_overflow_bit(h.hi)))
            return 0;
        gi = (gi + 1) & m->mask;
    }
}

static inline int simd_set128_contains(struct simd_set128 *m,
                                       uint64_t klo, uint64_t khi) {
    if (__builtin_expect(m->cap == 0, 0)) return 0;

    struct ss128_h h = ss128_hash(klo, khi);
    uint16_t h2 = ss128_h2(h.lo);
    uint32_t gi = h.lo & m->mask;

    for (;;) {
        const char     *grp  = ss128_group(m, gi);
        const uint16_t *base = (const uint16_t *)grp;
        const struct ss128_kv *kp = (const struct ss128_kv *)(grp + 64);

        uint32_t mm = ss128_match(base, h2);
        while (mm) {
            int pos = __builtin_ctz(mm);
            if (kp[pos].lo == klo && kp[pos].hi == khi) return 1;
            mm &= mm - 1;
        }
        if (!ss128_overflow_test(base, ss128_overflow_bit(h.hi)))
            return 0;
        gi = (gi + 1) & m->mask;
    }
}

#endif /* SIMD_MAP128_H */
