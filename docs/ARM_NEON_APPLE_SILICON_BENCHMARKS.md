# ARM NEON Benchmark Results — Apple Silicon (M-series)

**CPU**: Apple Silicon (ARM, M-series)
**ISA**: ARMv8.2+ with NEON (128-bit SIMD), CRC32
**Cache lines**: 128 bytes (vs 64B on x86) — 64B groups fit in half a cache line
**Hugepages**: macOS does not support explicit hugepages; uses superpage promotion
**Date**: 2026-03-17

---

## 1. NEON Backend Implementation

Added `#elif defined(__ARM_NEON)` backend to all 11 headers, between the AVX2
and scalar fallback paths. The NEON backend uses a movemask emulation pattern:

```c
// vshrq extracts MSB per lane, vmulq weights by position, vaddvq sums
static inline uint32_t neon_movemask_u16(uint16x8_t v) {
    static const uint16_t weights[] = {1, 2, 4, 8, 16, 32, 64, 128};
    return vaddvq_u16(vmulq_u16(vshrq_n_u16(v, 15), vld1q_u16(weights)));
}
```

`vaddvq` (horizontal add across vector) is available on all Apple Silicon
(ARMv8.2+). Three variants: `neon_movemask_u16` (8-bit mask from 8x u16),
`neon_movemask_u32` (4-bit mask from 4x u32), `neon_movemask_u64` (2-bit
mask from 2x u64 via lane extract).

The packed 48-bit layout (`simd_map48_packed.h`) uses `vld3q_u16` to natively
deinterleave stride-3 data into 3 separate registers — one per key word.
This avoids the broadcast+coalesce+PEXT pattern required on x86.

---

## 2. NEON vs Scalar — Same Machine Comparison

### simd_set64 (bench_map64_backends)

1M keys, 10M ops, Zipf s=1.0. NEON vs scalar on the same Apple Silicon.

| Operation | NEON (Mops/s) | Scalar (Mops/s) | Speedup |
|-----------|--------------|-----------------|---------|
| insert | 255.0 | 235.8 | 1.08x |
| lookup (hit) | 523.4 | 382.5 | **1.37x** |
| lookup (+/-) | 382.3 | 311.9 | **1.23x** |
| delete | 305.3 | 227.0 | **1.34x** |

**Mixed workloads:**

| Workload | NEON (Mops/s) | Scalar (Mops/s) | Speedup |
|----------|--------------|-----------------|---------|
| read-heavy 90/5/5 | 285.5 | 240.3 | 1.19x |
| balanced 50/25/25 | 178.5 | 156.0 | 1.14x |
| churn 33/33/34 | 164.3 | 147.6 | 1.11x |
| write-heavy 10/50/40 | 108.9 | 104.7 | 1.04x |

**Post-churn lookups:**

| Workload | NEON hit/miss | Scalar hit/miss | Speedup (hit) |
|----------|--------------|-----------------|---------------|
| read-heavy | 726 / 333 | 538 / 283 | **1.35x** |
| balanced | 744 / 327 | 527 / 280 | **1.41x** |
| churn | 748 / 409 | 561 / 315 | **1.33x** |

**map64 NEON improvement is modest** (1.1-1.4x) because the scalar loop over
8 u64 slots auto-vectorizes well on Apple Silicon's wide decode. The NEON
backend eliminates the loop entirely (4x `vceqq_u64` + `neon_movemask_u64`),
which helps most on pure lookup paths.

### Map48 architecture variants (bench_map48_arch)

2M keys, PF=24. **This is where NEON shines.**

| Variant | Op | NEON (ns) | Scalar (ns) | Speedup |
|---------|-----|-----------|-------------|---------|
| split | insert | 5.0 | 4.9 | 1.02x |
| split | contains | **2.9** | 11.2 | **3.86x** |
| split | mixed | **4.9** | 11.6 | **2.37x** |
| 3CL/31 | insert | 4.6 | 4.6 | 1.00x |
| 3CL/31 | contains | **4.0** | 21.7 | **5.43x** |
| 3CL/31 | mixed | **6.4** | 18.4 | **2.88x** |
| 2CL/20 | insert | 3.8 | 3.7 | 1.03x |
| 2CL/20 | contains | **4.4** | 15.0 | **3.41x** |
| 2CL/20 | mixed | **6.1** | 13.8 | **2.26x** |
| backshift | insert | 5.5 | 5.3 | 1.04x |
| backshift | contains | **2.7** | 7.9 | **2.93x** |
| backshift | mixed | **5.0** | 6.4 | 1.28x |
| lemire | insert | 4.9 | 4.6 | 1.07x |
| lemire | contains | **4.7** | 11.9 | **2.53x** |
| lemire | mixed | **4.6** | 12.4 | **2.70x** |
| lem+bs | insert | 6.0 | 6.4 | 1.07x |
| lem+bs | contains | **4.9** | 9.8 | **2.00x** |
| lem+bs | mixed | **4.9** | 5.6 | 1.14x |

**NEON gives 2-5.4x speedup on lookups** for 48-bit variants. The scalar
match functions loop over 10-31 slots with conditional branches; NEON
replaces these with 2-8 vector compares + movemask. The 3CL/31 variant
benefits most (5.4x) because it has the most SIMD lanes to vectorize.

Insert performance is ~unchanged because inserts are write-dominated — the
`sm48*_empty()` function (which checks occupancy bits or hi==0) is the fast
path, and the scalar version is already branch-free for that.

### Generic map layout grid (bench_map_layout)

2M keys, KW=2 (128-bit keys), VW=1, 16 layouts × 7 PF values. Best-PF
results for key variants (ns/op):

**Inline sentinel:**

| Operation | NEON (ns) | Scalar (ns) | Speedup |
|-----------|-----------|-------------|---------|
| insert | 15.7 (PF=40) | 16.7 (PF=36) | 1.06x |
| get_hit | **8.2** (PF=20) | 9.7 (PF=20) | **1.18x** |
| get_miss | **7.7** (PF=20) | 8.8 (PF=16) | **1.14x** |
| delete | **7.9** (PF=20) | 9.8 (PF=20) | **1.24x** |
| churn_del | **9.3** | 10.1 | 1.09x |
| churn_ins | **14.8** | 15.4 | 1.04x |
| post_churn_hit | **9.4** | 10.9 | **1.16x** |

**Inline bitstealing:**

| Operation | NEON (ns) | Scalar (ns) | Speedup |
|-----------|-----------|-------------|---------|
| insert | 16.0 (PF=32) | 16.8 (PF=36) | 1.05x |
| get_hit | **8.2** (PF=32) | 9.9 (PF=28) | **1.21x** |
| get_miss | **7.9** (PF=20) | 9.0 (PF=20) | **1.14x** |
| delete | **8.2** (PF=40) | 10.0 (PF=16) | **1.22x** |
| churn_del | **8.6** | 10.5 | **1.22x** |
| churn_ins | **12.8** | 14.1 | 1.10x |
| post_churn_hit | **8.9** | 11.6 | **1.30x** |

**Hybrid bitstealing N=1 (winner on EPYC):**

| Operation | NEON (ns) | Scalar (ns) | Speedup |
|-----------|-----------|-------------|---------|
| insert | 15.6 (PF=28) | 16.8 (PF=36) | 1.08x |
| get_hit | **8.1** (PF=28) | 10.0 (PF=28) | **1.23x** |
| get_miss | **7.8** (PF=28) | 9.0 (PF=28) | **1.15x** |
| delete | **7.9** (PF=16) | 9.9 (PF=16) | **1.25x** |
| churn_del | **8.7** | 10.7 | **1.23x** |
| churn_ins | **12.8** | 14.2 | 1.11x |
| post_churn_hit | **9.0** | 11.9 | **1.32x** |

**Generic map NEON improvement is 1.14-1.32x on read/delete paths** — more
modest than map48 (2-5.4x) because the generic map's bottleneck is DRAM
latency across 5-9 cache lines per probe, not metadata-match compute. NEON
saves ~1-2 ns on the 32×u16 metadata match (4× `vceqq_u16` + movemask
replaces 4× SWAR operations), but memory stalls dominate. Insert is nearly
unchanged (~1.05x) because inserts are write-dominated.

Sentinel and bitstealing are at performance parity on NEON, matching the AVX2
finding. Both achieve ~8.1-8.2 ns get_hit at best PF. Bitstealing's extra
`vandq_u16` to mask overflow bits before compare hides in OoO bubbles.

**Layout ranking is unchanged from AVX2**: inline wins insert decisively;
all layouts converge on reads. Hybrid N=1 (degenerate inline) is competitive.
Hybrid N≥2 regresses uniformly. The optimal PF window on Apple Silicon is
**PF=20-28** — tighter than EPYC's PF=36-40, consistent with Apple's
shallower memory hierarchy.

---

## 3. Apple Silicon NEON vs EPYC 9575F AVX-512

Cross-platform comparison. Different CPUs, memory systems, and ISAs.
PF=24 for both (note: EPYC optimal is PF=48-64, so these EPYC numbers
are not its best — Apple Silicon is closer to optimal at PF=24).

### simd_set64 (1M keys, Zipf s=1.0)

| Operation | Apple NEON | EPYC AVX-512 | EPYC advantage |
|-----------|-----------|--------------|----------------|
| insert | 255 Mops/s (3.9 ns) | 378 Mops/s (2.6 ns) | 1.48x |
| lookup+ | 523 Mops/s (1.9 ns) | 851 Mops/s (1.2 ns) | 1.63x |
| lookup+/- | 382 Mops/s (2.6 ns) | 579 Mops/s (1.7 ns) | 1.52x |
| delete | 305 Mops/s (3.3 ns) | 341 Mops/s (2.9 ns) | 1.12x |
| read-heavy 90/5/5 | 286 Mops/s | 374 Mops/s | 1.31x |
| balanced 50/25/25 | 179 Mops/s | 168 Mops/s | **Apple wins 1.07x** |
| post-churn hit | 726 Mops/s | 1073 Mops/s | 1.48x |

EPYC wins pure throughput by 1.1-1.6x — AVX-512's single-instruction
8-wide u64 compare (`vpcmpeqq zmm`) plus higher memory bandwidth. But
**Apple wins balanced 50/25/25** — likely because PF=24 is near-optimal
for Apple's shallow unified memory, while EPYC needs PF=48+ to reach
its best mixed numbers.

### Map48 architecture (2M keys, PF=24)

| Variant | Op | Apple NEON | EPYC AVX-512 | Winner |
|---------|-----|-----------|--------------|--------|
| split | insert | 5.0 ns | 3.2 ns | EPYC 1.56x |
| split | contains | **2.9 ns** | 3.5 ns | **Apple 1.21x** |
| split | mixed | **4.9 ns** | 6.5 ns | **Apple 1.33x** |
| 2CL/20 | insert | 3.8 ns | 2.3 ns | EPYC 1.65x |
| 2CL/20 | contains | 4.4 ns | 3.7 ns | EPYC 1.19x |
| 2CL/20 | mixed | **6.1 ns** | 8.2 ns | **Apple 1.34x** |
| 3CL/31 | insert | 4.6 ns | 3.1 ns | EPYC 1.48x |
| 3CL/31 | contains | 4.0 ns | 3.2 ns | EPYC 1.25x |
| 3CL/31 | mixed | **6.4 ns** | 10.1 ns | **Apple 1.58x** |
| backshift | insert | 5.5 ns | 4.5 ns | EPYC 1.22x |
| backshift | contains | **2.7 ns** | 4.3 ns | **Apple 1.59x** |
| backshift | mixed | **5.0 ns** | 7.0 ns | **Apple 1.40x** |
| lemire | insert | 4.9 ns | 3.9 ns | EPYC 1.22x |
| lemire | contains | 4.7 ns | 4.7 ns | Tie |
| lemire | mixed | **4.6 ns** | 7.0 ns | **Apple 1.52x** |

### Map48 direct-compare (2M keys, PF=24)

| Variant | Op | Apple NEON | EPYC AVX-512 | Winner |
|---------|-----|-----------|--------------|--------|
| packed | insert | 6.0 ns | 3.8 ns | EPYC 1.58x |
| packed | contains | **3.3 ns** | 3.9 ns | **Apple 1.18x** |
| packed | mixed | **4.7 ns** | 6.3 ns | **Apple 1.34x** |
| split | insert | 5.3 ns | 2.3 ns | EPYC 2.30x |
| split | contains | 4.1 ns | 3.3 ns | EPYC 1.24x |
| split | mixed | **4.5 ns** | 5.9 ns | **Apple 1.31x** |
| map64 | insert | 5.2 ns | 3.7 ns | EPYC 1.41x |
| map64 | contains | 3.0 ns | 2.8 ns | EPYC 1.07x |
| map64 | mixed | **5.0 ns** | 5.2 ns | **Apple 1.04x** |

### Map48 / sentinel / map64 comparison (2M keys, PF=24)

| Variant | Op | Apple NEON | EPYC AVX-512 | Winner |
|---------|-----|-----------|--------------|--------|
| map48 h2 | insert | 5.7 ns | 5.1 ns | EPYC 1.12x |
| map48 h2 | contains | 5.0 ns | 3.8 ns | EPYC 1.32x |
| map48 h2 | mixed | 8.5 ns | 6.6 ns | EPYC 1.29x |
| sentinel | insert | 4.8 ns | 4.0 ns | EPYC 1.20x |
| sentinel | contains | 5.7 ns | 5.0 ns | EPYC 1.14x |
| sentinel | mixed | 9.1 ns | 6.8 ns | EPYC 1.34x |
| map64 | insert | 4.7 ns | 3.7 ns | EPYC 1.27x |
| map64 | contains | 2.6 ns | 2.7 ns | **Apple 1.04x** |
| map64 | mixed | 5.0 ns | 5.2 ns | **Apple 1.04x** |

### TCP connection state (2M keys, PF=24)

| Variant | Workload | Apple NEON | EPYC AVX-512 | Winner |
|---------|----------|-----------|--------------|--------|
| map64 VW=1 | 90/5/5 | 5.2 ns | 5.4 ns | **Apple 1.04x** |
| map64 VW=2 | 90/5/5 | 7.8 ns | — | — |
| map64 VW=4 | 90/5/5 | 13.4 ns | — | — |
| split VW=1 | 90/5/5 | 5.1 ns | — | — |
| split VW=4 | 90/5/5 | 4.7 ns | — | — |
| lembs VW=1 | 90/5/5 | 5.6 ns | — | — |
| lembs VW=4 | 90/5/5 | 4.9 ns | — | — |
| map64 VW=1 | 50/25/25 | 7.6 ns | 7.6 ns | Tie |
| split VW=1 | 50/25/25 | 7.5 ns | — | — |
| lembs VW=1 | 50/25/25 | 10.1 ns | — | — |

*(EPYC TCP benchmark was bench_tcp_pareto with different variant set;
direct comparison only available for map64.)*

### Generic map layout (2M keys, KW=2, VW=1)

Best configs at optimal PF per platform. Apple uses PF=20-28; EPYC uses
PF=36-40. i9-12900HK (AVX2) uses PF=24.

| Operation | Apple NEON | i9-12900HK AVX2 | EPYC AVX-512 |
|-----------|-----------|-----------------|--------------|
| | inline-sent PF=20 | inline-sent PF=24 | hybrid-bs N=1 PF=40 |
| insert | 15.7 ns | 24.9 ns | **8.5 ns** |
| get_hit | **8.2 ns** | 16.8 ns | 6.1 ns |
| get_miss | **7.7 ns** | 16.4 ns | 3.8 ns |
| delete | **7.9 ns** | 17.1 ns | 6.1 ns |
| churn_del | 9.3 ns | 19.2 ns | 6.4 ns |
| churn_ins | 14.8 ns | 28.0 ns | **12.4 ns** |
| post_churn_hit | **9.4 ns** | 17.4 ns | 10.9 ns |

**Apple Silicon NEON is 2x faster than i9-12900HK AVX2** on all read/delete
operations for the generic map. 8.2 ns get_hit vs 16.8 ns — this is a
*memory system* advantage (unified low-latency memory + 128B cache lines),
not a SIMD advantage. The i9's 576B group (9 cache lines) is much more
expensive to fetch from DRAM than on Apple's wider cache lines.

**EPYC AVX-512 wins insert** (8.5 vs 15.7 ns, 1.85x) due to higher memory
bandwidth. But Apple **wins post-churn hit** (9.4 vs 10.9 ns, 1.16x) —
Apple's low memory latency benefits the probe-repaired chains after churn.
The read gap is moderate (8.2 vs 6.1 ns, EPYC 1.34x).

---

## 4. Key Findings

### EPYC wins insert throughput by 1.2-2.3x

Insert is write-dominated. The EPYC's higher memory bandwidth and AVX-512's
wider vectors allow faster group scanning for empty slots. The split insert
gap (2.3x) is the largest — EPYC's `vpcmpeqd zmm` covers all 10 hi-words
in 1 instruction vs NEON's 2x `vceqq_u32` + 2 scalar compares.

### Apple wins most mixed workloads by 1.0-1.6x

Mixed workloads (50/25/25 insert/contains/delete) favor Apple Silicon because:

1. **PF=24 is near-optimal for Apple**, but suboptimal for EPYC (which needs
   PF=48-64). The EPYC numbers at PF=24 leave significant performance on
   the table.
2. **128B cache lines**: Apple's 128B lines mean 64B groups always share a
   line with the neighbor — effectively free adjacent-group prefetch for
   linear probing.
3. **Low unified memory latency**: Apple's unified memory has lower base
   latency than EPYC's DDR5 over the memory controller, reducing the
   penalty for cache misses that prefetch doesn't cover.

### 3CL/31 contains: 5.4x NEON vs scalar

The largest NEON-vs-scalar improvement. The scalar 3CL match loops over 31
slots with per-slot branches; NEON replaces this with 8x `vceqq_u32` +
4x `vceqq_u16` + mask merge. The instruction count drops from ~100+ to ~30.

### map64 is architecture-neutral

map64 shows the smallest NEON-vs-scalar improvement (1.08-1.37x) because
the compiler auto-vectorizes the 8-element scalar loop effectively on Apple
Silicon. It also shows the smallest Apple-vs-EPYC gap on mixed workloads
(Apple wins by 1.04x). This confirms map64's design: the bottleneck is
memory latency, not compute — both NEON and AVX-512 hide in stalls.

### Generic map (sentinel/bitstealing) benefits modestly from NEON

The 32×u16 metadata match in sentinel/bitstealing shows 1.14-1.32x NEON
speedup — between map64 (1.1-1.4x) and map48 (2-5.4x). The SWAR scalar
pattern is already efficient on Apple Silicon's wide decode, and the 5-9
cache line probe footprint makes DRAM latency the dominant cost. Cross-
platform, Apple NEON is 2x faster than i9 AVX2 (memory system advantage)
and within 1.3x of EPYC AVX-512 on reads.

### packed 48-bit benefits from vld3q

The packed layout's `vld3q_u16` deinterleave on NEON natively separates
stride-3 key words — cleaner than x86's broadcast+coalesce+PEXT. Packed
contains at 3.3 ns on NEON vs 3.9 ns on EPYC AVX-512 demonstrates this.

---

## 5. Platform Comparison Summary

| Metric | Apple Silicon NEON | EPYC 9575F AVX-512 |
|--------|-------------------|---------------------|
| Best map64 lookup | 1.9 ns (PF=24) | 1.2 ns (PF=64) |
| Best map48 lookup | 2.7 ns (backshift) | 3.2 ns (3CL/31) |
| Best mixed (48-bit) | 4.5 ns (split direct) | 5.9 ns (split direct) |
| Best generic get_hit | 8.1 ns (hybrid-bs N=1) | 6.1 ns (hybrid-bs N=1) |
| Best generic post-churn | **9.0 ns** | 10.9 ns |
| SIMD width | 128-bit (NEON) | 512-bit (AVX-512) |
| Cache line | 128B | 64B |
| Optimal PF (map64) | 24 | 48-64 |
| Optimal PF (generic) | 20-28 | 36-40 |
| Memory model | Unified (low latency) | DDR5 (high bandwidth) |
| NEON vs scalar gain | 2-5.4x (map48), 1.1-1.3x (generic/map64) | N/A (no scalar baseline) |

Apple Silicon with NEON is a first-class platform for this library. The
combination of low memory latency, 128B cache lines, and efficient NEON
vector ops makes it competitive with — and sometimes faster than — a
64-core EPYC server CPU with AVX-512, particularly on mixed workloads
at PF=24.

---

## 6. Files Modified

| File | Change |
|------|--------|
| `include/simd_compat.h` | Added `arm_neon.h`, 3 movemask helpers (u16, u32, u64) |
| `include/simd_map64.h` | NEON match/empty: 4x `vceqq_u64` |
| `include/simd_sentinel.h` | NEON match/empty: 4x `vceqq_u16` → 32-bit mask |
| `include/simd_bitstealing.h` | NEON match/empty + overflow_test/propagate |
| `include/simd_map48_split.h` | NEON: 2x `vceqq_u32` (hi) + `vceqq_u16` (lo) + scalar tail |
| `include/simd_map48_packed.h` | NEON: `vld3q_u16` deinterleave + 3-way compare |
| `include/simd_map48_lembs.h` | NEON: same as split pattern |
| `include/simd_map48_bs.h` | NEON: same as split pattern |
| `include/simd_map48_lemire.h` | NEON: same as split pattern (standalone path) |
| `include/simd_map48_3cl.h` | NEON: 8x `vceqq_u32` + 4x `vceqq_u16` |
| `include/simd_map48_2cl.h` | NEON: 4x `vceqq_u32` + 2x `vceqq_u16` + scalar tail |
| `bench/bench_map64_backends.c` | Added NEON backend label |

All 7 correctness tests pass (39 sub-tests), including generic set (128-bit
+ 256-bit × sentinel + bitstealing) and generic map (3 layouts × 2 overflow).
