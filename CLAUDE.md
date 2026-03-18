# hmap/ — SIMD hash map library

Header-only C hash sets and maps with AVX-512, AVX2, ARM NEON, and scalar backends, designed for
memory-latency-bound workloads at 100K+ keys. All containers use open
addressing with group probing, hugepage allocation (2MB), and prefetch
pipelining.

## Hash containers

### `simd_map64.h` — unified uint64_t set/map, superblock layout

Unified set/map header for uint64_t keys. X-include pattern. Set mode when
`VAL_WORDS` is omitted or 0, map mode when `VAL_WORDS >= 1`:

```c
// Set mode (simd_set64.h is a thin wrapper for this):
#define SIMD_MAP_NAME  simd_set64
#include "simd_map64.h"

// Map mode:
#define SIMD_MAP_NAME           my_map64
#define SIMD_MAP64_VAL_WORDS    1
#define SIMD_MAP64_BLOCK_STRIDE 1   // power of 2, default 1
#include "simd_map64.h"
```

Direct-key comparison: keys stored in 8-wide groups (one cache line), SIMD
compares all 8 at once. Zero false positives, no metadata, no scalar verify.
Key=0 is reserved (empty sentinel). Delete uses backshift to repair probe
chains (moves values alongside keys in map mode).

Superblock layout (map mode): groups N key cache lines before N value cache
lines. N=1 degenerates to inline (key + value adjacent, 128B per group for
VW=1).

- Key: `uint64_t` (0 reserved)
- Value: `uint64_t[VAL_WORDS]` (map mode only)
- Group: 8 key slots (64B) + 8 value slots (8×VW×8B, map mode)
- Load factor: 75%
- h2: none (direct compare)
- Backends: AVX-512 (1 instr/group), AVX2 (5 instr), NEON (4 vceqq), scalar (portable)
- Delete: backshift (no tombstones)
- Probe termination: empty slot
- Prefetch: `_prefetch()` (key + value lines, read paths),
  `_prefetch_insert()` (key line only, write paths)

Set API: `_insert(m, key)`, `_contains(m, key)`, `_delete(m, key)`,
`_op(m, key, op)`, `_prefetch2(m, key)`.
Map API: `_insert(m, key, val)`, `_get(m, key)` → `uint64_t *`.

`simd_set64.h` is a thin `#pragma once` wrapper that defines
`SIMD_MAP_NAME simd_set64` and includes `simd_map64.h`.

**Block stride result** (2M keys, AVX2, VW=1): **N=1 wins every mixed
workload** by 6–14% over N=2/4. In true interleaved pipelines, the
alternating 1-line/2-line prefetch dispatch breaks the hardware
prefetcher's stride detection for N≥2. N=2/4 only win pure insert-only
(24% faster). Default: N=1, PF=24. See `docs/KV64_LAYOUT_ANALYSIS.md`.

### `simd_map48_packed.h` — direct-compare 48-bit set, packed 3×u16 (10 keys/CL)

10 keys in a single 64B cache line using packed 3×uint16 interleaved layout.
Direct SIMD comparison via broadcast+coalesce+PEXT — no metadata filtering.

```c
#define SIMD_MAP_NAME my_set48p
#include "simd_map48_packed.h"
```

Group layout (`uint16_t grp[32]`): ctrl (occupancy + reserved) at [0],
overflow partitions at [1], 10 keys × 3 words interleaved at [2..31].
AVX2 match: 3 broadcasts + 6 cmpeq + shift+AND coalesce + PEXT (~31 instr).

- Key: `uint64_t` (lower 48 bits, 0 reserved)
- Group: 10 data slots, 64B (1 cache line)
- Load factor: 7/8 (87.5%)
- h2: none (direct compare, zero false positives)
- Overflow: 16 partitions in `grp[1]`, `hash.hi & 15`
- Backends: AVX2 (broadcast+coalesce), NEON (vld3q deinterleave), scalar (loop)
- Delete: clear occupancy bit, O(1), no tombstones
- Probe termination: overflow bit check
- Prefetch: 1 cache line for both read and write paths

Set API: `_init`, `_init_cap`, `_destroy`, `_insert`, `_insert_unique`,
`_contains`, `_delete`, `_prefetch`, `_prefetch_insert`.

`simd_set48_packed.h` is a thin `#pragma once` wrapper.

### `simd_map48_split.h` — direct-compare 48-bit set, split hi32+lo16 (10 keys/CL)

10 keys in a single 64B cache line using split layout: `uint32_t hi[10]`
(key >> 16) + `uint16_t lo[10]` (key & 0xFFFF). SIMD matches hi and lo
separately then ANDs results.

```c
#define SIMD_MAP_NAME my_set48s
#include "simd_map48_split.h"
```

Group layout (64B): `uint32_t ctrl` at offset 0 (occupancy [9:0] + overflow
[25:10]), `uint32_t hi[10]` at offset 4, `uint16_t lo[10]` at offset 44.
AVX2 match: vpcmpeqd on hi[0..7] + scalar [8..9], vpcmpeqw on lo[0..7] +
scalar [8..9], AND (~14 instr).

- Key: `uint64_t` (lower 48 bits, 0 reserved)
- Group: 10 data slots, 64B (1 cache line)
- Load factor: 7/8 (87.5%)
- h2: none (direct compare, zero false positives)
- Overflow: 16 partitions in ctrl bits [25:10], `hash.hi & 15`
- Backends: AVX2 (vpcmpeqd + vpcmpeqw), NEON (vceqq), scalar (loop)
- Delete: clear occupancy bit, O(1), no tombstones
- Probe termination: overflow bit check
- Prefetch: 1 cache line for both read and write paths

Set API: same as packed. `simd_set48_split.h` is a thin `#pragma once` wrapper.

### `simd_map48_lembs.h` — direct-compare 48-bit set/map, Lemire + backshift + ghost overflow

Combines Lemire non-pow2 addressing, backshift deletion, and ghost overflow
bits. Uses the 4B padding at offset 60 as a 16-partition overflow word.

```c
// Set mode:
#define SIMD_MAP_NAME my_set48lb
#include "simd_map48_lembs.h"

// Map mode:
#define SIMD_MAP_NAME           my_map48lb
#define SIMD_MAP48LB_VAL_WORDS  2
#include "simd_map48_lembs.h"
```

Group layout (64B key block): `uint32_t hi[10]` at offset 0 (stored as
`(key>>16)+1`, 0=empty), `uint16_t lo[10]` at offset 40, `uint32_t ovf` at
offset 60 (ghost overflow, 16 partitions in bits [15:0]). Map mode: values
stored inline after keys (64B keys + 10×VW×8B values).

- Key: `uint64_t` (lower 48 bits, 0 reserved, `(key>>16)==0xFFFFFFFF` excluded)
- Value: `uint64_t[VAL_WORDS]` (map mode only)
- Group: 10 data slots, 64B key block + 10×VW×8B value block (map mode)
- Load factor: 7/8 (87.5%)
- h2: none (direct compare, zero false positives)
- Overflow: 16-partition ghost bits at offset 60, `hash.hi & 15`
- Addressing: Lemire fast range reduction (non-pow2 group count)
- Backends: AVX2 (vpcmpeqd + vpcmpeqw), NEON (vceqq), scalar (loop)
- Delete: backshift (no tombstones), ghost overflow bits never cleared
- Probe termination: overflow bit check (read paths), empty slot (backshift)
- Prefetch: `_prefetch()` (key + value lines, read paths),
  `_prefetch_insert()` (key line only, write paths)

Ghost overflow bits are set when insert overflows past a group, never
cleared on delete. Backshift repairs probe chains; ghost bits are a
conservative acceleration structure for miss-path termination. Grow resets
all ghost bits (fresh table). Same correctness guarantee as bitstealing.

Set API: `_insert(m, key)`, `_contains(m, key)`, `_delete(m, key)`.
Map API: `_insert(m, key, val)`, `_get(m, key)` → `uint64_t *`.

**Set-mode architecture benchmark** (2M keys, AVX2, PF=24, ns/op):

| Variant | Insert | Contains | Mixed 50/25/25 | Alloc |
|-----------|--------|----------|----------------|-------|
| split (baseline) | **3.9** | **3.4** | 7.4 | 16 MiB |
| 3CL/31 | 6.5 | 5.3 | 11.4 | 24 |
| 2CL/20 | 5.9 | 5.1 | 9.0 | 16 |
| backshift | 5.0 | 4.9 | 7.8 | 32 |
| lemire | 5.1 | 5.3 | 7.8 | 14 |
| lem+bs (ghost ovf) | 6.3 | 5.3 | **6.6** | 14 |

Split wins insert/contains. Lem+bs wins mixed at 14 MiB (12.5% less memory
than split, 8-9% better mixed). The ghost overflow bits stabilized mixed
(previous empty-slot termination at 75% LF spiked 7.9-12.4 ns). Cache line
count is the dominant factor — all 1CL designs cluster together, multi-CL
designs are 30-65% slower. See `docs/MAP48_ARCH_ANALYSIS.md`.

**Map-mode TCP state benchmark** (2M connections, AVX2, PF=24, 4M ops):

| Variant | 90/5/5 | 50/25/25 | Alloc | B/conn |
|-----------|--------|----------|-------|--------|
| map64 VW=1 | 9.5 | 10.7 | 128 MiB | 67 |
| map64 VW=2 | 13.9 | 15.4 | 192 MiB | 101 |
| map64 VW=4 | 20.8 | 22.7 | 320 MiB | 168 |
| split VW=1 | 9.2 | 12.1 | 72 MiB | 38 |
| split VW=2 | 9.6 | 12.6 | 112 MiB | 59 |
| split VW=4 | 9.1 | 13.0 | 192 MiB | 101 |
| **lembs VW=1** | **8.8** | 12.3 | **64 MiB** | **34** |
| **lembs VW=2** | **9.3** | 13.0 | **98 MiB** | **51** |
| **lembs VW=4** | **9.2** | 13.7 | **168 MiB** | **88** |

Map64 collapses at VW≥2 (128B→192B entry breaks ACP, 3+ CLs per probe).
Split and lembs are speed-equivalent — same 1CL key block, same value
layout. Lembs uses 11-14% less memory at every VW from non-pow2 groups.
At VW=4: lembs 9.2 ns / 168 MiB vs map64 20.8 ns / 320 MiB (2.3× faster,
47% less memory).

### Architecture variant headers (benchmark-only)

| Header | Design | Slots | LF | Overflow | Delete |
|--------|--------|-------|----|----------|--------|
| `simd_map48_3cl.h` | 3CL/31, fully vectorized | 31 | 87.5% | sentinel | clear bit |
| `simd_map48_2cl.h` | 2CL/20, ACP | 20 | 87.5% | sentinel | clear bit |
| `simd_map48_bs.h` | 1CL/10, backshift | 10 | 75% | none | backshift |
| `simd_map48_lemire.h` | 1CL/10, Lemire non-pow2 | 10 | 87.5% | sentinel | clear bit |

### `simd_sentinel.h` — unified set/map, sentinel overflow (31 data slots)

Unified set/map header. Set mode when `VAL_WORDS` is omitted or 0, map mode
when `VAL_WORDS >= 1`:

```c
// Set mode:
#define SIMD_MAP_NAME       my_set
#define SIMD_MAP_KEY_WORDS  2
#include "simd_sentinel.h"

// Map mode:
#define SIMD_MAP_NAME       my_map
#define SIMD_MAP_KEY_WORDS  2
#define SIMD_MAP_VAL_WORDS  1
#define SIMD_MAP_LAYOUT     1           // 1=inline, 2=separate, 3=hybrid
#include "simd_sentinel.h"
```

Interleaved group layout: 64B metadata + 32×(KW×8)B keys per group. SIMD
operates on 16-bit metadata only; scalar key compare fires only on h2 match.
Slot 31 is a dedicated overflow sentinel with 16 partition bits. Key width is
irrelevant to the SIMD hot path. All key parameters are `const uint64_t *`.

- Key: `uint64_t[KEY_WORDS]` (any width)
- Value: `uint64_t[VAL_WORDS]` (map mode only)
- Group: 31 data slots + 1 sentinel, `64 + 256×KW` bytes (set),
  `+ 256×VW` bytes (map inline)
- Load factor: 7/8 (87.5%)
- h2: 15-bit (1/32768 false positive rate)
- Overflow: 16 partitions in sentinel slot, `hash.hi & 15`
- Hash: chained CRC32 across all words (3 cy), overflow partition deferred
- Backends: AVX-512, AVX2 (movemask+pext, BMI2), NEON (vceqq+movemask), scalar (SWAR)
- Delete: set slot to 0, no backshift, no tombstones
- Probe termination: sentinel overflow bit check (not empty-slot scan)
- Prefetch: `_prefetch()` (5 cache lines, read paths),
  `_prefetch_insert()` (1 cache line, insert paths)

Set API: `_insert(m, key)`, `_contains(m, key)`, `_delete(m, key)`.
Map API adds: `_insert(m, key, val)`, `_get(m, key)` → `uint64_t *`.

Map mode supports three value layout strategies:
- **Strategy 1 (inline)**: values stored after keys in same group.
- **Strategy 2 (separate)**: values in a separate flat mmap.
- **Strategy 3 (hybrid)**: value blocks every N key groups (superblock layout).

### `simd_bitstealing.h` — unified set/map, bit-stealing overflow (32 data slots)

Same unified set/map pattern as sentinel. Overflow info is encoded in the
data slots themselves instead of a dedicated sentinel, reclaiming all 32
slots for data.

- Key: `uint64_t[KEY_WORDS]` (any width)
- Value: `uint64_t[VAL_WORDS]` (map mode only)
- Group: 32 data slots, `64 + 256×KW` bytes (set)
- Load factor: 7/8 (87.5%)
- h2: 11-bit (1/2048 false positive rate)
- Metadata: `[15] OCC [14:11] overflow (4 bits) [10:0] h2`
- Overflow: 4 partitions in bits [14:11], `hash.hi & 3`
- Hash: chained CRC32, same as sentinel
- Backends: AVX-512, AVX2 (movemask+pext, BMI2), NEON (vceqq+movemask), scalar (SWAR)
- Delete: `&= OVF_FIELD_MASK` (preserves ghost overflow bits)
- Insert: inherits ghost overflow bits from reused slots via OR
- Probe termination: SIMD overflow test across all slots (vtestmw / vtestz)
- Prefetch: `_prefetch()` (5 cache lines, read paths),
  `_prefetch_insert()` (1 cache line, insert paths)

**Bitstealing and sentinel are at performance parity** (~17-18 ns/op pipelined
at 2M keys on AVX2, i9-12900HK). Bitstealing executes ~12% more instructions
(extra `vpand` to mask overflow bits before `vpcmpeqw`) but achieves ~8%
higher IPC — the extra ALU ops hide in OoO memory-wait bubbles. Both are
83-84% backend-bound with ~40% memory-bound (topdown). Cache profiles are
identical (L1d, LLC, dTLB misses). Ghost overflow bits cause no measurable
degradation after 10 rounds of 200K-key churn. Bitstealing has 32 vs 31 data
slots per group (~3.2% better capacity) and is correct under all insert/delete
sequences (ghost overflow bits are monotonically preserved).

## Shared design patterns

- **Prefetch pipelining**: `_prefetch()` / `_prefetch_insert()` issued PF
  iterations ahead of the operation. Overlaps DRAM latency with computation.
  PF=20 is optimal for insert-only (1-line prefetch has less fill buffer
  pressure than the old 5-line, so closer PF works); PF=24-28 for mixed
  workloads. **Dual-mode prefetch**: `_prefetch()` issues 5 cache lines
  (metadata + 4 key lines) for read paths; `_prefetch_insert()` issues
  1 cache line (metadata only) for insert/delete paths. Delete uses
  metadata-only despite reading key data — full 5-line for deletes adds
  6-10% regression from fill buffer pressure (benchmarked: meta wins at
  every PF value in 50/25/25 churn). Key/value writes use write-allocate
  through the store buffer.
  The 5-line read prefetch was empirically optimal for lookups — tested
  alternatives that all regressed: two-tier L1/L2 (+60%), full 9-line L1
  (+50%), post-match key prefetch (no effect). Using 5-line prefetch for
  inserts caused 72% of cycles to stall on fill buffer saturation (PF=24 ×
  5 = 120 outstanding vs 12 L1 fill entries); switching to 1-line for
  inserts yielded a 47% speedup (26→14 ns/op).
- **init_cap + insert_unique**: Pre-allocate for known size, skip duplicate
  check. Combined with prefetch, achieves peak insert throughput.
- **Hugepage allocation**: mmap with MAP_HUGETLB first, fallback to
  MAP_POPULATE + MADV_HUGEPAGE. Eliminates TLB thrashing at scale.
- **Chained CRC32 hash**: `crc32(crc32(..., w[0]), w[1])` chains across all
  key words in a single round (3 cy for 128-bit, +3 cy per extra word).
  Round b = `crc32(a, w[last])` gives overflow partition, executes in
  parallel with memory load via OoO — never on critical path. Scalar
  fallback uses boost::hash_combine-style fold + murmur3 finalizer.
- **AVX2 movemask+pext pattern**: `_mm256_movemask_epi8` gives 2 bits per
  epi16 element; `_pext_u32(..., 0xAAAAAAAA)` extracts odd bits for 1-bit-per-
  slot mask. Two halves (lo/hi) merged with shift+OR for 32-bit result.
- **NEON movemask emulation**: `vandq` with positional weights (1,2,4,...),
  `vaddvq` horizontal sum → bitmask. Comparison results are all-ones or
  all-zeros per lane, so AND extracts the bit directly — no shift or
  multiply needed (2 ops vs the old 3: USHR+MUL+ADDV). Benchmarked 10%
  faster than shift+multiply on Apple Silicon. Three variants in
  `simd_compat.h`: u16 (8-bit), u32 (4-bit), u64 (2-bit via lane extract).
  Packed 48-bit uses `vld3q_u16` native deinterleave.
- **Scalar SWAR pattern**: 4 × 16-bit lanes per `uint64_t`. Zero-detection
  uses MSB-guard subtraction (`(v|0x8000...) - 0x0001...`) to prevent
  cross-lane borrows, then `~sub & ~v & 0x8000...` isolates true zeros.
  Multiply-shift packs 4 MSBs to 4 consecutive bits. Empty detection is
  simpler: `~word & 0x8000...` directly tests OCC_BIT per lane.

## Project layout

```
include/     Library headers (the product). Use -Iinclude when compiling.
vendor/      Third-party headers (benchmark comparison targets).
test/        Correctness tests (must exit 0).
bench/       Performance benchmarks (re-runnable, comparable).
scripts/     Analysis scripts (Python).
docs/        Benchmark analysis write-ups.
archive/     Historical 128-bit prototypes (superseded by generic headers).
```

Source naming: `{purpose}_{type}_{topic}.c` where `test_` = correctness,
`bench_` = benchmark.

### Correctness tests (`test/`)

| File | Type | What it does |
|------|------|-------------|
| `test_set_generic.c` | generic set | Full suite at 128-bit + 256-bit, both sentinel and bitstealing |
| `test_map_generic.c` | generic map | Map correctness: 3 layouts × 2 overflow schemes, insert/get/delete/re-insert with value verification |
| `test_map64.c` | map64 | Map64 correctness: N=1,2,4,8 strides, VW=1 and VW=2, insert/get/delete/re-insert with value verification |
| `test_map48.c` | map48 | Map48 correctness: set + map VW=1,2, insert/dup/contains/delete/re-insert |
| `test_set48_direct.c` | set48 direct | Packed + split direct-compare 48-bit set correctness (2M keys) |
| `test_map48_direct.c` | map48 direct | Split/packed map-mode: VW=1,2 × N=1,2,4, insert/get/delete with value verification |
| `test_set48_arch.c` | set48 arch | 6 architecture variants: split, 3CL, 2CL, backshift, lemire, lem+bs (2M keys) |

### Benchmarks (`bench/`)

| File | Type | What it does |
|------|------|-------------|
| `bench_map64_libs.c` | set64 | simd_set64 vs verstable vs khashl (C side). Linked with C++ driver. |
| `bench_map64_libs_main.cpp` | set64 | C++ driver: adds boost::unordered_flat_set to above comparison. |
| `bench_map64_backends.c` | set64 | AVX-512 vs AVX2 vs scalar backend comparison + post-churn lookup. Build two binaries: one native, one with `-mno-avx512f`. |
| `bench_map_layout.c` | generic map | Grid search: 3 layouts × 2 overflow × PF sweep. 16 instantiations. TSV output. |
| `bench_map_pf_tuning.c` | generic map | PF distance sweep + delete prefetch mode A/B. Sentinel inline (KW=2, VW=1). TSV output. |
| `bench_map64_layout.c` | map64 | 2D grid search: block stride (N=1,2,4) × PF distance (4–64). 7 workloads. TSV output. |
| `bench_map_vs_boost.c` | generic map | Map sentinel/bitstealing vs boost::unordered_flat_map (C side). Linked with C++ driver. |
| `bench_map_vs_boost_main.cpp` | generic map | C++ driver: boost benchmark + orchestration. 10 workloads: insert-only, 5 read/write ratios, 4 churn profiles. TSV output. |
| `bench_map48.c` | map48 | map48 vs sentinel(KW=1) vs map64, insert/contains/mixed. |
| `bench_map48_direct.c` | map48 direct | packed vs split vs map48 vs map64, insert/contains/mixed. |
| `bench_map48_arch.c` | map48 arch | 6 architecture variants: split, 3CL, 2CL, backshift, lemire, lem+bs. Speed + memory. |
| `bench_tcp_pareto.c` | map48 tcp | Map-mode pareto: map64 vs split vs packed × N=1,2 × 90/5/5 + 50/25/25. PF sweep. |
| `bench_tcp_state.c` | map48 tcp | Map-mode state: map64 vs split vs lembs × VW=1,2,4 × 90/5/5 + 50/25/25. |

### Third-party headers (`vendor/`)

| File | What it is |
|------|-----------|
| `verstable.h` | Verstable v2.2.1 — C99 open-addressing hash table (benchmark comparison target) |

### Documentation (`docs/`)

| File | What it does |
|------|-------------|
| `WHY_NOT_PHF.md` | Analysis of why perfect hash maps lose to open addressing at scale |
| `KV_LAYOUT_ANALYSIS.md` | Map value layout strategy benchmark: inline vs separate vs hybrid, ANOVA analysis |
| `KV_BOOST_COMPARISON.md` | Map vs boost::unordered_flat_map: 10 workloads, ANOVA analysis, crossover analysis |
| `KV64_LAYOUT_ANALYSIS.md` | Map64 superblock layout benchmark: N=1,2,4 × PF 2D grid, read vs write tradeoff |
| `MAP48_ARCH_ANALYSIS.md` | Map48 architecture comparison: 6 set-mode variants + map-mode TCP benchmark |
| `ARM_NEON_APPLE_SILICON_BENCHMARKS.md` | ARM NEON Apple Silicon benchmarks: NEON vs scalar, cross-platform vs EPYC AVX-512 |

### Analysis scripts (`scripts/`)

| File | What it does |
|------|-------------|
| `analyze_kv_bench.py` | Tabular analysis of map layout benchmark TSV output |
| `analyze_kv_pf_tuning.py` | Tabular analysis of map PF tuning TSV runs: median ns/op vs PF, optimal PF, meta vs full pairwise |
| `analyze_kv_vs_boost.py` | Tabular analysis of map vs boost comparison TSV output |

### Archive (`archive/`)

128-bit prototype headers and their test/bench consumers. Superseded by the
generic set headers at `SIMD_SET_WORDS=2` (identical performance, identical
assembly). Retained for historical reference.

| File | What it does |
|------|-------------|
| `simd_set128_sentinel.h` | 128-bit prototype set, sentinel overflow, `(klo, khi)` API |
| `simd_set128_bitstealing.h` | 128-bit prototype set, bit-stealing overflow, `(klo, khi)` API |
| `test_map128.c` | 11-test correctness suite for prototype |
| `bench_map128_throughput.c` | Mixed workload benchmark, `-DBITSTEALING` for variant |
| `bench_map128_delete.c` | A/B: rehash vs backshift delete |
| `bench_map128_delete_pf.c` | Raw vs prefetch-pipelined delete throughput |

## Performance characteristics (2M keys, AVX2, i9-12900HK)

Both 128-bit set variants achieve ~17-18 ns/op pipelined (PF=24) across all
operations (insert, contains-hit, contains-miss, delete, churn). The
bottleneck is memory latency, not compute:

- **Topdown**: 83-84% backend-bound, 40% memory-bound, 14% retiring, 1.5% frontend
- **L1d miss rate**: ~15% (6M/40M loads) — working set ~40MB exceeds L2 (1.25MB)
- **LLC miss rate**: ~20% of L1 misses go to DRAM (500K/2.5M LLC loads)
- **dTLB misses**: ~3.8M per 2M-key lookup (hugepages mitigate, ~2 misses/op)
- **Branch misses**: negligible (<0.1% miss rate for both variants)
- **IPC**: 0.74-0.89 (low due to memory stalls, not instruction overhead)

The 576B group (9 cache lines) is the fundamental cost unit. Each lookup
touches 1 cache line (metadata) for miss, 2 cache lines (metadata + key) for
hit. Prefetch hides this latency PF=24 iterations ahead.

## Build

Requires `-march=native` (AVX2/AVX-512/NEON backend selection) and BMI2 (`pext`,
x86 only). On ARM, CRC32 hash and NEON SIMD are auto-detected via
`__ARM_FEATURE_CRC32` and `__ARM_NEON`.

```sh
make            # build all tests (native + scalar)
make test       # build + run all 13 tests
make bench      # build all benchmarks
make clean      # rm -rf build/
make test_map64 # build individual target
make -j$(nproc) # parallel build
make archive    # build 5 archive targets (opt-in)
```

Binaries output to `build/` (gitignored). `-Iinclude` resolves library
headers. Scalar variants add `-mno-avx2 -mno-avx512f` (keeps SSE4.2 for
CRC32 hash + `_mm_prefetch`).

<details>
<summary>Manual build commands (reference)</summary>

```sh
# generic set correctness (128-bit + 256-bit, sentinel + bitstealing)
cc -O3 -march=native -std=gnu11 -Iinclude -o build/test_set_generic test/test_set_generic.c

# generic set scalar backend
cc -O3 -march=native -mno-avx2 -mno-avx512f -std=gnu11 -Iinclude -o build/test_set_generic_scalar test/test_set_generic.c

# set64 backend comparison (AVX-512 vs AVX2)
cc -O3 -march=native -std=gnu11 -Iinclude -o build/bench_512 bench/bench_map64_backends.c -lm
cc -O3 -march=native -mno-avx512f -std=gnu11 -Iinclude -o build/bench_avx2 bench/bench_map64_backends.c -lm

# map64 correctness (N=1,2,4,8, VW=1 and VW=2)
cc -O3 -march=native -std=gnu11 -Iinclude -o build/test_map64 test/test_map64.c

# map64 scalar backend correctness
cc -O3 -march=native -mno-avx2 -mno-avx512f -std=gnu11 -Iinclude -o build/test_map64_scalar test/test_map64.c

# generic map correctness (all 6 variants: 3 layouts × 2 overflow)
cc -O3 -march=native -std=gnu11 -Iinclude -o build/test_map_generic test/test_map_generic.c

# generic map scalar backend correctness
cc -O3 -march=native -mno-avx2 -mno-avx512f -std=gnu11 -Iinclude -o build/test_map_generic_scalar test/test_map_generic.c

# map64 layout benchmark (N=1,2,4 × PF 2D grid)
cc -O3 -march=native -std=gnu11 -Iinclude -o build/bench_map64_layout bench/bench_map64_layout.c

# generic map layout benchmark (16 instantiations, PF sweep, churn)
cc -O3 -march=native -std=gnu11 -Iinclude -o build/bench_map_layout bench/bench_map_layout.c

# generic map PF tuning (PF sweep + delete prefetch A/B)
cc -O3 -march=native -std=gnu11 -Iinclude -o build/bench_map_pf_tuning bench/bench_map_pf_tuning.c

# map48 correctness (sentinel + direct-compare packed/split)
cc -O3 -march=native -std=gnu11 -Iinclude -o build/test_map48 test/test_map48.c
cc -O3 -march=native -std=gnu11 -Iinclude -o build/test_set48_direct test/test_set48_direct.c
cc -O3 -march=native -mno-avx2 -mno-avx512f -std=gnu11 -Iinclude -o build/test_set48_direct_scalar test/test_set48_direct.c

# map48 direct-compare map correctness (split/packed map-mode)
cc -O3 -march=native -std=gnu11 -Iinclude -o build/test_map48_direct test/test_map48_direct.c
cc -O3 -march=native -mno-avx2 -mno-avx512f -std=gnu11 -Iinclude -o build/test_map48_direct_scalar test/test_map48_direct.c

# map48 direct-compare benchmark (packed vs split vs map48 vs map64)
cc -O3 -march=native -std=gnu11 -Iinclude -o build/bench_map48_direct bench/bench_map48_direct.c

# set48 architecture variants (6 variants: split, 3CL, 2CL, backshift, lemire, lem+bs)
cc -O3 -march=native -std=gnu11 -Iinclude -o build/test_set48_arch test/test_set48_arch.c
cc -O3 -march=native -mno-avx2 -mno-avx512f -std=gnu11 -Iinclude -o build/test_set48_arch_scalar test/test_set48_arch.c
cc -O3 -march=native -std=gnu11 -Iinclude -o build/bench_map48_arch bench/bench_map48_arch.c

# map48 TCP state benchmark (map64 vs split vs lembs × VW=1,2,4)
cc -O3 -march=native -std=gnu11 -Iinclude -o build/bench_tcp_state bench/bench_tcp_state.c

# generic map vs boost::unordered_flat_map (C/C++ linkage)
cc -O3 -march=native -std=gnu11 -Iinclude -c -o build/bench_map_vs_boost.o bench/bench_map_vs_boost.c
c++ -O3 -march=native -std=c++17 -c -o build/bench_map_vs_boost_main.o bench/bench_map_vs_boost_main.cpp
c++ -O3 -o build/bench_map_vs_boost build/bench_map_vs_boost.o build/bench_map_vs_boost_main.o
```

</details>
