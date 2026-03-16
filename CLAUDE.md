# hmap/ — SIMD hash map library

Header-only C hash sets and maps with AVX-512 and AVX2 backends, designed for
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
- Backends: AVX-512 (1 instr/group), AVX2 (5 instr), scalar (portable)
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
(24% faster). Default: N=1, PF=24. See `KV64_LAYOUT_ANALYSIS.md`.

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
- Backends: AVX2 (broadcast+coalesce), scalar (loop)
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
- Backends: AVX2 (vpcmpeqd + vpcmpeqw), scalar (loop)
- Delete: clear occupancy bit, O(1), no tombstones
- Probe termination: overflow bit check
- Prefetch: 1 cache line for both read and write paths

Set API: same as packed. `simd_set48_split.h` is a thin `#pragma once` wrapper.

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
- Backends: AVX-512, AVX2 (movemask+pext, BMI2), scalar (SWAR)
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
- Backends: AVX-512, AVX2 (movemask+pext, BMI2), scalar (SWAR)
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
- **Scalar SWAR pattern**: 4 × 16-bit lanes per `uint64_t`. Zero-detection
  uses MSB-guard subtraction (`(v|0x8000...) - 0x0001...`) to prevent
  cross-lane borrows, then `~sub & ~v & 0x8000...` isolates true zeros.
  Multiply-shift packs 4 MSBs to 4 consecutive bits. Empty detection is
  simpler: `~word & 0x8000...` directly tests OCC_BIT per lane.

## File naming convention

`{purpose}_{type}_{topic}.c` where:
- `test_` = correctness tests (must exit 0)
- `bench_` = performance benchmarks (re-runnable, comparable)

### Correctness tests

| File | Type | What it does |
|------|------|-------------|
| `test_map_generic.c` | generic set | Full suite at 128-bit + 256-bit, both sentinel and bitstealing |
| `test_kv_generic.c` | generic map | Map correctness: 3 layouts × 2 overflow schemes, insert/get/delete/re-insert with value verification |
| `test_kv64.c` | map64 | Map64 correctness: N=1,2,4,8 strides, VW=1 and VW=2, insert/get/delete/re-insert with value verification |
| `test_map48.c` | map48 | Map48 correctness: set + map VW=1,2, insert/dup/contains/delete/re-insert |
| `test_map48_direct.c` | map48 direct | Packed + split direct-compare 48-bit set correctness (2M keys) |

### Benchmarks

| File | Type | What it does |
|------|------|-------------|
| `bench_map64_libs.c` | set64 | simd_set64 vs verstable vs khashl (C side). Linked with C++ driver. |
| `bench_map64_libs_main.cpp` | set64 | C++ driver: adds boost::unordered_flat_set to above comparison. |
| `bench_map64_backends.c` | set64 | AVX-512 vs AVX2 vs scalar backend comparison + post-churn lookup. Build two binaries: one native, one with `-mno-avx512f`. |
| `bench_kv_layout.c` | generic map | Grid search: 3 layouts × 2 overflow × PF sweep. 16 instantiations. TSV output. |
| `bench_kv_pf_tuning.c` | generic map | PF distance sweep + delete prefetch mode A/B. Sentinel inline (KW=2, VW=1). TSV output. |
| `bench_kv64_layout.c` | map64 | 2D grid search: block stride (N=1,2,4) × PF distance (4–64). 7 workloads. TSV output. |
| `bench_kv_vs_boost.c` | generic map | Map sentinel/bitstealing vs boost::unordered_flat_map (C side). Linked with C++ driver. |
| `bench_kv_vs_boost_main.cpp` | generic map | C++ driver: boost benchmark + orchestration. 10 workloads: insert-only, 5 read/write ratios, 4 churn profiles. TSV output. |
| `bench_map48.c` | map48 | map48 vs sentinel(KW=1) vs map64, insert/contains/mixed. |
| `bench_map48_direct.c` | map48 direct | packed vs split vs map48 vs map64, insert/contains/mixed. |

### Third-party headers (vendored for benchmarks)

| File | What it is |
|------|-----------|
| `verstable.h` | Verstable v2.2.1 — C99 open-addressing hash table (benchmark comparison target) |

### Documentation

| File | What it does |
|------|-------------|
| `WHY_NOT_PHF.md` | Analysis of why perfect hash maps lose to open addressing at scale |
| `KV_LAYOUT_ANALYSIS.md` | Map value layout strategy benchmark: inline vs separate vs hybrid, ANOVA analysis across sentinel and bitstealing |
| `KV_BOOST_COMPARISON.md` | Map vs boost::unordered_flat_map benchmark: 10 workloads, ANOVA analysis, crossover analysis |
| `KV64_LAYOUT_ANALYSIS.md` | Map64 superblock layout benchmark: N=1,2,4 × PF 2D grid, read vs write tradeoff analysis |
| `analyze_kv_pf_tuning.py` | Tabular analysis of PF tuning TSV runs: median ns/op vs PF, optimal PF, meta vs full pairwise |

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

```sh
# generic set correctness (128-bit + 256-bit, sentinel + bitstealing)
cc -O3 -march=native -std=gnu11 -o test_generic test_map_generic.c

# generic set scalar backend
cc -O3 -march=native -mno-avx2 -mno-avx512f -std=gnu11 -o test_generic_scalar test_map_generic.c

# set64 backend comparison (AVX-512 vs AVX2)
cc -O3 -march=native -std=gnu11 -o bench_512 bench_map64_backends.c
cc -O3 -march=native -mno-avx512f -std=gnu11 -o bench_avx2 bench_map64_backends.c

# map64 correctness (N=1,2,4,8, VW=1 and VW=2)
cc -O3 -march=native -std=gnu11 -o test_kv64 test_kv64.c

# map64 scalar backend correctness
cc -O3 -march=native -mno-avx2 -mno-avx512f -std=gnu11 -o test_kv64_scalar test_kv64.c

# generic map correctness (all 6 variants: 3 layouts × 2 overflow)
cc -O3 -march=native -std=gnu11 -o test_kv test_kv_generic.c

# generic map scalar backend correctness
cc -O3 -march=native -mno-avx2 -mno-avx512f -std=gnu11 -o test_kv_scalar test_kv_generic.c

# map64 layout benchmark (N=1,2,4 × PF 2D grid)
cc -O3 -march=native -std=gnu11 -o bench_kv64 bench_kv64_layout.c

# generic map layout benchmark (16 instantiations, PF sweep, churn)
cc -O3 -march=native -std=gnu11 -o bench_kv bench_kv_layout.c

# generic map PF tuning (PF sweep + delete prefetch A/B)
cc -O3 -march=native -std=gnu11 -o bench_pf bench_kv_pf_tuning.c

# map48 correctness (sentinel + direct-compare packed/split)
cc -O3 -march=native -std=gnu11 -o test_map48 test_map48.c
cc -O3 -march=native -std=gnu11 -o test_map48_direct test_map48_direct.c
cc -O3 -march=native -mno-avx2 -mno-avx512f -std=gnu11 -o test_map48_direct_scalar test_map48_direct.c

# map48 direct-compare benchmark (packed vs split vs map48 vs map64)
cc -O3 -march=native -std=gnu11 -o bench_map48_direct bench_map48_direct.c

# generic map vs boost::unordered_flat_map (C/C++ linkage)
cc -O3 -march=native -std=gnu11 -c bench_kv_vs_boost.c
c++ -O3 -march=native -std=c++17 -c bench_kv_vs_boost_main.cpp
c++ -O3 -o bench_kv_vs_boost bench_kv_vs_boost.o bench_kv_vs_boost_main.o
```

`-march=native` is required (selects AVX2 or AVX-512 backend). BMI2 (`pext`)
is present on all AVX2 CPUs that matter (Haswell+, Zen2+).
`-mno-avx2 -mno-avx512f` forces the scalar (SWAR) backend while keeping
SSE4.2 (CRC32 hash, `_mm_prefetch`).
