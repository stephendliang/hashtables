# hmap/ — SIMD hash map library

Header-only C hash sets with AVX-512 and AVX2 backends, designed for
memory-latency-bound workloads at 100K+ keys. All maps use open addressing
with group probing, hugepage allocation (2MB), and prefetch pipelining.

## Hash maps

### `simd_map64.h` — uint64_t set, zero metadata

Direct-key comparison: keys stored in 8-wide groups (one cache line), SIMD
compares all 8 at once. Zero false positives, no metadata, no scalar verify.
Key=0 is reserved (empty sentinel). Delete uses backshift to repair probe
chains.

- Key: `uint64_t` (0 reserved)
- Group: 8 slots, 64B (1 cache line)
- Load factor: 75%
- h2: none (direct compare)
- Backends: AVX-512 (1 instr/group), AVX2 (5 instr), scalar (portable)
- Delete: backshift (no tombstones)
- Probe termination: empty slot

### `simd_map_sentinel.h` — size-agnostic set, sentinel overflow (31 data slots)

Macro-generated via X-include pattern. Supports arbitrary key widths
(128-bit, 192-bit, 256-bit, etc.) parameterized by word count:

```c
#define SIMD_MAP_NAME  simd_map256
#define SIMD_MAP_WORDS 4
#include "simd_map_sentinel.h"
// generates: struct simd_map256, simd_map256_init(), simd_map256_insert(), ...
```

Interleaved group layout: 64B metadata + 32×(WORDS×8)B keys per group. SIMD
operates on 16-bit metadata only; scalar key compare fires only on h2 match.
Slot 31 is a dedicated overflow sentinel with 16 partition bits. Key width is
irrelevant to the SIMD hot path. All key parameters are `const uint64_t *`.

- Key: `uint64_t[WORDS]` (any width)
- Group: 31 data slots + 1 sentinel, `64 + 256×WORDS` bytes
- Load factor: 7/8 (87.5%)
- h2: 15-bit (1/32768 false positive rate)
- Overflow: 16 partitions in sentinel slot, `hash.hi & 15`
- Hash: chained CRC32 across all words (3 cy), overflow partition deferred
- Backends: AVX-512, AVX2 (movemask+pext, BMI2), scalar (SWAR)
- Delete: set slot to 0, no backshift, no tombstones
- Probe termination: sentinel overflow bit check (not empty-slot scan)
- Prefetch: `_prefetch()` (5 cache lines, read paths),
  `_prefetch_insert()` (1 cache line, insert paths)

### `simd_map_bitstealing.h` — size-agnostic set, bit-stealing overflow (32 data slots)

Same X-include pattern and API as sentinel variant. Overflow info is encoded
in the data slots themselves instead of a dedicated sentinel, reclaiming all
32 slots for data.

- Key: `uint64_t[WORDS]` (any width)
- Group: 32 data slots, `64 + 256×WORDS` bytes
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

### `simd_kv_sentinel.h` / `simd_kv_bitstealing.h` — size-agnostic KV maps

Key-value extensions of the set headers. Same X-include pattern with additional
parameters for value width and layout strategy:

```c
#define SIMD_MAP_NAME       my_kv
#define SIMD_MAP_KEY_WORDS  2           // key width in uint64_t
#define SIMD_MAP_VAL_WORDS  1           // value width in uint64_t
#define SIMD_KV_LAYOUT      1           // 1=inline, 2=separate, 3=hybrid
#define SIMD_KV_BLOCK_STRIDE 4          // strategy 3 only, power of 2
#include "simd_kv_sentinel.h"
```

Three value layout strategies for benchmarking memory access patterns:
- **Strategy 1 (inline)**: values stored after keys in same group.
  Group: `64 + 32*KW*8 + 32*VW*8` bytes.
- **Strategy 2 (separate)**: values in a separate flat mmap. Key groups
  unchanged from set mode. Struct gains `val_data` pointer.
- **Strategy 3 (hybrid)**: value blocks every N key groups (superblock layout).
  N must be power of 2. N=1 degenerates to strategy 1.

API extends set version: `_insert(m, key, val)`, `_insert_unique(m, key, val)`,
`_get(m, key)` → `uint64_t *` (NULL on miss), `_contains`, `_delete` unchanged.
`_prefetch_insert(m, key)` prefetches metadata only (1 cache line) for
insert/delete paths; `_prefetch(m, key)` prefetches metadata + key data
(5 cache lines) for read paths.

### `simd_map128_sentinel.h` / `simd_map128_bitstealing.h` — 128-bit prototypes

Original fixed-size implementations. Hardcoded to 128-bit keys with separate
`klo, khi` arguments. Retained for reference; new code should use the generic
headers above.

**Bitstealing and sentinel are at performance parity** (~17-18 ns/op pipelined
at 2M keys on AVX2, i9-12900HK). Bitstealing executes ~12% more instructions
(extra `vpand` to mask overflow bits before `vpcmpeqw`) but achieves ~8%
higher IPC — the extra ALU ops hide in OoO memory-wait bubbles. Both are
83-84% backend-bound with ~40% memory-bound (topdown). Cache profiles are
identical (L1d, LLC, dTLB misses). Ghost overflow bits cause no measurable
degradation after 10 rounds of 200K-key churn. Bitstealing has 32 vs 31 data
slots per group (~3.2% better capacity) and is correct under all insert/delete
sequences (ghost overflow bits are monotonically preserved).

**Generic vs prototype assembly parity**: the macro-generated version at
WORDS=2 produces identical performance to the hand-written 128-bit prototype
(~17-18 ns/op, within run-to-run noise on all operations). The compiler fully
unrolls `for (i < WORDS)` loops, optimizes compound literal key arguments to
register pairs, and emits structurally identical assembly. The only difference
is the hash chains from `crc32(0, w[0])` instead of `crc32(khi[31:0], klo)`
— one extra instruction that hides behind memory latency.

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

`{purpose}_{map}_{topic}.c` where:
- `test_` = correctness tests (must exit 0)
- `bench_` = performance benchmarks (re-runnable, comparable)

### Correctness tests

| File | Map | What it does |
|------|-----|-------------|
| `test_map128.c` | map128 (proto) | 11-test suite: insert, dup, hit, miss, partial, delete-hit, delete-miss, re-insert, delete-all, init_cap, insert_unique |
| `test_map_generic.c` | generic | Full suite at 128-bit + 256-bit, both sentinel and bitstealing |
| `test_kv_generic.c` | generic KV | KV correctness: 3 layouts × 2 overflow schemes, insert/get/delete/re-insert with value verification |

### Benchmarks

| File | Map | What it does |
|------|-----|-------------|
| `bench_map128_throughput.c` | map128 | Mixed workload: bulk insert, hit/miss, churn rounds, post-churn lookups, delete-all. `-DBITSTEALING` for variant. |
| `bench_map64_libs.c` | map64 | simd_map64 vs verstable vs khashl (C side). Linked with C++ driver. |
| `bench_map64_libs_main.cpp` | map64 | C++ driver: adds boost::unordered_flat_set to above comparison. |
| `bench_map64_backends.c` | map64 | AVX-512 vs AVX2 vs scalar backend comparison + post-churn lookup. Build two binaries: one native, one with `-mno-avx512f`. |
| `bench_map128_delete.c` | map128 | A/B: rehash vs displacement backshift delete |
| `bench_map128_delete_pf.c` | map128 | Raw vs prefetch-pipelined delete throughput |
| `bench_kv_layout.c` | generic KV | Grid search: 3 layouts × 2 overflow × PF sweep. 16 instantiations. TSV output. |
| `bench_kv_pf_tuning.c` | generic KV | PF distance sweep + delete prefetch mode A/B. Sentinel inline (KW=2, VW=1). TSV output. |
| `bench_kv_vs_boost.c` | generic KV | KV sentinel/bitstealing vs boost::unordered_flat_map (C side). Linked with C++ driver. |
| `bench_kv_vs_boost_main.cpp` | generic KV | C++ driver: boost benchmark + orchestration. 10 workloads: insert-only, 5 read/write ratios, 4 churn profiles. TSV output. |

### Third-party headers (vendored for benchmarks)

| File | What it is |
|------|-----------|
| `verstable.h` | Verstable v2.2.1 — C99 open-addressing hash table (benchmark comparison target) |

### Documentation

| File | What it does |
|------|-------------|
| `WHY_NOT_PHF.md` | Analysis of why perfect hash maps lose to open addressing at scale |
| `KV_LAYOUT_ANALYSIS.md` | KV value layout strategy benchmark: inline vs separate vs hybrid, ANOVA analysis across sentinel and bitstealing |
| `KV_BOOST_COMPARISON.md` | KV map vs boost::unordered_flat_map benchmark: 10 workloads, ANOVA analysis, crossover analysis |
| `analyze_kv_pf_tuning.py` | Tabular analysis of PF tuning TSV runs: median ns/op vs PF, optimal PF, meta vs full pairwise |

## Performance characteristics (2M keys, AVX2, i9-12900HK)

Both 128-bit variants achieve ~17-18 ns/op pipelined (PF=24) across all
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
# generic correctness (128-bit + 256-bit, sentinel + bitstealing)
cc -O3 -march=native -std=gnu11 -o test_generic test_map_generic.c

# generic scalar backend
cc -O3 -march=native -mno-avx2 -mno-avx512f -std=gnu11 -o test_generic_scalar test_map_generic.c

# 128-bit prototype correctness (sentinel)
cc -O3 -march=native -std=gnu11 -o test_s test_map128.c

# 128-bit prototype correctness (bitstealing)
sed 's|simd_map128_sentinel\.h|simd_map128_bitstealing.h|' test_map128.c | \
  cc -O3 -march=native -std=gnu11 -x c -o test_bs -

# throughput benchmark (sentinel vs bitstealing)
cc -O3 -march=native -std=gnu11 -o bench_s bench_map128_throughput.c
cc -O3 -march=native -std=gnu11 -DBITSTEALING -o bench_bs bench_map128_throughput.c

# map64 backend comparison (AVX-512 vs AVX2)
cc -O3 -march=native -std=gnu11 -o bench_512 bench_map64_backends.c
cc -O3 -march=native -mno-avx512f -std=gnu11 -o bench_avx2 bench_map64_backends.c

# 128-bit scalar backend (SWAR, still uses CRC32 hash via SSE4.2)
cc -O3 -march=native -mno-avx2 -mno-avx512f -std=gnu11 -o test_scalar test_map128.c

# KV map correctness (all 6 variants: 3 layouts × 2 overflow)
cc -O3 -march=native -std=gnu11 -o test_kv test_kv_generic.c

# KV map scalar backend correctness
cc -O3 -march=native -mno-avx2 -mno-avx512f -std=gnu11 -o test_kv_scalar test_kv_generic.c

# KV layout benchmark (16 instantiations, PF sweep, churn)
cc -O3 -march=native -std=gnu11 -o bench_kv bench_kv_layout.c

# KV PF tuning (PF sweep + delete prefetch A/B)
cc -O3 -march=native -std=gnu11 -o bench_pf bench_kv_pf_tuning.c

# KV vs boost::unordered_flat_map (C/C++ linkage)
cc -O3 -march=native -std=gnu11 -c bench_kv_vs_boost.c
c++ -O3 -march=native -std=c++17 -c bench_kv_vs_boost_main.cpp
c++ -O3 -o bench_kv_vs_boost bench_kv_vs_boost.o bench_kv_vs_boost_main.o
```

`-march=native` is required (selects AVX2 or AVX-512 backend). BMI2 (`pext`)
is present on all AVX2 CPUs that matter (Haswell+, Zen2+).
`-mno-avx2 -mno-avx512f` forces the scalar (SWAR) backend while keeping
SSE4.2 (CRC32 hash, `_mm_prefetch`).
