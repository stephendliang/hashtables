# KV64 Superblock Layout Analysis

## Overview

`simd_map64.h` extends the zero-metadata `simd_set64` hash set to a key-value
map. Keys remain in 64B groups (8 × uint64_t, one cache line) with direct
SIMD comparison. Values are stored in separate 64B groups (8 × VW × uint64_t).

The **block stride** parameter N controls how key and value groups are
interleaved in memory:

```
N=1 (inline):  [K0 V0] [K1 V1] [K2 V2] ...         128B per unit
N=2:           [K0 K1 V0 V1] [K2 K3 V2 V3] ...      256B superblocks
N=4:           [K0 K1 K2 K3 V0 V1 V2 V3] ...         512B superblocks
```

Where each K is a 64B key group and each V is a 64B value group (VW=1).

The hypothesis: larger N improves write performance by grouping key cache
lines contiguously (better spatial prefetch during probe chains and
backshift), at the cost of separating keys from their values (hurting read
paths that need both).

## Setup

- **CPU**: i9-12900HK (Alder Lake), AVX2 + BMI2, pinned to P-core 0
- **Key**: uint64_t (0 reserved), **Value**: uint64_t (VW=1)
- **N**: 2M keys, 5 runs per configuration, median reported
- **Grid**: 3 strides (N=1, 2, 4) × 11 PF distances (4–64)
- **Prefetch modes**: `_prefetch_insert` (1 line, key only) for insert/delete;
  `_prefetch` (2 lines, key + value) for get
- **Load factor**: 75% (8 slots/group)
- **Delete**: backshift (no tombstones)

## True mixed workloads (interleaved pipeline)

**This is the primary result.** Mixed workloads interleave GET, INSERT, and
DELETE operations in a single prefetch pipeline, with per-op dispatch:
GET → `_prefetch` (2 lines), INSERT/DELETE → `_prefetch_insert` (1 line).

Four churn profiles tested: 80/10/10, 50/25/25, 33/33/34, 20/40/40
(read/insert/delete %). 1M keys pre-inserted, 2M timed mixed ops.

### Results (median ns/op, 5 runs)

#### insert_only (2M unique keys, insert_unique)

```
 PF     N=1     N=2     N=4
  4    12.2    14.2    14.1
  8    11.3    11.4    11.5
 12    10.8    10.3    10.8
 16    11.1     9.5     9.8
 20    10.4     8.7     9.2
 24    10.4     8.7     8.3
 28    10.7     8.0     8.2
 32    10.5     8.0     8.0
 40    10.6     8.0     7.8
 48    10.9    11.1    12.1
 64    11.5    12.2    12.3
best   10.4     8.0     7.8   >> N=4/PF=40 (7.8 ns/op)
```

N=2/4 win pure inserts by 24% — the only workload where superblocks help.

#### churn_80_10_10 (80% read, 10% insert, 10% delete)

```
 PF     N=1     N=2     N=4
  4    15.5    18.0    18.5
  8    13.8    17.0    17.3
 12    13.1    16.3    16.1
 16    13.3    16.0    15.8
 20    13.6    15.4    15.7
 24    13.1    15.3    15.7
 28    13.8    15.3    15.1
 32    13.1    15.4    15.4
 40    13.2    16.0    14.8
 48    13.8    15.4    15.5
 64    13.2    15.3    15.6
best   13.1    15.3    14.8   >> N=1/PF=24 (13.1 ns/op)
```

**N=1 wins by 14% over N=2.**

#### churn_50_25_25 (50% read, 25% insert, 25% delete)

```
 PF     N=1     N=2     N=4
  4    18.3    19.7    20.6
  8    16.4    17.8    17.8
 12    15.6    17.2    17.6
 16    15.7    16.9    17.2
 20    16.1    16.6    17.6
 24    15.9    16.6    17.3
 28    15.8    16.9    16.9
 32    15.6    17.6    17.5
 40    16.2    17.6    17.8
 48    16.6    17.2    17.8
 64    16.3    18.2    18.2
best   15.6    16.6    16.9   >> N=1/PF=32 (15.6 ns/op)
```

**N=1 wins by 6% over N=2.**

#### churn_33_33_34 (33% read, 33% insert, 34% delete)

```
 PF     N=1     N=2     N=4
  4    18.9    21.1    21.4
  8    16.4    18.8    19.0
 12    16.5    18.2    18.2
 16    15.9    18.2    18.2
 20    15.9    18.0    17.4
 24    15.9    17.9    17.1
 28    16.0    17.7    17.7
 32    16.0    18.1    17.7
 40    16.3    18.9    18.7
 48    17.1    18.6    18.3
 64    16.7    18.2    17.8
best   15.9    17.7    17.1   >> N=1/PF=24 (15.9 ns/op)
```

**N=1 wins by 10% over N=2.**

#### churn_20_40_40 (20% read, 40% insert, 40% delete)

```
 PF     N=1     N=2     N=4
  4    19.2    21.6    21.8
  8    17.2    18.9    19.1
 12    16.6    18.2    17.9
 16    16.2    17.3    17.7
 20    16.4    17.2    17.7
 24    16.5    17.4    17.4
 28    16.4    17.9    17.3
 32    16.2    17.8    17.1
 40    16.5    18.7    18.2
 48    17.0    19.1    18.1
 64    16.7    18.6    18.2
best   16.2    17.2    17.1   >> N=1/PF=16 (16.2 ns/op)
```

**N=1 wins by 6% over N=2**, even at 80% writes.

### Summary: mixed workload winners

| Workload | N=1 | N=2 | N=4 | N=1 advantage |
|----------|-----|-----|-----|---------------|
| insert_only | 10.4 | 8.0 | **7.8** | N=4 wins (pure writes only) |
| churn_80_10_10 | **13.1** | 15.3 | 14.8 | 14% over N=2 |
| churn_50_25_25 | **15.6** | 16.6 | 16.9 | 6% over N=2 |
| churn_33_33_34 | **15.9** | 17.7 | 17.1 | 10% over N=2 |
| churn_20_40_40 | **16.2** | 17.2 | 17.1 | 6% over N=2 |

**N=1 wins every mixed workload, at every read/write ratio.**

### Why N≥2 loses in mixed pipelines

In a true mixed pipeline, the prefetch dispatch alternates between 1-line
(`_prefetch_insert` for insert/delete) and 2-line (`_prefetch` for get)
on consecutive iterations. This creates two interacting effects:

1. **Spatial coherence**: with N=1, both prefetch modes target cache lines
   within the same 128B unit (key at offset 0, value at offset 64). The
   hardware prefetcher sees a consistent stride-1 pattern regardless of
   which prefetch mode fires. With N≥2, the 2-line get prefetch reaches
   into a value block 128–256B away from the key, creating scattered
   access that breaks the hardware prefetcher's stride detection.

2. **Pipeline interference**: when a get prefetch (2 lines) is followed
   by an insert prefetch (1 line to a different group), N=1 keeps both
   accesses in a compact address range. N≥2 creates a wider scatter:
   key groups in one region, value groups in another. The resulting
   address stream has no exploitable spatial pattern for the hardware
   prefetcher.

The isolated-phase benchmarks (§ Appendix) masked this because bulk
operations have a uniform prefetch pattern — all 1-line or all 2-line —
which the hardware prefetcher can track even with N≥2. The mixed pipeline
destroys that uniformity.

## Recommendation

**Use N=1 (inline) with PF=24.** This is the correct default for all
workloads that include any read operations.

N≥2 only wins pure insert-only workloads (bulk loading with no reads).
If your workload is exclusively `init_cap` + `insert_unique` (e.g.,
building a lookup table), N=2/PF=32 saves 24%. For everything else,
N=1 is 6–14% faster.

## Appendix: isolated-phase benchmarks

The following results measure each operation in isolation (all-insert,
then all-get, then all-delete) rather than interleaved. These show the
theoretical per-operation advantage of each layout, but overstate N≥2's
benefit because uniform prefetch patterns are unrealistically favorable
to the superblock layout.

### Insert (isolated bulk)

```
 PF     N=1     N=2     N=4
  4    11.5    13.3    13.3
  8    10.8    10.8    11.0
 12    10.1     9.6    10.1
 16    10.0     8.9     9.1
 20    10.0     8.2     8.3
 24     9.9     7.7     7.6
 28    10.0     7.4     7.5
 32     9.9     7.4     7.3
 40    10.0     7.2     7.1
 48     9.8     9.3    11.2
 64    11.1    11.8    11.8
```

Winner: N=4/PF=40 (7.1 ns/op). N=2/4 beat N=1 by 30%.

N=1 plateaus at ~10 ns regardless of PF, while N=2/4 drop to 7.1–7.2 ns
at PF=32–40. Insert uses `_prefetch_insert` (1 line, key only). With N=1,
consecutive groups alternate [key, value, key, value, ...], so the hardware
prefetcher pulls in interleaved value lines that are never read. With N≥2,
consecutive key groups are contiguous, and the hardware prefetcher assists
within the key block.

### Get hit (isolated bulk)

```
 PF     N=1     N=2     N=4
  4     8.8    11.4    11.9
  8     8.4    10.6    10.9
 12     8.2    10.1    10.4
 16     8.0     9.7     9.6
 20     7.9     9.5     9.6
 24     7.8     9.6     9.6
 28     7.9     9.6     9.4
 32     7.9     9.5     9.4
 40     8.4     9.8     9.5
 48     8.5     9.6     9.5
 64     8.5     9.7     9.6
```

Winner: N=1/PF=24 (7.8 ns/op). N=1 beats N=2 by 18%.

Get hits require both key (SIMD match) and value (return pointer) cache
lines. With N=1, these are adjacent (offsets 0 and 64 within 128B). With
N≥2, the value is 128–256B away, defeating spatial prefetch coherence.

### Get miss (isolated bulk)

```
 PF     N=1     N=2     N=4
  4     9.5    12.0    12.7
  8     8.9    10.9    11.6
 12     8.7    10.5    11.3
 16     8.6    10.1    10.3
 20     8.5    10.1    10.2
 24     8.4    10.0    10.2
 28     8.4    10.1    10.1
 32     8.4    10.1    10.0
 40     8.8    10.0    10.1
 48     8.8    10.1    10.0
 64     8.7    10.3    10.1
```

Winner: N=1/PF=24 (8.4 ns/op). N=1 wins by 16%.

Misses only touch the key line (empty-slot termination), yet N=2/4 are
still slower because `_prefetch` issues 2 lines — the wasted value
prefetch competes for fill buffer entries.

### Delete (isolated bulk)

```
 PF     N=1     N=2     N=4
  4    10.7    12.0    11.7
  8    10.0     9.9     9.6
 12     9.7     8.6     8.6
 16     9.4     8.2     8.1
 20     9.2     7.6     7.7
 24     9.4     7.1     7.1
 28     9.2     6.8     6.9
 32     9.3     6.7     6.8
 40     9.5     6.6     6.6
 48     9.6     6.6     6.7
 64     9.5     6.9     6.8
```

Winner: N=2/PF=40 (6.6 ns/op). N=2/4 beat N=1 by 28%.

Delete uses `_prefetch_insert` (1 line) and performs backshift — a
multi-group traversal scanning consecutive key groups. With N≥2,
consecutive key groups are contiguous, turning backshift into a
sequential access pattern.

### Churn delete (isolated rounds)

```
 PF     N=1     N=2     N=4
  4    13.9    15.9    15.0
  8    12.6    12.9    12.4
 12    12.2    11.6    11.2
 16    12.0    11.2    10.6
 20    11.9    10.9    10.6
 24    11.9    10.6    10.5
 28    11.9    10.6    10.4
 32    12.0    10.6    10.3
 40    12.3    10.6    10.4
 48    12.2    10.7    10.5
 64    12.8    11.0    10.7
```

Winner: N=4/PF=32 (10.3 ns/op). N=4 beats N=1 by 13%.

### Churn insert (isolated rounds)

```
 PF     N=1     N=2     N=4
  4    12.6    14.4    14.4
  8    11.0    11.5    11.6
 12    10.4    10.2    10.6
 16    10.2     9.5    10.2
 20    10.0     9.0     9.3
 24    10.1     8.4     8.7
 28    10.1     8.2     8.3
 32    10.1     8.2     8.3
 40    10.5     8.1     8.1
 48    10.4     8.5     9.4
 64    11.2    12.0    12.1
```

Winner: N=2/PF=40 (8.1 ns/op). N=2 beats N=1 by 19%.

### Post-churn get hit (isolated)

```
 PF     N=1     N=2     N=4
  4     8.8    11.3    11.8
  8     8.3    10.5    10.9
 12     8.1     9.8    10.4
 16     7.9     9.6     9.6
 20     7.9     9.4     9.5
 24     7.8     9.5     9.6
 28     7.8     9.4     9.5
 32     7.8     9.4     9.5
 40     8.5     9.5     9.4
 48     8.5     9.4     9.6
 64     8.4     9.6     9.6
```

Winner: N=1/PF=24 (7.8 ns/op). Identical to fresh get_hit — backshift
keeps probe chains intact, no churn penalty.

### PF distance interaction (isolated phases)

| Path | Prefetch lines | Optimal PF | Reason |
|------|---------------|------------|--------|
| Read (get) | 2 (key + value) | 20–24 | More lines per op = fill buffer fills faster |
| Write (insert) | 1 (key only) | 32–40 | Fewer lines = more runway before saturation |
| Delete | 1 (key only) | 32–40 | Same as insert, plus backshift spatial scan |

This matches the KV sentinel/bitstealing PF tuning results: 1-line
prefetch tolerates higher PF distances because it generates less fill
buffer pressure per operation.

## Methodology

- 5 independent runs per (N, PF) pair, median reported
- Same key/value data across all configurations (identical RNG seed)
- `taskset -c 0` pinned to P-core
- **Mixed workloads** (§ True mixed): xoshiro256** RNG, gen_mixed workload
  generator simulates map state to produce valid op sequences (no deletes
  on empty, no inserts past pool). Per-op prefetch dispatch: GET → 2 lines,
  INSERT/DELETE → 1 line. Timed phase excludes pre-insertion.
- **Isolated workloads** (§ Appendix): 2M bulk insert, 2M get hit, 2M get
  miss (disjoint keys), 2M bulk delete, 10×200K churn rounds, post-churn
  get hit on churned-in keys
- Final counts validated: identical across all strides for same workload
