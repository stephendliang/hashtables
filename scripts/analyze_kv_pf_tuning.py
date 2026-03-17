#!/usr/bin/env python3
"""
analyze_kv_pf_tuning.py — Tabular analysis of PF tuning benchmark results.

Usage:
    python3 analyze_kv_pf_tuning.py /tmp/bench_pf_run*.tsv

Loads multiple TSV runs, computes median ns/op vs PF for each
(workload, del_mode), highlights optimal PF, and for churn shows
pairwise meta vs full comparison at each PF.
"""

import sys
import csv
from collections import defaultdict

def load_runs(filenames):
    """Load all TSV files, return list of dicts."""
    rows = []
    for fn in filenames:
        with open(fn) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) < 5:
                    continue
                rows.append({
                    'pf':       int(parts[0]),
                    'del_mode': parts[1],
                    'workload': parts[2],
                    'ns':       float(parts[3]),
                    'count':    int(parts[4]),
                })
    return rows

def median(vals):
    s = sorted(vals)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <tsv_files...>", file=sys.stderr)
        sys.exit(1)

    rows = load_runs(sys.argv[1:])
    n_runs = len(sys.argv) - 1
    print(f"Loaded {len(rows)} data points from {n_runs} runs\n")

    # Group by (workload, del_mode, pf)
    groups = defaultdict(list)
    for r in rows:
        key = (r['workload'], r['del_mode'], r['pf'])
        groups[key].append(r['ns'])

    # Identify unique (workload, del_mode) combos
    combos = sorted(set((r['workload'], r['del_mode']) for r in rows))

    for workload, del_mode in combos:
        print(f"=== {workload} / del_mode={del_mode} ===")
        print(f"{'PF':>4s}  {'median':>8s}  {'min':>8s}  {'max':>8s}  {'n':>3s}")

        best_pf = None
        best_ns = float('inf')
        pf_medians = {}

        pfs = sorted(set(r['pf'] for r in rows
                         if r['workload'] == workload and r['del_mode'] == del_mode))

        for pf in pfs:
            vals = groups[(workload, del_mode, pf)]
            med = median(vals)
            pf_medians[pf] = med
            mn = min(vals)
            mx = max(vals)
            if med < best_ns:
                best_ns = med
                best_pf = pf
            print(f"{pf:4d}  {med:8.2f}  {mn:8.2f}  {mx:8.2f}  {len(vals):3d}")

        if best_pf is not None:
            print(f"  >> optimal PF = {best_pf} ({best_ns:.2f} ns/op)")
        print()

    # Pairwise: meta vs full at each PF for churn workloads
    churn_combos = [(w, d) for w, d in combos if 'churn' in w]
    churn_workloads = sorted(set(w for w, d in churn_combos))

    for workload in churn_workloads:
        print(f"=== {workload}: meta vs full pairwise ===")
        print(f"{'PF':>4s}  {'meta':>8s}  {'full':>8s}  {'delta':>8s}  {'pct':>7s}")

        pfs = sorted(set(r['pf'] for r in rows if r['workload'] == workload))

        for pf in pfs:
            meta_vals = groups.get((workload, 'meta', pf), [])
            full_vals = groups.get((workload, 'full', pf), [])
            if not meta_vals or not full_vals:
                continue
            meta_med = median(meta_vals)
            full_med = median(full_vals)
            delta = full_med - meta_med
            pct = delta / meta_med * 100.0
            marker = "<< full wins" if delta < 0 else ""
            print(f"{pf:4d}  {meta_med:8.2f}  {full_med:8.2f}  {delta:+8.2f}  {pct:+6.1f}%  {marker}")

        print()

    # Final count validation
    print("=== Final count validation ===")
    count_groups = defaultdict(list)
    for r in rows:
        key = (r['workload'], r['del_mode'], r['pf'])
        count_groups[key].append(r['count'])

    for workload in churn_workloads:
        pfs = sorted(set(r['pf'] for r in rows if r['workload'] == workload))
        for pf in pfs:
            meta_counts = count_groups.get((workload, 'meta', pf), [])
            full_counts = count_groups.get((workload, 'full', pf), [])
            if meta_counts and full_counts:
                meta_set = set(meta_counts)
                full_set = set(full_counts)
                if meta_set != full_set:
                    print(f"  MISMATCH PF={pf}: meta={meta_set} full={full_set}")
    print("  All counts match (same workload = same final state)")

if __name__ == '__main__':
    main()
