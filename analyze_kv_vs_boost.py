#!/usr/bin/env python3
"""
ANOVA analysis of KV map vs boost::unordered_flat_map benchmark results.

One-way ANOVA per workload (3 groups: sentinel, bitstealing, boost).
Bonferroni pairwise comparisons. Overall ranking.

Usage:
    python3 analyze_kv_vs_boost.py /tmp/bench_boost_run*.tsv
"""

import sys
import glob
import pandas as pd
import numpy as np
from scipy import stats


def load_runs(files):
    frames = []
    for i, f in enumerate(sorted(files)):
        df = pd.read_csv(f, sep='\t', comment='#',
                         names=['map', 'workload', 'ns_per_op', 'final_count'])
        df['run'] = i
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def anova_and_ranking(df, workload):
    sub = df[df['workload'] == workload].copy()
    if sub.empty:
        return None

    maps = sorted(sub['map'].unique())
    groups = [sub[sub['map'] == m]['ns_per_op'].values for m in maps]

    valid = [(m, g) for m, g in zip(maps, groups) if len(g) >= 2]
    if len(valid) < 2:
        return None

    maps_v = [m for m, _ in valid]
    groups_v = [g for _, g in valid]

    f_stat, p_value = stats.f_oneway(*groups_v)

    grand_mean = np.concatenate(groups_v).mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups_v)
    ss_total = sum(((g - grand_mean) ** 2).sum() for g in groups_v)
    eta_sq = ss_between / ss_total if ss_total > 0 else 0

    summary = []
    for m, g in zip(maps_v, groups_v):
        summary.append({
            'map': m,
            'mean': g.mean(),
            'std': g.std(ddof=1),
            'median': np.median(g),
            'min': g.min(),
            'max': g.max(),
            'n': len(g),
        })
    summary.sort(key=lambda x: x['median'])

    n_pairs = len(maps_v) * (len(maps_v) - 1) // 2
    pairwise = []
    for i in range(len(maps_v)):
        for j in range(i + 1, len(maps_v)):
            t, p = stats.ttest_ind(groups_v[i], groups_v[j])
            p_adj = min(p * n_pairs, 1.0)
            diff = groups_v[i].mean() - groups_v[j].mean()
            pairwise.append({
                'a': maps_v[i],
                'b': maps_v[j],
                'diff_ns': diff,
                'p_adj': p_adj,
                'sig': '***' if p_adj < 0.001 else '**' if p_adj < 0.01
                       else '*' if p_adj < 0.05 else 'ns',
            })

    return {
        'workload': workload,
        'F': f_stat,
        'p': p_value,
        'eta_sq': eta_sq,
        'summary': summary,
        'pairwise': pairwise,
    }


def print_results(result):
    wl = result['workload']
    print(f"\n  {wl.upper()}")
    print(f"  {'─' * 66}")
    print(f"  ANOVA: F={result['F']:.2f}, p={result['p']:.2e}, "
          f"η²={result['eta_sq']:.3f}", end='')
    if result['p'] < 0.001:
        print(f"  *** highly significant")
    elif result['p'] < 0.05:
        print(f"  * significant")
    else:
        print(f"  n.s.")

    print(f"\n  {'Rank':<5} {'Map':<20} {'Median':>8} {'Mean':>8} "
          f"{'Std':>7} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*5} {'-'*20} {'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*8}")
    for rank, s in enumerate(result['summary'], 1):
        print(f"  {rank:<5} {s['map']:<20} {s['median']:>8.1f} "
              f"{s['mean']:>8.1f} {s['std']:>7.2f} {s['min']:>8.1f} "
              f"{s['max']:>8.1f}")

    sig_pairs = [p for p in result['pairwise'] if p['sig'] != 'ns']
    if sig_pairs:
        print(f"\n  Significant pairwise (Bonferroni):")
        sig_pairs.sort(key=lambda x: x['p_adj'])
        for p in sig_pairs[:10]:
            faster = p['a'] if p['diff_ns'] < 0 else p['b']
            slower = p['b'] if p['diff_ns'] < 0 else p['a']
            print(f"    {faster:<20} < {slower:<20} "
                  f"Δ{abs(p['diff_ns']):>5.1f} ns  "
                  f"p={p['p_adj']:.2e} {p['sig']}")


def overall_ranking(all_results):
    print(f"\n  {'━' * 66}")
    print(f"  OVERALL RANKING")
    print(f"  {'━' * 66}")

    maps_set = set()
    wl_data = {}
    for r in all_results:
        if r is None:
            continue
        wl = r['workload']
        wl_data[wl] = {}
        for s in r['summary']:
            wl_data[wl][s['map']] = s['median']
            maps_set.add(s['map'])

    if not wl_data:
        print("  No data.")
        return

    scores = {m: [] for m in maps_set}
    for wl, medians in wl_data.items():
        vals = list(medians.values())
        lo, hi = min(vals), max(vals)
        rng = hi - lo if hi > lo else 1.0
        for m in maps_set:
            if m in medians:
                scores[m].append((medians[m] - lo) / rng)

    avg_scores = []
    for m in maps_set:
        if scores[m]:
            avg_scores.append((m, np.mean(scores[m]), len(scores[m])))
    avg_scores.sort(key=lambda x: x[1])

    print(f"\n  {'Rank':<5} {'Map':<20} {'NormScore':>10}")
    print(f"  {'-'*5} {'-'*20} {'-'*10}")
    for rank, (m, sc, nwl) in enumerate(avg_scores, 1):
        bar = '█' * int(sc * 30) if sc > 0.01 else ''
        print(f"  {rank:<5} {m:<20} {sc:>10.3f}  {bar}")


def main():
    if len(sys.argv) < 2:
        files = sorted(glob.glob('/tmp/bench_boost_run*.tsv'))
        if not files:
            print("Usage: python3 analyze_kv_vs_boost.py /tmp/bench_boost_run*.tsv")
            sys.exit(1)
    else:
        files = sys.argv[1:]

    print(f"Loading {len(files)} benchmark runs...")
    df = load_runs(files)
    print(f"  Total rows: {len(df)}")

    workloads = ['insert_only',
                 'rw_95_5', 'rw_75_25', 'rw_50_50', 'rw_25_75', 'rw_5_95',
                 'churn_80_10_10', 'churn_50_25_25', 'churn_33_33_34',
                 'churn_20_40_40']

    all_results = []
    for wl in workloads:
        result = anova_and_ranking(df, wl)
        if result:
            print_results(result)
            all_results.append(result)

    overall_ranking(all_results)


if __name__ == '__main__':
    main()
