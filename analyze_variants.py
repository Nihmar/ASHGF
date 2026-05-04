"""Analyze benchmark results: ASGF variants vs baseline ASGF."""
import csv
import os

base_dir = r"results_variants"
files = sorted(os.listdir(base_dir))

algorithms = ["ASGF", "ASGF-RS", "ASGF-LS", "ASGF-CD", "ASGF-SS", "ASGF-AQ", "ASGF-BW", "ASHGF"]

data = {}
for fname in files:
    if not fname.endswith(".csv"):
        continue
    with open(os.path.join(base_dir, fname)) as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        continue
    final_val = float(rows[-1][1])

    stem = fname.replace(".csv", "")
    for algo in algorithms:
        prefix = algo + "_"
        if stem.startswith(prefix):
            func = stem[len(prefix):]
            data.setdefault(func, {})[algo] = final_val
            break

variants = [a for a in algorithms if a != "ASGF"]

print("=" * 60)
print(f"COMPARISON: {len(data)} functions, dim=10, iter=500, patience=50")
print("=" * 60)
print(f"{'Variant':<10s} {'wins':>6s} {'loses':>6s} {'ties':>6s}  Description")
print("-" * 60)

for variant in variants:
    wins = 0
    loses = 0
    ties = 0
    ratio_sum = 0.0
    ratio_count = 0
    for func in data:
        asgf_val = data[func].get("ASGF")
        var_val = data[func].get(variant)
        if asgf_val is None or var_val is None:
            continue

        # Lower is better (minimization)
        if var_val < asgf_val:
            wins += 1
        elif var_val > asgf_val:
            loses += 1
        else:
            ties += 1

        if asgf_val != 0 and var_val != 0:
            ratio_sum += abs(var_val / asgf_val)
            ratio_count += 1

    avg_ratio = ratio_sum / ratio_count if ratio_count else 0
    desc = {
        "ASGF-RS": "Restart Scheduling",
        "ASGF-LS": "Line Search",
        "ASGF-CD": "Conjugate Directions",
        "ASGF-SS": "Smooth Sigma",
        "ASGF-AQ": "Adaptive Quadrature",
        "ASGF-BW": "Blended Warm-start",
        "ASHGF": "Original ASHGF",
    }.get(variant, "")
    print(f"{variant:<10s} {wins:6d} {loses:6d} {ties:6d}  {desc}")

# Which variant achieved best value on most functions?
print(f"\n{'='*40}")
print("PER-FUNCTION BEST ALGORITHM")
print(f"{'='*40}")
winners = {a: 0 for a in algorithms}
for func in data:
    vals = {}
    for a in algorithms:
        if a in data[func] and data[func][a] is not None:
            vals[a] = data[func][a]
    if vals:
        best = min(vals, key=vals.get)
        winners[best] += 1

for a in sorted(winners, key=winners.get, reverse=True):
    bar = "#" * winners[a]
    print(f"  {a:<10s}: {winners[a]:3d}  {bar}")

# Cumulative dominance: how many functions does each variant beat ASGF on by > 10%
print(f"\n{'='*40}")
print("FUNCTIONS WHERE VARIANT BEATS ASGF BY >10%")
print(f"{'='*40}")
for variant in variants:
    big = []
    for func in data:
        a = data[func].get("ASGF")
        v = data[func].get(variant)
        if a is None or v is None or a == 0:
            continue
        if v < a and abs(v / a) < 0.9:
            big.append((func, a, v, v / a))
    print(f"  {variant:<10s}: {len(big):3d} functions")
    if big and len(big) <= 5:
        for fn, av, vv, r in sorted(big, key=lambda x: x[3]):
            print(f"             {fn:40s} {av:.2e} -> {vv:.2e}  ({r:.2f}x)")

# Functions where ASGF-LS or ASGF-SS significantly outperform ASGF
print(f"\n{'='*60}")
print("TOP WINS: ASGF-LS (Line Search) best improvements over ASGF")
print(f"{'='*60}")
ls_wins = []
for func in data:
    a = data[func].get("ASGF")
    v = data[func].get("ASGF-LS")
    if a is None or v is None or a == 0:
        continue
    if v < a and v / a < 1.0:
        ls_wins.append((func, a, v, v / a))
for fn, av, vv, r in sorted(ls_wins, key=lambda x: x[3])[:15]:
    print(f"  {fn:45s} ASGF={av:.3e} -> LS={vv:.3e}  ({r:.2f}x better)")

print(f"\n{'='*60}")
print("TOP WINS: ASGF-SS (Smooth Sigma) best improvements over ASGF")
print(f"{'='*60}")
ss_wins = []
for func in data:
    a = data[func].get("ASGF")
    v = data[func].get("ASGF-SS")
    if a is None or v is None or a == 0:
        continue
    if v < a and v / a < 1.0:
        ss_wins.append((func, a, v, v / a))
for fn, av, vv, r in sorted(ss_wins, key=lambda x: x[3])[:15]:
    print(f"  {fn:45s} ASGF={av:.3e} -> SS={vv:.3e}  ({r:.2f}x better)")
