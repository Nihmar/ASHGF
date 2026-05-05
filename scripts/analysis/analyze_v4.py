"""Analyze v3 results: ASGF vs 6 variants at dim=10 and dim=100."""
import csv, os

algos = ["ASGF", "ASHGF", "ASGF-2X", "ASGF-2A", "ASGF-2F", "ASGF-2G", "ASHGF-2X"]
names = {
    "ASGF": "ASGF baseline",
    "ASHGF": "ASHGF (thesis)",
    "ASGF-2X": "Try-2x",
    "ASGF-2A": "2A adapt-dim",
    "ASGF-2F": "2F frequency",
    "ASGF-2G": "2G guard-rail",
    "ASHGF-2X": "ASHGF+2x",
}

def load(base):
    data = {}
    csv_dir = os.path.join(base, "csv")
    for fn in sorted(os.listdir(csv_dir)):
        if not fn.endswith(".csv"): continue
        with open(os.path.join(csv_dir, fn)) as f:
            rows = list(csv.reader(f))
        if len(rows) < 2: continue
        stem = fn.replace(".csv", "")
        for a in algos:
            if stem.startswith(a + "_"):
                data.setdefault(stem[len(a)+1:], {})[a] = float(rows[-1][1])
                break
    return data

for dim in [10, 100]:
    data = load(f"results_v3/dim_{dim}")
    variants = [a for a in algos if a != "ASGF"]

    print(f"\n{'='*75}")
    print(f"  DIM = {dim}   ({len(data)} funzioni)")
    print(f"{'='*75}")
    print(f"{'Algorithm':<15s} {'wins':>5s} {'loss':>5s} {'ties':>5s}  {'best_on':>6s}  {'eval':>5s}")
    print("-" * 50)

    for v in variants:
        wins = sum(1 for f in data if data[f].get(v, 1e99) < data[f].get("ASGF", 1e99))
        loss = sum(1 for f in data if data[f].get(v, 1e99) > data[f].get("ASGF", 1e99))
        ties = sum(1 for f in data if data[f].get(v, 1e99) == data[f].get("ASGF", 1e99))
        best = sum(1 for f in data if min((data[f].get(a, 1e99), a) for a in algos)[1] == v)
        evals = "1" if v.startswith("ASGF-2") else ("2" if v == "ASHGF-2X" else "0")
        print(f"{names[v]:<15s} {wins:5d} {loss:5d} {ties:5d}  {best:6d}  {evals:>5s}")

    best_algos = {a: 0 for a in algos}
    for f in data:
        vals = [(a, data[f][a]) for a in algos if a in data[f]]
        if vals:
            best_algos[min(vals, key=lambda x: x[1])[0]] += 1
    print(f"\n  Miglior algoritmo per funzione:")
    for a in sorted(best_algos, key=best_algos.get, reverse=True):
        bar = "#" * (best_algos[a] // 2)
        print(f"    {names[a]:<15s} {best_algos[a]:3d}  {bar}")

# Detail: where does 2G beat 2X?
data100 = load("results_v3/dim_100")
print(f"\n{'='*75}")
print("  FUNZIONI DOVE 2G (guard-rail) MIGLIORA SU 2X A DIM=100")
print(f"{'='*75}")
improved = [(f, data100[f]["ASGF-2X"], data100[f]["ASGF-2G"])
            for f in data100
            if data100[f].get("ASGF-2G", 1e99) < data100[f].get("ASGF-2X", 1e99)]
improved.sort(key=lambda x: x[1]/x[2] if x[2] != 0 else 999, reverse=True)
print(f"  2G better than 2X on {len(improved)} functions")
for fn, v2x, v2g in improved[:10]:
    ratio = v2x/v2g if v2g != 0 else 999
    print(f"    {fn:45s} 2X={v2x:.3e} -> 2G={v2g:.3e}  ({ratio:.1f}x better)")

# Detail: levy at dim=100
for dim in [10, 100]:
    data = load(f"results_v3/dim_{dim}")
    print(f"\n  levy dim={dim}:")
    for a in algos:
        val = data.get("levy", {}).get(a, float('nan'))
        print(f"    {names[a]:<15s} {val:.4e}")
