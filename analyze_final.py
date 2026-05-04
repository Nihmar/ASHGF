"""Analyze dim=10 and dim=100 results."""
import csv, os

algos = ["ASGF", "ASHGF", "ASGF-2X"]

def load_results(base):
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

data10 = load_results("results_compare/dim_10")
data100 = load_results("results_compare/dim_100")

for dim, data in [(10, data10), (100, data100)]:
    print(f"\n{'='*65}")
    print(f"  DIM = {dim}   ({len(data)} funzioni)")
    print(f"{'='*65}")
    print(f"{'Algorithm':<12s} {'wins':>5s} {'loss':>5s} {'ties':>5s}  {'best_on':>6s}  note")
    print("-" * 50)

    for v in ["ASHGF", "ASGF-2X"]:
        wins = sum(1 for f in data if data[f].get(v, 1e99) < data[f].get("ASGF", 1e99))
        loss = sum(1 for f in data if data[f].get(v, 1e99) > data[f].get("ASGF", 1e99))
        ties = sum(1 for f in data if data[f].get(v, 1e99) == data[f].get("ASGF", 1e99))
        best = sum(1 for f in data if min((data[f].get(a, 1e99), a) for a in algos)[1] == v)
        note = "1 eval extra" if v == "ASGF-2X" else "0 eval extra"
        print(f"{v:<12s} {wins:5d} {loss:5d} {ties:5d}  {best:6d}  ({note})")

    # Per-function best
    best_algos = {a: 0 for a in algos}
    for f in data:
        vals = [(a, data[f][a]) for a in algos if a in data[f]]
        if vals:
            best_algos[min(vals, key=lambda x: x[1])[0]] += 1
    print(f"\n  Migliore algoritmo per funzione:")
    for a in sorted(best_algos, key=best_algos.get, reverse=True):
        bar = "#" * (best_algos[a] // 2)
        print(f"    {a:<12s} {best_algos[a]:3d}  {bar}")

# Where does 2X lose at dim=100?
print(f"\n{'='*65}")
print("  FUNZIONI DOVE 2X PERDE vs ASGF A DIM=100 (losses)")
print(f"{'='*65}")
losses100 = [(f, data100[f]["ASGF"], data100[f]["ASGF-2X"],
              data100[f]["ASGF-2X"]/data100[f]["ASGF"] if data100[f]["ASGF"]!=0 else 999)
             for f in data100
             if data100[f].get("ASGF-2X",1e99) > data100[f].get("ASGF",1e99)]
losses100.sort(key=lambda x: x[3], reverse=True)
print(f"{'Function':<45s} {'ASGF':>12s} {'2X':>12s} {'ratio':>10s}")
print("-" * 80)
for fn, a, l, r in losses100:
    print(f"{fn:<45s} {a:12.4e} {l:12.4e} {r:10.4f}")

# Where does 2X win BIG at dim=100?
print(f"\n{'='*65}")
print("  TOP WINS 2X vs ASGF A DIM=100")
print(f"{'='*65}")
wins100 = [(f, data100[f]["ASGF"], data100[f]["ASGF-2X"],
            data100[f]["ASGF"]/data100[f]["ASGF-2X"] if data100[f]["ASGF-2X"]!=0 else 999)
           for f in data100
           if data100[f].get("ASGF-2X",1e99) < data100[f].get("ASGF",1e99)]
wins100.sort(key=lambda x: x[3], reverse=True)
for fn, a, l, r in wins100[:10]:
    print(f"  {fn:<45s} ASGF={a:.2e} -> 2X={l:.2e}  ({r:.1f}x better)")
