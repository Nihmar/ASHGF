"""Analyze v4 results: 2H and 2I vs baselines."""
import csv, os

algos = ["ASGF","ASHGF","ASGF-2X","ASGF-2F","ASGF-2H","ASGF-2I"]
names = {"ASGF":"baseline","ASHGF":"ASHGF","ASGF-2X":"Try-2x","ASGF-2F":"2F freq","ASGF-2H":"2H adapF","ASGF-2I":"2I freq+dim"}

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
    data = load(f"results_v4/dim_{dim}")
    variants = [a for a in algos if a != "ASGF"]

    print(f"\n{'='*80}")
    print(f"  DIM = {dim}   ({len(data)} funzioni)")
    print(f"{'='*80}")
    print(f"{'Algorithm':<15s} {'wins':>5s} {'loss':>5s} {'ties':>5s}  best_on  vs2F       note")
    print("-" * 65)

    f2_wins = sum(1 for f in data if data[f].get("ASGF-2F",1e99) < data[f].get("ASGF",1e99))
    f2_loss = sum(1 for f in data if data[f].get("ASGF-2F",1e99) > data[f].get("ASGF",1e99))

    for v in variants:
        wins = sum(1 for f in data if data[f].get(v, 1e99) < data[f].get("ASGF", 1e99))
        loss = sum(1 for f in data if data[f].get(v, 1e99) > data[f].get("ASGF", 1e99))
        ties = sum(1 for f in data if data[f].get(v, 1e99) == data[f].get("ASGF", 1e99))
        best = sum(1 for f in data if min((data[f].get(a, 1e99), a) for a in algos)[1] == v)
        # vs 2F
        vs2f_w = sum(1 for f in data if data[f].get(v,1e99) < data[f].get("ASGF-2F",1e99)) if v != "ASGF-2F" else 0
        vs2f = f"vs2F: +{vs2f_w}" if v != "ASGF-2F" else "---"

        if v == "ASGF-2F": tag = "(baseline)"
        elif v == "ASGF-2H": tag = "self-tuning warmup/cooldown"
        elif v == "ASGF-2I": tag = "2F + dim-adaptive k"
        else: tag = ""
        print(f"{names[v]:<15s} {wins:5d} {loss:5d} {ties:5d}  {best:6d}  {vs2f:<10s} {tag}")

    # Winners
    best_algos = {a: 0 for a in algos}
    for f in data:
        vals = [(a, data[f][a]) for a in algos if a in data[f]]
        if vals: best_algos[min(vals, key=lambda x: x[1])[0]] += 1
    print(f"\n  Best algorithm per function:")
    for a in sorted(best_algos, key=best_algos.get, reverse=True):
        bar = "#" * (best_algos[a] // 2)
        print(f"    {names[a]:<15s} {best_algos[a]:3d}  {bar}")

# Direct comparison 2H vs 2F and 2I vs 2F at dim=100
print(f"\n{'='*80}")
print("  2H and 2I vs 2F at dim=100 — where do they differ?")
print(f"{'='*80}")
data100 = load("results_v4/dim_100")
for child in ["ASGF-2H","ASGF-2I"]:
    better = [(f, data100[f]["ASGF-2F"], data100[f][child])
              for f in data100
              if data100[f].get(child,1e99) < data100[f].get("ASGF-2F",1e99)]
    worse = [(f, data100[f]["ASGF-2F"], data100[f][child])
             for f in data100
             if data100[f].get(child,1e99) > data100[f].get("ASGF-2F",1e99)]
    print(f"\n  {names[child]} is BETTER than 2F on {len(better)} functions:")
    for fn, v2f, vch in sorted(better, key=lambda x: x[1]/x[2] if x[2]!=0 else 999, reverse=True)[:6]:
        print(f"    {fn:45s} {v2f:.2e} -> {vch:.2e}")
    print(f"  {names[child]} is WORSE  than 2F on {len(worse)} functions:")
    for fn, v2f, vch in sorted(worse, key=lambda x: x[1]/x[2] if x[2]!=0 else 999, reverse=True)[:6]:
        print(f"    {fn:45s} {v2f:.2e} -> {vch:.2e}")
