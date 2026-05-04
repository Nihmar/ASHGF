import csv, os

algos = ["ASGF","ASGF-2F","ASGF-2S","ASGF-2T"]
names = {"ASGF":"baseline","ASGF-2F":"2F","ASGF-2S":"2S","ASGF-2T":"2T"}

for dim in [10, 100]:
    base = f"results_v9/dim_{dim}/csv"
    data = {}
    for fn in sorted(os.listdir(base)):
        if not fn.endswith(".csv"): continue
        with open(os.path.join(base, fn)) as f:
            rows = list(csv.reader(f))
        if len(rows) < 2: continue
        stem = fn.replace(".csv", "")
        for a in algos:
            if stem.startswith(a + "_"):
                data.setdefault(stem[len(a)+1:], {})[a] = float(rows[-1][1])
                break

    print(f"\n{'='*55}")
    print(f"  DIM = {dim}   ({len(data)} funzioni)")
    print(f"{'='*55}")
    print(f"{'Algo':<6s} {'wins':>5s} {'loss':>5s} {'ties':>5s}  {'best':>5s}  vs2S(better/worse)")
    print("-" * 55)

    for v in ["ASGF-2F","ASGF-2S","ASGF-2T"]:
        wins = sum(1 for f in data if data[f].get(v,1e99) < data[f].get("ASGF",1e99))
        loss = sum(1 for f in data if data[f].get(v,1e99) > data[f].get("ASGF",1e99))
        ties = sum(1 for f in data if data[f].get(v,1e99) == data[f].get("ASGF",1e99))
        best = sum(1 for f in data if min((data[f].get(a,1e99),a) for a in algos)[1]==v)
        b = sum(1 for f in data if data[f].get(v,1e99) < data[f].get("ASGF-2S",1e99)) if v!="ASGF-2S" else 0
        w = sum(1 for f in data if data[f].get(v,1e99) > data[f].get("ASGF-2S",1e99)) if v!="ASGF-2S" else 0
        vs = f"{b}/{w}" if v!="ASGF-2S" else "--"
        print(f"{names[v]:<6s} {wins:5d} {loss:5d} {ties:5d}  {best:5d}  {vs:>10s}")

    # 2T vs 2S detail
    better = [(f, data[f]["ASGF-2S"], data[f]["ASGF-2T"]) for f in data if data[f].get("ASGF-2T",1e99) < data[f].get("ASGF-2S",1e99)]
    better.sort(key=lambda x: x[1]/x[2] if x[2]!=0 else 999, reverse=True)
    worse = [(f, data[f]["ASGF-2S"], data[f]["ASGF-2T"]) for f in data if data[f].get("ASGF-2T",1e99) > data[f].get("ASGF-2S",1e99)]
    worse.sort(key=lambda x: x[2]/x[1] if x[1]!=0 else 999, reverse=True)
    if better:
        print(f"\n  2T > 2S ({len(better)}):")
        for fn, v2s, v2t in better[:4]:
            r = v2s/v2t if v2t!=0 else 999
            print(f"    {fn:45s} {v2s:.2e} -> {v2t:.2e}  ({r:.0f}x)")
    if worse:
        print(f"  2T < 2S ({len(worse)}):")
        for fn, v2s, v2t in worse[:4]:
            r = v2t/v2s if v2s!=0 else 999
            print(f"    {fn:45s} {v2s:.2e} -> {v2t:.2e}  ({r:.0f}x)")
