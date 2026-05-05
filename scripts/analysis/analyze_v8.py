import csv, os

algos = ["ASGF", "ASGF-2F", "ASGF-2S"]
names = {"ASGF":"baseline", "ASGF-2F":"2F", "ASGF-2S":"2S safe"}

for dim in [10, 100]:
    base = f"results_v8/dim_{dim}/csv"
    if not os.path.isdir(base): continue
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
    print(f"{'Algo':<10s} {'wins':>5s} {'loss':>5s} {'ties':>5s}  {'best':>5s}")
    print("-" * 36)

    for v in ["ASGF-2F", "ASGF-2S"]:
        wins = sum(1 for f in data if data[f].get(v,1e99) < data[f].get("ASGF",1e99))
        loss = sum(1 for f in data if data[f].get(v,1e99) > data[f].get("ASGF",1e99))
        ties = sum(1 for f in data if data[f].get(v,1e99) == data[f].get("ASGF",1e99))
        best = sum(1 for f in data if min((data[f].get(a,1e99),a) for a in algos)[1]==v)
        print(f"{names[v]:<10s} {wins:5d} {loss:5d} {ties:5d}  {best:5d}")

    s_vs_f = sum(1 for f in data if data[f].get("ASGF-2S",1e99) < data[f].get("ASGF-2F",1e99))
    s_vs_f_w = sum(1 for f in data if data[f].get("ASGF-2S",1e99) > data[f].get("ASGF-2F",1e99))
    s_vs_f_t = sum(1 for f in data if data[f].get("ASGF-2S",1e99) == data[f].get("ASGF-2F",1e99))
    print(f"  2S vs 2F:  better={s_vs_f}  worse={s_vs_f_w}  ties={s_vs_f_t}")

    # Key improvements
    better = [(f, data[f]["ASGF-2F"], data[f]["ASGF-2S"])
              for f in data
              if data[f].get("ASGF-2S",1e99) < data[f].get("ASGF-2F",1e99)]
    better.sort(key=lambda x: x[1]/x[2] if x[2]!=0 else 999, reverse=True)
    if better:
        print(f"\n  Top 2S improvements over 2F:")
        for fn, v2f, v2s in better[:5]:
            ratio = v2f/v2s if v2s!=0 else 999
            print(f"    {fn:45s} {v2f:.2e} -> {v2s:.2e}  ({ratio:.1f}x)")
