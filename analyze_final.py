import csv, os
base = r"results_final"
algos = ["ASGF","ASHGF","ASGF-LS","ASGF-2X","ASGF-HX"]
names = {"ASGF":"base","ASHGF":"thesis","ASGF-LS":"LS greedy","ASGF-2X":"Try-2x","ASGF-HX":"Hybrid"}

data = {}
for fn in sorted(os.listdir(base)):
    if not fn.endswith(".csv"): continue
    with open(os.path.join(base,fn)) as f: rows = list(csv.reader(f))
    if len(rows)<2: continue
    stem = fn.replace(".csv","")
    for a in algos:
        if stem.startswith(a+"_"): data.setdefault(stem[len(a)+1:],{})[a]=float(rows[-1][1]); break

print(f"{'Algorithm':<15s} {'wins':>5s} {'loss':>5s} {'ties':>5s}  {'best':>5s}  extra_eval")
print("-"*55)
for v in [a for a in algos if a!="ASGF"]:
    wins=sum(1 for f in data if data[f].get(v,1e99)<data[f].get("ASGF",1e99))
    loss=sum(1 for f in data if data[f].get(v,1e99)>data[f].get("ASGF",1e99))
    ties=sum(1 for f in data if data[f].get(v,1e99)==data[f].get("ASGF",1e99))
    best=sum(1 for f in data if min((data[f].get(a,1e99),a) for a in algos)[1]==v)
    cost={"ASGF-LS":"4","ASGF-2X":"1","ASGF-HX":"2","ASHGF":"0"}[v]
    print(f"{names[v]:<15s} {wins:5d} {loss:5d} {ties:5d}  {best:5d}  {cost}")
