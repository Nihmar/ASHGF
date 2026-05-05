"""analyze_rust_bench.py — Analizza i risultati del benchmark Rust (tesi + ASGF-2S/2SW)."""

import os
import sys

import numpy as np
import pandas as pd

RESULTS_DIR = "results/bench_full"
DIMS = [10, 100]
THESIS_ALGOS = ["GD", "SGES", "ASGF", "ASHGF", "ASEBO"]
NEW_ALGOS = ["ASGF-2S", "ASGF-2SW"]
ALL_ALGOS = THESIS_ALGOS + NEW_ALGOS


def load_dim(dim: int) -> pd.DataFrame:
    path = os.path.join(RESULTS_DIR, f"dim_{dim}", "benchmark_results.csv")
    return pd.read_csv(path)


def analyze_dim(df: pd.DataFrame, dim: int):
    print(f"\n{'=' * 80}")
    print(f"  ANALISI DIMENSIONE {dim}")
    print(f"{'=' * 80}")

    # ---- 1. Classifica per best_value medio (normalizzato per funzione) ----
    print(f"\n--- Riepilogo per algoritmo (d={dim}) ---")
    print(
        f"{'Algo':<12} {'Best medio':>14} {'Final medio':>14} {'Iter medi':>10} {'Convergenza %':>13}"
    )
    print("-" * 65)

    for algo in ALL_ALGOS:
        sub = df[df["algorithm"] == algo]
        if len(sub) == 0:
            continue
        # Replace inf/nan with large numbers for ranking
        best_vals = sub["best_value"].replace([np.inf, -np.inf], np.nan).dropna()
        final_vals = sub["final_value"].replace([np.inf, -np.inf], np.nan).dropna()
        converged = sub["converged"].mean() * 100
        iter_mean = sub["iterations"].mean()

        best_mean = best_vals.mean()
        final_mean = final_vals.mean()

        print(
            f"{algo:<12} {best_mean:>14.4e} {final_mean:>14.4e} {iter_mean:>10.1f} {converged:>12.1f}%"
        )

    # ---- 2. Performance profile: per ogni funzione, chi è migliore ----
    print(
        f"\n--- Confronto diretto: ASGF-2S vs ASGF-2SW vs ASGF vs ASHGF (best_value) ---"
    )
    functions = sorted(df["function"].unique())

    wins_2s = 0
    wins_2sw = 0
    wins_asgf = 0
    wins_ashgf = 0
    total_comparable = 0

    for func in functions:
        sub = df[df["function"] == func]
        bests = {}
        for algo in ["ASGF-2S", "ASGF-2SW", "ASGF", "ASHGF"]:
            row = sub[sub["algorithm"] == algo]
            if len(row) > 0:
                val = row["best_value"].values[0]
                if np.isfinite(val):
                    bests[algo] = val

        if len(bests) >= 2:
            total_comparable += 1
            best_algo = min(bests, key=bests.get)
            if best_algo == "ASGF-2S":
                wins_2s += 1
            elif best_algo == "ASGF-2SW":
                wins_2sw += 1
            elif best_algo == "ASGF":
                wins_asgf += 1
            elif best_algo == "ASHGF":
                wins_ashgf += 1

    print(f"  Funzioni comparabili: {total_comparable}")
    print(
        f"  ASGF-2S  vince in {wins_2s} ({100 * wins_2s / total_comparable:.1f}%) funzioni"
    )
    print(
        f"  ASGF-2SW vince in {wins_2sw} ({100 * wins_2sw / total_comparable:.1f}%) funzioni"
    )
    print(
        f"  ASGF     vince in {wins_asgf} ({100 * wins_asgf / total_comparable:.1f}%) funzioni"
    )
    print(
        f"  ASHGF    vince in {wins_ashgf} ({100 * wins_ashgf / total_comparable:.1f}%) funzioni"
    )

    # ---- 3. Top-5 funzioni dove ASGF-2S/2SW battono tutti ----
    print(
        f"\n--- Top-10 funzioni dove ASGF-2S/2SW migliorano ASGF (per best_value) ---"
    )

    improvements = []
    for func in functions:
        sub = df[df["function"] == func]
        asgf_val = None
        s2_val = None
        sw_val = None
        for _, row in sub.iterrows():
            if row["algorithm"] == "ASGF":
                asgf_val = row["best_value"]
            elif row["algorithm"] == "ASGF-2S":
                s2_val = row["best_value"]
            elif row["algorithm"] == "ASGF-2SW":
                sw_val = row["best_value"]

        if (
            asgf_val
            and s2_val
            and np.isfinite(asgf_val)
            and np.isfinite(s2_val)
            and asgf_val > 0
            and s2_val > 0
        ):
            ratio_2s = asgf_val / s2_val
            improvements.append((func, "2S", ratio_2s, asgf_val, s2_val))
        if (
            asgf_val
            and sw_val
            and np.isfinite(asgf_val)
            and np.isfinite(sw_val)
            and asgf_val > 0
            and sw_val > 0
        ):
            ratio_sw = asgf_val / sw_val
            improvements.append((func, "2SW", ratio_sw, asgf_val, sw_val))

    improvements.sort(key=lambda x: x[2], reverse=True)
    print(
        f"{'Funzione':<40} {'Variante':<8} {'ASGF':>14} {'Variante':>14} {'Ratio':>10}"
    )
    print("-" * 90)
    for func, var, ratio, asgf_v, var_v in improvements[:15]:
        print(f"{func:<40} {var:<8} {asgf_v:>14.4e} {var_v:>14.4e} {ratio:>10.2f}x")

    # ---- 4. Confronto convergenza (iterazioni) ----
    print(f"\n--- Iterazioni medie per convergenza ---")
    for algo in ALL_ALGOS:
        sub = df[df["algorithm"] == algo]
        conv = sub[sub["converged"] == True]
        if len(conv) > 0:
            print(
                f"  {algo:<12}: {conv['iterations'].mean():>8.1f} iter (converged in {len(conv)}/{len(sub)} funzioni)"
            )

    return {
        "dim": dim,
        "wins_2s": wins_2s,
        "wins_2sw": wins_2sw,
        "wins_asgf": wins_asgf,
        "wins_ashgf": wins_ashgf,
        "total": total_comparable,
    }


def main():
    all_stats = []
    for dim in DIMS:
        df = load_dim(dim)
        stats = analyze_dim(df, dim)
        all_stats.append(stats)

    # ---- Confronto cross-dimensionale ----
    print(f"\n{'=' * 80}")
    print(f"  RIEPILOGO CROSS-DIMENSIONALE")
    print(f"{'=' * 80}")
    print(
        f"{'Dim':<8} {'2S wins':<12} {'2SW wins':<12} {'ASGF wins':<12} {'ASHGF wins':<12} {'Tot comparable':<16}"
    )
    print("-" * 72)
    for s in all_stats:
        t = s["total"]
        print(
            f"{s['dim']:<8} {s['wins_2s']} ({100 * s['wins_2s'] / t:5.1f}%)   "
            f"{s['wins_2sw']} ({100 * s['wins_2sw'] / t:5.1f}%)   "
            f"{s['wins_asgf']} ({100 * s['wins_asgf'] / t:5.1f}%)   "
            f"{s['wins_ashgf']} ({100 * s['wins_ashgf'] / t:5.1f}%)   "
            f"{t}"
        )

    # ---- Classifica finale ----
    print(f"\n--- CLASSIFICA FINALE (basata su best_value medio) ---")
    all_dfs = []
    for dim in DIMS:
        df = load_dim(dim)
        df["dim"] = dim
        all_dfs.append(df)
    full = pd.concat(all_dfs, ignore_index=True)

    # Normalize per function and dimension: rank within each (function, dim) group
    full["rank"] = full.groupby(["function", "dim"])["best_value"].rank(method="min")

    print(
        f"{'Algo':<12} {'Rank medio':>12} {'Best medio (log)':>16} {'% Converged':>12}"
    )
    print("-" * 55)
    for algo in ALL_ALGOS:
        sub = full[full["algorithm"] == algo].copy()
        if len(sub) == 0:
            continue
        rank_mean = sub["rank"].mean()
        # Log-mean of best (ignore non-positive)
        best_pos = sub["best_value"][sub["best_value"] > 0]
        if len(best_pos) > 0:
            log_mean = np.exp(np.log(best_pos).mean())
        else:
            log_mean = np.nan
        conv_pct = sub["converged"].mean() * 100
        print(f"{algo:<12} {rank_mean:>12.2f} {log_mean:>16.4e} {conv_pct:>11.1f}%")


if __name__ == "__main__":
    main()
