"""
Experiment A — REDO (2026-05-11) under the current Recipe K version.

Reproduces the headline canonical-n table from
`RECIPE_K_MANUSCRIPT_EXPERIMENT_A.md` §6.1 with the current code, which
incorporates the 2026-05-11 Recipe K corrections:
  - Rayleigh-trace VE^D (exact `tr(P_k(p) S)` numerator).
  - PAV monotonization of delta_emp(p).
  - Adaptive BBP floor `p_floor = lambda_{k+1}^2 / (lambda_k^2 + lambda_{k+1}^2)`.

Trim relative to the prior n-scan (435 min / 1320 cells): we run each dataset
ONLY at its design n (no n-scan), consistent with the §6.1 table.
A5 / A5h stay at n=400 (their default in ADVERSARIAL_DATASETS_SEEDED); the
multi-hour n=1600 outliers from the prior n-scan are not revisited.

Scope:
  - 33 unique seeded datasets from `all_seeded_datasets()`:
      12 ADVERSARIAL_DATASETS_SEEDED (A1-A8 + harder variants),
       4 EGAP_ADVERSARIAL_SEEDED   (E1-E4),
       4 RECIPE_K_ADVERSARIAL_SEEDED (R1-R4),
       4 R4_VARIANTS_SEEDED        (R4a-R4d),
       4 HIGH_DIM_SEEDED           (HD1-HD4 @ n=2000),
       5 X_SUITE_SEEDED            (10block, DCSBM, Multiscale, Temporal, Hornfail).
  - 11 benchmarks (current `BENCHMARKS` registry):
      kneedle, eigengap, donoho_gavish, screenot, horn_pa, recipe_K_kcut,
      twonn, mle, danco, evbmf, owen_perry_bicv.
  - 10 seeds per (dataset).
  - Per seed: spectral_pass(S, B=20, smooth_window=10) + full recipe_K() with
    `p_floor_mode='adaptive'` so the manuscript's p^*, p_cv, adaptive floor and
    bulk-edge diagnostics are recorded.

Output: results/experiment_A_REDO_2026-05-11/
  - run.log
  - results.pkl     (per-dataset row: κ̂_r, eigvals, F-stat k_cut, recipe_K diag, all methods)
  - summary.json    (slim numerical summary)
  - figures/AGGREGATE_heatmap.png, AGGREGATE_metrics.png
  - figures/kappa/kappa_<label>.png  (median κ_hat + 2-segment fit overlay)
"""
import os, sys, time, json, pickle, warnings
import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

HERE = os.path.dirname(os.path.abspath(__file__))   # RECIPE_K/exp/
ROOT = os.path.dirname(HERE)                          # RECIPE_K/
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import spectral_pass, recipe_K, env_info
from _benchmarks import BENCHMARKS
from _synthetic_extras import all_seeded_datasets


RESULTS_DIR = os.path.join(ROOT, "output", "experiment_A")
os.makedirs(RESULTS_DIR, exist_ok=True)
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
KAPPA_FIG_DIR = os.path.join(FIG_DIR, "kappa")
os.makedirs(KAPPA_FIG_DIR, exist_ok=True)


LOG = open(os.path.join(RESULTS_DIR, "run.log"), "w", buffering=1)
def log(msg):
    print(msg, flush=True); LOG.write(msg + "\n")


log(f"=== Experiment A REDO 2026-05-11 — start {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
with open(os.path.join(RESULTS_DIR, "env.txt"), "w") as f:
    json.dump(env_info(), f, indent=2)


DATASETS = all_seeded_datasets()
N_SEEDS = 10
METHOD_ORDER = [bn for bn, _ in BENCHMARKS]
log(f"datasets={len(DATASETS)}  methods={len(METHOD_ORDER)}  seeds={N_SEEDS}")
log(f"methods: {METHOD_ORDER}")

METHOD_COLORS = {
    "kneedle":        "C1",
    "eigengap":       "C3",
    "donoho_gavish":  "C4",
    "screenot":       "C5",
    "horn_pa":        "C6",
    "recipe_K_kcut":  "C7",
    "twonn":          "C8",
    "mle":            "C9",
    "danco":          "tab:olive",
    "evbmf":          "tab:brown",
    "owen_perry_bicv":"tab:cyan",
}


# ============================================================
# Per-dataset processing
# ============================================================
def process_dataset(label, build_fn, ds_idx, total, t_global):
    log(f"\n{'='*70}\n[{ds_idx+1}/{total}] {label}  ({time.time()-t_global:.0f}s elapsed)\n{'='*70}")
    seed_kappa = []
    seed_eigs = []
    seed_kcut = []
    seed_klist = []
    method_results = {m: [] for m in METHOD_ORDER}
    recipe_K_per_seed = []
    true_rank = None
    n = None

    for seed in range(N_SEEDS):
        t_seed = time.time()
        try:
            out = build_fn(seed)
            S, true = out if isinstance(out, tuple) else (out, None)
        except Exception as e:
            log(f"  seed {seed}: GENERATOR FAILED {e}")
            continue

        if true_rank is None:
            true_rank = true
            n = S.shape[0]

        try:
            spec = spectral_pass(S, B=20, smooth_window=10)
            spec["k_cut"] = spec["k_smooth"]
        except Exception as e:
            log(f"  seed {seed}: SPECTRAL_PASS FAILED {e}")
            continue

        seed_kappa.append(np.asarray(spec["kappa_hat"]))
        seed_eigs.append(np.asarray(spec["evals_ref"]))
        seed_klist.append(np.asarray(spec["k_list"]))
        seed_kcut.append(int(spec["k_smooth"]))

        # --- Full recipe_K with adaptive BBP floor + Rayleigh-trace VE^D + PAV ---
        try:
            rk = recipe_K(spec, delta=0.1, k_cv=5,
                          p_floor_mode="adaptive",
                          use_rayleigh_trace=True,
                          apply_pav=True,
                          tr_S_mode="full")
            # Trim large arrays out before storage (keep scalar diagnostics).
            rk_slim = {k: v for k, v in rk.items()
                       if k not in {"delta_emp_raw", "delta_emp", "delta_emp_iso", "p_grid"}}
            recipe_K_per_seed.append(rk_slim)
        except Exception as e:
            log(f"  seed {seed}: RECIPE_K FAILED {e}")
            recipe_K_per_seed.append(None)

        # --- All benchmarks (sharing the spectral_pass) ---
        for bn, bfn in BENCHMARKS:
            try:
                r = bfn(S, spec=spec) if bn == "recipe_K_kcut" else bfn(S)
                method_results[bn].append(int(r["k_hat"]))
            except Exception:
                method_results[bn].append(-1)

        log(f"  seed {seed}: {time.time() - t_seed:.1f}s  k_cut={spec['k_smooth']}  "
            f"p*={recipe_K_per_seed[-1].get('p_star', float('nan')):.3f}  "
            f"p_floor={recipe_K_per_seed[-1].get('p_floor', float('nan')):.3f}  "
            f"floor_bind={recipe_K_per_seed[-1].get('floor_binding', None)}"
            if recipe_K_per_seed[-1] is not None else
            f"  seed {seed}: {time.time() - t_seed:.1f}s  k_cut={spec['k_smooth']}  (recipe_K failed)")

    log(f"  {label}: true={true_rank}, n={n}, seeds_done={len(seed_kappa)}/{N_SEEDS}")
    for m in METHOD_ORDER:
        arr = np.array(method_results[m])
        valid = arr[arr > 0]
        if len(valid) > 0:
            mean = float(np.mean(valid))
            sem = float(np.std(valid, ddof=1) / np.sqrt(len(valid))) if len(valid) > 1 else 0.0
            log(f"    {m:18s} mean={mean:7.2f} sem={sem:5.2f} n={len(valid)}/{N_SEEDS}")
        else:
            log(f"    {m:18s} all failed")

    if len(seed_kappa) == 0:
        return None

    K_common = min(len(k) for k in seed_kappa)
    seed_kappa = np.array([k[:K_common] for k in seed_kappa])
    seed_eigs = np.array([e[:K_common] for e in seed_eigs])
    k_list = seed_klist[0][:K_common]

    # Aggregate recipe_K diagnostics across seeds.
    def _agg(field, cast=float):
        vals = [cast(rk[field]) for rk in recipe_K_per_seed
                if rk is not None and rk.get(field) is not None]
        if not vals:
            return dict(mean=None, std=None, n=0)
        return dict(mean=float(np.mean(vals)),
                    std=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    n=len(vals))
    rk_agg = dict(
        p_star_raw=_agg("p_star_raw"),
        p_star=_agg("p_star"),
        p_floor=_agg("p_floor"),
        p_cv=_agg("p_cv"),
        delta_eff=_agg("delta_eff"),
        gap_ratio=_agg("gap_ratio"),
        bulk_flatness=_agg("bulk_flatness"),
        floor_binding_rate=float(np.mean([1.0 if (rk is not None and rk.get("floor_binding"))
                                          else 0.0
                                          for rk in recipe_K_per_seed]))
        if recipe_K_per_seed else None,
        cap_binding_rate=float(np.mean([1.0 if (rk is not None and rk.get("cap_binding"))
                                        else 0.0
                                        for rk in recipe_K_per_seed]))
        if recipe_K_per_seed else None,
    )

    row = dict(
        label=label, n=int(n), true_rank=true_rank,
        seed_kappa=seed_kappa, seed_eigs=seed_eigs,
        seed_kcut=np.array(seed_kcut), k_list=k_list,
        method_results=method_results,
        recipe_K_per_seed=recipe_K_per_seed,
        recipe_K_agg=rk_agg,
    )
    plot_kappa(label, row)
    return row


# ============================================================
# Median F-stat 2-segment constant fit
# ============================================================
def fstat_2seg_constants(kappa_med, k_cut):
    if k_cut < 1 or k_cut >= len(kappa_med):
        return float("nan"), float("nan")
    return float(np.mean(kappa_med[:k_cut])), float(np.mean(kappa_med[k_cut:]))


def plot_kappa(label, row):
    seed_kappa = row["seed_kappa"]
    seed_eigs = row["seed_eigs"]
    k_list = row["k_list"]
    seed_kcut = row["seed_kcut"]
    n = row["n"]; true_rank = row["true_rank"]

    kappa_med = np.median(seed_kappa, axis=0)
    kappa_min = np.min(seed_kappa, axis=0)
    kappa_max = np.max(seed_kappa, axis=0)
    eigs_med = np.median(seed_eigs, axis=0)
    eigs_min = np.min(seed_eigs, axis=0)
    eigs_max = np.max(seed_eigs, axis=0)
    k_cut_med = int(np.median(seed_kcut))
    k_cut_mean = float(np.mean(seed_kcut))
    k_cut_std = float(np.std(seed_kcut, ddof=1)) if len(seed_kcut) > 1 else 0.0

    method_means = {}
    for m in METHOD_ORDER:
        arr = np.array(row["method_results"][m])
        valid = arr[arr > 0]
        method_means[m] = float(np.mean(valid)) if len(valid) > 0 else None

    mu_low, mu_high = fstat_2seg_constants(kappa_med, k_cut_med)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Left: eigenspectrum
    ax = axes[0]
    ax.fill_between(k_list, eigs_min, eigs_max, color="C0", alpha=0.15,
                     label="seed min-max")
    ax.plot(k_list, eigs_med, "o-", color="C0", markersize=4, alpha=0.85,
             label="median eigvals")
    if true_rank is not None:
        ax.axvline(true_rank, color="C2", linewidth=2.5, alpha=0.9,
                    label=f"true={true_rank}")
    for method, mean_k in method_means.items():
        if mean_k is not None and mean_k > 0:
            ax.axvline(mean_k, color=METHOD_COLORS.get(method, "k"),
                        linestyle=":", alpha=0.6,
                        label=f"{method}={mean_k:.1f}")
    ax.set_xlabel("rank"); ax.set_ylabel("eigenvalue")
    if eigs_med.min() > 0:
        ax.set_yscale("log")
    ax.set_title(f"{label} — eigenspectrum (n={n}, n_seeds={N_SEEDS})", fontsize=10)
    ax.legend(fontsize=6, loc="upper right", ncol=2); ax.grid(alpha=0.3)

    # Right: κ profile + median F-stat 2-segment constant fit
    ax = axes[1]
    ax.fill_between(k_list, kappa_min, kappa_max, color="C0", alpha=0.15,
                     label="seed min-max")
    ax.plot(k_list, kappa_med, "o-", color="C0", markersize=4, alpha=0.85,
             label=r"median $\hat\kappa_k$")
    ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5,
                label="bulk-asymptote (1.0)")
    for kc in seed_kcut:
        ax.axvline(kc, color="black", alpha=0.15, linewidth=0.8)
    ax.axvline(k_cut_med, color="black", linewidth=2.5, alpha=0.9,
                label=f"F-stat k_cut median={k_cut_med} (mean={k_cut_mean:.1f}, std={k_cut_std:.2f})")
    if not np.isnan(mu_low):
        ax.plot([k_list[0], k_cut_med], [mu_low, mu_low], "-",
                 color="orange", linewidth=2.5, alpha=0.85,
                 label=f"F-stat fit μ_low={mu_low:.3f}")
        ax.plot([k_cut_med, k_list[-1]], [mu_high, mu_high], "-",
                 color="red", linewidth=2.5, alpha=0.85,
                 label=f"F-stat fit μ_high={mu_high:.3f}")
        residual_low_std = float(np.std(kappa_med[:k_cut_med])) if k_cut_med > 1 else 0.0
        residual_high_std = float(np.std(kappa_med[k_cut_med:])) if (len(kappa_med) - k_cut_med) > 1 else 0.0
        ax.fill_between([k_list[0], k_cut_med],
                          [mu_low - residual_low_std]*2,
                          [mu_low + residual_low_std]*2,
                          color="orange", alpha=0.15)
        ax.fill_between([k_cut_med, k_list[-1]],
                          [mu_high - residual_high_std]*2,
                          [mu_high + residual_high_std]*2,
                          color="red", alpha=0.15)

    if true_rank is not None:
        ax.axvline(true_rank, color="C2", linewidth=2.5, alpha=0.9,
                    label=f"true={true_rank}")
    ax.set_xlabel("rank k"); ax.set_ylabel(r"$\hat\kappa_k$")
    seed_cuts_str = sorted(seed_kcut.tolist())
    rk_agg = row.get("recipe_K_agg", {})
    pstar_txt = ""
    if rk_agg.get("p_star", {}).get("mean") is not None:
        pstar_txt = (f"  p*={rk_agg['p_star']['mean']:.3f}±{rk_agg['p_star']['std']:.3f}"
                     f"  p_floor={rk_agg['p_floor']['mean']:.3f}"
                     f"  floor_bind={rk_agg['floor_binding_rate']:.1f}")
    ax.set_title(f"κ profile + 2-segment constant fit (cuts: {seed_cuts_str}){pstar_txt}",
                  fontsize=8)
    ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=0.3)

    fig.suptitle(f"{label}", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(KAPPA_FIG_DIR, f"kappa_{label}.png"),
                 dpi=110, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main loop
# ============================================================
all_rows = []
t0 = time.time()
for ds_idx, (label, build_fn) in enumerate(DATASETS):
    row = process_dataset(label, build_fn, ds_idx, len(DATASETS), t0)
    if row is not None:
        all_rows.append(row)
    with open(os.path.join(RESULTS_DIR, "results.pkl"), "wb") as f:
        pickle.dump(all_rows, f)


# ============================================================
# Slim summary JSON
# ============================================================
slim = {}
for r in all_rows:
    slim[r["label"]] = dict(
        true_rank=r["true_rank"], n=r["n"],
        kcut_seeds=r["seed_kcut"].tolist(),
        kcut_median=int(np.median(r["seed_kcut"])),
        kcut_mean=float(np.mean(r["seed_kcut"])),
        kcut_std=float(np.std(r["seed_kcut"], ddof=1)) if len(r["seed_kcut"]) > 1 else 0.0,
        recipe_K_agg=r.get("recipe_K_agg"),
        per_method={},
    )
    for m, arr in r["method_results"].items():
        a = np.array(arr); valid = a[a > 0]
        slim[r["label"]]["per_method"][m] = dict(
            raw=list(a),
            mean=float(np.mean(valid)) if len(valid) > 0 else None,
            std=float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
            n_valid=int(len(valid)),
        )
with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
    json.dump(slim, f, indent=2, default=str)


# ============================================================
# Aggregate plots
# ============================================================
def plot_aggregate_heatmap(rows, out_path):
    n_m = len(METHOD_ORDER); n_d = len(rows)
    labels = [r["label"] for r in rows]
    mat = np.full((n_m, n_d), np.nan)
    for j, r in enumerate(rows):
        true = r["true_rank"]
        for i, m in enumerate(METHOD_ORDER):
            arr = np.array(r["method_results"][m]); valid = arr[arr > 0]
            if len(valid) > 0 and true is not None:
                mat[i, j] = np.mean(valid - true)
    fig, ax = plt.subplots(figsize=(max(14, 0.6 * n_d + 2), 0.7 * n_m + 2))
    finite = mat[np.isfinite(mat)]
    vmax = max(np.nanpercentile(np.abs(finite), 95), 1) if finite.size else 1
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n_d))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_m)); ax.set_yticklabels(METHOD_ORDER, fontsize=10)
    ax.set_title(f"mean(k_hat − true), n_seeds={N_SEEDS}")
    for i in range(n_m):
        for j in range(n_d):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.0f}", ha="center", va="center",
                         color="white" if abs(v) > vmax * 0.5 else "black",
                         fontsize=8)
            else:
                ax.text(j, i, "NaN", ha="center", va="center",
                         color="orange", fontsize=8)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_aggregate_metrics(rows, out_path):
    metrics = {m: dict(mae=[], within1=0, catastrophic=0, total=0) for m in METHOD_ORDER}
    for r in rows:
        true = r["true_rank"]
        if true is None: continue
        for m in METHOD_ORDER:
            arr = np.array(r["method_results"][m]); valid = arr[arr > 0]
            for v in valid:
                e = abs(int(v) - int(true))
                metrics[m]["mae"].append(e)
                if e <= 1: metrics[m]["within1"] += 1
                if e > 20: metrics[m]["catastrophic"] += 1
                metrics[m]["total"] += 1
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    means = [np.mean(metrics[m]["mae"]) if metrics[m]["mae"] else 0 for m in METHOD_ORDER]
    sems = [np.std(metrics[m]["mae"], ddof=1) / np.sqrt(len(metrics[m]["mae"]))
            if len(metrics[m]["mae"]) > 1 else 0 for m in METHOD_ORDER]
    axes[0].bar(METHOD_ORDER, means, yerr=sems, color="C0", alpha=0.7, capsize=4)
    axes[0].set_title("mean |k_hat − true|"); axes[0].set_ylabel("MAE")
    axes[0].tick_params(axis="x", rotation=30); axes[0].grid(axis="y", alpha=0.3)
    fracs = [metrics[m]["within1"] / max(metrics[m]["total"], 1) for m in METHOD_ORDER]
    axes[1].bar(METHOD_ORDER, fracs, color="C2", alpha=0.7)
    for i, f in enumerate(fracs):
        axes[1].text(i, f + 0.01, f"{f:.0%}", ha="center", fontsize=9)
    axes[1].set_title("within ±1 of true"); axes[1].set_ylabel("hit rate")
    axes[1].tick_params(axis="x", rotation=30); axes[1].set_ylim(0, 1.05); axes[1].grid(axis="y", alpha=0.3)
    cnts = [metrics[m]["catastrophic"] for m in METHOD_ORDER]
    axes[2].bar(METHOD_ORDER, cnts, color="C3", alpha=0.7)
    for i, c in enumerate(cnts):
        axes[2].text(i, c + 0.5, str(c), ha="center", fontsize=9)
    axes[2].set_title("catastrophic misses (|err|>20)"); axes[2].set_ylabel("count")
    axes[2].tick_params(axis="x", rotation=30); axes[2].grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


plot_aggregate_heatmap(all_rows, os.path.join(FIG_DIR, "AGGREGATE_heatmap.png"))
plot_aggregate_metrics(all_rows, os.path.join(FIG_DIR, "AGGREGATE_metrics.png"))
log(f"\nAggregate figures saved.")


# ============================================================
# Final per-dataset table
# ============================================================
log(f"\n=== Done {time.strftime('%Y-%m-%d %H:%M:%S')}  total {(time.time() - t0)/60:.1f} min ===")
log(f"\nPer-dataset, mean k_hat over {N_SEEDS} seeds:")
header = (f"{'dataset':28s}|{'n':>5s}|{'true':>5s}|"
          + "|".join(f"{m[:10]:>10s}" for m in METHOD_ORDER))
log(header); log("-" * len(header))
for r in all_rows:
    parts = [f"{r['label']:28s}", f"{r['n']:>5d}", f"{str(r['true_rank']):>5s}"]
    for m in METHOD_ORDER:
        arr = np.array(r["method_results"][m]); valid = arr[arr > 0]
        v = float(np.mean(valid)) if len(valid) > 0 else float("nan")
        parts.append(f"{v:>10.1f}" if not np.isnan(v) else f"{'NaN':>10s}")
    log("|".join(parts))

# Recipe-K diagnostics table.
log(f"\nRecipe K diagnostics (mean ± std over {N_SEEDS} seeds):")
header = f"{'dataset':28s}|{'k_cut':>6s}|{'p*_raw':>10s}|{'p*':>10s}|{'p_floor':>10s}|{'floor_bind':>10s}|{'p_cv':>10s}|{'cap_bind':>10s}"
log(header); log("-" * len(header))
for r in all_rows:
    rk = r.get("recipe_K_agg", {})
    def _fmt(d):
        if d is None or d.get("mean") is None: return "NaN"
        return f"{d['mean']:.3f}±{d['std']:.3f}"
    line = (f"{r['label']:28s}"
            f"|{int(np.median(r['seed_kcut'])):>6d}"
            f"|{_fmt(rk.get('p_star_raw')):>10s}"
            f"|{_fmt(rk.get('p_star')):>10s}"
            f"|{_fmt(rk.get('p_floor')):>10s}"
            f"|{rk.get('floor_binding_rate', float('nan')):>10.2f}"
            f"|{_fmt(rk.get('p_cv')):>10s}"
            f"|{rk.get('cap_binding_rate', float('nan')):>10.2f}")
    log(line)

LOG.close()
