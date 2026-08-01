#!/usr/bin/env python3
"""
Coordinate-shuffle permutation test of hypoxia front organization.

Note: shuffling coordinates among a fixed set of positions is mathematically
equivalent to shuffling labels on a fixed point cloud / kNN graph. We implement
the coordinate shuffle explicitly for clarity, but compute statistics on the
fixed positions with permuted labels (exact equivalence, far faster).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors

OUT = Path("/ix1/ylee/shared/MC38_Hypoxia_001/reseq_finetune/deep_dive/spatial_niches/front_figures")
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260801)
N_PERM = 2000
K = 30
ENTER = {"entering_hypoxia", "entering_deep_hypoxia"}


def load():
    adata = sc.read_h5ad("/ix1/ylee/shared/MC38_Hypoxia_001/reseq_finetune/mc38_tumor_reseq_finetuned.h5ad")
    dyn = pd.read_csv(
        "/ix1/ylee/shared/MC38_Hypoxia_001/reseq_finetune/hypoxia_velocity/hypoxia_dynamics_per_cell.csv"
    ).set_index("barcode")
    adata = adata[adata.obs_names.intersection(dyn.index)].copy()
    for c in ["hypoxia_dynamics", "hypoxia_score", "v_hypoxia"]:
        adata.obs[c] = dyn.loc[adata.obs_names, c].values
    if "spatial" not in adata.obsm:
        adata.obsm["spatial"] = adata.obs[["spatial_coordinate_x", "spatial_coordinate_y"]].to_numpy()
    adata.obs["is_tumor"] = (
        adata.obs["cell_type_finetuned"].astype(str).str.contains("Tumor", case=False)
        | adata.obs["cell_type"].astype(str).str.contains("Tumor", case=False)
    ).to_numpy()
    return adata


def metrics_from_labels(labs, tumor, knn_idx):
    """Compute front metrics given labels on fixed positions / fixed knn."""
    enter = np.isin(labs, list(ENTER)) & tumor
    deep = (labs == "entering_deep_hypoxia") & tumor
    exit_ = (labs == "exiting_hypoxia") & tumor
    persist = (labs == "persistent_hypoxia") & tumor
    norm = (labs == "normoxic_stable") & tumor

    def enr(src, tgt):
        if src.sum() < 5 or tgt.sum() < 5:
            return np.nan
        frac = tgt[knn_idx[src]].mean()
        return float(frac / max(tgt.mean(), 1e-12))

    def contact(src, tgt):
        if src.sum() < 5 or tgt.sum() < 5:
            return np.nan
        return float(tgt[knn_idx[src]].any(axis=1).mean())

    # front ribbon: mean sqrt(local enter * local exit) over all positions
    ef = enter.astype(float)[knn_idx].mean(axis=1)
    xf = exit_.astype(float)[knn_idx].mean(axis=1)
    # include self in local density approximately by also counting own label
    # (knn_idx excludes self; add self contribution)
    ef = (ef * K + enter.astype(float)) / (K + 1)
    xf = (xf * K + exit_.astype(float)) / (K + 1)
    front = float(np.mean(np.sqrt(ef * xf)))

    return {
        "enr_deep_exit": enr(deep, exit_),
        "enr_enter_exit": enr(enter, exit_),
        "enr_exit_deep": enr(exit_, deep),
        "enr_persist_exit": enr(persist, exit_),
        "enr_norm_exit": enr(norm, exit_),
        "contact_deep_exit": contact(deep, exit_),
        "contact_enter_exit": contact(enter, exit_),
        "front_score": front,
        "n_deep": int(deep.sum()),
        "n_exit": int(exit_.sum()),
        "n_enter": int(enter.sum()),
    }


def permute_labels_equivalent_to_coord_shuffle(labs, tumor, mode):
    """
    Exact equivalents of coordinate shuffles on a fixed point cloud:
      coord_all    == shuffle all labels across all positions
      coord_tumor  == shuffle labels among tumor positions only
                     (non-tumor labels/positions fixed)
      label_tumor  == same as coord_tumor (explicit name)
    """
    out = labs.copy()
    if mode == "coord_all":
        out = RNG.permutation(labs)
    elif mode in ("coord_tumor", "label_tumor"):
        tidx = np.where(tumor)[0]
        out[tidx] = RNG.permutation(labs[tidx])
    else:
        raise ValueError(mode)
    return out


def run_sample(adata, sample, mode):
    m = (adata.obs["sample"] == sample).to_numpy() & np.isfinite(adata.obsm["spatial"]).all(1)
    sub = adata[m]
    xy = np.asarray(sub.obsm["spatial"], float)
    labs = sub.obs["hypoxia_dynamics"].astype(str).to_numpy()
    tumor = sub.obs["is_tumor"].to_numpy()

    nn = NearestNeighbors(n_neighbors=min(K + 1, len(xy))).fit(xy)
    knn_idx = nn.kneighbors(xy, return_distance=False)[:, 1:]

    obs = metrics_from_labels(labs, tumor, knn_idx)
    keys = [k for k in obs if k.startswith("enr_") or k.startswith("contact_") or k == "front_score"]
    null = {k: np.empty(N_PERM, float) for k in keys}

    for i in range(N_PERM):
        labs_p = permute_labels_equivalent_to_coord_shuffle(labs, tumor, mode)
        st = metrics_from_labels(labs_p, tumor, knn_idx)
        for k in keys:
            null[k][i] = st[k]

    out = {
        "sample": sample,
        "mode": mode,
        "n_perm": N_PERM,
        "k": K,
        "equivalence_note": (
            "Shuffling coordinates among a fixed set of tissue positions is exactly "
            "equivalent to shuffling labels on the fixed point cloud / kNN graph."
        ),
        "observed": obs,
        "null": {},
    }
    for k in keys:
        a = null[k]
        a = a[np.isfinite(a)]
        o = obs[k]
        p = float((np.sum(a >= o) + 1) / (len(a) + 1)) if np.isfinite(o) and len(a) else np.nan
        out["null"][k] = {
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "q95": float(np.quantile(a, 0.95)),
            "q99": float(np.quantile(a, 0.99)),
            "p_ge_obs": p,
            "z": float((o - np.mean(a)) / (np.std(a) + 1e-12)),
        }
    return out, null


def main():
    print("loading…")
    adata = load()
    results = []
    null_store = {}
    # coord_tumor and label_tumor are identical; run both names once each for clarity in output
    for sample in ["E14S", "E15S"]:
        for mode in ["coord_all", "coord_tumor"]:
            print(f"running {sample} {mode} ({N_PERM} perms)…")
            r, nulls = run_sample(adata, sample, mode)
            results.append(r)
            null_store[(sample, mode)] = nulls
            print(
                f"  deep→exit obs={r['observed']['enr_deep_exit']:.3f} "
                f"null={r['null']['enr_deep_exit']['mean']:.3f} "
                f"p={r['null']['enr_deep_exit']['p_ge_obs']:.4g} "
                f"z={r['null']['enr_deep_exit']['z']:.2f}"
            )
            print(
                f"  front_score obs={r['observed']['front_score']:.5f} "
                f"null={r['null']['front_score']['mean']:.5f} "
                f"p={r['null']['front_score']['p_ge_obs']:.4g} "
                f"z={r['null']['front_score']['z']:.2f}"
            )

    # also store label_tumor as alias of coord_tumor results for documentation
    for sample in ["E14S", "E15S"]:
        base = next(r for r in results if r["sample"] == sample and r["mode"] == "coord_tumor")
        alias = json.loads(json.dumps(base))
        alias["mode"] = "label_tumor"
        alias["alias_of"] = "coord_tumor"
        results.append(alias)
        null_store[(sample, "label_tumor")] = null_store[(sample, "coord_tumor")]

    (OUT / "coord_shuffle_front_test.json").write_text(json.dumps(results, indent=2))

    rows = []
    for r in results:
        for metric in [
            "enr_deep_exit",
            "enr_enter_exit",
            "enr_exit_deep",
            "enr_persist_exit",
            "enr_norm_exit",
            "contact_deep_exit",
            "front_score",
        ]:
            rows.append(
                {
                    "sample": r["sample"],
                    "mode": r["mode"],
                    "metric": metric,
                    "observed": r["observed"][metric],
                    "null_mean": r["null"][metric]["mean"],
                    "null_q95": r["null"][metric]["q95"],
                    "z": r["null"][metric]["z"],
                    "p": r["null"][metric]["p_ge_obs"],
                }
            )
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "coord_shuffle_front_test.csv", index=False)
    print("\nKEY RESULTS")
    print(tab[tab.metric.isin(["enr_deep_exit", "front_score", "contact_deep_exit", "enr_norm_exit"])].to_string(index=False))

    # enrichment null histograms
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    for row, sample in enumerate(["E14S", "E15S"]):
        for col, mode in enumerate(["coord_all", "coord_tumor"]):
            ax = axes[row, col]
            r = next(x for x in results if x["sample"] == sample and x["mode"] == mode)
            a = null_store[(sample, mode)]["enr_deep_exit"]
            a = a[np.isfinite(a)]
            o = r["observed"]["enr_deep_exit"]
            ax.hist(a, bins=50, color="#bdc3c7", density=True, alpha=0.9)
            ax.axvline(o, color="#7b241c", lw=2.4, label=f"obs={o:.2f}")
            ax.axvline(1.0, color="#7f8c8d", ls="--", lw=1, label="random=1")
            p = r["null"]["enr_deep_exit"]["p_ge_obs"]
            z = r["null"]["enr_deep_exit"]["z"]
            title = "shuffle all coordinates" if mode == "coord_all" else "shuffle tumor coordinates only"
            ax.set_title(f"{sample}: {title}\np={p:.1e}  z={z:.1f}", fontsize=9)
            ax.set_xlabel("deep→exit neighbor enrichment")
            if col == 0:
                ax.set_ylabel("Density")
            ax.legend(fontsize=7, frameon=False)
    fig.suptitle(
        "Coordinate-shuffle permutation test of hypoxia front adjacency\n"
        "(enter-deep cells neighboring exiting cells)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT / "coord_shuffle_deep_exit_enrichment.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # front score
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    for row, sample in enumerate(["E14S", "E15S"]):
        for col, mode in enumerate(["coord_all", "coord_tumor"]):
            ax = axes[row, col]
            r = next(x for x in results if x["sample"] == sample and x["mode"] == mode)
            a = null_store[(sample, mode)]["front_score"]
            a = a[np.isfinite(a)]
            o = r["observed"]["front_score"]
            ax.hist(a, bins=50, color="#bdc3c7", density=True, alpha=0.9)
            ax.axvline(o, color="#e67e22", lw=2.4, label=f"obs={o:.4f}")
            p = r["null"]["front_score"]["p_ge_obs"]
            z = r["null"]["front_score"]["z"]
            title = "shuffle all coordinates" if mode == "coord_all" else "shuffle tumor coordinates only"
            ax.set_title(f"{sample}: {title}\np={p:.1e}  z={z:.1f}", fontsize=9)
            ax.set_xlabel("front score  mean√(local enter × local exit)")
            if col == 0:
                ax.set_ylabel("Density")
            ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Coordinate-shuffle test of front-ribbon score", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "coord_shuffle_front_score.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # summary
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    plot_df = tab[(tab.metric == "enr_deep_exit") & (tab.mode.isin(["coord_all", "coord_tumor"]))].copy()
    y = np.arange(len(plot_df))
    ax.hlines(y, plot_df.null_mean, plot_df.observed, color="#95a5a6", lw=2)
    ax.scatter(plot_df.null_mean, y, c="#bdc3c7", s=55, label="null mean", zorder=3)
    cols = ["#c0392b" if p < 0.05 else "#7f8c8d" for p in plot_df.p]
    ax.scatter(plot_df.observed, y, c=cols, s=70, label="observed", zorder=3)
    ax.axvline(1, color="#888", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [
            f"{r.sample} | {'all coords' if r.mode == 'coord_all' else 'tumor coords'}"
            for r in plot_df.itertuples()
        ],
        fontsize=9,
    )
    for yi, r in zip(y, plot_df.itertuples()):
        ax.text(max(r.observed, r.null_mean) + 0.04, yi, f"p={r.p:.1e}  z={r.z:.1f}", va="center", fontsize=8)
    ax.set_xlabel("deep → exit neighbor enrichment (obs/exp)")
    ax.set_title("Does coordinate shuffling destroy the hypoxia front? (yes ⇒ organization is real)")
    ax.legend(frameon=False, loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT / "coord_shuffle_summary.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # explanation figure
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))
    # observed cartoon
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_title("Observed: a front", fontsize=11)
    for x0, col in [(1.2, "#27ae60"), (3.2, "#2980b9"), (5.1, "#8e44ad"), (6.8, "#c0392b"), (8.3, "#7b241c")]:
        ax.add_patch(plt.Rectangle((x0, 1.5), 1.5, 7, color=col, alpha=0.85))
    ax.text(5, 0.5, "exit abuts enter along a continuum", ha="center", fontsize=8, color="#7f8c8d")
    ax.axis("off")

    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_title("After shuffling coordinates", fontsize=11)
    rng = np.random.default_rng(0)
    cols = ["#27ae60", "#2980b9", "#8e44ad", "#c0392b", "#7b241c"]
    for _ in range(80):
        ax.scatter(rng.uniform(0.5, 9.5), rng.uniform(1.5, 8.5), s=40, c=cols[rng.integers(0, 5)], alpha=0.85)
    ax.text(5, 0.5, "states mixed — adjacency gone", ha="center", fontsize=8, color="#7f8c8d")
    ax.axis("off")

    ax = axes[2]
    ax.axis("off")
    txt = (
        "What “front” means\n\n"
        "Not scattered hypoxic islands, but a spatial\n"
        "continuum where cells entering hypoxia sit next\n"
        "to cells exiting hypoxia (with persistent cells\n"
        "in between) — like a moving oxygen boundary.\n\n"
        "Permutation null\n\n"
        "Shuffle cell coordinates (≡ shuffle dynamics\n"
        "labels on fixed positions). If enter↔exit\n"
        "adjacency vanishes under the null, the front\n"
        "is real spatial organization, not chance."
    )
    ax.text(0.0, 0.95, txt, va="top", family="serif", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "what_front_means.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("wrote", OUT)


if __name__ == "__main__":
    main()
