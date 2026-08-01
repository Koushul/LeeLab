#!/usr/bin/env python3
"""RNA velocity of MC38 hypoxia dynamics (E14S + E15S USA alevin-fry).

Maps hypoxic tumor cells onto a continuum:
  entering → persistent → exiting/reverting → reverted/post-hypoxic
using spliced/unspliced velocity projected onto a hypoxia gene module.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
from scipy import sparse

# Avoid umap→tensorflow crash under numpy2
_fake = types.ModuleType("umap.parametric_umap")
_fake.ParametricUMAP = None
_fake.load_ParametricUMAP = None
sys.modules.setdefault("umap.parametric_umap", _fake)

ROOT = Path("/ix1/ylee/shared/MC38_Hypoxia_001")
OUT = ROOT / "reseq_finetune" / "hypoxia_velocity"
OUT.mkdir(parents=True, exist_ok=True)

ANNOT = ROOT / "reseq_finetune" / "mc38_tumor_reseq_finetuned.h5ad"
Q = {
    "E14S": Path(
        "/ix1/ylee/shared/organized_experiments/simpleleaf_runs/E14S/"
        "gex_quant/af_quant/alevin/quants.h5ad"
    ),
    "E15S": Path(
        "/ix1/ylee/shared/organized_experiments/simpleleaf_runs/E15S/"
        "gex_quant/af_quant/alevin/quants.h5ad"
    ),
}

HYPOXIA = [
    "Slc2a1",
    "Hk2",
    "Ldha",
    "Pgk1",
    "Bnip3",
    "Vegfa",
    "Egln3",
    "Eno1",
    "Pdk1",
    "Gapdh",
    "Aldoa",
    "Pfkl",
    "Ndrg1",
    "Fam162a",
    "P4ha1",
    "Higd1a",
]
PROLIF = ["Mki67", "Top2a", "Cdk1", "Ccnb1", "Stmn1", "Pcna", "Ube2c"]
TUMOR_LABELS = {"Tumor", "Hypoxic Tumor", "Tumor Proliferating"}


def _strip(bc: str) -> str:
    return str(bc).split("-")[0]


def _as_csr(X):
    if sparse.issparse(X):
        return X.tocsr()
    return sparse.csr_matrix(X)


def load_usa_for_annotated(prior: sc.AnnData) -> sc.AnnData:
    parts = []
    for sample, path in Q.items():
        print(f"loading USA {sample} …")
        q = sc.read_h5ad(path)
        # gene symbols as var_names for scvelo / scoring
        if "gene_symbol" in q.var.columns:
            q.var_names = q.var["gene_symbol"].astype(str)
            q.var_names_make_unique()
        q.obs["barcode"] = [_strip(b) for b in q.obs_names.astype(str)]
        q.obs["sample"] = sample
        q.obs_names = q.obs["barcode"].astype(str) + "_" + sample

        want = prior.obs_names[prior.obs["sample"].astype(str) == sample]
        keep = q.obs_names.intersection(want)
        print(f"  overlap {len(keep)} / {len(want)} annotated")
        q = q[keep].copy()

        # USA convention for scvelo: spliced += ambiguous
        S = _as_csr(q.layers["spliced"]) + _as_csr(q.layers["ambiguous"])
        U = _as_csr(q.layers["unspliced"])
        q.layers["spliced"] = S
        q.layers["unspliced"] = U
        q.X = S.copy()
        parts.append(q)

    adata = ad.concat(parts, join="inner", merge="same")
    adata.obs_names_make_unique()
    # attach annotations + UMAP
    shared = adata.obs_names.intersection(prior.obs_names)
    adata = adata[shared].copy()
    for col in [
        "cell_type",
        "cell_type_finetuned",
        "sample",
        "annotation_source",
        "score_hypoxia",
        "score_prolif",
        "score_epi",
        "batch",
    ]:
        if col in prior.obs.columns:
            adata.obs[col] = prior.obs.loc[adata.obs_names, col].to_numpy()
    if "X_umap" in prior.obsm:
        adata.obsm["X_umap"] = prior[adata.obs_names].obsm["X_umap"].copy()
    if "X_pca_harmony" in prior.obsm:
        adata.obsm["X_pca_harmony"] = prior[adata.obs_names].obsm["X_pca_harmony"].copy()
    return adata


def score_module(adata: sc.AnnData, genes: list[str], key: str, layer: str = "Ms") -> list[str]:
    present = [g for g in genes if g in adata.var_names]
    if len(present) < 2:
        adata.obs[key] = 0.0
        return present
    X = adata.layers[layer] if layer in adata.layers else adata.X
    X = _as_csr(X)
    idx = [adata.var_names.get_loc(g) for g in present]
    sub = np.asarray(X[:, idx].todense(), dtype=np.float64)
    mu, sd = sub.mean(0), sub.std(0)
    sd[sd < 1e-8] = 1.0
    adata.obs[key] = ((sub - mu) / sd).mean(1)
    return present


def hypoxia_velocity_delta(adata: sc.AnnData, genes: list[str], key: str = "v_hypoxia") -> None:
    """Per-cell projected hypoxia change: mean velocity of hypoxia genes (Ms-normalized scale)."""
    present = [g for g in genes if g in adata.var_names and "velocity" in adata.layers]
    if not present:
        # velocity layer uses gene subset
        present = [g for g in genes if g in adata.var_names]
    V = _as_csr(adata.layers["velocity"])
    idx = []
    for g in present:
        if g in adata.var_names:
            j = adata.var_names.get_loc(g)
            # only genes with velocity computed
            if sparse.issparse(V):
                if V[:, j].nnz == 0 and np.allclose(V[:, j].toarray(), 0):
                    continue
            idx.append(j)
    if not idx:
        adata.obs[key] = 0.0
        return
    # mean velocity across hypoxia genes (already in Ms space typically)
    vsub = np.asarray(V[:, idx].todense(), dtype=np.float64)
    # mask genes that are all-nan
    good = ~np.isnan(vsub).all(0)
    vsub = vsub[:, good]
    adata.obs[key] = np.nanmean(vsub, axis=1)


def classify_hypoxia_dynamics(adata: sc.AnnData) -> pd.DataFrame:
    """Systematic state calls on tumor cells from (score, velocity) phase plane."""
    s = adata.obs["hypoxia_score"].to_numpy(dtype=float)
    v = adata.obs["v_hypoxia"].to_numpy(dtype=float)
    # robust thresholds on tumor compartment
    tumor = adata.obs["cell_type"].astype(str).isin(TUMOR_LABELS).to_numpy()
    s_t = s[tumor]
    v_t = v[tumor]
    s_hi = np.quantile(s_t, 0.65)
    s_lo = np.quantile(s_t, 0.35)
    v_pos = np.quantile(np.abs(v_t), 0.40)  # magnitude floor
    # directed thresholds
    v_up = max(v_pos, np.quantile(v_t, 0.60))
    v_dn = min(-v_pos, np.quantile(v_t, 0.40))

    states = np.full(adata.n_obs, "non_tumor", dtype=object)
    for i in np.where(tumor)[0]:
        si, vi = s[i], v[i]
        if si >= s_hi and vi >= v_up:
            st = "entering_deep_hypoxia"
        elif si >= s_hi and vi <= v_dn:
            st = "exiting_hypoxia"
        elif si >= s_hi and abs(vi) < v_pos:
            st = "persistent_hypoxia"
        elif si <= s_lo and vi >= v_up:
            st = "entering_hypoxia"
        elif si <= s_lo and vi <= v_dn:
            st = "reverted_stable"  # low hypoxia, still decreasing / settled
        elif s_lo < si < s_hi and vi >= v_up:
            st = "entering_hypoxia"
        elif s_lo < si < s_hi and vi <= v_dn:
            st = "exiting_hypoxia"
        elif si <= s_lo and abs(vi) < v_pos:
            # low hypoxia, quiescent velocity — likely post-hypoxic or never-hypoxic
            # use prior label: if was Hypoxic Tumor or high finetuned hypoxia → reverted
            lab = str(adata.obs["cell_type"].iloc[i])
            st = "reverted_posthypoxic" if lab == "Hypoxic Tumor" else "normoxic_stable"
        else:
            st = "transitional"
        states[i] = st

    adata.obs["hypoxia_dynamics"] = pd.Categorical(states)
    thr = {
        "s_hi": float(s_hi),
        "s_lo": float(s_lo),
        "v_pos": float(v_pos),
        "v_up": float(v_up),
        "v_dn": float(v_dn),
    }
    return thr


def main() -> None:
    scv.settings.verbosity = 2
    scv.settings.set_figure_params("scvelo", dpi=120, facecolor="white")

    print("Loading annotated MC38 …")
    prior = sc.read_h5ad(ANNOT)
    adata = load_usa_for_annotated(prior)
    print(adata)

    # Focus analysis object: all cells for embedding, tumor for dynamics emphasis
    print("scVelo preprocess …")
    # Keep hypoxia / prolif genes even if not highly variable
    retain = sorted(set(HYPOXIA + PROLIF))
    scv.pp.filter_and_normalize(
        adata, min_shared_counts=20, n_top_genes=3000, retain_genes=retain
    )
    # Scanpy neighbors on harmony if present, else PCA
    if "X_pca_harmony" in adata.obsm:
        sc.pp.neighbors(adata, use_rep="X_pca_harmony", n_neighbors=30)
    else:
        sc.tl.pca(adata, n_comps=30)
        sc.pp.neighbors(adata, n_neighbors=30)
    scv.pp.moments(adata, n_pcs=None, n_neighbors=None)

    print("velocity (stochastic) …")
    scv.tl.velocity(adata, mode="stochastic")
    scv.tl.velocity_graph(adata, n_jobs=8, show_progress_bar=False)

    # Prefer full-transcriptome hypoxia score transferred from annotation when present
    hyp_genes = score_module(adata, HYPOXIA, "hypoxia_score_ms", layer="Ms")
    prolif_genes = score_module(adata, PROLIF, "prolif_score", layer="Ms")
    if "score_hypoxia" in adata.obs.columns and adata.obs["score_hypoxia"].notna().any():
        adata.obs["hypoxia_score"] = pd.to_numeric(adata.obs["score_hypoxia"], errors="coerce")
        # z-score within tumor for comparable phase plane
        tumor_m = adata.obs["cell_type"].astype(str).isin(TUMOR_LABELS)
        mu = adata.obs.loc[tumor_m, "hypoxia_score"].mean()
        sd = adata.obs.loc[tumor_m, "hypoxia_score"].std() or 1.0
        adata.obs["hypoxia_score_z"] = (adata.obs["hypoxia_score"] - mu) / sd
        adata.obs["hypoxia_score"] = adata.obs["hypoxia_score_z"]
    else:
        adata.obs["hypoxia_score"] = adata.obs["hypoxia_score_ms"]
    print("hypoxia genes used:", hyp_genes)
    hypoxia_velocity_delta(adata, hyp_genes, "v_hypoxia")
    # z-score v_hypoxia within tumor for thresholds
    tumor_m = adata.obs["cell_type"].astype(str).isin(TUMOR_LABELS).to_numpy()
    vh = adata.obs["v_hypoxia"].to_numpy(dtype=float)
    mu_v, sd_v = vh[tumor_m].mean(), vh[tumor_m].std() or 1.0
    adata.obs["v_hypoxia_raw"] = vh
    adata.obs["v_hypoxia"] = (vh - mu_v) / sd_v

    # Velocity pseudotime — root = top hypoxic cells
    scv.tl.velocity_pseudotime(adata)
    # root-ish: push high hypoxia to early latent for "time since hypoxia onset" alternative
    # Also compute a hypoxia-rooted time: flip so high hypoxia ~ early
    vpt = adata.obs["velocity_pseudotime"].to_numpy(dtype=float)
    # correlate with hypoxia to orient
    mask = np.isfinite(vpt)
    corr = np.corrcoef(adata.obs["hypoxia_score"].to_numpy()[mask], vpt[mask])[0, 1]
    if corr > 0:
        # high hypoxia at late time → flip so hypoxia is early (stress onset → resolve)
        adata.obs["hypoxia_latent_time"] = 1.0 - vpt
    else:
        adata.obs["hypoxia_latent_time"] = vpt
    print("vpt~hypoxia corr", corr, "→ oriented hypoxia_latent_time")

    thr = classify_hypoxia_dynamics(adata)
    print("thresholds", thr)
    print(adata.obs["hypoxia_dynamics"].value_counts())

    palette = {
        "entering_hypoxia": "#2a9d8f",
        "entering_deep_hypoxia": "#1d3557",
        "persistent_hypoxia": "#e76f51",
        "exiting_hypoxia": "#f4a261",
        "reverted_posthypoxic": "#9b5de5",
        "reverted_stable": "#bdb2ff",
        "normoxic_stable": "#8d99ae",
        "transitional": "#cad2c5",
        "non_tumor": "#dddddd",
    }

    # Dynamical model on tumor only (richer kinetics) if enough cells
    tumor = adata[adata.obs["cell_type"].astype(str).isin(TUMOR_LABELS)].copy()
    print("tumor subset", tumor.shape)
    tumor_ok = False
    try:
        sc.pp.neighbors(tumor, use_rep="X_pca_harmony" if "X_pca_harmony" in tumor.obsm else "X_pca", n_neighbors=30)
        scv.pp.moments(tumor, n_pcs=None, n_neighbors=None)
        scv.tl.recover_dynamics(tumor, n_jobs=8, var_names="velocity_genes")
        scv.tl.velocity(tumor, mode="dynamical")
        scv.tl.velocity_graph(tumor, n_jobs=8, show_progress_bar=False)
        scv.tl.latent_time(tumor)
        adata.obs["dynamical_latent_time"] = np.nan
        adata.obs.loc[tumor.obs_names, "dynamical_latent_time"] = tumor.obs["latent_time"].to_numpy()
        # recompute tumor hypoxia velocity under dynamical model
        hypoxia_velocity_delta(tumor, hyp_genes, "v_hypoxia_dyn")
        adata.obs["v_hypoxia_dyn"] = np.nan
        adata.obs.loc[tumor.obs_names, "v_hypoxia_dyn"] = tumor.obs["v_hypoxia_dyn"].to_numpy()
        tumor_ok = True
    except Exception as e:
        print("dynamical model skipped:", e)
        tumor = adata[adata.obs["cell_type"].astype(str).isin(TUMOR_LABELS)].copy()

    # --------- plots ---------
    # Ensure velocity embedding exists for streams
    try:
        scv.tl.velocity_embedding(adata, basis="umap")
    except Exception as e:
        print("velocity_embedding:", e)

    def _stream(ax, color, title, **kw):
        scv.pl.velocity_embedding_stream(
            adata,
            basis="umap",
            color=color,
            ax=ax,
            show=False,
            title=title,
            density=1.2,
            alpha=0.65,
            **kw,
        )

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)
    _stream(axes[0, 0], "hypoxia_score", "Velocity · hypoxia score", color_map="magma_r")
    _stream(
        axes[0, 1],
        "hypoxia_dynamics",
        "Velocity · hypoxia dynamics states",
        legend_loc="right margin",
    )

    umap = np.asarray(adata.obsm["X_umap"])
    vh = adata.obs["v_hypoxia"].to_numpy(dtype=float)
    hs = adata.obs["hypoxia_score"].to_numpy(dtype=float)
    lt = adata.obs["hypoxia_latent_time"].to_numpy(dtype=float)

    pts = axes[1, 0].scatter(
        umap[:, 0],
        umap[:, 1],
        c=vh,
        s=3,
        cmap="RdBu_r",
        vmin=np.nanpercentile(vh, 5),
        vmax=np.nanpercentile(vh, 95),
        linewidths=0,
        rasterized=True,
    )
    axes[1, 0].set_title("Projected hypoxia velocity (v_hypoxia)")
    axes[1, 0].set_xticks([])
    axes[1, 0].set_yticks([])
    fig.colorbar(pts, ax=axes[1, 0], fraction=0.046, pad=0.04)

    pts2 = axes[1, 1].scatter(
        umap[:, 0],
        umap[:, 1],
        c=lt,
        s=3,
        cmap="viridis",
        linewidths=0,
        rasterized=True,
    )
    axes[1, 1].set_title("Hypoxia-oriented latent time")
    axes[1, 1].set_xticks([])
    axes[1, 1].set_yticks([])
    fig.colorbar(pts2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "hypoxia_velocity_umap.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 2) phase plane: hypoxia score vs v_hypoxia for tumor
    fig, ax = plt.subplots(figsize=(7.2, 6), dpi=150)
    dyn = adata.obs.loc[tumor.obs_names, "hypoxia_dynamics"].astype(str)
    hs = adata.obs.loc[tumor.obs_names, "hypoxia_score"].to_numpy()
    vh = adata.obs.loc[tumor.obs_names, "v_hypoxia"].to_numpy()
    for st, c in palette.items():
        m = dyn.to_numpy() == st
        if not np.any(m):
            continue
        ax.scatter(hs[m], vh[m], s=10, alpha=0.75, c=c, label=f"{st} ({m.sum()})", linewidths=0)
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax.axvline(thr["s_lo"], color="k", lw=0.5, ls="--", alpha=0.4)
    ax.axvline(thr["s_hi"], color="k", lw=0.5, ls="--", alpha=0.4)
    ax.set_xlabel("Hypoxia module score (z within tumor)")
    ax.set_ylabel("v_hypoxia (z within tumor)")
    ax.set_title("Tumor hypoxia phase plane")
    ax.legend(fontsize=7, frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "hypoxia_phase_plane.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 3) tumor-only stream
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=150)
    tumor.obsm["X_umap"] = adata[tumor.obs_names].obsm["X_umap"].copy()
    plot_obj = tumor if "velocity" in tumor.layers else adata[tumor.obs_names].copy()
    if "hypoxia_dynamics" not in plot_obj.obs.columns:
        plot_obj.obs["hypoxia_dynamics"] = adata.obs.loc[plot_obj.obs_names, "hypoxia_dynamics"]
    try:
        if "velocity_umap" not in plot_obj.obsm:
            scv.tl.velocity_embedding(plot_obj, basis="umap")
        scv.pl.velocity_embedding_stream(
            plot_obj,
            basis="umap",
            color="hypoxia_dynamics",
            ax=ax,
            show=False,
            title="Tumor only · hypoxia dynamics",
            legend_loc="right margin",
            density=1.4,
        )
    except Exception as e:
        print("tumor stream fallback:", e)
        dyn = adata.obs.loc[tumor.obs_names, "hypoxia_dynamics"].astype(str)
        u = np.asarray(tumor.obsm["X_umap"])
        for st, c in palette.items():
            m = dyn.to_numpy() == st
            if np.any(m):
                ax.scatter(u[m, 0], u[m, 1], s=8, c=c, label=f"{st} ({m.sum()})", linewidths=0)
        ax.legend(fontsize=7, frameon=False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Tumor only · hypoxia dynamics")
    fig.tight_layout()
    fig.savefig(OUT / "tumor_hypoxia_stream.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # export tables
    cols = [
        "sample",
        "cell_type",
        "hypoxia_dynamics",
        "hypoxia_score",
        "v_hypoxia",
        "prolif_score",
        "hypoxia_latent_time",
        "velocity_pseudotime",
    ]
    if "dynamical_latent_time" in adata.obs.columns:
        cols.append("dynamical_latent_time")
    export = adata.obs[cols].copy()
    export.to_csv(OUT / "hypoxia_dynamics_per_cell.csv")

    # summary counts
    summary = {
        "n_cells": int(adata.n_obs),
        "n_tumor": int(tumor.n_obs),
        "hypoxia_genes": hyp_genes,
        "prolif_genes": prolif_genes,
        "thresholds": thr,
        "dynamics_counts_all": {k: int(v) for k, v in adata.obs["hypoxia_dynamics"].value_counts().items()},
        "dynamics_counts_tumor": {
            k: int(v)
            for k, v in adata.obs.loc[
                adata.obs["cell_type"].astype(str).isin(TUMOR_LABELS), "hypoxia_dynamics"
            ]
            .value_counts()
            .items()
        },
        "vpt_hypoxia_corr": float(corr),
        "dynamical_model": tumor_ok,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    # save adata (tumor+all)
    # sanitize
    for c in adata.obs.columns:
        if adata.obs[c].dtype == object:
            adata.obs[c] = adata.obs[c].astype(str)
    adata.obs.index.name = None
    adata.write_h5ad(OUT / "mc38_hypoxia_velocity.h5ad", compression="gzip")

    # JSON for web (tumor points on UMAP)
    web = adata.obs_names
    umap = np.asarray(adata.obsm["X_umap"])
    payload = {
        "n": int(adata.n_obs),
        "umap1": [round(float(x), 3) for x in umap[:, 0]],
        "umap2": [round(float(x), 3) for x in umap[:, 1]],
        "hypoxia_score": [round(float(x), 4) for x in adata.obs["hypoxia_score"]],
        "v_hypoxia": [round(float(x), 5) for x in adata.obs["v_hypoxia"]],
        "latent": [round(float(x), 4) for x in adata.obs["hypoxia_latent_time"].fillna(0)],
        "dynamics": adata.obs["hypoxia_dynamics"].astype(str).tolist(),
        "cell_type": adata.obs["cell_type"].astype(str).tolist(),
        "sample": adata.obs["sample"].astype(str).tolist(),
        "is_tumor": adata.obs["cell_type"].astype(str).isin(TUMOR_LABELS).astype(int).tolist(),
        "palette": palette,
        "summary": summary,
    }
    (OUT / "velocity_web.json").write_text(json.dumps(payload, separators=(",", ":")))
    print(json.dumps(summary, indent=2))
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
