#!/usr/bin/env python3
"""Incorporate resequenced E15S GEX into MC38 and fine-tune existing annotations.

E14S is unchanged (no reseq). E15S swaps the April Cell Ranger matrix for Palak's
July reseq run (S2 only under ResequencedE15S_E27_E28S). Existing curated labels
are transferred by barcode; new / unlabeled cells get kNN labels; tumor subtypes
(Tumor / Hypoxic Tumor / Tumor Proliferating) are refined with module scores on
the deeper counts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

ROOT = Path(__file__).resolve().parent

E14S_H5 = ROOT / "E14S" / "filtered_feature_bc_matrix.h5"
E15S_RESEQ_H5 = Path(
    "/ix1/ylee/Palak/cellranger_apps/e15s/outs/per_sample_outs/e15s/"
    "sample_filtered_feature_bc_matrix.h5"
)
ANNOT_H5AD = ROOT / "finer_annot.h5ad"
FALLBACK_ANNOT = ROOT / "mc38_tumor.h5ad"

MODULES = {
    "epi": ["Epcam", "Krt8", "Krt18", "Krt19", "Cdh1", "Cldn7", "Tspan8"],
    "prolif": ["Mki67", "Top2a", "Cdk1", "Ccnb1", "Stmn1", "Pcna", "Ube2c"],
    "hypoxia": ["Slc2a1", "Hk2", "Ldha", "Pgk1", "Bnip3", "Vegfa", "Egln3", "Eno1"],
    "fibro": ["Col1a1", "Col1a2", "Col3a1", "Dcn", "Pdgfra", "Acta2", "Fn1"],
    "immune": ["Ptprc", "Cd3e", "Cd79a", "Cd68", "Itgam", "S100a8"],
}

TUMOR_LABELS = {"Tumor", "Hypoxic Tumor", "Tumor Proliferating"}
LABEL_COLS = [
    "cell_type",
    "tiffany_cell_type_fine",
    "tiffany_cell_type_broad",
    "malt_label_subcluster_annotation",
    "malt_confidence_subcluster_annotation",
    "knn_label_subcluster_annotation",
    "malt_label_final_annotation",
    "malt_confidence_final_annotation",
    "knn_label_final_annotation",
    "leiden",
    "confident_r1.4",
    "confident_r1.6",
    "confident_r1.8",
    "confident_r2.0",
    "spatial_coordinate_x",
    "spatial_coordinate_y",
]


def _strip_bc(idx) -> pd.Index:
    return pd.Index(pd.Index(idx).astype(str).str.replace(r"-\d+$", "", regex=True))


def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    s = x.std()
    if s < 1e-9:
        return np.zeros_like(x)
    return (x - x.mean()) / s


def _read_gex(path: Path, sample: str) -> sc.AnnData:
    a = sc.read_10x_h5(path)
    a.var_names_make_unique()
    if "feature_types" in a.var.columns:
        a = a[:, a.var["feature_types"].to_numpy() == "Gene Expression"].copy()
    bc = _strip_bc(a.obs_names)
    a.obs["barcode"] = bc.astype(str)
    a.obs["sample"] = sample
    a.obs_names = a.obs["barcode"].astype(str) + "_" + sample
    a.layers["counts"] = a.X.copy()
    return a


def _load_prior() -> sc.AnnData:
    src = ANNOT_H5AD if ANNOT_H5AD.is_file() else FALLBACK_ANNOT
    prior = sc.read_h5ad(src)
    # Historical encoding: batch 0 = E14S, batch 1 = E15S
    sample = np.where(prior.obs["batch"].astype(str).to_numpy() == "0", "E14S", "E15S")
    prior.obs["sample"] = sample
    prior.obs["barcode"] = _strip_bc(prior.obs_names)
    prior.obs_names = prior.obs["barcode"].astype(str) + "_" + prior.obs["sample"].astype(str)
    return prior


def _transfer_obs(new: sc.AnnData, prior: sc.AnnData) -> dict:
    stats = {"n_prior": int(prior.n_obs), "n_new": int(new.n_obs)}
    shared = new.obs_names.intersection(prior.obs_names)
    stats["n_transferred"] = int(len(shared))
    stats["n_unlabeled"] = int(new.n_obs - len(shared))

    for col in LABEL_COLS:
        if col not in prior.obs.columns:
            continue
        new.obs[col] = pd.NA
        new.obs.loc[shared, col] = prior.obs.loc[shared, col].to_numpy()

    new.obs["annotation_source"] = "unlabeled"
    new.obs.loc[shared, "annotation_source"] = "transferred"

    if "spatial" in prior.obsm:
        xy = np.full((new.n_obs, 2), np.nan, dtype=np.float64)
        pmap = {n: i for i, n in enumerate(prior.obs_names)}
        for i, name in enumerate(new.obs_names):
            j = pmap.get(name)
            if j is not None:
                xy[i] = prior.obsm["spatial"][j]
        new.obsm["spatial"] = xy

    return stats


def _knn_graph(X: np.ndarray, n_neighbors: int = 15):
    """Build scanpy-compatible knn connectivities without importing umap/tensorflow."""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import normalize

    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    nn.fit(X)
    dists, idx = nn.kneighbors(X)
    dists, idx = dists[:, 1:], idx[:, 1:]
    n = X.shape[0]
    rows = np.repeat(np.arange(n), n_neighbors)
    cols = idx.ravel()
    # adaptive Gaussian kernel
    sig = np.maximum(dists[:, -1], 1e-6)
    vals = np.exp(-(dists.ravel() ** 2) / (sig.repeat(n_neighbors) ** 2))
    conn = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    conn = conn.maximum(conn.T)
    conn = normalize(conn, norm="l1", axis=1)
    dist = sparse.csr_matrix((dists.ravel(), (rows, cols)), shape=(n, n))
    return conn, dist


def _preprocess(adata: sc.AnnData, n_hvg: int = 3000) -> sc.AnnData:
    a = adata.copy()
    sc.pp.filter_genes(a, min_cells=3)
    sc.pp.calculate_qc_metrics(a, percent_top=None, log1p=False, inplace=True)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=min(n_hvg, a.n_vars - 1), subset=False)
    a.raw = a
    a = a[:, a.var["highly_variable"]].copy()
    sc.pp.scale(a, max_value=10)
    sc.tl.pca(a, n_comps=min(50, a.n_obs - 2, a.n_vars - 1))
    sc.external.pp.harmony_integrate(a, key="sample", basis="X_pca", adjusted_basis="X_pca_harmony")
    conn, dist = _knn_graph(a.obsm["X_pca_harmony"], n_neighbors=15)
    a.obsp["connectivities"] = conn
    a.obsp["distances"] = dist
    a.uns["neighbors"] = {
        "connectivities_key": "connectivities",
        "distances_key": "distances",
        "params": {"n_neighbors": 15, "method": "sklearn", "metric": "euclidean", "use_rep": "X_pca_harmony"},
    }
    # 2D embedding via PCA of harmony (avoid umap/tensorflow)
    a.obsm["X_umap"] = a.obsm["X_pca_harmony"][:, :2].copy()
    sc.tl.leiden(a, resolution=1.0, key_added="leiden_reseq", flavor="igraph", directed=False)
    return a


def _score_modules(adata: sc.AnnData) -> None:
    # score on log-normalized full gene space via .raw when present
    ref = adata.raw.to_adata() if adata.raw is not None else adata
    for name, genes in MODULES.items():
        present = [g for g in genes if g in ref.var_names]
        if len(present) < 2:
            adata.obs[f"score_{name}"] = 0.0
            continue
        tmp = ref.copy()
        sc.tl.score_genes(tmp, gene_list=present, score_name=f"score_{name}")
        adata.obs[f"score_{name}"] = tmp.obs[f"score_{name}"].to_numpy()


def _finetune_tumor_subtypes(adata: sc.AnnData) -> dict:
    """Propose tumor subtype updates from hypoxia/prolif scores; keep sticky labels separate."""
    ct = adata.obs["cell_type"].astype(object)
    mask = ct.isin(TUMOR_LABELS).to_numpy()
    if mask.sum() < 50:
        adata.obs["cell_type_finetuned"] = ct.astype("string")
        return {"n_tumor_refined": 0}

    zh = _z(adata.obs["score_hypoxia"].to_numpy())
    zp = _z(adata.obs["score_prolif"].to_numpy())
    # Among tumor cells only
    hypo_hi = np.quantile(zh[mask], 0.70)
    prolif_hi = np.quantile(zp[mask], 0.70)
    hypo_lo = np.quantile(zh[mask], 0.30)
    prolif_lo = np.quantile(zp[mask], 0.30)

    proposed = ct.to_numpy().copy()
    changed = 0
    for i in np.where(mask)[0]:
        old = proposed[i]
        # Strong evidence only — otherwise keep curated label
        if zh[i] >= hypo_hi and zp[i] < prolif_hi:
            lab = "Hypoxic Tumor"
        elif zp[i] >= prolif_hi and zh[i] < hypo_hi:
            lab = "Tumor Proliferating"
        elif zh[i] <= hypo_lo and zp[i] <= prolif_lo:
            lab = "Tumor"
        else:
            lab = old
        if lab != old:
            changed += 1
        proposed[i] = lab

    adata.obs["cell_type_pre_finetune"] = ct.astype("string")
    adata.obs["cell_type_finetuned"] = pd.Categorical(proposed)
    # Default cell_type stays curated+knn; callers can opt into finetuned
    return {
        "n_tumor_input": int(mask.sum()),
        "n_tumor_label_changed": int(changed),
        "hypoxia_hi_z": float(hypo_hi),
        "prolif_hi_z": float(prolif_hi),
        "hypoxia_lo_z": float(hypo_lo),
        "prolif_lo_z": float(prolif_lo),
        "tumor_subtype_counts_sticky": {
            k: int(v) for k, v in pd.Series(ct[mask]).value_counts().items()
        },
        "tumor_subtype_counts_finetuned": {
            k: int(v) for k, v in pd.Series(proposed[mask]).value_counts().items()
        },
    }


def _knn_label_unlabeled(adata: sc.AnnData, k: int = 15) -> dict:
    unlabeled = adata.obs["cell_type"].isna().to_numpy()
    labeled = ~unlabeled
    n_u = int(unlabeled.sum())
    if n_u == 0:
        return {"n_knn_labeled": 0}

    # neighbors graph from preprocess
    conn = adata.obsp["connectivities"]
    labels = adata.obs["cell_type"].astype(object).to_numpy()
    assigned = 0
    conf = np.full(adata.n_obs, np.nan)
    for i in np.where(unlabeled)[0]:
        row = conn[i].toarray().ravel()
        nbrs = np.argsort(-row)[: max(k * 3, k)]
        nbrs = [j for j in nbrs if labeled[j] and row[j] > 0][:k]
        if not nbrs:
            continue
        vals = pd.Series(labels[nbrs]).dropna()
        if vals.empty:
            continue
        top = vals.value_counts()
        lab = top.index[0]
        conf[i] = float(top.iloc[0] / len(vals))
        labels[i] = lab
        assigned += 1

    adata.obs["cell_type"] = pd.Categorical(labels)
    adata.obs["knn_label_confidence"] = conf
    adata.obs.loc[unlabeled & pd.notna(adata.obs["cell_type"]), "annotation_source"] = "knn"
    return {"n_unlabeled": n_u, "n_knn_labeled": int(assigned)}


def _broad_from_fine(fine: str) -> str:
    if pd.isna(fine):
        return "Unknown"
    f = str(fine)
    if f in TUMOR_LABELS:
        return "Tumor"
    if "TAM" in f or "Macrophage" in f or f in {"Monocyte", "Neutrophil", "cDC", "mDC"}:
        return "Myeloid"
    if f.startswith("CD") or f in {"NK", "NK T", "B"}:
        return "Lymphoid"
    if f in {"Stromal", "CAF"}:
        return "Stroma"
    return "Other"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, default=ROOT / "reseq_finetune")
    p.add_argument("--e15s-h5", type=Path, default=E15S_RESEQ_H5)
    p.add_argument("--e14s-h5", type=Path, default=E14S_H5)
    args = p.parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    if not args.e14s_h5.is_file():
        raise FileNotFoundError(args.e14s_h5)
    if not args.e15s_h5.is_file():
        raise FileNotFoundError(args.e15s_h5)

    print("Loading matrices…")
    e14 = _read_gex(args.e14s_h5, "E14S")
    e15 = _read_gex(args.e15s_h5, "E15S")
    print(f"  E14S {e14.shape}  E15S_reseq {e15.shape}")

    adata = ad.concat({"E14S": e14, "E15S": e15}, join="outer", merge="same")
    adata.obs_names_make_unique()
    adata.obs["batch"] = (adata.obs["sample"].astype(str) == "E15S").astype(int)
    # restore dense-ish counts layer after concat
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    print("Loading prior annotations…")
    prior = _load_prior()
    stats = _transfer_obs(adata, prior)
    print(stats)

    print("Preprocess + Harmony…")
    # keep counts on full object; work on a processed view for graph
    processed = _preprocess(adata)
    # copy embeddings / leiden back
    adata.obsm["X_pca"] = processed.obsm["X_pca"]
    adata.obsm["X_pca_harmony"] = processed.obsm["X_pca_harmony"]
    adata.obsm["X_umap"] = processed.obsm["X_umap"]
    adata.obsp["connectivities"] = processed.obsp["connectivities"]
    adata.obsp["distances"] = processed.obsp["distances"]
    adata.obs["leiden_reseq"] = processed.obs["leiden_reseq"].to_numpy()
    adata.uns["neighbors"] = processed.uns["neighbors"]
    adata.uns["umap"] = processed.uns.get("umap", {})
    # log-normalized X for scoring: rebuild from counts
    adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    print("Module scores + tumor fine-tune…")
    _score_modules(adata)
    # Label new cells first so tumor finetune sees full set
    knn_stats = _knn_label_unlabeled(adata)
    tumor_stats = _finetune_tumor_subtypes(adata)
    print(tumor_stats)
    print(knn_stats)

    adata.obs["cell_type_broad"] = (
        adata.obs["cell_type"].astype(str).map(_broad_from_fine).astype("category")
    )
    if "cell_type_finetuned" in adata.obs.columns:
        adata.obs["cell_type_finetuned_broad"] = (
            adata.obs["cell_type_finetuned"].astype(str).map(_broad_from_fine).astype("category")
        )

    # still-unlabeled → leiden majority within cluster among labeled
    still = adata.obs["cell_type"].isna()
    if still.any():
        for cl in adata.obs.loc[still, "leiden_reseq"].unique():
            m = adata.obs["leiden_reseq"] == cl
            labs = adata.obs.loc[m & ~adata.obs["cell_type"].isna(), "cell_type"]
            if labs.empty:
                continue
            maj = labs.value_counts().index[0]
            idx = m & still
            adata.obs.loc[idx, "cell_type"] = maj
            adata.obs.loc[idx, "annotation_source"] = "leiden_majority"
        still2 = int(adata.obs["cell_type"].isna().sum())
    else:
        still2 = 0

    # Sanitize obs for h5ad (nullable / mixed types from transfer)
    for col in adata.obs.columns:
        s = adata.obs[col]
        if pd.api.types.is_bool_dtype(s) or str(s.dtype) == "boolean":
            adata.obs[col] = s.astype(object).where(s.notna(), other=None)
            continue
        if pd.api.types.is_numeric_dtype(s):
            adata.obs[col] = pd.to_numeric(s, errors="coerce")
            continue
        adata.obs[col] = s.astype("string").fillna("").astype(str)
        adata.obs.loc[adata.obs[col] == "<NA>", col] = ""
        adata.obs.loc[adata.obs[col] == "nan", col] = ""

    out_h5ad = outdir / "mc38_tumor_reseq_finetuned.h5ad"
    adata.obs.index.name = None
    if "barcode" in adata.obs.columns and (adata.obs.index.astype(str) == adata.obs["barcode"].astype(str)).all():
        # index is sample-suffixed; barcode column is unsuffixed — fine, just clear name clash risk
        pass
    adata.write_h5ad(out_h5ad, compression="gzip")

    ann_out = outdir / "annotations_reseq_finetuned.csv"
    export = adata.obs[
        ["barcode", "sample", "cell_type", "cell_type_finetuned", "annotation_source", "cell_type_broad"]
    ].copy()
    export.to_csv(ann_out, index=True)

    summary = {
        "inputs": {
            "e14s": str(args.e14s_h5),
            "e15s_reseq": str(args.e15s_h5),
            "prior": str(ANNOT_H5AD if ANNOT_H5AD.is_file() else FALLBACK_ANNOT),
        },
        "shapes": {"E14S": list(e14.shape), "E15S": list(e15.shape), "merged": list(adata.shape)},
        "transfer": stats,
        "tumor_finetune": tumor_stats,
        "knn": knn_stats,
        "n_still_unlabeled": still2,
        "cell_type_counts": {k: int(v) for k, v in adata.obs["cell_type"].value_counts(dropna=False).items()},
        "annotation_source_counts": {
            k: int(v) for k, v in adata.obs["annotation_source"].value_counts().items()
        },
        "outputs": {"h5ad": str(out_h5ad), "annotations_csv": str(ann_out)},
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(json.dumps(summary, indent=2, default=str))
    print(f"Wrote {out_h5ad}")


if __name__ == "__main__":
    main()
