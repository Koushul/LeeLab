#!/usr/bin/env python3
"""Shared H≤8 MAP microwell spatial placement for front figures/animations."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

H_CUT = 8.0
OUT = Path(
    "/ix1/ylee/shared/MC38_Hypoxia_001/reseq_finetune/deep_dive/spatial_niches/front_figures"
)
ENTROPY_CSV = OUT / "microwell_entropy_per_cell.csv"
H5AD = "/ix1/ylee/shared/MC38_Hypoxia_001/reseq_finetune/mc38_tumor_reseq_finetuned.h5ad"
DYN_CSV = (
    "/ix1/ylee/shared/MC38_Hypoxia_001/reseq_finetune/hypoxia_velocity/hypoxia_dynamics_per_cell.csv"
)


def jitter_xy(row_idx: np.ndarray, col_idx: np.ndarray, keys: np.ndarray, radius: float = 0.28):
    xy = np.column_stack([row_idx.astype(float), col_idx.astype(float)])
    groups: dict[tuple, list[int]] = {}
    for i, key in enumerate(zip(row_idx.tolist(), col_idx.tolist())):
        groups.setdefault(key, []).append(i)
    for idxs in groups.values():
        n = len(idxs)
        if n <= 1:
            continue
        for rank, i in enumerate(idxs):
            ang = 2 * np.pi * rank / n
            h = abs(hash(str(keys[i]))) % 1000 / 1000.0
            r = radius * (0.55 + 0.45 * h)
            xy[i, 0] += r * np.cos(ang)
            xy[i, 1] += r * np.sin(ang)
    return xy


def load_h8_adata(h_cut: float = H_CUT):
    adata = sc.read_h5ad(H5AD)
    dyn = pd.read_csv(DYN_CSV).set_index("barcode")
    common = adata.obs_names.intersection(dyn.index)
    adata = adata[common].copy()
    for c in ["hypoxia_dynamics", "hypoxia_score", "v_hypoxia"]:
        if c in dyn.columns:
            adata.obs[c] = dyn.loc[adata.obs_names, c].values
    adata.obs["is_tumor"] = (
        adata.obs["cell_type_finetuned"].astype(str).str.contains("Tumor", case=False)
        | adata.obs["cell_type"].astype(str).str.contains("Tumor", case=False)
    ).to_numpy()

    ent = pd.read_csv(ENTROPY_CSV)
    ent = ent[ent["entropy_bits"] <= h_cut].copy()
    ent["bc_raw"] = ent["barcode"].astype(str).str.split("-").str[0]
    ent["obs_name"] = ent["bc_raw"] + "_" + ent["sample"]
    keep = adata.obs_names.intersection(ent["obs_name"])
    adata = adata[keep].copy()
    emap = ent.set_index("obs_name")
    for c in ["entropy_bits", "map_prob", "map_row_idx", "map_col_idx", "adt_umi_total"]:
        if c in emap.columns:
            adata.obs[c] = emap.loc[adata.obs_names, c].to_numpy()

    xy = jitter_xy(
        adata.obs["map_row_idx"].to_numpy(),
        adata.obs["map_col_idx"].to_numpy(),
        adata.obs_names.to_numpy(),
    )
    adata.obsm["spatial"] = xy
    adata.obs["spatial_filter"] = f"entropy_bits<={h_cut}"
    return adata


def subset_xy(adata, sample: str | None):
    """sample=None/'OVERLAY' returns all cells on shared chip coords."""
    if sample is None or sample == "OVERLAY":
        sub = adata
        label = "OVERLAY"
    else:
        sub = adata[adata.obs["sample"].to_numpy() == sample]
        label = sample
    xy = np.asarray(sub.obsm["spatial"], float)
    labs = sub.obs["hypoxia_dynamics"].astype(str).to_numpy()
    tumor = sub.obs["is_tumor"].to_numpy()
    return sub, xy, labs, tumor, label
