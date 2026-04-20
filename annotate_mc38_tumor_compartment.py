#!/usr/bin/env python3
"""
After MALT, assign MC38 tumor vs microenvironment compartments using gene modules.

MC38 cells are epithelial/carcinoma-like; colon-reference MALT often maps them to
Epithelium_* types. This script adds:

  - obs['compartment']: Tumor_epithelial | Tumor_epithelial_lowconf | Immune | Stroma | Endothelial | Ambiguous
  - obs['annotation_coarse']: Tumor | Tumor_candidate | TME_* | Other
  - obs['annotation_detailed']: compartment + hypoxia tag + malt=...
  - obs['is_tumor_epithelial']: bool (high tumor_epithelial_score among epithelial MALT calls)

Uses scanpy.tl.score_genes on log-normalized counts (normalize_total + log1p if needed).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


def _ensure_lognorm(adata: sc.AnnData) -> None:
    if "log1p" in adata.uns:
        return
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)


def _z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    s = x.std()
    if s < 1e-9:
        return np.zeros_like(x)
    return (x - x.mean()) / s


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--in",
        dest="inp",
        type=Path,
        required=True,
        help="AnnData with malt_label (e.g. malt_out/query_malt_labeled.h5ad).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (default: same as input with _tumor_annot suffix).",
    )
    p.add_argument(
        "--tumor-epi-quantile",
        type=float,
        default=0.85,
        help="Cells above this quantile on tumor_epithelial_score among Epithelium_* MALT calls "
        "are marked is_tumor_epithelial (default 0.85).",
    )
    args = p.parse_args()

    adata = sc.read_h5ad(args.inp)
    if "malt_label" not in adata.obs.columns:
        raise KeyError("Expected obs column 'malt_label' from MALT output.")

    adata = adata.copy()
    _ensure_lognorm(adata)

    modules = {
        "epi_crc": [
            "Epcam",
            "Krt8",
            "Krt18",
            "Krt19",
            "Cdh1",
            "Cldn7",
            "Tspan8",
            "Ceacam1",
        ],
        "prolif": ["Mki67", "Top2a", "Cdk1", "Ccnb1", "Stmn1", "Pcna", "Ube2c"],
        "hypoxia": ["Slc2a1", "Hk2", "Ldha", "Pgk1", "Bnip3", "Vegfa", "Egln3", "Eno1"],
        "fibro": ["Col1a1", "Col1a2", "Col3a1", "Dcn", "Pdgfra", "Acta2", "Fn1"],
        "immune_core": ["Ptprc", "Cd3e", "Cd79a", "Cd68", "Itgam", "S100a8"],
        "endo": ["Pecam1", "Cdh5", "Kdr", "Eng", "Vwf"],
    }

    for name, genes in modules.items():
        present = [g for g in genes if g in adata.var_names]
        if len(present) < 2:
            adata.obs[f"score_{name}"] = 0.0
            continue
        sc.tl.score_genes(adata, gene_list=present, score_name=f"score_{name}")

    epi = adata.obs["score_epi_crc"].to_numpy()
    pr = adata.obs["score_prolif"].to_numpy()
    hy = adata.obs["score_hypoxia"].to_numpy()
    fb = adata.obs["score_fibro"].to_numpy()
    im = adata.obs["score_immune_core"].to_numpy()
    en = adata.obs["score_endo"].to_numpy()

    ze, zp, zh, zf, zi, zn = map(_z, (epi, pr, hy, fb, im, en))
    tumor_epithelial_score = ze + zp - zf

    adata.obs["tumor_epithelial_score"] = tumor_epithelial_score
    adata.obs["hypoxia_score"] = hy

    malt = adata.obs["malt_label"].astype(str)
    is_epi_malt = malt.str.startswith("Epithelium_")
    is_imm_malt = malt.str.startswith("Immune_")
    is_mes_malt = malt.str.startswith("Mesenchyme_")
    is_endo_malt = malt.str.startswith("Endothelium_")

    q = float(args.tumor_epi_quantile)
    mask_epi = is_epi_malt.to_numpy()
    thr = np.quantile(tumor_epithelial_score[mask_epi], q) if mask_epi.sum() > 200 else np.quantile(
        tumor_epithelial_score, q
    )
    is_tumor_epi = (tumor_epithelial_score >= thr) & mask_epi
    adata.obs["is_tumor_epithelial"] = is_tumor_epi

    comp = np.full(adata.n_obs, "Ambiguous", dtype=object)
    comp[is_tumor_epi] = "Tumor_epithelial"
    comp[(comp == "Ambiguous") & is_endo_malt.to_numpy()] = "Endothelial"
    comp[(comp == "Ambiguous") & is_imm_malt.to_numpy()] = "Immune"
    comp[(comp == "Ambiguous") & is_mes_malt.to_numpy()] = "Stroma"
    comp[(comp == "Ambiguous") & is_epi_malt.to_numpy()] = "Tumor_epithelial_lowconf"
    comp[(comp == "Ambiguous") & (zi > zn + 0.3) & (zi > zf + 0.3)] = "Immune"
    comp[(comp == "Ambiguous") & (zf > zi + 0.3) & (zf > zn + 0.3)] = "Stroma"
    comp[(comp == "Ambiguous") & (zn > zi + 0.3) & (zn > zf + 0.3)] = "Endothelial"
    adata.obs["compartment"] = pd.Categorical(comp)

    coarse_map = {
        "Tumor_epithelial": "Tumor",
        "Tumor_epithelial_lowconf": "Tumor_candidate",
        "Immune": "TME_Immune",
        "Stroma": "TME_Stroma",
        "Endothelial": "TME_Endothelial",
        "Ambiguous": "Other",
    }
    adata.obs["annotation_coarse"] = adata.obs["compartment"].astype(str).map(coarse_map).astype("category")

    hypo_hi = zh >= np.quantile(zh, 0.75)
    det = []
    for i in range(adata.n_obs):
        c = comp[i]
        ml = malt.iloc[i]
        if c == "Tumor_epithelial":
            tag = "hypoxic" if hypo_hi[i] else "normoxic"
            det.append(f"MC38_tumor_epithelial_{tag}|malt={ml}")
        elif c == "Tumor_epithelial_lowconf":
            det.append(f"MC38_tumor_epithelial_candidate|malt={ml}")
        elif c in ("Immune", "Stroma", "Endothelial"):
            suf = "_hypoxic" if hypo_hi[i] else ""
            det.append(f"{c}{suf}|malt={ml}")
        else:
            det.append(f"Other|malt={ml}")
    adata.obs["annotation_detailed"] = det

    out = args.out
    if out is None:
        stem = args.inp.stem
        out = args.inp.with_name(f"{stem}_tumor_annot.h5ad")
    out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out, compression="gzip")

    print(adata.obs["compartment"].value_counts())
    print("\nis_tumor_epithelial:", int(adata.obs["is_tumor_epithelial"].sum()), "cells")
    print(f"\nWrote {out.resolve()}")


if __name__ == "__main__":
    main()
