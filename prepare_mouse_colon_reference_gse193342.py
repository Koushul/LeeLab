#!/usr/bin/env python3
"""
Build a mouse colon mucosa reference AnnData from GEO GSE193342 (Liu et al., 2023;
single-cell colitis mucosa). Suitable as an scRNA reference for MC38 / tumor
microenvironment label transfer: rich Immune + Mesenchyme + Epithelium labels.

Downloads (once) into --cache-dir:
  - GSE193342_exprsData.mtx.gz
  - GSE193342_rowData.txt.gz
  - GSE193342_colData.txt.gz

Labels (--label-column):
  - annotation.2  : fine types within Immune/Epithelium/Mesenchyme (recommended)
  - annotation    : coarse (Endothelium, Epithelium, Immune, ...)

For MALT we export obs column 'cell_type' (or --output-obs-column).
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import anndata as ad
import pandas as pd
from scipy.io import mmread


GEO_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE193nnn/GSE193342/suppl"


def download(url: str, dest: Path) -> None:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def main() -> None:
    p = argparse.ArgumentParser(description="Build GSE193342 reference h5ad for MALT.")
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("references/gse193342_cache"),
        help="Directory for downloaded GEO files.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("references/mouse_colon_gse193342_reference.h5ad"),
        help="Output .h5ad path.",
    )
    p.add_argument(
        "--label-column",
        choices=("annotation.2", "annotation"),
        default="annotation.2",
        help="Which colData label to use as biological type.",
    )
    p.add_argument(
        "--output-obs-column",
        default="cell_type",
        help="Name of obs column written for MALT --groupby (default: cell_type).",
    )
    args = p.parse_args()

    cache = args.cache_dir.resolve()
    mtx_gz = cache / "GSE193342_exprsData.mtx.gz"
    row_gz = cache / "GSE193342_rowData.txt.gz"
    col_gz = cache / "GSE193342_colData.txt.gz"

    download(f"{GEO_BASE}/GSE193342_exprsData.mtx.gz", mtx_gz)
    download(f"{GEO_BASE}/GSE193342_rowData.txt.gz", row_gz)
    download(f"{GEO_BASE}/GSE193342_colData.txt.gz", col_gz)

    print("Loading rowData / colData...")
    row_df = pd.read_csv(row_gz, sep="\t", compression="gzip")
    col_df = pd.read_csv(col_gz, sep="\t", compression="gzip")

    if "gene_short_name" not in row_df.columns:
        raise KeyError(f"Unexpected rowData columns: {row_df.columns.tolist()}")
    if args.label_column not in col_df.columns:
        raise KeyError(f"Missing label column {args.label_column!r} in colData")

    print("Loading matrix (sparse, genes × cells)...")
    X = mmread(gzip.open(mtx_gz, "rb")).tocsr()

    genes = row_df["gene_short_name"].astype(str).tolist()
    n_genes, n_cells = X.shape
    if len(genes) != n_genes:
        raise ValueError(f"Gene count mismatch: rowData {len(genes)} vs matrix {n_genes}")

    adata = ad.AnnData(X.T)
    obs_ids = (
        col_df["sample"].astype(str).values + "_" + col_df["barcode"].astype(str).values
    )
    adata.obs_names = obs_ids
    adata.var_names = genes
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    coarse = col_df["annotation"].astype(str)
    fine = col_df[args.label_column].astype(str)
    adata.obs["annotation"] = coarse.values
    adata.obs[args.label_column] = fine.values

    lab = fine if args.label_column == "annotation.2" else coarse
    if args.label_column == "annotation.2":
        lab = coarse.astype(str) + "_" + fine.astype(str)

    adata.obs[args.output_obs_column] = lab.values

    mask = adata.obs[args.output_obs_column].notna()
    mask &= ~adata.obs[args.output_obs_column].isin(["nan", "NaN", "NA", ""])
    if args.label_column == "annotation.2":
        mask &= ~adata.obs["annotation.2"].isin(["nan", "NaN", "NA", ""])

    adata = adata[mask.values].copy()
    vc = adata.obs[args.output_obs_column].value_counts()
    keep = vc[vc >= 50].index
    adata = adata[adata.obs[args.output_obs_column].isin(keep)].copy()
    adata.obs[args.output_obs_column] = adata.obs[args.output_obs_column].astype("category")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(adata)
    print(adata.obs[args.output_obs_column].value_counts())
    adata.write_h5ad(args.out, compression="gzip")
    print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
