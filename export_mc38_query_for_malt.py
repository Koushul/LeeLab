#!/usr/bin/env python3
"""Merge E14S/E15S filtered_feature_bc_matrix.h5 (GEX only) into one query .h5ad for MALT."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import scanpy as sc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project root containing E14S/ and E15S/.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("query_mc38_e14s_e15s_gex.h5ad"),
        help="Output path for merged query AnnData.",
    )
    args = p.parse_args()
    root = args.root.resolve()

    paths = {
        "E14S": root / "E14S" / "filtered_feature_bc_matrix.h5",
        "E15S": root / "E15S" / "filtered_feature_bc_matrix.h5",
    }
    for sid, pth in paths.items():
        if not pth.is_file():
            raise FileNotFoundError(pth)

    parts = []
    for sid, pth in paths.items():
        a = sc.read_10x_h5(pth, gex_only=True)
        a.var_names_make_unique()
        a.obs_names = a.obs_names.astype(str) + "_" + sid
        a.obs["sample"] = sid
        parts.append(a)

    adata = ad.concat(parts, join="outer", merge="same")
    adata.obs_names_make_unique()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.out, compression="gzip")
    print(adata)
    print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
