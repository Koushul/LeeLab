#!/usr/bin/env python3
"""Place E30S / E31S spatial cells with 4-barcode ADT entropy (E28S method).

E31S OB3/OB4 (standalone perturb-seq, spatial=False) are excluded.
V11 and V70 are separate chips — never overlaid.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGS = ROOT / "figures"
for p in (DATA, FIGS):
    p.mkdir(parents=True, exist_ok=True)

E30 = Path("/ix1/ylee/kor11/spatial_crispr/analysis/outputs/E30S_gex_adt_guide_ocm.h5ad")
E31 = Path("/ix1/ylee/kor11/spatial_crispr/analysis/outputs/E31S_gex_adt_guide_ocm.h5ad")
EXPORT_DIR = Path("/ix1/ylee/kor11/tools/spac_analysis/e28s_4bc_spatial")
DATA_JS = EXPORT_DIR / "data.js"

H_CUT = 8.0
GRID = 48


def entropy_bits_counts(counts: np.ndarray) -> np.ndarray:
    """Shannon entropy (bits) of non-negative count rows."""
    tot = counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.divide(counts, tot, out=np.zeros_like(counts, dtype=float), where=tot > 0)
        logp = np.log2(p, out=np.zeros_like(p), where=p > 0)
        return -np.sum(p * logp, axis=1)


def peak_mass(counts: np.ndarray) -> np.ndarray:
    tot = counts.sum(axis=1)
    mx = counts.max(axis=1)
    return np.where(tot > 0, mx / np.maximum(tot, 1e-12), 0.0)


def compute_adt_entropy(adata) -> pd.DataFrame:
    av = pd.DataFrame(adata.uns["ADT_var"])
    names = av["feature_name"].astype(str).tolist()
    sbc_idx = {int(n[3:]): i for i, n in enumerate(names) if n.startswith("sbc")}
    missing = [i for i in range(1, 193) if i not in sbc_idx]
    if missing:
        raise SystemExit(f"Missing sbc features: {missing[:10]}...")
    X = adata.obsm["ADT"]
    if sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=float)
    S = np.stack([X[:, sbc_idx[i]] for i in range(1, 193)], axis=1)
    row = S[:, 0:48] + S[:, 144:192]
    col = S[:, 48:96] + S[:, 96:144]
    H_row = entropy_bits_counts(row)
    H_col = entropy_bits_counts(col)
    H = H_row + H_col
    map_r = row.argmax(1) + 1
    map_c = col.argmax(1) + 1
    conf = peak_mass(row) * peak_mass(col)
    return pd.DataFrame(
        {
            "H_row_adt": H_row,
            "H_col_adt": H_col,
            "H_adt": H,
            "map_row_adt": map_r.astype(int),
            "map_col_adt": map_c.astype(int),
            "conf_adt": conf,
            "umi_sbc": S.sum(1).astype(int),
        },
        index=adata.obs_names.astype(str),
    )


def compute_4bc_soft(adata) -> pd.DataFrame:
    sys.path.insert(0, str(EXPORT_DIR))
    from export_cells import assign, load_layout, plate_axis_counts

    layout = load_layout(DATA_JS)
    blocks = plate_axis_counts(adata, layout)
    payload = assign(blocks, adata.obs_names.astype(str).tolist())
    return pd.DataFrame(
        {
            "map_row_4bc": payload["map_row"],
            "map_col_4bc": payload["map_col"],
            "conf_4bc": payload["confidence"],
            "H_row_4bc": payload["row_entropy"],
            "H_col_4bc": payload["col_entropy"],
            "H_4bc": payload["total_entropy"],
            "layout_umi_4bc": payload["total_counts"],
        },
        index=payload["obs_names"],
    )


def spatial_mask(obs: pd.DataFrame) -> np.ndarray:
    if "spatial" not in obs.columns:
        return np.ones(len(obs), dtype=bool)
    s = obs["spatial"]
    if s.dtype == bool:
        return s.to_numpy()
    return s.astype(str).str.lower().isin(["true", "1", "yes"]).to_numpy()


def meta_cols(obs: pd.DataFrame) -> pd.DataFrame:
    keep = [
        c
        for c in [
            "ocm_barcode_id",
            "sample_id",
            "vector",
            "spatial",
            "chip",
            "replicate_group",
            "standalone_perturbseq",
            "gex_counts",
            "adt_counts",
            "description",
        ]
        if c in obs.columns
    ]
    out = obs[keep].copy()
    out.index = obs.index.astype(str)
    return out


def place_dataset(name: str, path: Path) -> tuple[pd.DataFrame, dict]:
    print(f"\n=== {name} ===\nLoading {path}")
    adata = sc.read_h5ad(path)
    mask = spatial_mask(adata.obs)
    n_excl = int((~mask).sum())
    if n_excl:
        print(f"Excluding {n_excl} non-spatial cells (perturb-seq / spatial=False)")
    adata = adata[mask].copy()
    print(f"Spatial cells: {adata.n_obs}")

    ent = compute_adt_entropy(adata)
    soft = compute_4bc_soft(adata)
    meta = meta_cols(adata.obs)
    df = meta.join(ent).join(soft)
    df["experiment"] = name
    df["barcode"] = df.index
    df["placed_H8"] = df["H_adt"] <= H_CUT
    df["x"] = df["map_row_adt"].astype(float)
    df["y"] = df["map_col_adt"].astype(float)

    # attach to AnnData
    for c in [
        "H_adt",
        "H_row_adt",
        "H_col_adt",
        "map_row_adt",
        "map_col_adt",
        "conf_adt",
        "umi_sbc",
        "H_4bc",
        "map_row_4bc",
        "map_col_4bc",
        "conf_4bc",
        "placed_H8",
    ]:
        adata.obs[c] = df.loc[adata.obs_names.astype(str), c].values
    xy = np.column_stack([df.loc[adata.obs_names.astype(str), "x"], df.loc[adata.obs_names.astype(str), "y"]])
    adata.obsm["spatial"] = xy
    adata.obsm["spatial_H8"] = xy.copy()
    adata.obsm["spatial_H8"][~df.loc[adata.obs_names.astype(str), "placed_H8"].to_numpy()] = np.nan
    adata.uns["spatial_placement"] = {
        "method": "ADT 4-barcode entropy (row/col plate aggregates)",
        "H_cut": H_CUT,
        "grid": GRID,
        "layout": "row=sbc1-48+sbc145-192; col=sbc49-96+sbc97-144",
        "chip_rule": "never overlay v11_spatial_chip with v70_spatial_chip",
    }

    out_csv = DATA / f"{name}_assignments.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(df)} cells)")

    out_h5 = DATA / f"{name}_placed.h5ad"
    adata.write_h5ad(out_h5)
    print(f"Wrote {out_h5}")

    # also drop filtered object
    placed = adata[adata.obs["placed_H8"]].copy()
    out_h5f = DATA / f"{name}_placed_Hle{int(H_CUT)}.h5ad"
    placed.write_h5ad(out_h5f)
    print(f"Wrote {out_h5f} ({placed.n_obs} cells)")

    summary = summarize(name, df)
    return df, summary


def summarize(name: str, df: pd.DataFrame) -> dict:
    def block(sub: pd.DataFrame) -> dict:
        return {
            "n": int(len(sub)),
            "n_pass_H8": int(sub["placed_H8"].sum()),
            "frac_pass_H8": float(sub["placed_H8"].mean()) if len(sub) else 0.0,
            "H_adt_median": float(sub["H_adt"].median()) if len(sub) else None,
            "H_adt_mean": float(sub["H_adt"].mean()) if len(sub) else None,
            "umi_sbc_median": float(sub["umi_sbc"].median()) if len(sub) else None,
            "agree_map_adt_vs_4bc": float(
                ((sub["map_row_adt"] == sub["map_row_4bc"]) & (sub["map_col_adt"] == sub["map_col_4bc"])).mean()
            )
            if len(sub)
            else None,
            "agree_map_adt_vs_4bc_H8": float(
                (
                    (sub.loc[sub["placed_H8"], "map_row_adt"] == sub.loc[sub["placed_H8"], "map_row_4bc"])
                    & (sub.loc[sub["placed_H8"], "map_col_adt"] == sub.loc[sub["placed_H8"], "map_col_4bc"])
                ).mean()
            )
            if sub["placed_H8"].any()
            else None,
        }

    by_chip = {}
    if "chip" in df.columns:
        for chip, sub in df.groupby("chip", observed=True):
            by_chip[str(chip)] = block(sub)
    by_ocm = {}
    if "ocm_barcode_id" in df.columns:
        for ocm, sub in df.groupby("ocm_barcode_id", observed=True):
            by_ocm[str(ocm)] = block(sub)

    return {
        "experiment": name,
        "H_cut": H_CUT,
        "overall": block(df),
        "by_chip": by_chip,
        "by_ocm_barcode_id": by_ocm,
    }


def plot_qc(df: pd.DataFrame, name: str):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    ax = axes[0]
    ax.hist(df["H_adt"], bins=40, color="#2a9d8f", edgecolor="white", lw=0.4)
    ax.axvline(H_CUT, color="#e76f51", ls="--", lw=1.2, label=f"H≤{H_CUT:g}")
    ax.set_xlabel("H_adt (bits)")
    ax.set_ylabel("cells")
    ax.set_title(f"{name} ADT entropy")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    chips = sorted(df["chip"].astype(str).unique()) if "chip" in df.columns else [name]
    cmap = {"v11_spatial_chip": "#264653", "v70_spatial_chip": "#e9c46a"}
    for chip in chips:
        sub = df[df["chip"].astype(str) == chip] if "chip" in df.columns else df
        ax.hist(sub["H_adt"], bins=30, alpha=0.55, label=chip, color=cmap.get(chip, "#457b9d"))
    ax.axvline(H_CUT, color="#e76f51", ls="--", lw=1.2)
    ax.set_xlabel("H_adt (bits)")
    ax.set_title("by chip")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[2]
    for chip in chips:
        sub = df[(df["chip"].astype(str) == chip) & df["placed_H8"]] if "chip" in df.columns else df[df["placed_H8"]]
        ax.scatter(
            sub["map_col_adt"],
            sub["map_row_adt"],
            s=4,
            alpha=0.35,
            c=cmap.get(chip, "#457b9d"),
            label=f"{chip} (H≤{H_CUT:g})",
            lw=0,
        )
    ax.set_xlim(0.5, GRID + 0.5)
    ax.set_ylim(GRID + 0.5, 0.5)
    ax.set_xlabel("map_col_adt")
    ax.set_ylabel("map_row_adt")
    ax.set_title("MAP wells (do not overlay chips)")
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    fig.tight_layout()
    out = FIGS / f"{name}_qc.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}")

    # per-chip separate MAP panels
    if "chip" in df.columns:
        uniq = sorted(df["chip"].astype(str).unique())
        fig, axes = plt.subplots(1, len(uniq), figsize=(4.2 * len(uniq), 4.0), squeeze=False)
        for i, chip in enumerate(uniq):
            ax = axes[0, i]
            sub = df[(df["chip"].astype(str) == chip) & df["placed_H8"]]
            ax.scatter(sub["map_col_adt"], sub["map_row_adt"], s=6, alpha=0.4, c=cmap.get(chip, "#457b9d"), lw=0)
            ax.set_xlim(0.5, GRID + 0.5)
            ax.set_ylim(GRID + 0.5, 0.5)
            ax.set_title(f"{name} · {chip}\nn={len(sub)} H≤{H_CUT:g}")
            ax.set_xlabel("col")
            ax.set_ylabel("row")
        fig.tight_layout()
        out = FIGS / f"{name}_map_by_chip.png"
        fig.savefig(out, dpi=160)
        plt.close(fig)
        print(f"Wrote {out}")


def main():
    all_dfs = []
    summaries = {
        "algorithm": {
            "primary": "H_adt = H(row)+H(col) on 4-BC ADT aggregates; MAP=argmax",
            "H_cut": H_CUT,
            "secondary": "soft multinomial 4-BC posterior (export_cells.py / data.js layout)",
            "exclude": "E31S OB3/OB4 non-spatial perturb-seq",
            "chip_rule": "v11_spatial_chip and v70_spatial_chip must never be overlaid",
        },
        "datasets": {},
    }
    for name, path in (("E30S", E30), ("E31S", E31)):
        df, summary = place_dataset(name, path)
        plot_qc(df, name)
        summaries["datasets"][name] = summary
        all_dfs.append(df)

    combined = pd.concat(all_dfs, axis=0)
    combined.to_csv(DATA / "E30S_E31S_assignments.csv", index=False)

    # cross-experiment chip replicate tables
    chip_rep = []
    for chip, sub in combined.groupby("chip", observed=True):
        chip_rep.append(
            {
                "chip": str(chip),
                "n": int(len(sub)),
                "n_pass_H8": int(sub["placed_H8"].sum()),
                "frac_pass_H8": float(sub["placed_H8"].mean()),
                "experiments": sorted(sub["experiment"].unique().tolist()),
                "ocm_lanes": sorted(sub["ocm_barcode_id"].astype(str).unique().tolist())
                if "ocm_barcode_id" in sub.columns
                else [],
            }
        )
    summaries["chip_replicates"] = chip_rep

    out_json = DATA / "placement_summary.json"
    out_json.write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"\nWrote {out_json}")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
