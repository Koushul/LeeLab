#!/usr/bin/env python3
"""Microwell posteriors from raw spatial ADT counts + per-cell entropy."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

OUT = Path(
    "/ix1/ylee/shared/MC38_Hypoxia_001/reseq_finetune/deep_dive/spatial_niches/front_figures"
)
DIAG = OUT / "diagnostics"
DIAG.mkdir(parents=True, exist_ok=True)

ROWS = [f"sbc{i}" for i in range(1, 43)]
COLS = [f"sbc{i}" for i in range(97, 139)]
N_WELLS = len(ROWS) * len(COLS)
ALPHA = 0.05
LOG2_N = float(np.log2(N_WELLS))

RP = {
    "base": "#191724",
    "surface": "#1f1d2e",
    "overlay": "#26233a",
    "muted": "#6e6a86",
    "subtle": "#908caa",
    "text": "#e0def4",
    "love": "#eb6f92",
    "gold": "#f6c177",
    "foam": "#9ccfd8",
    "iris": "#c4a7e7",
    "pine": "#31748f",
    "hl": "#403d52",
}


def load_adt(sample: str) -> pd.DataFrame:
    path = f"/ix1/ylee/shared/MC38_Hypoxia_001/{sample}/filtered_feature_bc_matrix.h5"
    ad = sc.read_10x_h5(path, gex_only=False)
    ad.var_names_make_unique()
    ft = ad.var["feature_types"].astype(str)
    prot = ad[:, ft == "Antibody Capture"].to_df()
    missing = [c for c in ROWS + COLS if c not in prot.columns]
    if missing:
        raise RuntimeError(f"{sample}: missing ADTs {missing[:5]}...")
    return prot[ROWS + COLS].astype(float)


def well_posterior(row_counts: np.ndarray, col_counts: np.ndarray, alpha: float = ALPHA):
    """P(well i,j) = p_row[i] * p_col[j] with Dirichlet pseudocounts."""
    pr = row_counts + alpha
    pc = col_counts + alpha
    pr = pr / pr.sum()
    pc = pc / pc.sum()
    h_row = float(-np.sum(pr * np.log2(pr)))
    h_col = float(-np.sum(pc * np.log2(pc)))
    h_joint = h_row + h_col
    map_i = int(np.argmax(pr))
    map_j = int(np.argmax(pc))
    map_p = float(pr[map_i] * pc[map_j])
    return {
        "p_row": pr,
        "p_col": pc,
        "entropy_bits": h_joint,
        "entropy_row_bits": h_row,
        "entropy_col_bits": h_col,
        "entropy_norm": h_joint / LOG2_N,
        "map_row": ROWS[map_i],
        "map_col": COLS[map_j],
        "map_row_idx": map_i + 1,
        "map_col_idx": map_j + 1,
        "map_prob": map_p,
        "p_row_max": float(pr[map_i]),
        "p_col_max": float(pc[map_j]),
    }


def score_sample(sample: str) -> pd.DataFrame:
    prot = load_adt(sample)
    R = prot[ROWS].to_numpy()
    C = prot[COLS].to_numpy()
    rows = []
    for i, bc in enumerate(prot.index):
        out = well_posterior(R[i], C[i])
        rows.append(
            {
                "barcode": bc,
                "barcode_raw": str(bc).split("-")[0],
                "sample": sample,
                "adt_umi_row": float(R[i].sum()),
                "adt_umi_col": float(C[i].sum()),
                "adt_umi_total": float(R[i].sum() + C[i].sum()),
                "n_nonzero_adt": int(((R[i] > 0).sum() + (C[i] > 0).sum())),
                "entropy_bits": out["entropy_bits"],
                "entropy_row_bits": out["entropy_row_bits"],
                "entropy_col_bits": out["entropy_col_bits"],
                "entropy_norm": out["entropy_norm"],
                "map_row": out["map_row"],
                "map_col": out["map_col"],
                "map_row_idx": out["map_row_idx"],
                "map_col_idx": out["map_col_idx"],
                "map_prob": out["map_prob"],
                "p_row_max": out["p_row_max"],
                "p_col_max": out["p_col_max"],
            }
        )
    return pd.DataFrame(rows)


def load_hard_coords() -> pd.DataFrame:
    ad = sc.read_h5ad(
        "/ix1/ylee/shared/MC38_Hypoxia_001/reseq_finetune/mc38_tumor_reseq_finetuned.h5ad",
        backed="r",
    )
    obs = ad.obs[["sample", "spatial_coordinate_x", "spatial_coordinate_y"]].copy()
    obs["barcode_raw"] = [n.split("_")[0].split("-")[0] for n in ad.obs_names]
    ad.file.close()
    return obs


def style_ax(ax):
    ax.set_facecolor(RP["surface"])
    for sp in ax.spines.values():
        sp.set_color(RP["hl"])
    ax.tick_params(colors=RP["subtle"])
    ax.xaxis.label.set_color(RP["subtle"])
    ax.yaxis.label.set_color(RP["subtle"])
    ax.title.set_color(RP["text"])


def make_figures(df: pd.DataFrame, obs: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2))
    fig.patch.set_facecolor(RP["base"])

    for si, sample in enumerate(["E14S", "E15S"]):
        sub = df[df["sample"] == sample]
        gate = "GFP−" if sample == "E14S" else "GFP+"

        ax = axes[si, 0]
        style_ax(ax)
        ax.hist(
            sub["entropy_bits"],
            bins=40,
            color=RP["foam"] if sample == "E15S" else RP["love"],
            edgecolor=RP["base"],
            alpha=0.92,
        )
        ax.axvline(sub["entropy_bits"].median(), color=RP["gold"], ls="--", lw=1.2)
        ax.set_xlabel("microwell entropy (bits)")
        ax.set_ylabel("cells")
        ax.set_title(f"{sample} {gate} · H(well | ADT)")

        ax = axes[si, 1]
        style_ax(ax)
        ax.hist(
            sub["entropy_norm"],
            bins=40,
            color=RP["iris"],
            edgecolor=RP["base"],
            alpha=0.92,
        )
        ax.axvline(sub["entropy_norm"].median(), color=RP["gold"], ls="--", lw=1.2)
        ax.set_xlabel(f"H / log2({N_WELLS} wells)")
        ax.set_ylabel("cells")
        ax.set_title(f"{sample} · normalized entropy")

        ax = axes[si, 2]
        style_ax(ax)
        hb = ax.hexbin(
            np.log1p(sub["adt_umi_total"]),
            sub["entropy_bits"],
            gridsize=35,
            cmap="magma",
            mincnt=1,
        )
        ax.set_xlabel("log1p(ADT UMI)")
        ax.set_ylabel("entropy (bits)")
        ax.set_title(f"{sample} · depth vs entropy")
        cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_tick_params(color=RP["subtle"])
        plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=RP["subtle"])

    fig.suptitle(
        "Microwell posterior entropy from raw spatial ADTs",
        color=RP["text"],
        fontsize=14,
        fontweight="600",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "microwell_entropy_distribution.png", dpi=180, facecolor=RP["base"])
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 10.2))
    fig.patch.set_facecolor(RP["base"])
    for row_i, use_map in enumerate([False, True]):
        for col_i, sample in enumerate(["E14S", "E15S"]):
            ax = axes[row_i, col_i]
            style_ax(ax)
            sub = df[df["sample"] == sample].copy()
            if use_map:
                x = pd.to_numeric(sub["map_row_idx"], errors="coerce")
                y = pd.to_numeric(sub["map_col_idx"], errors="coerce")
                vals = sub["entropy_bits"].to_numpy()
                title_suffix = "MAP microwells"
            else:
                o = obs[obs["sample"] == sample].copy()
                merged = o.join(
                    sub.set_index("barcode_raw")[["entropy_bits", "map_prob"]],
                    on="barcode_raw",
                    how="inner",
                )
                x = pd.to_numeric(merged["spatial_coordinate_x"], errors="coerce")
                y = pd.to_numeric(merged["spatial_coordinate_y"], errors="coerce")
                vals = merged["entropy_bits"].to_numpy()
                title_suffix = "hard demux (artifact)"
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() == 0:
                ax.set_title(f"{sample} · no coord overlap")
                continue
            vmax = float(np.nanpercentile(vals[m], 99)) if m.sum() else 8.0
            sca = ax.scatter(
                x[m],
                y[m],
                c=vals[m],
                s=6,
                cmap="magma",
                linewidths=0,
                alpha=0.85,
                vmin=0,
                vmax=max(4.0, vmax),
            )
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            gate = "GFP−" if sample == "E14S" else "GFP+"
            ax.set_title(f"{sample} {gate} · {title_suffix}")
            cb = fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.02)
            cb.set_label("bits", color=RP["subtle"])
            cb.ax.yaxis.set_tick_params(color=RP["subtle"])
            plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=RP["subtle"])
    fig.suptitle(
        "High entropy = ambiguous microwell · top=old demux · bottom=MAP (analysis coords)",
        color=RP["text"],
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "microwell_entropy_spatial.png", dpi=180, facecolor=RP["base"])
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    fig.patch.set_facecolor(RP["base"])
    for ax, sample in zip(axes, ["E14S", "E15S"]):
        style_ax(ax)
        sub = df[df["sample"] == sample]
        zero = (sub["adt_umi_row"] == 0) | (sub["adt_umi_col"] == 0)
        ax.hist(
            sub.loc[~zero, "entropy_bits"],
            bins=35,
            color=RP["pine"],
            alpha=0.85,
            label=f"both halves >0 (n={(~zero).sum()})",
            density=True,
        )
        ax.hist(
            sub.loc[zero, "entropy_bits"],
            bins=35,
            color=RP["love"],
            alpha=0.65,
            label=f"empty half (n={zero.sum()})",
            density=True,
        )
        ax.set_xlabel("entropy (bits)")
        ax.set_ylabel("density")
        ax.set_title(sample)
        leg = ax.legend(fontsize=8, frameon=False)
        for t in leg.get_texts():
            t.set_color(RP["subtle"])
    fig.suptitle(
        "Empty ADT halves → near-uniform posterior → high entropy",
        color=RP["text"],
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "microwell_entropy_empty_vs_signal.png", dpi=180, facecolor=RP["base"])
    plt.close(fig)


def summarize(df: pd.DataFrame) -> dict:
    out = {
        "model": {
            "description": (
                "Each microwell (i,j) is the known pair (row_i, col_j). "
                "With Dirichlet pseudocount α on raw ADT UMIs, "
                "p_row ∝ row_counts+α, p_col ∝ col_counts+α, "
                "P(well=i,j)=p_row[i]*p_col[j]. "
                "Entropy H=-∑ P log2 P = H_row+H_col (bits)."
            ),
            "rows": ROWS[0] + ".." + ROWS[-1],
            "cols": COLS[0] + ".." + COLS[-1],
            "n_wells": N_WELLS,
            "alpha": ALPHA,
            "max_entropy_bits": LOG2_N,
        },
        "samples": {},
    }
    for sample in ["E14S", "E15S"]:
        sub = df[df["sample"] == sample]
        zero = (sub["adt_umi_row"] == 0) | (sub["adt_umi_col"] == 0)
        out["samples"][sample] = {
            "n_cells": int(len(sub)),
            "entropy_bits_mean": round(float(sub["entropy_bits"].mean()), 3),
            "entropy_bits_median": round(float(sub["entropy_bits"].median()), 3),
            "entropy_norm_median": round(float(sub["entropy_norm"].median()), 3),
            "map_prob_median": round(float(sub["map_prob"].median()), 4),
            "frac_entropy_norm_lt_0.25": round(float((sub["entropy_norm"] < 0.25).mean()), 3),
            "frac_entropy_norm_gt_0.75": round(float((sub["entropy_norm"] > 0.75).mean()), 3),
            "empty_half_n": int(zero.sum()),
            "empty_half_entropy_median": round(float(sub.loc[zero, "entropy_bits"].median()), 3)
            if zero.any()
            else None,
            "both_halves_entropy_median": round(float(sub.loc[~zero, "entropy_bits"].median()), 3),
        }
    return out


def main():
    frames = [score_sample(s) for s in ["E14S", "E15S"]]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(OUT / "microwell_entropy_per_cell.csv", index=False)
    df.to_csv(DIAG / "microwell_entropy_per_cell.csv", index=False)

    obs = load_hard_coords()
    make_figures(df, obs)
    summary = summarize(df)
    (OUT / "microwell_entropy_summary.json").write_text(json.dumps(summary, indent=2))
    (DIAG / "microwell_entropy_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("wrote figures to", OUT)


if __name__ == "__main__":
    main()
