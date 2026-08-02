#!/usr/bin/env python3
"""Regenerate key MC38 hypoxia figures in an elegant Rose Pine theme."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors

# Rose Pine
RP = {
    "base": "#191724",
    "surface": "#1f1d2e",
    "overlay": "#26233a",
    "muted": "#6e6a86",
    "subtle": "#908caa",
    "text": "#e0def4",
    "love": "#eb6f92",
    "gold": "#f6c177",
    "rose": "#ebbcba",
    "pine": "#31748f",
    "foam": "#9ccfd8",
    "iris": "#c4a7e7",
    "hl_low": "#21202e",
    "hl_med": "#403d52",
    "hl_high": "#524f67",
}

STATE_COLORS = {
    "entering_hypoxia": RP["love"],
    "entering_deep_hypoxia": "#b4637a",
    "persistent_hypoxia": RP["iris"],
    "exiting_hypoxia": RP["foam"],
    "normoxic_stable": RP["pine"],
    "transitional": RP["gold"],
    "reverted_stable": RP["rose"],
    "reverted_posthypoxic": RP["muted"],
}

ENTER = {"entering_hypoxia", "entering_deep_hypoxia"}
EXIT = {"exiting_hypoxia"}

OUT = Path("/ix1/ylee/shared/MC38_Hypoxia_001/reseq_finetune/deep_dive/spatial_niches/front_figures")
OUT_DD = Path("/ix1/ylee/shared/MC38_Hypoxia_001/reseq_finetune/deep_dive")
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(11)


def apply_theme():
    mpl.rcParams.update(
        {
            "figure.facecolor": RP["base"],
            "axes.facecolor": RP["surface"],
            "savefig.facecolor": RP["base"],
            "axes.edgecolor": RP["hl_med"],
            "axes.labelcolor": RP["subtle"],
            "xtick.color": RP["muted"],
            "ytick.color": RP["muted"],
            "text.color": RP["text"],
            "axes.titlecolor": RP["text"],
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
        }
    )


def style_ax(ax, grid=False):
    ax.set_facecolor(RP["surface"])
    for sp in ax.spines.values():
        sp.set_color(RP["hl_med"])
        sp.set_linewidth(0.8)
    ax.tick_params(colors=RP["muted"], length=3)
    if grid:
        ax.grid(True, color=RP["hl_low"], lw=0.6, alpha=0.8)


def load():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from h8_spatial import load_h8_adata

    adata = load_h8_adata()
    dyn = pd.read_csv(
        "/ix1/ylee/shared/MC38_Hypoxia_001/reseq_finetune/hypoxia_velocity/hypoxia_dynamics_per_cell.csv"
    ).set_index("barcode")
    for c in ["prolif_score"]:
        if c in dyn.columns:
            common = adata.obs_names.intersection(dyn.index)
            adata.obs.loc[common, c] = dyn.loc[common, c].values
    return adata


def sample_xy(adata, sample):
    if sample in (None, "OVERLAY"):
        m = np.isfinite(adata.obsm["spatial"]).all(1)
        sub = adata[m]
    else:
        m = (adata.obs["sample"] == sample).to_numpy() & np.isfinite(adata.obsm["spatial"]).all(1)
        sub = adata[m]
    return sub, np.asarray(sub.obsm["spatial"], float)


def draw_edges(ax, xy, src_mask, tgt_mask, k=3, max_dist=None, color=RP["gold"], lw=0.55, alpha=0.45):
    if src_mask.sum() < 2 or tgt_mask.sum() < 2:
        return 0
    nn = NearestNeighbors(n_neighbors=min(k, int(tgt_mask.sum()))).fit(xy[tgt_mask])
    d, idx = nn.kneighbors(xy[src_mask])
    tgt_xy = xy[tgt_mask]
    src_xy = xy[src_mask]
    n = 0
    for i in range(len(src_xy)):
        for j, dist in zip(idx[i], d[i]):
            if max_dist is not None and dist > max_dist:
                continue
            ax.plot(
                [src_xy[i, 0], tgt_xy[j, 0]],
                [src_xy[i, 1], tgt_xy[j, 1]],
                color=color,
                lw=lw,
                alpha=alpha,
                zorder=2,
                solid_capstyle="round",
            )
            n += 1
    return n


def fig_front_bridges(adata):
    for sample, gate in [("E14S", "GFP−"), ("E15S", "GFP+")]:
        sub, xy = sample_xy(adata, sample)
        labs = sub.obs["hypoxia_dynamics"].astype(str).to_numpy()
        tumor = sub.obs["is_tumor"].to_numpy()
        nn = NearestNeighbors(n_neighbors=2).fit(xy)
        dmed = float(np.median(nn.kneighbors(xy)[0][:, 1]))
        max_dist = dmed * 4.0

        fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.9), facecolor=RP["base"])
        # left continuum
        ax = axes[0]
        style_ax(ax)
        ax.scatter(xy[~tumor, 0], xy[~tumor, 1], s=1.8, c=RP["hl_med"], alpha=0.35, linewidths=0, zorder=0)
        order = [
            "normoxic_stable",
            "exiting_hypoxia",
            "persistent_hypoxia",
            "entering_hypoxia",
            "entering_deep_hypoxia",
        ]
        for st in order:
            m = (labs == st) & tumor
            if m.sum() == 0:
                continue
            ax.scatter(
                xy[m, 0],
                xy[m, 1],
                s=16,
                c=STATE_COLORS[st],
                alpha=0.92,
                linewidths=0,
                label=st.replace("_", " "),
                zorder=3,
                rasterized=True,
            )
        ax.set_aspect("equal")
        ax.set_title(f"{sample} {gate}  ·  hypoxia continuum", pad=10)
        leg = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7.5, labelcolor=RP["subtle"])
        for t in leg.get_texts():
            t.set_color(RP["subtle"])
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

        # right bridges
        ax = axes[1]
        style_ax(ax)
        ax.scatter(xy[:, 0], xy[:, 1], s=1.5, c=RP["hl_med"], alpha=0.28, linewidths=0, zorder=0)
        enter_m = np.isin(labs, list(ENTER)) & tumor
        exit_m = np.isin(labs, list(EXIT)) & tumor
        persist_m = (labs == "persistent_hypoxia") & tumor
        ax.scatter(xy[persist_m, 0], xy[persist_m, 1], s=9, c=RP["iris"], alpha=0.4, linewidths=0, zorder=2)
        draw_edges(ax, xy, enter_m, exit_m, k=3, max_dist=max_dist, color=RP["gold"], lw=0.65, alpha=0.5)
        ax.scatter(xy[exit_m, 0], xy[exit_m, 1], s=18, c=RP["foam"], alpha=0.95, linewidths=0, label="exiting", zorder=4)
        ax.scatter(xy[enter_m, 0], xy[enter_m, 1], s=18, c=RP["love"], alpha=0.95, linewidths=0, label="entering", zorder=4)
        ax.set_aspect("equal")
        ax.set_title(f"{sample} {gate}  ·  enter↔exit front", pad=10)
        leg = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, labelcolor=RP["subtle"])
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

        fig.suptitle("Hypoxia organized as a front", color=RP["text"], fontsize=14, y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(OUT / f"{sample}_front_bridges.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def fig_enrichment_bars():
    df = pd.read_csv(OUT / "front_enrichment_table.csv")
    pairs = ["deep → exit", "enter → exit", "persist → exit", "normoxic → exit"]
    fig, ax = plt.subplots(figsize=(9.2, 4.8), facecolor=RP["base"])
    style_ax(ax, grid=True)
    x = np.arange(len(pairs))
    w = 0.34
    for i, sample, col in [(0, "E14S", RP["love"]), (1, "E15S", RP["foam"])]:
        vals, ps = [], []
        for lab in pairs:
            hit = df[(df["sample"] == sample) & (df["pair"] == lab)]
            vals.append(float(hit.iloc[0]["enrichment"]) if len(hit) else np.nan)
            ps.append(float(hit.iloc[0]["p_perm"]) if len(hit) else np.nan)
        ax.bar(x + (i - 0.5) * w, vals, width=w, color=col, label=sample, alpha=0.92, edgecolor=RP["base"], linewidth=0.4)
        for xi, v, p in zip(x + (i - 0.5) * w, vals, ps):
            if np.isfinite(v) and np.isfinite(p) and p < 0.05:
                ax.text(xi, v + 0.08, "✦", ha="center", va="bottom", color=col, fontsize=11)
    ax.axhline(1, color=RP["muted"], ls="--", lw=0.9, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pairs)
    ax.set_ylabel("Neighbor enrichment (obs / exp)")
    ax.set_title("Dynamics adjacency along the hypoxia front")
    ax.legend(loc="upper right", labelcolor=RP["subtle"])
    fig.tight_layout()
    fig.savefig(OUT / "front_enrichment_bars.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_coord_shuffle_summary():
    tab = pd.read_csv(OUT / "coord_shuffle_front_test.csv")
    plot_df = tab[(tab["metric"] == "enr_deep_exit") & (tab["mode"].isin(["coord_all", "coord_tumor"]))].copy()
    fig, ax = plt.subplots(figsize=(9.0, 4.4), facecolor=RP["base"])
    style_ax(ax, grid=True)
    y = np.arange(len(plot_df))
    ax.hlines(y, plot_df.null_mean, plot_df.observed, color=RP["hl_high"], lw=1.8)
    ax.scatter(plot_df.null_mean, y, c=RP["muted"], s=48, label="null mean", zorder=3, edgecolors=RP["base"], linewidths=0.4)
    cols = [RP["love"] if p < 0.05 else RP["subtle"] for p in plot_df.p]
    ax.scatter(plot_df.observed, y, c=cols, s=64, label="observed", zorder=3, edgecolors=RP["base"], linewidths=0.4)
    ax.axvline(1, color=RP["muted"], ls="--", lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{r.sample} · {'all coords' if r.mode == 'coord_all' else 'tumor coords'}" for r in plot_df.itertuples()],
        fontsize=9,
        color=RP["subtle"],
    )
    for yi, r in zip(y, plot_df.itertuples()):
        ax.text(
            max(r.observed, r.null_mean) + 0.05,
            yi,
            f"p={r.p:.3g}  z={r.z:.1f}",
            va="center",
            fontsize=8,
            color=RP["subtle"],
        )
    ax.set_xlabel("deep → exit neighbor enrichment")
    ax.set_title("Coordinate-shuffle test of front adjacency")
    ax.legend(loc="lower right", labelcolor=RP["subtle"])
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT / "coord_shuffle_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_coord_shuffle_hists(adata):
    # rebuild nulls quickly for pretty plots
    N_PERM = 2000
    K = 30

    def get_nulls(sample, mode):
        sub, xy = sample_xy(adata, sample)
        labs = sub.obs["hypoxia_dynamics"].astype(str).to_numpy()
        tumor = sub.obs["is_tumor"].to_numpy()
        knn = NearestNeighbors(n_neighbors=min(K + 1, len(xy))).fit(xy).kneighbors(xy, return_distance=False)[:, 1:]

        def enr(labs_arr):
            deep = (labs_arr == "entering_deep_hypoxia") & tumor
            exit_ = (labs_arr == "exiting_hypoxia") & tumor
            if deep.sum() < 5 or exit_.sum() < 5:
                return np.nan
            return float(exit_[knn[deep]].mean() / max(exit_.mean(), 1e-12))

        def front(labs_arr):
            enter = np.isin(labs_arr, list(ENTER)) & tumor
            exit_ = (labs_arr == "exiting_hypoxia") & tumor
            ef = (enter.astype(float)[knn].mean(1) * K + enter.astype(float)) / (K + 1)
            xf = (exit_.astype(float)[knn].mean(1) * K + exit_.astype(float)) / (K + 1)
            return float(np.mean(np.sqrt(ef * xf)))

        obs_e, obs_f = enr(labs), front(labs)
        null_e = np.empty(N_PERM)
        null_f = np.empty(N_PERM)
        for i in range(N_PERM):
            lp = labs.copy()
            if mode == "coord_all":
                lp = RNG.permutation(labs)
            else:
                tidx = np.where(tumor)[0]
                lp[tidx] = RNG.permutation(labs[tidx])
            null_e[i] = enr(lp)
            null_f[i] = front(lp)
        return obs_e, null_e, obs_f, null_f

    for metric, fname, xlab, obs_col in [
        ("enr", "coord_shuffle_deep_exit_enrichment.png", "deep → exit enrichment", RP["love"]),
        ("front", "coord_shuffle_front_score.png", "front score", RP["gold"]),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.0), facecolor=RP["base"])
        for row, sample in enumerate(["E14S", "E15S"]):
            for col, mode in enumerate(["coord_all", "coord_tumor"]):
                ax = axes[row, col]
                style_ax(ax)
                oe, ne, of, nf = get_nulls(sample, mode)
                a = ne if metric == "enr" else nf
                o = oe if metric == "enr" else of
                a = a[np.isfinite(a)]
                ax.hist(a, bins=48, color=RP["overlay"], edgecolor=RP["hl_med"], density=True, alpha=0.95)
                ax.axvline(o, color=obs_col, lw=2.2, label=f"obs = {o:.3g}")
                if metric == "enr":
                    ax.axvline(1.0, color=RP["muted"], ls="--", lw=0.9)
                p = (np.sum(a >= o) + 1) / (len(a) + 1)
                z = (o - a.mean()) / (a.std() + 1e-12)
                title = "all coordinates" if mode == "coord_all" else "tumor coordinates only"
                ax.set_title(f"{sample} · {title}\np = {p:.3g}   z = {z:.1f}", fontsize=10, pad=8)
                ax.set_xlabel(xlab)
                if col == 0:
                    ax.set_ylabel("Density")
                ax.legend(loc="upper right", labelcolor=RP["subtle"])
        fig.suptitle(
            "Coordinate-shuffle permutation test",
            fontsize=13,
            color=RP["text"],
            y=0.98,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(OUT / fname, dpi=220, bbox_inches="tight")
        plt.close(fig)


def fig_what_front_means():
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9), facecolor=RP["base"])
    # continuum
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_facecolor(RP["surface"])
    ax.set_title("Observed front", color=RP["text"], pad=8)
    cols = [RP["pine"], RP["foam"], RP["iris"], RP["love"], "#b4637a"]
    labels = ["norm", "exit", "persist", "enter", "deep"]
    for i, (col, lab) in enumerate(zip(cols, labels)):
        ax.add_patch(
            mpatches.FancyBboxPatch(
                (0.7 + i * 1.75, 1.6),
                1.55,
                6.8,
                boxstyle="round,pad=0.02,rounding_size=0.25",
                facecolor=col,
                edgecolor=RP["base"],
                linewidth=0.6,
                alpha=0.92,
            )
        )
        ax.text(0.7 + i * 1.75 + 0.77, 0.7, lab, ha="center", color=RP["subtle"], fontsize=8)
    ax.annotate(
        "",
        xy=(9.2, 5),
        xytext=(1.0, 5),
        arrowprops=dict(arrowstyle="->", color=RP["gold"], lw=1.6),
    )
    ax.axis("off")

    # shuffled
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_facecolor(RP["surface"])
    ax.set_title("After coordinate shuffle", color=RP["text"], pad=8)
    rng = np.random.default_rng(1)
    for _ in range(90):
        ax.scatter(
            rng.uniform(0.7, 9.3),
            rng.uniform(1.5, 8.5),
            s=36,
            c=cols[rng.integers(0, 5)],
            alpha=0.88,
            linewidths=0,
        )
    ax.text(5, 0.7, "adjacency dissolves", ha="center", color=RP["muted"], fontsize=8)
    ax.axis("off")

    # copy
    ax = axes[2]
    ax.set_facecolor(RP["surface"])
    ax.axis("off")
    ax.set_title("Meaning", color=RP["text"], pad=8)
    ax.text(
        0.04,
        0.92,
        "A front is a spatial continuum\n"
        "along an oxygen boundary —\n"
        "not scattered hypoxic islands.\n\n"
        "ImageIT+ (E15S GFP+) marks\n"
        "hypoxia history; velocity tells\n"
        "whether a cell is entering,\n"
        "persistent, exiting, or reverted.\n\n"
        "Tumor-coordinate shuffle asks:\n"
        "is enter↔exit adjacency real?",
        transform=ax.transAxes,
        va="top",
        ha="left",
        color=RP["subtle"],
        fontsize=9.5,
        linespacing=1.45,
    )
    for a in axes:
        for sp in a.spines.values():
            sp.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "what_front_means.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_hero_board(adata):
    tab_path = OUT / "h8_coord_shuffle_front_test.csv"
    if not tab_path.exists():
        tab_path = OUT / "coord_shuffle_front_test.csv"
    tab = pd.read_csv(tab_path)
    enr = pd.read_csv(OUT / "front_enrichment_table.csv")
    fig = plt.figure(figsize=(12.2, 9.6), facecolor=RP["base"])
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)

    # A enrichment
    ax = fig.add_subplot(gs[0, 0])
    style_ax(ax, grid=True)
    pairs = ["deep → exit", "enter → exit", "persist → exit", "normoxic → exit"]
    x = np.arange(len(pairs))
    w = 0.34
    for i, sample, col in [(0, "E14S", RP["love"]), (1, "E15S", RP["foam"])]:
        vals = []
        for p in pairs:
            hit = enr[(enr["sample"] == sample) & (enr["pair"] == p)]
            vals.append(float(hit.iloc[0]["enrichment"]) if len(hit) else np.nan)
        ax.bar(x + (i - 0.5) * w, vals, width=w, color=col, label=sample, alpha=0.92)
    ax.axhline(1, color=RP["muted"], ls="--", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=12, ha="right")
    ax.set_ylabel("obs / exp")
    ax.set_title("A   Adjacency enrichment", loc="left", color=RP["gold"])
    ax.legend(labelcolor=RP["subtle"])

    # B perm summary
    ax = fig.add_subplot(gs[0, 1])
    style_ax(ax, grid=True)
    if "mode" in tab.columns:
        sub = tab[(tab["metric"] == "enr_deep_exit") & (tab["mode"] == "coord_tumor")]
    else:
        sub = tab[tab["metric"].astype(str).str.contains("deep", case=False)]
        if "sample" not in sub.columns and "library" in sub.columns:
            sub = sub.rename(columns={"library": "sample"})
    if len(sub) == 0:
        sub = tab.copy()
    xs = np.arange(len(sub))
    null_col = "null_mean" if "null_mean" in sub.columns else ("null" if "null" in sub.columns else None)
    obs_col = "observed" if "observed" in sub.columns else ("obs" if "obs" in sub.columns else None)
    if null_col and obs_col and len(sub):
        ax.bar(xs - 0.18, sub[null_col], width=0.36, color=RP["overlay"], label="null", edgecolor=RP["hl_med"])
        cols = [RP["love"], RP["foam"], RP["iris"]][: len(sub)]
        ax.bar(xs + 0.18, sub[obs_col], width=0.36, color=cols, label="observed")
        for i, r in enumerate(sub.itertuples()):
            obs_v = getattr(r, obs_col)
            null_v = getattr(r, null_col)
            p = getattr(r, "p", getattr(r, "p_ge_obs", getattr(r, "pval", np.nan)))
            ax.text(i, max(obs_v, null_v) + 0.08, f"p={p:.3g}", ha="center", fontsize=8, color=RP["subtle"])
        ax.set_xticks(xs)
        labels = []
        for s in sub["sample"].astype(str):
            if s == "E14S":
                labels.append("E14S\nGFP−")
            elif s == "E15S":
                labels.append("E15S\nGFP+")
            else:
                labels.append(s)
        ax.set_xticklabels(labels)
    ax.set_ylabel("deep → exit enrichment")
    ax.set_title("B   Tumor-label shuffle · H≤8", loc="left", color=RP["gold"])
    ax.legend(labelcolor=RP["subtle"])

    # C/D maps on H≤8 MAP microwells
    for col_i, sample in enumerate(["E14S", "E15S"]):
        ax = fig.add_subplot(gs[1, col_i])
        style_ax(ax)
        sub, xy = sample_xy(adata, sample)
        labs = sub.obs["hypoxia_dynamics"].astype(str).to_numpy()
        tumor = sub.obs["is_tumor"].to_numpy()
        knn = NearestNeighbors(n_neighbors=min(31, len(xy))).fit(xy).kneighbors(xy, return_distance=False)[:, 1:]
        deep = (labs == "entering_deep_hypoxia") & tumor
        exit_ = (labs == "exiting_hypoxia") & tumor
        persist = (labs == "persistent_hypoxia") & tumor
        ax.scatter(xy[:, 0], xy[:, 1], s=1.6, c=RP["hl_med"], alpha=0.3, linewidths=0)
        ax.scatter(xy[persist, 0], xy[persist, 1], s=7, c=RP["iris"], alpha=0.35, linewidths=0)
        exit_set = set(np.where(exit_)[0])
        for i in np.where(deep)[0]:
            for j in knn[i]:
                if j in exit_set:
                    ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], color=RP["gold"], lw=0.45, alpha=0.55, zorder=2)
                    break
        ax.scatter(xy[exit_, 0], xy[exit_, 1], s=11, c=RP["foam"], linewidths=0, zorder=3)
        ax.scatter(xy[deep, 0], xy[deep, 1], s=11, c=RP["love"], linewidths=0, zorder=4)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        gate = "GFP−" if sample == "E14S" else "GFP+"
        ax.set_title(f"{'C' if col_i == 0 else 'D'}   {sample} {gate} · H≤8 MAP", loc="left", color=RP["gold"])

    handles = [
        mpatches.Patch(color=RP["love"], label="enter-deep"),
        mpatches.Patch(color=RP["foam"], label="exiting"),
        mpatches.Patch(color=RP["iris"], label="persistent"),
        Line2D([0], [0], color=RP["gold"], lw=2, label="kNN bridge"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.01), labelcolor=RP["subtle"])
    fig.suptitle("Point 1 · Hypoxia is a front, not an island · H≤8", fontsize=15, color=RP["text"], y=0.98)
    fig.savefig(OUT / "point1_hero_board.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_entry_exit_programs(adata):
    """Elegant stress vs metabolism and program heatmap for deep-dive site."""
    from scipy import sparse, stats

    # scores
    genes = list(adata.var_names)
    g2i = {g: i for i, g in enumerate(genes)}

    def present(gs):
        return [g for g in gs if g in g2i]

    def score(gs):
        idx = [g2i[g] for g in present(gs)]
        if not idx:
            return np.zeros(adata.n_obs)
        sub = adata.X[:, idx]
        v = np.asarray(sub.mean(1)).ravel() if sparse.issparse(sub) else sub.mean(1)
        s = v.std()
        return (v - v.mean()) / s if s > 0 else v * 0

    # check if already log-normalized
    mx = float(adata.X[:50].max()) if sparse.issparse(adata.X) else float(np.max(adata.X[:50]))
    if mx > 50:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    adata.obs["stress"] = score(["Ppp1r15a", "Ddit3", "Atf3", "Atf4", "Slc38a2", "Asns", "Chka", "Jun", "Jund", "Ier3"])
    adata.obs["met_ox"] = score(["Cox5a", "Cox4i1", "Ndufa4", "Ndufb9", "Atp5pb", "Atp5c1", "Uqcrb", "Slc25a3", "Slc25a4"])
    adata.obs["met_gly"] = score(["Ldha", "Pkm", "Gapdh", "Pgk1", "Pgam1", "Aldoa", "Eno1", "Slc2a1"])
    adata.obs["met_sum"] = adata.obs["met_ox"] + adata.obs["met_gly"]
    adata.var["mt"] = adata.var_names.str.startswith("mt-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    hq = (adata.obs["n_genes_by_counts"] >= 1000) & (adata.obs["pct_counts_mt"] < 15) & adata.obs["is_tumor"]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), facecolor=RP["base"])
    for ax, sample, gate in zip(axes, ["E14S", "E15S"], ["GFP−", "GFP+"]):
        style_ax(ax)
        m = hq & (adata.obs["sample"] == sample)
        for state, lab in [
            ("entering_hypoxia", "enter"),
            ("entering_deep_hypoxia", "enter-deep"),
            ("persistent_hypoxia", "persist"),
            ("exiting_hypoxia", "exit"),
            ("normoxic_stable", "normoxic"),
        ]:
            mm = m & (adata.obs["hypoxia_dynamics"] == state)
            if mm.sum() < 5:
                continue
            ax.scatter(
                adata.obs.loc[mm, "stress"],
                adata.obs.loc[mm, "met_sum"],
                s=12,
                c=STATE_COLORS[state],
                alpha=0.75,
                linewidths=0,
                label=lab,
                rasterized=True,
            )
        ax.axhline(0, color=RP["hl_high"], lw=0.7)
        ax.axvline(0, color=RP["hl_high"], lw=0.7)
        ax.set_xlabel("Stress  (ISR · AA · Chka · IEG)")
        ax.set_ylabel("Metabolic engines  (OXPHOS + glycolysis)")
        ax.set_title(f"{sample} {gate}", color=RP["text"])
        ax.legend(loc="best", labelcolor=RP["subtle"], markerscale=1.4)
    fig.suptitle("Entry = stress↑ / metabolism↓   ·   Exit = stress↓ / metabolism↑", color=RP["text"], fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DD / "v3_stress_vs_metabolism.png", dpi=220, bbox_inches="tight")
    # also into report_site
    fig.savefig(OUT_DD / "report_site" / "v3_stress_vs_metabolism.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # program heatmap
    progs = {
        "ISR": ["Ppp1r15a", "Ddit3", "Atf3", "Atf4", "Trib3", "Slc38a2"],
        "IEG": ["Jun", "Jund", "Fos", "Ier3", "Nr4a1"],
        "AA transport": ["Slc38a2", "Slc7a5", "Slc3a2", "Slc1a5"],
        "Lipid/Chka": ["Chka", "Hilpda", "Acsl4", "Lpin1"],
        "Glycolysis": ["Ldha", "Gapdh", "Pgk1", "Slc2a1"],
        "OXPHOS": ["Cox5a", "Ndufa4", "Atp5pb", "Slc25a3"],
        "Ribosome": ["Rps4x", "Rpl7", "Rps9", "Eef1b2"],
    }
    for name, gs in progs.items():
        adata.obs[f"p_{name}"] = score(gs)
    states = ["entering_hypoxia", "entering_deep_hypoxia", "persistent_hypoxia", "exiting_hypoxia", "normoxic_stable"]
    mat = []
    for st in states:
        m = hq & (adata.obs["hypoxia_dynamics"] == st)
        mat.append([float(adata.obs.loc[m, f"p_{n}"].mean()) if m.sum() else np.nan for n in progs])
    mat = np.asarray(mat)
    fig, ax = plt.subplots(figsize=(8.6, 4.2), facecolor=RP["base"])
    style_ax(ax)
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-0.8, vmax=0.8)
    ax.set_xticks(range(len(progs)))
    ax.set_xticklabels(list(progs), rotation=30, ha="right", color=RP["subtle"])
    ax.set_yticks(range(len(states)))
    ax.set_yticklabels([s.replace("_", " ") for s in states], color=RP["subtle"])
    ax.set_title("HQ tumor programs across hypoxia dynamics", color=RP["text"])
    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color=RP["muted"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=RP["subtle"])
    cbar.outline.set_edgecolor(RP["hl_med"])
    fig.tight_layout()
    fig.savefig(OUT_DD / "v2_program_state_heatmap.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / "program_state_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_zooms_and_ribbons(adata):
    for sample, gate in [("E14S", "GFP−"), ("E15S", "GFP+")]:
        sub, xy = sample_xy(adata, sample)
        labs = sub.obs["hypoxia_dynamics"].astype(str).to_numpy()
        tumor = sub.obs["is_tumor"].to_numpy()
        enter_m = np.isin(labs, list(ENTER)) & tumor
        exit_m = (labs == "exiting_hypoxia") & tumor
        nn = NearestNeighbors(n_neighbors=2).fit(xy)
        dmed = float(np.median(nn.kneighbors(xy)[0][:, 1]))
        max_dist = dmed * 4.0

        # ribbons
        knn = NearestNeighbors(n_neighbors=min(40, len(xy))).fit(xy).kneighbors(xy, return_distance=False)
        enter_f = enter_m.astype(float)[knn].mean(1)
        exit_f = exit_m.astype(float)[knn].mean(1)
        persist_f = ((labs == "persistent_hypoxia") & tumor).astype(float)[knn].mean(1)
        front_score = np.sqrt(enter_f * exit_f)
        polarity = exit_f - enter_f

        fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), facecolor=RP["base"])
        for ax, val, title, cmap, vmin, vmax in [
            (axes[0], front_score, "Front ribbon\nenter × exit", "magma", 0, np.percentile(front_score, 99)),
            (axes[1], polarity, "Polarity\nexit − enter", "RdBu_r", -np.percentile(np.abs(polarity), 98), np.percentile(np.abs(polarity), 98)),
            (axes[2], persist_f, "Persistent\nlocal density", "Purples", 0, max(np.percentile(persist_f, 99), 1e-6)),
        ]:
            style_ax(ax)
            sca = ax.scatter(xy[:, 0], xy[:, 1], c=val, s=3.2, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0, alpha=0.92, rasterized=True)
            cb = fig.colorbar(sca, ax=ax, shrink=0.78, fraction=0.046, pad=0.02)
            cb.outline.set_edgecolor(RP["hl_med"])
            plt.setp(cb.ax.yaxis.get_ticklabels(), color=RP["subtle"])
            ax.set_aspect("equal")
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        fig.suptitle(f"{sample} {gate} · local front geometry", color=RP["text"], fontsize=12)
        fig.tight_layout()
        fig.savefig(OUT / f"{sample}_front_ribbons.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        # zooms
        if enter_m.sum() < 5 or exit_m.sum() < 5:
            continue
        nn1 = NearestNeighbors(n_neighbors=1).fit(xy[exit_m])
        d_e = nn1.kneighbors(xy[enter_m])[0].ravel()
        enter_xy = xy[enter_m]
        close = d_e <= max_dist
        if close.sum() < 3:
            close = d_e <= np.percentile(d_e, 40)
        hx = enter_xy[close]
        nn2 = NearestNeighbors(n_neighbors=min(8, len(hx))).fit(hx)
        dens = 1.0 / (nn2.kneighbors(hx)[0].mean(axis=1) + 1e-9)
        order = np.argsort(dens)[::-1]
        c1 = hx[order[0]]
        top = hx[order[: max(5, len(order) // 3)]]
        c2 = top[np.argmax(np.linalg.norm(top - c1, axis=1))]
        centers = [c1, c2]
        win = max_dist * 8

        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), facecolor=RP["base"])
        ax = axes[0]
        style_ax(ax)
        ax.scatter(xy[:, 0], xy[:, 1], s=1.5, c=RP["hl_med"], alpha=0.3, linewidths=0)
        ax.scatter(xy[enter_m, 0], xy[enter_m, 1], s=8, c=RP["love"], alpha=0.9, linewidths=0, label="enter")
        ax.scatter(xy[exit_m, 0], xy[exit_m, 1], s=8, c=RP["foam"], alpha=0.9, linewidths=0, label="exit")
        for i, c in enumerate(centers):
            rect = mpatches.Rectangle(
                (c[0] - win / 2, c[1] - win / 2),
                win,
                win,
                fill=False,
                edgecolor=RP["gold"],
                lw=1.3,
                linestyle="--",
            )
            ax.add_patch(rect)
            ax.text(c[0] - win / 2, c[1] + win / 2 + 0.02 * win, f"zoom {i+1}", color=RP["gold"], fontsize=8)
        ax.set_aspect("equal")
        ax.set_title(f"{sample} · front hotspots")
        ax.legend(labelcolor=RP["subtle"])
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

        for ax, c, title in zip(axes[1:], centers, ["Zoom 1", "Zoom 2"]):
            style_ax(ax)
            xmin, xmax = c[0] - win / 2, c[0] + win / 2
            ymin, ymax = c[1] - win / 2, c[1] + win / 2
            inwin = (xy[:, 0] >= xmin) & (xy[:, 0] <= xmax) & (xy[:, 1] >= ymin) & (xy[:, 1] <= ymax)
            ax.scatter(xy[inwin, 0], xy[inwin, 1], s=5, c=RP["hl_med"], alpha=0.35, linewidths=0)
            persist = (labs == "persistent_hypoxia") & tumor & inwin
            ax.scatter(xy[persist, 0], xy[persist, 1], s=26, c=RP["iris"], alpha=0.7, linewidths=0, label="persistent")
            draw_edges(ax, xy, enter_m & inwin, exit_m & inwin, k=4, max_dist=max_dist, color=RP["gold"], lw=0.95, alpha=0.7)
            deep = (labs == "entering_deep_hypoxia") & tumor & inwin
            ent = (labs == "entering_hypoxia") & tumor & inwin
            ax.scatter(xy[exit_m & inwin, 0], xy[exit_m & inwin, 1], s=36, c=RP["foam"], linewidths=0, label="exiting", zorder=4)
            ax.scatter(xy[ent, 0], xy[ent, 1], s=36, c=RP["love"], linewidths=0, label="entering", zorder=4)
            ax.scatter(xy[deep, 0], xy[deep, 1], s=36, c="#b4637a", linewidths=0, label="enter-deep", zorder=4)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect("equal")
            ax.set_title(title)
            ax.legend(fontsize=7, labelcolor=RP["subtle"])
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        fig.suptitle(f"{sample} {gate}: entering and exiting share local fronts", color=RP["text"], fontsize=12)
        fig.tight_layout()
        fig.savefig(OUT / f"{sample}_front_zooms.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def fig_knn_contact(adata):
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.0), facecolor=RP["base"])
    K = 30
    N_PERM = 400
    rows = []
    for row, sample in enumerate(["E14S", "E15S"]):
        sub, xy = sample_xy(adata, sample)
        labs = sub.obs["hypoxia_dynamics"].astype(str).to_numpy()
        tumor = sub.obs["is_tumor"].to_numpy()
        knn = NearestNeighbors(n_neighbors=min(K + 1, len(xy))).fit(xy).kneighbors(xy, return_distance=False)[:, 1:]
        deep = (labs == "entering_deep_hypoxia") & tumor
        exit_ = (labs == "exiting_hypoxia") & tumor
        norm = (labs == "normoxic_stable") & tumor

        def frac(src, tgt):
            if src.sum() < 5 or tgt.sum() < 5:
                return np.nan
            return float(tgt[knn[src]].any(axis=1).mean())

        obs = frac(deep, exit_)
        null = []
        tidx = np.where(tumor)[0]
        for _ in range(N_PERM):
            shuf = labs.copy()
            shuf[tidx] = RNG.permutation(labs[tidx])
            null.append(frac((shuf == "entering_deep_hypoxia") & tumor, (shuf == "exiting_hypoxia") & tumor))
        null = np.asarray(null, float)
        p = (np.sum(null >= obs) + 1) / (len(null) + 1)

        ax = axes[row, 0]
        style_ax(ax)
        ax.hist(null, bins=28, color=RP["overlay"], edgecolor=RP["hl_med"], density=True)
        ax.axvline(obs, color=RP["love"], lw=2.2, label=f"observed = {obs:.2f}")
        ax.axvline(frac(norm, exit_), color=RP["pine"], lw=1.5, ls="--", label=f"normoxic = {frac(norm, exit_):.2f}")
        ax.set_title(f"{sample}: enter-deep contacting exit\np = {p:.3g}", fontsize=10)
        ax.set_xlabel("Contact fraction in 30-NN")
        ax.set_ylabel("Density")
        ax.legend(labelcolor=RP["subtle"])

        ax = axes[row, 1]
        style_ax(ax)
        ax.scatter(xy[:, 0], xy[:, 1], s=1.5, c=RP["hl_med"], alpha=0.3, linewidths=0)
        exit_set = set(np.where(exit_)[0])
        for i in np.where(deep)[0]:
            hits = [j for j in knn[i] if j in exit_set]
            for j in hits[:2]:
                ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], color=RP["gold"], lw=0.7, alpha=0.65, zorder=2)
        s = 12 if sample == "E15S" else 20
        ax.scatter(xy[exit_, 0], xy[exit_, 1], s=s, c=RP["foam"], linewidths=0, label="exiting", zorder=3)
        ax.scatter(xy[deep, 0], xy[deep, 1], s=s, c=RP["love"], linewidths=0, label="enter-deep", zorder=4)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{sample}: deep↔exit bridges")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), labelcolor=RP["subtle"])
        rows.append({"sample": sample, "deep_has_exit": obs, "p": p})
    fig.suptitle("Enter-deep and exiting cells are spatial neighbors · H≤8 MAP", color=RP["text"], fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "point1_knn_contact_both.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(boards_only: bool = False):
    apply_theme()
    print("loading H≤8 MAP…")
    adata = load()
    if not boards_only:
        print("bridges…")
        fig_front_bridges(adata)
        print("enrichment…")
        fig_enrichment_bars()
        print("coord shuffle summary…")
        fig_coord_shuffle_summary()
        print("coord shuffle hists…")
        fig_coord_shuffle_hists(adata)
        print("zooms/ribbons…")
        fig_zooms_and_ribbons(adata)
        print("entry/exit programs…")
        fig_entry_exit_programs(adata.copy())
    print("what front means…")
    fig_what_front_means()
    print("hero…")
    fig_hero_board(adata)
    print("knn contact…")
    fig_knn_contact(adata)
    print("done")


if __name__ == "__main__":
    import sys

    main(boards_only="--boards-only" in sys.argv)
