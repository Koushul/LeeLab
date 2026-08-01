#!/usr/bin/env python3
"""Hypoxia front analysis with ADT entropy filter H≤8 and E14/E15 chip overlay."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from sklearn.neighbors import NearestNeighbors

OUT = Path(
    "/ix1/ylee/shared/MC38_Hypoxia_001/reseq_finetune/deep_dive/spatial_niches/front_figures"
)
OUT.mkdir(parents=True, exist_ok=True)
H_CUT = 8.0
K = 30
N_PERM = 2000
RNG = np.random.default_rng(20260801)

ENTER = {"entering_hypoxia", "entering_deep_hypoxia"}
EXIT = {"exiting_hypoxia"}
FRONT_STATES = [
    "normoxic_stable",
    "exiting_hypoxia",
    "persistent_hypoxia",
    "entering_hypoxia",
    "entering_deep_hypoxia",
]

RP = {
    "base": "#191724",
    "surface": "#1f1d2e",
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
COLORS = {
    "entering_hypoxia": "#eb6f92",
    "entering_deep_hypoxia": "#b4637a",
    "persistent_hypoxia": "#c4a7e7",
    "exiting_hypoxia": "#9ccfd8",
    "normoxic_stable": "#31748f",
    "transitional": "#f6c177",
    "other": "#403d52",
    "E14S": "#eb6f92",
    "E15S": "#9ccfd8",
}


def style_ax(ax):
    ax.set_facecolor(RP["surface"])
    for sp in ax.spines.values():
        sp.set_color(RP["hl"])
    ax.tick_params(colors=RP["subtle"])
    ax.xaxis.label.set_color(RP["subtle"])
    ax.yaxis.label.set_color(RP["subtle"])
    ax.title.set_color(RP["text"])


def jitter_xy(row_idx: np.ndarray, col_idx: np.ndarray, keys: np.ndarray, radius=0.28):
    xy = np.column_stack([row_idx.astype(float), col_idx.astype(float)])
    groups = {}
    for i, key in enumerate(zip(row_idx, col_idx)):
        groups.setdefault(key, []).append(i)
    for idxs in groups.values():
        n = len(idxs)
        if n == 1:
            continue
        for rank, i in enumerate(idxs):
            ang = 2 * np.pi * rank / n
            # stable hash offset from barcode
            h = abs(hash(str(keys[i]))) % 1000 / 1000.0
            r = radius * (0.55 + 0.45 * h)
            xy[i, 0] += r * np.cos(ang)
            xy[i, 1] += r * np.sin(ang)
    return xy


def load_filtered():
    adata = sc.read_h5ad(
        "/ix1/ylee/shared/MC38_Hypoxia_001/reseq_finetune/mc38_tumor_reseq_finetuned.h5ad"
    )
    dyn = pd.read_csv(
        "/ix1/ylee/shared/MC38_Hypoxia_001/reseq_finetune/hypoxia_velocity/hypoxia_dynamics_per_cell.csv"
    ).set_index("barcode")
    common = adata.obs_names.intersection(dyn.index)
    adata = adata[common].copy()
    for c in ["hypoxia_dynamics", "hypoxia_score", "v_hypoxia"]:
        adata.obs[c] = dyn.loc[adata.obs_names, c].values
    adata.obs["is_tumor"] = (
        adata.obs["cell_type_finetuned"].astype(str).str.contains("Tumor", case=False)
        | adata.obs["cell_type"].astype(str).str.contains("Tumor", case=False)
    ).to_numpy()

    ent = pd.read_csv(OUT / "microwell_entropy_per_cell.csv")
    ent = ent[ent["entropy_bits"] <= H_CUT].copy()
    ent["bc_raw"] = ent["barcode"].astype(str).str.split("-").str[0]
    ent["obs_name"] = ent["bc_raw"] + "_" + ent["sample"]

    keep = adata.obs_names.intersection(ent["obs_name"])
    adata = adata[keep].copy()
    emap = ent.set_index("obs_name")
    for c in [
        "entropy_bits",
        "map_prob",
        "map_row_idx",
        "map_col_idx",
        "adt_umi_total",
    ]:
        adata.obs[c] = emap.loc[adata.obs_names, c].to_numpy()

    xy = jitter_xy(
        adata.obs["map_row_idx"].to_numpy(),
        adata.obs["map_col_idx"].to_numpy(),
        adata.obs_names.to_numpy(),
    )
    adata.obsm["spatial"] = xy
    adata.obs["spatial_filter"] = f"entropy_bits<={H_CUT}"
    return adata, {
        "h_cut": H_CUT,
        "n_after_dyn": int(len(common)),
        "n_entropy_pass": int(len(ent)),
        "n_analysis": int(adata.n_obs),
        "n_by_sample": adata.obs["sample"].value_counts().to_dict(),
    }


def subset(adata, sample: str | None):
    if sample is None or sample == "OVERLAY":
        sub = adata
        label = "OVERLAY"
    else:
        sub = adata[adata.obs["sample"].to_numpy() == sample]
        label = sample
    xy = np.asarray(sub.obsm["spatial"], float)
    return sub, xy, label


def local_fields(xy, labs, tumor, k=K):
    nn = NearestNeighbors(n_neighbors=min(k, len(xy))).fit(xy)
    idx = nn.kneighbors(xy, return_distance=False)
    enter = np.isin(labs, list(ENTER)) & tumor
    exit_ = (labs == "exiting_hypoxia") & tumor
    persist = (labs == "persistent_hypoxia") & tumor
    deep = (labs == "entering_deep_hypoxia") & tumor
    ef = enter.astype(float)[idx].mean(1)
    xf = exit_.astype(float)[idx].mean(1)
    pf = persist.astype(float)[idx].mean(1)
    return {
        "enter_f": ef,
        "exit_f": xf,
        "persist_f": pf,
        "front": np.sqrt(ef * xf),
        "polarity": xf - ef,
        "enter": enter,
        "exit": exit_,
        "persist": persist,
        "deep": deep,
        "idx": idx,
    }


def enrichment(src, tgt, knn_idx):
    if src.sum() < 5 or tgt.sum() < 5:
        return np.nan
    return float(tgt[knn_idx[src]].mean() / max(tgt.mean(), 1e-12))


def contact_frac(src, tgt, knn_idx):
    if src.sum() < 5 or tgt.sum() < 5:
        return np.nan
    return float(tgt[knn_idx[src]].any(1).mean())


def fig_overlay_gates(adata):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    fig.patch.set_facecolor(RP["base"])
    for ax, sample, title in [
        (axes[0], "E14S", "E14S GFP−"),
        (axes[1], "E15S", "E15S GFP+"),
        (axes[2], "OVERLAY", "Same chip overlay"),
    ]:
        style_ax(ax)
        if sample == "OVERLAY":
            for s, col, lab in [("E14S", COLORS["E14S"], "GFP−"), ("E15S", COLORS["E15S"], "GFP+")]:
                m = adata.obs["sample"].to_numpy() == s
                xy = np.asarray(adata.obsm["spatial"][m], float)
                ax.scatter(xy[:, 0], xy[:, 1], s=3, c=col, alpha=0.55, linewidths=0, label=lab)
            ax.legend(fontsize=8, frameon=False, labelcolor=RP["subtle"])
        else:
            sub, xy, _ = subset(adata, sample)
            ax.scatter(xy[:, 0], xy[:, 1], s=3, c=COLORS[sample], alpha=0.7, linewidths=0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
    fig.suptitle(
        f"H≤{H_CUT} MAP microwells · E14/E15 share one chip",
        color=RP["text"],
        fontsize=13,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "h8_chip_overlay.png", dpi=180, facecolor=RP["base"])
    plt.close(fig)


def fig_dynamics_map(adata, sample):
    sub, xy, label = subset(adata, sample)
    labs = sub.obs["hypoxia_dynamics"].astype(str).to_numpy()
    tumor = sub.obs["is_tumor"].to_numpy()
    gate = {"E14S": "GFP−", "E15S": "GFP+", "OVERLAY": "GFP− + GFP+"}.get(label, label)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4))
    fig.patch.set_facecolor(RP["base"])

    ax = axes[0]
    style_ax(ax)
    ax.scatter(xy[~tumor, 0], xy[~tumor, 1], s=2, c=RP["hl"], alpha=0.35, linewidths=0)
    for st in FRONT_STATES:
        m = (labs == st) & tumor
        if m.sum() == 0:
            continue
        ax.scatter(
            xy[m, 0],
            xy[m, 1],
            s=10,
            c=COLORS[st],
            alpha=0.9,
            linewidths=0,
            label=st.replace("_", " "),
        )
    if label == "OVERLAY":
        # faint gate edge via marker edge on a subsample legend only
        pass
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{label} {gate} · dynamics")
    leg = ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    for t in leg.get_texts():
        t.set_color(RP["subtle"])

    ax = axes[1]
    style_ax(ax)
    fields = local_fields(xy, labs, tumor)
    enter_m, exit_m = fields["enter"], fields["exit"]
    ax.scatter(xy[~tumor, 0], xy[~tumor, 1], s=2, c=RP["hl"], alpha=0.3, linewidths=0)
    ax.scatter(xy[tumor, 0], xy[tumor, 1], s=3, c=RP["muted"], alpha=0.25, linewidths=0)
    # bridges
    if enter_m.sum() >= 2 and exit_m.sum() >= 2:
        nn = NearestNeighbors(n_neighbors=min(8, int(exit_m.sum()))).fit(xy[exit_m])
        d, idx = nn.kneighbors(xy[enter_m])
        dmed = float(np.median(NearestNeighbors(n_neighbors=2).fit(xy).kneighbors(xy)[0][:, 1]))
        max_d = dmed * 4
        tgt = xy[exit_m]
        src = xy[enter_m]
        for i in range(len(src)):
            for j, dist in zip(idx[i], d[i]):
                if dist <= max_d:
                    ax.plot(
                        [src[i, 0], tgt[j, 0]],
                        [src[i, 1], tgt[j, 1]],
                        color=RP["gold"],
                        lw=0.45,
                        alpha=0.45,
                        zorder=2,
                    )
    ax.scatter(xy[exit_m, 0], xy[exit_m, 1], s=12, c=COLORS["exiting_hypoxia"], linewidths=0, zorder=3, label="exit")
    ax.scatter(xy[enter_m, 0], xy[enter_m, 1], s=12, c=COLORS["entering_hypoxia"], linewidths=0, zorder=3, label="enter")
    deep = fields["deep"]
    ax.scatter(xy[deep, 0], xy[deep, 1], s=14, c=COLORS["entering_deep_hypoxia"], linewidths=0, zorder=4, label="enter-deep")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{label} · enter↔exit bridges")
    leg = ax.legend(fontsize=7, frameon=False, labelcolor=RP["subtle"])

    fig.suptitle(f"H≤{H_CUT} front maps", color=RP["text"], fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / f"h8_{label}_front_bridges.png", dpi=180, facecolor=RP["base"])
    plt.close(fig)
    return fields


def fig_ribbons(adata, sample):
    sub, xy, label = subset(adata, sample)
    labs = sub.obs["hypoxia_dynamics"].astype(str).to_numpy()
    tumor = sub.obs["is_tumor"].to_numpy()
    f = local_fields(xy, labs, tumor)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
    fig.patch.set_facecolor(RP["base"])
    panels = [
        (f["front"], "Front ribbon √(enter×exit)", plt.cm.magma, 0, np.percentile(f["front"], 99) if len(f["front"]) else 0.1),
        (
            f["polarity"],
            "Polarity exit − enter",
            plt.cm.RdBu_r,
            -np.percentile(np.abs(f["polarity"]), 98) if len(f["polarity"]) else -0.1,
            np.percentile(np.abs(f["polarity"]), 98) if len(f["polarity"]) else 0.1,
        ),
        (
            f["persist_f"],
            "Persistent local density",
            plt.cm.Purples,
            0,
            np.percentile(f["persist_f"], 99) if len(f["persist_f"]) else 0.1,
        ),
    ]
    for ax, (val, title, cmap, vmin, vmax) in zip(axes, panels):
        style_ax(ax)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin, vmax = 0.0, 1.0
        sca = ax.scatter(xy[:, 0], xy[:, 1], c=val, s=4, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0, alpha=0.9)
        cb = fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.yaxis.set_tick_params(color=RP["subtle"])
        plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=RP["subtle"])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=10)
    gate = {"E14S": "GFP−", "E15S": "GFP+", "OVERLAY": "chip overlay"}.get(label, label)
    fig.suptitle(f"{label} {gate} · H≤{H_CUT} local front geometry", color=RP["text"], fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / f"h8_{label}_front_ribbons.png", dpi=180, facecolor=RP["base"])
    plt.close(fig)


def persist_stats(adata, sample):
    sub, xy, label = subset(adata, sample)
    labs = sub.obs["hypoxia_dynamics"].astype(str).to_numpy()
    tumor = sub.obs["is_tumor"].to_numpy()
    f = local_fields(xy, labs, tumor)
    persist = f["persist"]
    enter = f["enter"]
    exit_ = f["exit"]
    if persist.sum() < 5:
        return {"sample": label, "n_persist": int(persist.sum())}

    def med_dist(src, tgt):
        if tgt.sum() < 1:
            return np.nan
        nn = NearestNeighbors(n_neighbors=1).fit(xy[tgt])
        d, _ = nn.kneighbors(xy[src])
        return float(np.median(d))

    front = f["front"]
    pos = front[front > 0]
    if len(pos) >= 20:
        thr = float(np.quantile(pos, 0.8))
        on_front = front >= thr
    else:
        thr = np.nan
        on_front = np.zeros_like(front, dtype=bool)
    abs_pol = np.abs(f["polarity"])
    both = f["enter"][f["idx"]].any(1) & f["exit"][f["idx"]].any(1)

    d_exit = np.full(persist.sum(), np.nan)
    d_enter = np.full(persist.sum(), np.nan)
    if exit_.sum():
        d_exit = NearestNeighbors(n_neighbors=1).fit(xy[exit_]).kneighbors(xy[persist])[0][:, 0]
    if enter.sum():
        d_enter = NearestNeighbors(n_neighbors=1).fit(xy[enter]).kneighbors(xy[persist])[0][:, 0]

    out = {
        "sample": label,
        "n_persist": int(persist.sum()),
        "n_enter": int(enter.sum()),
        "n_exit": int(exit_.sum()),
        "median_dist_to_exit": float(np.median(d_exit)) if len(d_exit) else np.nan,
        "median_dist_to_enter": float(np.median(d_enter)) if len(d_enter) else np.nan,
        "frac_closer_to_exit": float(np.mean(d_exit < d_enter)) if len(d_exit) else np.nan,
        "frac_closer_to_enter": float(np.mean(d_enter < d_exit)) if len(d_enter) else np.nan,
        "p_paired_exit_vs_enter": float(stats.wilcoxon(d_exit, d_enter).pvalue)
        if len(d_exit) > 10
        else np.nan,
        "median_front_persist": float(np.median(front[persist])),
        "median_front_enter": float(np.median(front[enter])) if enter.sum() else np.nan,
        "median_front_exit": float(np.median(front[exit_])) if exit_.sum() else np.nan,
        "p_front_persist_vs_enter": float(stats.mannwhitneyu(front[persist], front[enter], alternative="two-sided").pvalue)
        if enter.sum() >= 5
        else np.nan,
        "frac_persist_on_front_top20": float(on_front[persist].mean()) if np.isfinite(thr) else np.nan,
        "frac_enter_on_front_top20": float(on_front[enter].mean()) if (np.isfinite(thr) and enter.sum()) else np.nan,
        "frac_exit_on_front_top20": float(on_front[exit_].mean()) if (np.isfinite(thr) and exit_.sum()) else np.nan,
        "front_top20_threshold": thr,
        "median_abs_polarity_persist": float(np.median(abs_pol[persist])),
        "median_abs_polarity_enter": float(np.median(abs_pol[enter])) if enter.sum() else np.nan,
        "frac_both_neighbors": float(both[persist].mean()),
        "enr_deep_exit": enrichment(f["deep"], exit_, f["idx"]),
        "enr_enter_exit": enrichment(enter, exit_, f["idx"]),
        "contact_deep_exit": contact_frac(f["deep"], exit_, f["idx"]),
        "contact_enter_exit": contact_frac(enter, exit_, f["idx"]),
        "mean_front_score": float(front.mean()),
    }
    return out


def coord_shuffle(adata, sample, n_perm=N_PERM):
    sub, xy, label = subset(adata, sample)
    labs = sub.obs["hypoxia_dynamics"].astype(str).to_numpy()
    tumor = sub.obs["is_tumor"].to_numpy()
    nn = NearestNeighbors(n_neighbors=min(K + 1, len(xy))).fit(xy)
    knn = nn.kneighbors(xy, return_distance=False)[:, 1:]

    def metrics(L):
        deep = (L == "entering_deep_hypoxia") & tumor
        exit_ = (L == "exiting_hypoxia") & tumor
        enter = np.isin(L, list(ENTER)) & tumor
        return {
            "enr_deep_exit": enrichment(deep, exit_, knn),
            "enr_enter_exit": enrichment(enter, exit_, knn),
            "contact_deep_exit": contact_frac(deep, exit_, knn),
            "front_score": float(
                np.mean(
                    np.sqrt(
                        ((enter.astype(float)[knn].mean(1) * K + enter.astype(float)) / (K + 1))
                        * ((exit_.astype(float)[knn].mean(1) * K + exit_.astype(float)) / (K + 1))
                    )
                )
            ),
        }

    obs = metrics(labs)
    nulls = {k: [] for k in obs}
    tidx = np.where(tumor)[0]
    for _ in range(n_perm):
        L = labs.copy()
        L[tidx] = RNG.permutation(labs[tidx])
        m = metrics(L)
        for k, v in m.items():
            nulls[k].append(v)

    rows = []
    for k, v in obs.items():
        arr = np.asarray(nulls[k], float)
        arr = arr[np.isfinite(arr)]
        if not np.isfinite(v) or len(arr) == 0:
            p = z = np.nan
        else:
            p = float((np.sum(arr >= v) + 1) / (len(arr) + 1))
            z = float((v - arr.mean()) / max(arr.std(), 1e-12))
        rows.append(
            {
                "sample": label,
                "metric": k,
                "observed": v,
                "null_mean": float(np.nanmean(nulls[k])),
                "null_std": float(np.nanstd(nulls[k])),
                "z": z,
                "p_ge_obs": p,
                "n_perm": n_perm,
                "null": "tumor_label_shuffle",
            }
        )
    return pd.DataFrame(rows), obs, nulls


def fig_enrichment(enr_df):
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    fig.patch.set_facecolor(RP["base"])
    style_ax(ax)
    pairs = [
        ("enr_deep_exit", "deep → exit"),
        ("enr_enter_exit", "enter → exit"),
    ]
    samples = ["E14S", "E15S", "OVERLAY"]
    x = np.arange(len(pairs))
    w = 0.25
    for i, sample in enumerate(samples):
        vals = []
        for metric, _ in pairs:
            hit = enr_df[(enr_df["sample"] == sample) & (enr_df["metric"] == metric)]
            vals.append(float(hit.iloc[0]["observed"]) if len(hit) else np.nan)
        ax.bar(
            x + (i - 1) * w,
            vals,
            width=w,
            color=COLORS.get(sample, RP["gold"]),
            label=sample,
            edgecolor=RP["base"],
        )
    ax.axhline(1, color=RP["muted"], ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([p[1] for p in pairs])
    ax.set_ylabel("Neighbor enrichment")
    ax.set_title(f"H≤{H_CUT} front enrichments (incl. chip overlay)")
    leg = ax.legend(frameon=False, labelcolor=RP["subtle"])
    fig.tight_layout()
    fig.savefig(OUT / "h8_front_enrichment_bars.png", dpi=180, facecolor=RP["base"])
    plt.close(fig)


def fig_shuffle_summary(shuf_df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor(RP["base"])
    for ax, metric, title in [
        (axes[0], "enr_deep_exit", "deep→exit enrichment"),
        (axes[1], "front_score", "front ribbon score"),
    ]:
        style_ax(ax)
        sub = shuf_df[shuf_df["metric"] == metric]
        xs = np.arange(len(sub))
        ax.bar(xs, sub["null_mean"], color=RP["hl"], label="null mean", width=0.55)
        ax.scatter(xs, sub["observed"], color=RP["gold"], s=60, zorder=3, label="observed")
        for i, r in enumerate(sub.itertuples()):
            ax.text(i, max(r.observed, r.null_mean) * 1.02, f"p={r.p_ge_obs:.3f}\nz={r.z:.1f}", ha="center", fontsize=7, color=RP["subtle"])
        ax.set_xticks(xs)
        ax.set_xticklabels(sub["sample"])
        ax.set_title(title)
        leg = ax.legend(fontsize=8, frameon=False, labelcolor=RP["subtle"])
    fig.suptitle(f"H≤{H_CUT} coordinate/label shuffle (tumor null, n={N_PERM})", color=RP["text"], fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "h8_coord_shuffle_summary.png", dpi=180, facecolor=RP["base"])
    plt.close(fig)


def fig_persist_board(stats_rows):
    df = pd.DataFrame(stats_rows)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    fig.patch.set_facecolor(RP["base"])
    metrics = [
        ("frac_persist_on_front_top20", "Persist on top-20% front"),
        ("frac_closer_to_enter", "Persist closer to enter"),
        ("frac_both_neighbors", "Persist with enter+exit neigh."),
    ]
    for ax, (col, title) in zip(axes, metrics):
        style_ax(ax)
        vals = [float(df.loc[df["sample"] == s, col].iloc[0]) if col in df.columns and len(df[df["sample"] == s]) else np.nan for s in ["E14S", "E15S", "OVERLAY"]]
        ax.bar(["E14S", "E15S", "OVERLAY"], vals, color=[COLORS["E14S"], COLORS["E15S"], RP["gold"]])
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=10)
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, color=RP["text"])
    fig.suptitle(f"Persistent vs front · H≤{H_CUT}", color=RP["text"], fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "h8_persistent_vs_front.png", dpi=180, facecolor=RP["base"])
    plt.close(fig)


def main():
    print("loading filtered…")
    adata, meta = load_filtered()
    print(json.dumps(meta, indent=2))

    fig_overlay_gates(adata)

    stats_rows = []
    shuf_all = []
    for sample in ["E14S", "E15S", "OVERLAY"]:
        print("===", sample, "n=", subset(adata, sample)[0].n_obs)
        fig_dynamics_map(adata, sample)
        fig_ribbons(adata, sample)
        st = persist_stats(adata, sample)
        stats_rows.append(st)
        print(sample, {k: st[k] for k in st if k.startswith(("n_", "enr_", "contact_", "mean_front", "frac_persist"))})
        shuf, obs, _ = coord_shuffle(adata, sample)
        shuf_all.append(shuf)
        print(sample, "shuffle deep→exit", obs.get("enr_deep_exit"), "front", obs.get("front_score"))

    shuf_df = pd.concat(shuf_all, ignore_index=True)
    shuf_df.to_csv(OUT / "h8_coord_shuffle_front_test.csv", index=False)
    pd.DataFrame(stats_rows).to_csv(OUT / "h8_persistent_vs_front_stats.csv", index=False)
    (OUT / "h8_persistent_vs_front_stats.json").write_text(json.dumps(stats_rows, indent=2, default=float))
    fig_enrichment(shuf_df)
    fig_shuffle_summary(shuf_df)
    fig_persist_board(stats_rows)

    summary = {
        "filter": meta,
        "note": "E14S (GFP−) and E15S (GFP+) are gates from the same chip; OVERLAY analyzes both on shared MAP microwell coordinates after H≤8.",
        "stats": stats_rows,
        "shuffle": shuf_df.to_dict(orient="records"),
    }
    (OUT / "h8_front_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
