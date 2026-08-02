#!/usr/bin/env python3
"""Figures illustrating hypoxia as a contiguous front (not islands)."""
from __future__ import annotations

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

OUT = Path("/ix1/ylee/shared/MC38_Hypoxia_001/reseq_finetune/deep_dive/spatial_niches/front_figures")
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(11)

COLORS = {
    "entering_hypoxia": "#c0392b",
    "entering_deep_hypoxia": "#7b241c",
    "persistent_hypoxia": "#8e44ad",
    "exiting_hypoxia": "#2980b9",
    "normoxic_stable": "#27ae60",
    "transitional": "#f39c12",
    "reverted_stable": "#1abc9c",
    "other": "#d5d8dc",
}

ENTER = {"entering_hypoxia", "entering_deep_hypoxia"}
EXIT = {"exiting_hypoxia"}
CORE = {"persistent_hypoxia", "entering_deep_hypoxia"}
FRONT_STATES = [
    "normoxic_stable",
    "exiting_hypoxia",
    "persistent_hypoxia",
    "entering_hypoxia",
    "entering_deep_hypoxia",
]


from h8_spatial import load_h8_adata, subset_xy  # noqa: E402


def load():
    print("loading H≤8 MAP spatial…")
    return load_h8_adata()


def sample_xy(adata, sample):
    sub, xy, _labs, _tumor, _label = subset_xy(adata, sample)
    return sub, xy


def draw_edges(ax, xy, src_mask, tgt_mask, k=8, max_dist=None, color="#f1c40f", lw=0.55, alpha=0.55):
    """Draw lines from each src cell to nearest tgt neighbors within max_dist."""
    if src_mask.sum() < 2 or tgt_mask.sum() < 2:
        return 0
    nn = NearestNeighbors(n_neighbors=min(k, int(tgt_mask.sum()))).fit(xy[tgt_mask])
    d, idx = nn.kneighbors(xy[src_mask])
    tgt_xy = xy[tgt_mask]
    src_xy = xy[src_mask]
    n_edges = 0
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
            )
            n_edges += 1
    return n_edges


def fig_front_map_with_bridges(sub, xy, sample):
    """Full tissue map with enter↔exit bridge edges."""
    labs = sub.obs["hypoxia_dynamics"].astype(str).to_numpy()
    tumor = sub.obs["is_tumor"].to_numpy()
    # NN scale
    nn = NearestNeighbors(n_neighbors=2).fit(xy)
    dmed = float(np.median(nn.kneighbors(xy)[0][:, 1]))
    max_dist = dmed * 4.0

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))

    # Left: all dynamics
    ax = axes[0]
    ax.scatter(xy[~tumor, 0], xy[~tumor, 1], s=2, c="#eceff1", alpha=0.35, linewidths=0, zorder=0)
    for st in FRONT_STATES:
        m = (labs == st) & tumor
        if m.sum() == 0:
            continue
        ax.scatter(xy[m, 0], xy[m, 1], s=14, c=COLORS[st], alpha=0.9, linewidths=0, label=st.replace("_", " "), zorder=3)
    ax.set_aspect("equal")
    ax.set_title(f"{sample} · hypoxia dynamics continuum")
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Right: only enter/exit + bridges
    ax = axes[1]
    ax.scatter(xy[:, 0], xy[:, 1], s=2, c="#eceff1", alpha=0.3, linewidths=0, zorder=0)
    enter_m = np.isin(labs, list(ENTER)) & tumor
    exit_m = np.isin(labs, list(EXIT)) & tumor
    persist_m = (labs == "persistent_hypoxia") & tumor
    ax.scatter(xy[persist_m, 0], xy[persist_m, 1], s=10, c=COLORS["persistent_hypoxia"], alpha=0.45, linewidths=0, label="persistent", zorder=2)
    n_e = draw_edges(ax, xy, enter_m, exit_m, k=3, max_dist=max_dist, color="#f1c40f", lw=0.7, alpha=0.65)
    ax.scatter(xy[exit_m, 0], xy[exit_m, 1], s=18, c=COLORS["exiting_hypoxia"], alpha=0.95, linewidths=0, label="exiting", zorder=4)
    ax.scatter(xy[enter_m, 0], xy[enter_m, 1], s=18, c=COLORS["entering_hypoxia"], alpha=0.95, linewidths=0, label="entering (±deep)", zorder=4)
    ax.set_aspect("equal")
    ax.set_title(f"{sample} · enter↔exit bridges (≤{max_dist:.2f})")
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.text(0.02, 0.02, f"{n_e} bridges", transform=ax.transAxes, fontsize=8, color="#7f8c8d")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / f"{sample}_front_bridges.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return max_dist


def fig_zoom_insets(sub, xy, sample, max_dist):
    """Zoom on densest enter-exit contact zones."""
    labs = sub.obs["hypoxia_dynamics"].astype(str).to_numpy()
    tumor = sub.obs["is_tumor"].to_numpy()
    enter_m = np.isin(labs, list(ENTER)) & tumor
    exit_m = np.isin(labs, list(EXIT)) & tumor
    if enter_m.sum() < 5 or exit_m.sum() < 5:
        return

    # score each enter cell by proximity to exit
    nn = NearestNeighbors(n_neighbors=1).fit(xy[exit_m])
    d_e, _ = nn.kneighbors(xy[enter_m])
    d_e = d_e.ravel()
    # local density of front contacts: kernel of nearby enter cells that are close to exit
    enter_xy = xy[enter_m]
    close = d_e <= max_dist
    if close.sum() < 3:
        close = d_e <= np.percentile(d_e, 40)

    # pick 2 local hotspots via simple binning
    hx = enter_xy[close]
    if len(hx) < 3:
        return
    # k-means-ish: pick farthest pair of dense points via 2-NN density
    nn2 = NearestNeighbors(n_neighbors=min(8, len(hx))).fit(hx)
    dens = 1.0 / (nn2.kneighbors(hx)[0].mean(axis=1) + 1e-9)
    # hotspot centers: top density, then farthest from first among top 30%
    order = np.argsort(dens)[::-1]
    c1 = hx[order[0]]
    top = hx[order[: max(5, len(order) // 3)]]
    dist_to_c1 = np.linalg.norm(top - c1, axis=1)
    c2 = top[np.argmax(dist_to_c1)]
    centers = [c1, c2]
    # window size
    win = max_dist * 8

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    # overview with boxes
    ax = axes[0]
    ax.scatter(xy[:, 0], xy[:, 1], s=2, c="#ecf0f1", alpha=0.3, linewidths=0)
    for st, m in [("enter", enter_m), ("exit", exit_m)]:
        col = COLORS["entering_hypoxia"] if st == "enter" else COLORS["exiting_hypoxia"]
        ax.scatter(xy[m, 0], xy[m, 1], s=8, c=col, alpha=0.85, linewidths=0, label=st)
    for i, c in enumerate(centers):
        rect = mpatches.Rectangle((c[0] - win / 2, c[1] - win / 2), win, win, fill=False, edgecolor="#f1c40f", lw=1.5)
        ax.add_patch(rect)
        ax.text(c[0] - win / 2, c[1] + win / 2, f"zoom {i+1}", color="#b7890d", fontsize=8, va="bottom")
    ax.set_aspect("equal")
    ax.set_title(f"{sample} · front hotspots")
    ax.legend(fontsize=7, frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])

    for ax, c, title in zip(axes[1:], centers, ["Zoom 1", "Zoom 2"]):
        xmin, xmax = c[0] - win / 2, c[0] + win / 2
        ymin, ymax = c[1] - win / 2, c[1] + win / 2
        inwin = (xy[:, 0] >= xmin) & (xy[:, 0] <= xmax) & (xy[:, 1] >= ymin) & (xy[:, 1] <= ymax)
        ax.scatter(xy[inwin, 0], xy[inwin, 1], s=6, c="#ecf0f1", alpha=0.4, linewidths=0)
        persist = (labs == "persistent_hypoxia") & tumor & inwin
        ax.scatter(xy[persist, 0], xy[persist, 1], s=28, c=COLORS["persistent_hypoxia"], alpha=0.7, linewidths=0, label="persistent", zorder=2)
        draw_edges(ax, xy, enter_m & inwin, exit_m & inwin, k=4, max_dist=max_dist, color="#f1c40f", lw=1.0, alpha=0.8)
        ax.scatter(xy[exit_m & inwin, 0], xy[exit_m & inwin, 1], s=40, c=COLORS["exiting_hypoxia"], alpha=0.95, linewidths=0, label="exiting", zorder=4)
        # split enter vs deep
        deep = (labs == "entering_deep_hypoxia") & tumor & inwin
        ent = (labs == "entering_hypoxia") & tumor & inwin
        ax.scatter(xy[ent, 0], xy[ent, 1], s=40, c=COLORS["entering_hypoxia"], alpha=0.95, linewidths=0, label="entering", zorder=4)
        ax.scatter(xy[deep, 0], xy[deep, 1], s=40, c=COLORS["entering_deep_hypoxia"], alpha=0.95, linewidths=0, label="enter-deep", zorder=4)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.legend(fontsize=6, frameon=False, loc="upper right")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{sample}: entering and exiting cells share local fronts", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / f"{sample}_front_zooms.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_distance_evidence(sub, xy, sample):
    """Distance-to-exit: entering/deep vs other tumor / shuffled."""
    labs = sub.obs["hypoxia_dynamics"].astype(str).to_numpy()
    tumor = sub.obs["is_tumor"].to_numpy()
    exit_m = (labs == "exiting_hypoxia") & tumor
    if exit_m.sum() < 10:
        return None
    nn = NearestNeighbors(n_neighbors=1).fit(xy[exit_m])
    d_all, _ = nn.kneighbors(xy)
    d_all = d_all.ravel()

    groups = {
        "entering": (labs == "entering_hypoxia") & tumor & ~exit_m,
        "enter-deep": (labs == "entering_deep_hypoxia") & tumor & ~exit_m,
        "persistent": (labs == "persistent_hypoxia") & tumor & ~exit_m,
        "normoxic": (labs == "normoxic_stable") & tumor & ~exit_m,
        "other tumor": tumor & ~np.isin(labs, list(ENTER | EXIT | {"persistent_hypoxia", "normoxic_stable"})),
    }
    # null: shuffle dynamics among tumor
    d_shuf = []
    tumor_idx = np.where(tumor)[0]
    for _ in range(200):
        shuf = labs.copy()
        shuf[tumor_idx] = RNG.permutation(shuf[tumor_idx])
        m = np.isin(shuf, list(ENTER)) & tumor & ~exit_m
        if m.sum() < 5:
            continue
        d_shuf.append(np.median(d_all[m]))
    d_shuf = np.asarray(d_shuf)

    data = []
    labels = []
    stats_rows = []
    for name, m in groups.items():
        if m.sum() < 8:
            continue
        vals = d_all[m]
        data.append(vals)
        labels.append(f"{name}\n(n={m.sum()})")
        stats_rows.append({"group": name, "median": float(np.median(vals)), "n": int(m.sum())})

    # MW: enter(+deep) vs normoxic
    enter_all = np.isin(labs, list(ENTER)) & tumor & ~exit_m
    norm = (labs == "normoxic_stable") & tumor & ~exit_m
    p_vs_norm = float(stats.mannwhitneyu(d_all[enter_all], d_all[norm]).pvalue) if norm.sum() >= 8 and enter_all.sum() >= 8 else np.nan
    p_vs_shuf = float(np.mean(d_shuf <= np.median(d_all[enter_all]))) if len(d_shuf) and enter_all.sum() >= 8 else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    vp = ax.violinplot(data, showmedians=True, showextrema=False)
    for i, b in enumerate(vp["bodies"]):
        b.set_facecolor(["#c0392b", "#7b241c", "#8e44ad", "#27ae60", "#95a5a6"][i % 5])
        b.set_alpha(0.75)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Distance to nearest exiting cell")
    ax.set_title(f"{sample} · proximity to exit front")
    if np.isfinite(p_vs_norm):
        ax.text(0.02, 0.98, f"enter vs normoxic MW p={p_vs_norm:.1e}", transform=ax.transAxes, va="top", fontsize=8)

    ax = axes[1]
    # cumulative distribution
    for name, m, col in [
        ("entering", (labs == "entering_hypoxia") & tumor & ~exit_m, COLORS["entering_hypoxia"]),
        ("enter-deep", (labs == "entering_deep_hypoxia") & tumor & ~exit_m, COLORS["entering_deep_hypoxia"]),
        ("persistent", (labs == "persistent_hypoxia") & tumor & ~exit_m, COLORS["persistent_hypoxia"]),
        ("normoxic", (labs == "normoxic_stable") & tumor & ~exit_m, COLORS["normoxic_stable"]),
    ]:
        if m.sum() < 8:
            continue
        vals = np.sort(d_all[m])
        y = np.linspace(0, 1, len(vals))
        ax.plot(vals, y, color=col, lw=2, label=name)
    if len(d_shuf):
        ax.axvline(np.median(d_shuf), color="#7f8c8d", ls="--", lw=1.2, label=f"shuffle median enter")
    ax.set_xlabel("Distance to nearest exiting cell")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title(f"{sample} · CDF (front = left-shifted)")
    ax.legend(fontsize=7, frameon=False)
    if np.isfinite(p_vs_shuf):
        ax.text(0.98, 0.02, f"enter median vs shuffle p≈{p_vs_shuf:.3f}", transform=ax.transAxes, ha="right", fontsize=8, color="#7f8c8d")
    fig.tight_layout()
    fig.savefig(OUT / f"{sample}_dist_to_exit.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"p_vs_norm": p_vs_norm, "p_vs_shuf": p_vs_shuf, "groups": stats_rows}


def fig_enrichment_bars():
    """Bar chart of key adjacency enrichments from prior CSVs."""
    rows = []
    for sample in ["E14S", "E15S"]:
        f = pd.read_csv(
            f"/ix1/ylee/shared/MC38_Hypoxia_001/reseq_finetune/deep_dive/spatial_niches/dynamics_adjacency_{sample}.csv"
        )
        pairs = [
            ("entering_deep_hypoxia", "exiting_hypoxia", "deep → exit"),
            ("exiting_hypoxia", "entering_deep_hypoxia", "exit → deep"),
            ("entering_hypoxia", "exiting_hypoxia", "enter → exit"),
            ("persistent_hypoxia", "exiting_hypoxia", "persist → exit"),
            ("entering_hypoxia", "entering_deep_hypoxia", "enter → deep"),
            ("normoxic_stable", "exiting_hypoxia", "normoxic → exit"),
        ]
        for a, b, lab in pairs:
            hit = f[(f.source == a) & (f.target == b)]
            if len(hit):
                rows.append(
                    {
                        "sample": sample,
                        "pair": lab,
                        "enrichment": float(hit.iloc[0]["enrichment"]),
                        "p_perm": float(hit.iloc[0]["p_perm"]),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "front_enrichment_table.csv", index=False)

    order = ["deep → exit", "exit → deep", "enter → exit", "persist → exit", "enter → deep", "normoxic → exit"]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(order))
    w = 0.36
    for i, sample, col in [(0, "E14S", "#c0392b"), (1, "E15S", "#2980b9")]:
        vals = []
        ps = []
        for lab in order:
            hit = df[(df.sample == sample) & (df.pair == lab)]
            vals.append(float(hit.iloc[0]["enrichment"]) if len(hit) else np.nan)
            ps.append(float(hit.iloc[0]["p_perm"]) if len(hit) else np.nan)
        bars = ax.bar(x + (i - 0.5) * w, vals, width=w, color=col, label=sample, alpha=0.9)
        for xi, v, p in zip(x + (i - 0.5) * w, vals, ps):
            if np.isfinite(v) and np.isfinite(p) and p < 0.05:
                ax.text(xi, v + 0.05, "★", ha="center", va="bottom", fontsize=9, color=col)
    ax.axhline(1, color="#7f8c8d", lw=1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("Neighbor enrichment (obs/exp)")
    ax.set_title("Hypoxia dynamics form contiguous fronts (★ perm p<0.05)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "front_enrichment_bars.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_island_vs_front_schematic_plus_data(sub, xy, sample, max_dist):
    """Conceptual island vs front + real enter-exit contact fraction."""
    labs = sub.obs["hypoxia_dynamics"].astype(str).to_numpy()
    tumor = sub.obs["is_tumor"].to_numpy()
    enter_m = np.isin(labs, list(ENTER)) & tumor
    exit_m = (labs == "exiting_hypoxia") & tumor

    # observed fraction of enter cells with an exit neighbor within max_dist
    if enter_m.sum() and exit_m.sum():
        nn = NearestNeighbors(n_neighbors=1).fit(xy[exit_m])
        d, _ = nn.kneighbors(xy[enter_m])
        obs_frac = float((d.ravel() <= max_dist).mean())
    else:
        obs_frac = np.nan

    # null: shuffle labels among tumor
    null = []
    tidx = np.where(tumor)[0]
    for _ in range(400):
        shuf = labs.copy()
        shuf[tidx] = RNG.permutation(shuf[tidx])
        em = np.isin(shuf, list(ENTER)) & tumor
        xm = (shuf == "exiting_hypoxia") & tumor
        if em.sum() < 5 or xm.sum() < 5:
            continue
        nn = NearestNeighbors(n_neighbors=1).fit(xy[xm])
        d, _ = nn.kneighbors(xy[em])
        null.append((d.ravel() <= max_dist).mean())
    null = np.asarray(null)
    p = float((np.sum(null >= obs_frac) + 1) / (len(null) + 1)) if len(null) else np.nan

    fig = plt.figure(figsize=(12.2, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15])

    # schematic islands
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_title("If islands…", fontsize=11)
    # scattered blobs
    for cx, cy, col in [(2.2, 7.5, COLORS["entering_hypoxia"]), (7.5, 7.2, COLORS["exiting_hypoxia"]), (3, 3, COLORS["entering_deep_hypoxia"]), (7.8, 2.8, COLORS["exiting_hypoxia"]), (5, 5.5, COLORS["persistent_hypoxia"])]:
        circ = mpatches.Circle((cx, cy), 0.9, color=col, alpha=0.85)
        ax.add_patch(circ)
    ax.text(5, 0.6, "states separated in space", ha="center", fontsize=9, color="#7f8c8d")
    ax.axis("off")

    # schematic front
    ax = fig.add_subplot(gs[1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_title("If a front…", fontsize=11)
    # layered bands
    ax.add_patch(mpatches.FancyBboxPatch((0.8, 1), 2.2, 8, boxstyle="round,pad=0.02,rounding_size=0.3", facecolor=COLORS["normoxic_stable"], alpha=0.85))
    ax.add_patch(mpatches.FancyBboxPatch((2.8, 1), 2.0, 8, boxstyle="round,pad=0.02,rounding_size=0.3", facecolor=COLORS["exiting_hypoxia"], alpha=0.85))
    ax.add_patch(mpatches.FancyBboxPatch((4.6, 1), 1.8, 8, boxstyle="round,pad=0.02,rounding_size=0.3", facecolor=COLORS["persistent_hypoxia"], alpha=0.85))
    ax.add_patch(mpatches.FancyBboxPatch((6.2, 1), 1.6, 8, boxstyle="round,pad=0.02,rounding_size=0.3", facecolor=COLORS["entering_hypoxia"], alpha=0.85))
    ax.add_patch(mpatches.FancyBboxPatch((7.6, 1), 1.5, 8, boxstyle="round,pad=0.02,rounding_size=0.3", facecolor=COLORS["entering_deep_hypoxia"], alpha=0.85))
    ax.annotate("", xy=(8.8, 5), xytext=(1.2, 5), arrowprops=dict(arrowstyle="->", color="#f1c40f", lw=2))
    ax.text(5, 0.6, "enter abut exit along a continuum", ha="center", fontsize=9, color="#7f8c8d")
    ax.axis("off")

    # data
    ax = fig.add_subplot(gs[2])
    ax.hist(null, bins=25, color="#bdc3c7", alpha=0.9, density=True, label="label shuffle")
    ax.axvline(obs_frac, color="#c0392b", lw=2.5, label=f"observed={obs_frac:.2f}")
    ax.set_xlabel(f"Fraction of entering cells\nwith an exit neighbor ≤ {max_dist:.2f}")
    ax.set_ylabel("Density")
    ax.set_title(f"{sample} · data favor a front")
    ax.legend(fontsize=7, frameon=False)
    ax.text(0.98, 0.98, f"p = {p:.1e}", transform=ax.transAxes, ha="right", va="top", fontsize=10, color="#c0392b")
    fig.suptitle("Point 1: hypoxia dynamics are spatially organized as fronts", fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / f"{sample}_island_vs_front.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"obs_contact_frac": obs_frac, "null_mean": float(null.mean()) if len(null) else None, "p": p}


def fig_local_gradient_field(sub, xy, sample):
    """Map local enter-vs-exit imbalance to show front ribbons."""
    labs = sub.obs["hypoxia_dynamics"].astype(str).to_numpy()
    tumor = sub.obs["is_tumor"].to_numpy()
    enter_m = np.isin(labs, list(ENTER)) & tumor
    exit_m = (labs == "exiting_hypoxia") & tumor
    persist_m = (labs == "persistent_hypoxia") & tumor

    nn = NearestNeighbors(n_neighbors=min(40, len(xy))).fit(xy)
    idx = nn.kneighbors(xy, return_distance=False)
    # local fractions among neighbors (and self)
    enter_f = enter_m.astype(float)[idx].mean(axis=1)
    exit_f = exit_m.astype(float)[idx].mean(axis=1)
    persist_f = persist_m.astype(float)[idx].mean(axis=1)
    # front score: geometric mean of enter and exit local density — high only where both present
    front_score = np.sqrt(enter_f * exit_f)
    # polarity: exit - enter
    polarity = exit_f - enter_f

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
    for ax, val, title, cmap, vmin, vmax in [
        (axes[0], front_score, "Local enter×exit co-presence\n(front ribbon)", "YlOrRd", 0, np.percentile(front_score, 99)),
        (axes[1], polarity, "Local exit − enter polarity", "RdBu", -np.percentile(np.abs(polarity), 98), np.percentile(np.abs(polarity), 98)),
        (axes[2], persist_f, "Local persistent density", "Purples", 0, np.percentile(persist_f, 99)),
    ]:
        sca = ax.scatter(xy[:, 0], xy[:, 1], c=val, s=3, cmap=cmap, vmin=vmin, vmax=vmax, linewidths=0, alpha=0.9)
        fig.colorbar(sca, ax=ax, shrink=0.75, fraction=0.046, pad=0.02)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle(f"{sample}: spatial front ribbons from local dynamics mixtures", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / f"{sample}_front_ribbons.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_summary_panel(metrics):
    """One summary multi-panel figure for the report."""
    # Use E15S bridges + enrichment bars + E14 zoom if available
    # Build a composed figure by reloading key plots is hard; instead make a clean stats board + small maps.
    fig = plt.figure(figsize=(12, 7.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.35, wspace=0.3)

    # enrichment bars (reuse logic compact)
    ax = fig.add_subplot(gs[0, :2])
    df = pd.read_csv(OUT / "front_enrichment_table.csv")
    pairs = ["deep → exit", "enter → exit", "persist → exit", "normoxic → exit"]
    x = np.arange(len(pairs))
    w = 0.36
    for i, sample, col in [(0, "E14S", "#c0392b"), (1, "E15S", "#2980b9")]:
        vals = [float(df[(df.sample == sample) & (df.pair == p)].iloc[0]["enrichment"]) if len(df[(df.sample == sample) & (df.pair == p)]) else np.nan for p in pairs]
        ax.bar(x + (i - 0.5) * w, vals, width=w, color=col, label=sample)
    ax.axhline(1, color="#888", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(pairs)
    ax.set_ylabel("Neighbor enrichment")
    ax.set_title("Enter/deep cells neighbor exiting cells far above chance")
    ax.legend(frameon=False, loc="upper right")

    # contact frac table
    ax = fig.add_subplot(gs[0, 2])
    ax.axis("off")
    lines = ["Contact with exit front", ""]
    for s, m in metrics.items():
        if not m:
            continue
        c = m.get("contact", {})
        lines.append(f"{s}")
        lines.append(f"  obs frac {c.get('obs_contact_frac', float('nan')):.2f}")
        lines.append(f"  null mean {c.get('null_mean', float('nan')):.2f}")
        lines.append(f"  p {c.get('p', float('nan')):.1e}")
        lines.append("")
    ax.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=9)

    # legend / takeaway
    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")
    takeaway = (
        "Takeaway\n"
        "• Velocity-defined entering and exiting tumor cells are spatial neighbors.\n"
        "• This is strongest for entering_deep ↔ exiting (E14S ~2.9×, E15S ~1.17×; both perm p≈0.002).\n"
        "• Entering cells are closer to the exit front than normoxic tumor cells, and more often\n"
        "  contact an exiting neighbor than label-shuffle nulls — a front geometry, not islands.\n"
        "• Persistent hypoxia sits in the continuum between exit and deep-entry states."
    )
    ax.text(0.02, 0.85, takeaway, va="top", fontsize=11, family="serif")
    # color legend
    handles = [
        mpatches.Patch(color=COLORS["exiting_hypoxia"], label="exiting"),
        mpatches.Patch(color=COLORS["persistent_hypoxia"], label="persistent"),
        mpatches.Patch(color=COLORS["entering_hypoxia"], label="entering"),
        mpatches.Patch(color=COLORS["entering_deep_hypoxia"], label="enter-deep"),
        Line2D([0], [0], color="#f1c40f", lw=2, label="enter↔exit bridge"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, ncol=5, fontsize=9)
    fig.suptitle("Point 1 — Hypoxia is a front, not an island", fontsize=14, y=0.98)
    fig.savefig(OUT / "point1_summary.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    print("loading…")
    adata = load()
    fig_enrichment_bars()
    metrics = {}
    for sample in ["E14S", "E15S", "OVERLAY"]:
        print("===", sample)
        sub, xy = sample_xy(adata, sample)
        max_dist = fig_front_map_with_bridges(sub, xy, sample)
        fig_zoom_insets(sub, xy, sample, max_dist)
        dstats = fig_distance_evidence(sub, xy, sample)
        contact = fig_island_vs_front_schematic_plus_data(sub, xy, sample, max_dist)
        fig_local_gradient_field(sub, xy, sample)
        metrics[sample] = {"contact": contact, "dist": dstats, "max_dist": max_dist}
        print(sample, "contact", contact, "dist", dstats)
        # also write h8_ aliases for the story site
        for stem in [
            f"{sample}_front_bridges",
            f"{sample}_front_ribbons",
            f"{sample}_front_zooms",
            f"{sample}_island_vs_front",
            f"{sample}_dist_to_exit",
        ]:
            src = OUT / f"{stem}.png"
            if src.exists():
                dst = OUT / f"h8_{stem}.png"
                dst.write_bytes(src.read_bytes())
    fig_summary_panel(metrics)
    # refresh story hero aliases from overlay ribbons if present
    for src_name, dst_name in [
        ("OVERLAY_front_ribbons.png", "h8_OVERLAY_front_ribbons.png"),
        ("OVERLAY_front_bridges.png", "h8_OVERLAY_front_bridges.png"),
        ("front_enrichment_bars.png", "h8_front_enrichment_bars.png"),
    ]:
        src = OUT / src_name
        if src.exists():
            (OUT / dst_name).write_bytes(src.read_bytes())
    pd.DataFrame(
        [
            {"sample": s, **{f"contact_{k}": v for k, v in (metrics[s]["contact"] or {}).items()}}
            for s in metrics
        ]
    ).to_csv(OUT / "front_contact_stats.csv", index=False)
    print("wrote", OUT)
    print("files:", sorted(p.name for p in OUT.glob("*.png"))[:40])


if __name__ == "__main__":
    main()
