#!/usr/bin/env python3
"""Animated GIFs of the hypoxia front (Rosé Pine)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.neighbors import NearestNeighbors

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
    "hl": "#403d52",
}
ENTER = {"entering_hypoxia", "entering_deep_hypoxia"}
OUT = Path("/ix1/ylee/shared/MC38_Hypoxia_001/reseq_finetune/deep_dive/spatial_niches/front_figures")
ANIM = OUT / "animations"
ANIM.mkdir(parents=True, exist_ok=True)

from h8_spatial import load_h8_adata, subset_xy  # noqa: E402

_ADATA = None


def get_adata():
    global _ADATA
    if _ADATA is None:
        print("loading H≤8 MAP spatial…")
        _ADATA = load_h8_adata()
        print("n_cells", _ADATA.n_obs, dict(_ADATA.obs["sample"].value_counts()))
    return _ADATA


def load_sample(sample: str):
    adata = get_adata()
    _sub, xy, labs, tumor, label = subset_xy(adata, sample)
    return xy, labs, tumor, label


def local_fields(xy, labs, tumor, k=40):
    enter = np.isin(labs, list(ENTER)) & tumor
    exit_ = (labs == "exiting_hypoxia") & tumor
    persist = (labs == "persistent_hypoxia") & tumor
    knn = NearestNeighbors(n_neighbors=min(k, len(xy))).fit(xy).kneighbors(xy, return_distance=False)
    enter_f = enter.astype(float)[knn].mean(1)
    exit_f = exit_.astype(float)[knn].mean(1)
    persist_f = persist.astype(float)[knn].mean(1)
    front = np.sqrt(enter_f * exit_f)
    polarity = exit_f - enter_f
    return {
        "enter": enter,
        "exit": exit_,
        "persist": persist,
        "enter_f": enter_f,
        "exit_f": exit_f,
        "persist_f": persist_f,
        "front": front,
        "polarity": polarity,
        "knn": knn,
    }


def pulse(t, period=1.0):
    """Smooth 0.55–1.0 pulse."""
    return 0.55 + 0.45 * (0.5 * (1 + np.sin(2 * np.pi * t / period)))


def gate_label(sample: str) -> str:
    return {"E14S": "GFP−", "E15S": "GFP+", "OVERLAY": "chip overlay"}.get(sample, sample)


def make_pulsating_front(sample: str, nframes=36, fps=12):
    xy, labs, tumor, label = load_sample(sample)
    f = local_fields(xy, labs, tumor)
    gate = gate_label(label)
    # subsample background for speed
    bg_idx = np.arange(len(xy))
    if len(bg_idx) > 8000:
        bg_idx = np.random.default_rng(0).choice(bg_idx, 8000, replace=False)

    fig, ax = plt.subplots(figsize=(7.2, 6.4), facecolor=RP["base"])
    ax.set_facecolor(RP["surface"])
    ax.scatter(xy[bg_idx, 0], xy[bg_idx, 1], s=1.2, c=RP["hl"], alpha=0.25, linewidths=0, zorder=0)
    # static persistent
    ax.scatter(
        xy[f["persist"], 0],
        xy[f["persist"], 1],
        s=8,
        c=RP["iris"],
        alpha=0.35,
        linewidths=0,
        zorder=1,
    )
    # front glow layer (will update sizes/alphas)
    # use front score to size points; highlight high-front cells
    hi = f["front"] >= np.percentile(f["front"], 85)
    glow = ax.scatter(
        xy[hi, 0],
        xy[hi, 1],
        s=20,
        c=RP["gold"],
        alpha=0.35,
        linewidths=0,
        zorder=2,
    )
    exit_sc = ax.scatter(
        xy[f["exit"], 0],
        xy[f["exit"], 1],
        s=14,
        c=RP["foam"],
        alpha=0.9,
        linewidths=0,
        zorder=4,
        label="exiting",
    )
    enter_sc = ax.scatter(
        xy[f["enter"], 0],
        xy[f["enter"], 1],
        s=14,
        c=RP["love"],
        alpha=0.9,
        linewidths=0,
        zorder=4,
        label="entering",
    )
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    title = ax.set_title(f"{label} {gate} · H≤8 front", color=RP["text"], fontsize=13, pad=10)
    ax.legend(loc="upper left", frameon=False, labelcolor=RP["subtle"], fontsize=8)
    fig.tight_layout()

    front_vals = f["front"][hi]
    base_sizes = 12 + 80 * (front_vals / (front_vals.max() + 1e-9))

    def update(frame):
        t = frame / nframes
        p = pulse(t, period=1.0)
        # breathing glow
        glow.set_sizes(base_sizes * (0.7 + 0.9 * p))
        glow.set_alpha(0.15 + 0.45 * p)
        # enter/exit pulse out of phase
        pe = pulse(t, period=1.0)
        px = pulse(t + 0.5, period=1.0)
        enter_sc.set_sizes(np.full(f["enter"].sum(), 10 + 18 * pe))
        exit_sc.set_sizes(np.full(f["exit"].sum(), 10 + 18 * px))
        enter_sc.set_alpha(0.55 + 0.4 * pe)
        exit_sc.set_alpha(0.55 + 0.4 * px)
        title.set_text(f"{label} {gate} · H≤8 front  ·  pulse")
        return glow, enter_sc, exit_sc, title

    anim = FuncAnimation(fig, update, frames=nframes, interval=1000 / fps, blit=False)
    path = ANIM / f"{label}_front_pulse.gif"
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print("wrote", path)
    return path


def make_bridge_pulse(sample: str, nframes=36, fps=12):
    xy, labs, tumor, label = load_sample(sample)
    f = local_fields(xy, labs, tumor)
    gate = gate_label(label)
    deep = (labs == "entering_deep_hypoxia") & tumor
    exit_ = f["exit"]
    knn = f["knn"]
    # collect bridge segments
    exit_set = set(np.where(exit_)[0])
    segs = []
    for i in np.where(deep)[0]:
        for j in knn[i]:
            if j in exit_set:
                segs.append((xy[i], xy[j]))
                break
    # limit segments for clarity
    if len(segs) > 400:
        idx = np.linspace(0, len(segs) - 1, 400).astype(int)
        segs = [segs[i] for i in idx]

    fig, ax = plt.subplots(figsize=(7.2, 6.4), facecolor=RP["base"])
    ax.set_facecolor(RP["surface"])
    ax.scatter(xy[:, 0], xy[:, 1], s=1.2, c=RP["hl"], alpha=0.22, linewidths=0, zorder=0)
    ax.scatter(xy[f["persist"], 0], xy[f["persist"], 1], s=7, c=RP["iris"], alpha=0.3, linewidths=0, zorder=1)
    lines = []
    for a, b in segs:
        (ln,) = ax.plot([a[0], b[0]], [a[1], b[1]], color=RP["gold"], lw=0.7, alpha=0.15, zorder=2, solid_capstyle="round")
        lines.append(ln)
    ax.scatter(xy[exit_, 0], xy[exit_, 1], s=12, c=RP["foam"], alpha=0.9, linewidths=0, zorder=4, label="exiting")
    ax.scatter(xy[deep, 0], xy[deep, 1], s=12, c=RP["love"], alpha=0.9, linewidths=0, zorder=4, label="enter-deep")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"{label} {gate} · H≤8 deep↔exit bridges", color=RP["text"], fontsize=13, pad=10)
    ax.legend(loc="upper left", frameon=False, labelcolor=RP["subtle"], fontsize=8)
    fig.tight_layout()

    # staggered pulse along bridges
    n = len(lines)
    phases = np.linspace(0, 1, n, endpoint=False) if n else np.array([])

    def update(frame):
        t = frame / nframes
        for i, ln in enumerate(lines):
            # traveling pulse along the set of bridges
            local = pulse(t - phases[i], period=1.0)
            ln.set_alpha(0.08 + 0.72 * local)
            ln.set_linewidth(0.45 + 1.6 * local)
        return lines

    anim = FuncAnimation(fig, update, frames=nframes, interval=1000 / fps, blit=False)
    path = ANIM / f"{label}_bridge_pulse.gif"
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print("wrote", path)
    return path


def make_continuum_wave(sample: str, nframes=48, fps=12):
    """Wave of emphasis traveling exit → persist → enter → deep."""
    xy, labs, tumor, label = load_sample(sample)
    gate = gate_label(label)
    states = [
        ("exiting_hypoxia", RP["foam"], "exiting"),
        ("persistent_hypoxia", RP["iris"], "persistent"),
        ("entering_hypoxia", RP["love"], "entering"),
        ("entering_deep_hypoxia", "#b4637a", "enter-deep"),
    ]
    masks = [(labs == s) & tumor for s, _, _ in states]
    fig, ax = plt.subplots(figsize=(7.2, 6.4), facecolor=RP["base"])
    ax.set_facecolor(RP["surface"])
    ax.scatter(xy[:, 0], xy[:, 1], s=1.2, c=RP["hl"], alpha=0.22, linewidths=0, zorder=0)
    scs = []
    for m, (st, col, lab) in zip(masks, states):
        sc = ax.scatter(xy[m, 0], xy[m, 1], s=10, c=col, alpha=0.35, linewidths=0, zorder=3, label=lab)
        scs.append(sc)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    title = ax.set_title(f"{label} {gate} · H≤8 continuum wave", color=RP["text"], fontsize=13, pad=10)
    ax.legend(loc="upper left", frameon=False, labelcolor=RP["subtle"], fontsize=8)
    fig.tight_layout()

    # phase centers for each state along continuum
    centers = np.array([0.0, 0.25, 0.5, 0.75])

    def update(frame):
        t = frame / nframes
        for i, sc in enumerate(scs):
            # gaussian bump traveling along continuum phase
            phase = (t - centers[i]) % 1.0
            # distance on circle
            d = min(phase, 1 - phase)
            amp = np.exp(-(d**2) / (2 * 0.08**2))
            n = masks[i].sum()
            sc.set_sizes(np.full(n, 8 + 28 * amp))
            sc.set_alpha(0.25 + 0.7 * amp)
        active = states[int((t * 4) % 4)][2]
        title.set_text(f"{label} {gate} · H≤8 continuum  ·  {active}")
        return scs + [title]

    anim = FuncAnimation(fig, update, frames=nframes, interval=1000 / fps, blit=False)
    path = ANIM / f"{label}_continuum_wave.gif"
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print("wrote", path)
    return path


def make_ribbon_breathe(sample: str, nframes=36, fps=12):
    xy, labs, tumor, label = load_sample(sample)
    f = local_fields(xy, labs, tumor)
    gate = gate_label(label)
    front = f["front"]
    vmax = np.percentile(front, 99)

    fig, ax = plt.subplots(figsize=(7.2, 6.4), facecolor=RP["base"])
    ax.set_facecolor(RP["surface"])
    # draw with plasma-like but using gold-love via custom: use magma and tint
    sc = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=front,
        s=4,
        cmap="magma",
        vmin=0,
        vmax=vmax,
        linewidths=0,
        alpha=0.85,
        rasterized=True,
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("front score", color=RP["subtle"])
    cb.ax.yaxis.set_tick_params(color=RP["muted"])
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=RP["subtle"])
    cb.outline.set_edgecolor(RP["hl"])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"{label} {gate} · H≤8 front ribbon breathing", color=RP["text"], fontsize=13, pad=10)
    fig.tight_layout()

    def update(frame):
        t = frame / nframes
        p = pulse(t)
        # modulate displayed field by pulse (visual breath)
        sc.set_array(front * (0.55 + 0.45 * p))
        sc.set_clim(0, vmax)
        sc.set_sizes(np.full(len(xy), 2.5 + 5.5 * p))
        sc.set_alpha(0.45 + 0.45 * p)
        return (sc,)

    anim = FuncAnimation(fig, update, frames=nframes, interval=1000 / fps, blit=False)
    path = ANIM / f"{label}_ribbon_breathe.gif"
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print("wrote", path)
    return path


def main():
    paths = []
    for sample in ["E14S", "E15S", "OVERLAY"]:
        paths.append(make_pulsating_front(sample))
        paths.append(make_bridge_pulse(sample))
        paths.append(make_continuum_wave(sample))
        paths.append(make_ribbon_breathe(sample))
    import json

    (ANIM / "manifest.json").write_text(
        json.dumps(
            {
                "theme": "rose-pine",
                "spatial": "H<=8 MAP microwells",
                "animations": [
                    {
                        "file": p.name,
                        "sample": p.name.split("_")[0],
                        "kind": "_".join(p.name.split("_")[1:]).replace(".gif", ""),
                    }
                    for p in paths
                ],
            },
            indent=2,
        )
    )
    print("done", ANIM)


if __name__ == "__main__":
    main()
