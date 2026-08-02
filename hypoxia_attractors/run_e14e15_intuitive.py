#!/usr/bin/env python3
"""HAL for E14S + E15S — intuitive hypoxia attractor / basin figures.

Pooled tumor cells from both ImageIT gates (max n). Shared landscape;
library-colored overlays. Language aims at basins you can read at a glance.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy import optimize, stats
from scipy.integrate import solve_ivp
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity, NearestNeighbors

ROOT = Path("/ix1/ylee/shared/MC38_Hypoxia_001/hypoxia_attractors")
FIGS = ROOT / "figures"
DATA = ROOT / "data"
FIGS.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

CSV = Path(
    "/ix1/ylee/shared/MC38_Hypoxia_001/reseq_finetune/hypoxia_velocity/hypoxia_dynamics_per_cell.csv"
)

C = {
    "bg": "#0b1320",
    "panel": "#101a2c",
    "ink": "#f2f5fa",
    "mute": "#93a0b8",
    "line": "#243049",
    "ox": "#2ec4b6",       # oxygenated / normoxic well
    "hyp": "#ff5a36",      # hypoxic well
    "saddle": "#ffc14d",   # ridge / decision point
    "enter": "#4ea1ff",
    "exit": "#8be0c8",
    "e14": "#ff7aa2",
    "e15": "#7ec8ff",
    "cell": "#d7deea",
}

STATE_COL = {
    "normoxic_stable": C["ox"],
    "reverted_stable": "#7aa89a",
    "reverted_posthypoxic": "#9bb8a8",
    "exiting_hypoxia": C["exit"],
    "transitional": C["saddle"],
    "persistent_hypoxia": C["hyp"],
    "entering_hypoxia": C["enter"],
    "entering_deep_hypoxia": "#2f6bff",
}


def style():
    mpl.rcParams.update(
        {
            "figure.facecolor": C["bg"],
            "axes.facecolor": C["panel"],
            "savefig.facecolor": C["bg"],
            "text.color": C["ink"],
            "axes.labelcolor": C["mute"],
            "axes.edgecolor": C["line"],
            "xtick.color": C["mute"],
            "ytick.color": C["mute"],
            "font.family": "DejaVu Sans",
            "axes.titlesize": 14,
            "figure.dpi": 160,
        }
    )


def stroke(txt_artist):
    txt_artist.set_path_effects([pe.withStroke(linewidth=3.2, foreground=C["bg"])])
    return txt_artist


def load_e14_e15():
    df = pd.read_csv(CSV)
    m = ~df["hypoxia_dynamics"].astype(str).eq("non_tumor")
    if "cell_type" in df.columns:
        m = m | df["cell_type"].astype(str).str.contains("Tumor", case=False, na=False)
    df = df.loc[m].copy()
    df["h"] = pd.to_numeric(df["hypoxia_score"], errors="coerce")
    df["v"] = pd.to_numeric(df["v_hypoxia"], errors="coerce")
    df = df[np.isfinite(df["h"]) & np.isfinite(df["v"])]
    lo, hi = np.quantile(df["v"], [0.01, 0.99])
    df = df[(df["v"] >= lo) & (df["v"] <= hi)].reset_index(drop=True)
    return df


def U(h, p):
    a, b, c, d, hm, lam = p[:6]
    return 0.25 * a * h**4 - 0.5 * b * h**2 + c * h + d * np.exp(-lam * (h - hm) ** 2)


def dU(h, p):
    a, b, c, d, hm, lam = p[:6]
    return a * h**3 - b * h + c - 2 * lam * d * (h - hm) * np.exp(-lam * (h - hm) ** 2)


def fit_U(h, v, w):
    kde = KernelDensity(bandwidth=0.35).fit(h.reshape(-1, 1))
    g = np.linspace(h.min() - 0.2, h.max() + 0.2, 400).reshape(-1, 1)
    dens = np.exp(kde.score_samples(g))
    dens /= dens.max()
    peaks = [float(g[i, 0]) for i in range(2, len(dens) - 2) if dens[i] > dens[i - 1] and dens[i] > dens[i + 1] and dens[i] > 0.28]
    peaks = sorted(peaks)
    if len(peaks) < 2:
        peaks = [float(np.quantile(h, 0.2)), float(np.quantile(h, 0.8))]
    hN, hH = peaks[0], peaks[-1]
    hm = 0.5 * (hN + hH)

    def loss(th):
        a, b, c, d, lam, gma = th
        if a <= 0 or lam <= 0 or gma <= 0:
            return 1e6
        p = (a, b, c, d, hm, lam, gma)
        e_fp = dU(hN, p) ** 2 + dU(hH, p) ** 2 + 0.2 * dU(hm, p) ** 2
        pred = -dU(h, p) / gma
        e_v = np.average((pred - v) ** 2, weights=w)
        depth = U(hm, p) - 0.5 * (U(hN, p) + U(hH, p))
        return e_fp + e_v + 0.25 * np.exp(-2.5 * max(depth, 0))

    th0 = np.array([0.4, 1.1, -0.1 * (hN + hH), 0.35, 1.4, 1.3])
    res = optimize.minimize(loss, th0, method="Nelder-Mead", options={"maxiter": 3000})
    a, b, c, d, lam, gma = res.x
    p = (
        float(np.clip(abs(a), 0.08, 2.0)),
        float(b),
        float(c),
        float(d),
        float(hm),
        float(np.clip(abs(lam), 0.3, 6.0)),
        float(np.clip(abs(gma), 0.5, 3.0)),
    )
    return p, {"hN": hN, "hH": hH, "hm": hm, "loss": float(res.fun)}


def fixed_points(p, hr):
    out = []
    for h0 in np.linspace(hr[0], hr[1], 50):
        sol = optimize.root(lambda x: dU(x[0], p), [h0])
        if not sol.success:
            continue
        h = float(sol.x[0])
        if not (hr[0] - 0.4 <= h <= hr[1] + 0.4):
            continue
        curv = (dU(h + 1e-4, p) - dU(h - 1e-4, p)) / 2e-4
        kind = "attractor" if curv > 0 else "saddle"
        if not out or abs(h - out[-1][0]) > 0.12:
            out.append((h, kind, curv))
        else:
            # keep more central
            out[-1] = (h, kind, curv)
    # unique sort
    uniq = []
    for r in sorted(out, key=lambda t: t[0]):
        if not uniq or abs(r[0] - uniq[-1][0]) > 0.12:
            uniq.append(r)
    return uniq


def basins_1d(p, hr, n=500):
    hs = np.linspace(hr[0], hr[1], n)
    h_end = hs.copy()
    for _ in range(500):
        h_end -= 0.02 * dU(h_end, p)
    fps = [fp for fp in fixed_points(p, hr) if fp[1] == "attractor"]
    ah = np.array([fp[0] for fp in fps])
    lab = np.argmin(np.abs(h_end[:, None] - ah[None, :]), axis=1)
    return hs, lab, fps


def soft_membership(X, centers, scales):
    logits = [-np.sum((X - c) ** 2, axis=1) / max(2 * s**2, 1e-6) for c, s in zip(centers, scales)]
    L = np.vstack(logits).T
    L -= L.max(1, keepdims=True)
    P = np.exp(L)
    return P / P.sum(1, keepdims=True)


def trajs(p, starts, t=7.0):
    out = []
    g = min(p[6], 2.2)

    def rhs(t, y):
        h, v = y
        return [v, -dU(h, p) - g * v]

    for s in starts:
        sol = solve_ivp(rhs, (0, t), s, rtol=1e-3, atol=1e-3, max_step=0.08)
        out.append(sol.y)
    return out


# ---------------- plots ----------------
def plot_hero_landscape(df, p, fps, path):
    """One glance: two wells, ridge, cells, enter/exit arrows."""
    h, v = df["h"].to_numpy(), df["v"].to_numpy()
    fig = plt.figure(figsize=(13.2, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.25], hspace=0.32, wspace=0.28)

    # top: 1D potential spanning full width
    ax = fig.add_subplot(gs[0, :])
    hg = np.linspace(h.min() - 0.5, h.max() + 0.5, 500)
    Ug = U(hg, p)
    Ug = Ug - Ug.min()
    # shade basins
    hs, lab, attrs = basins_1d(p, (hg.min(), hg.max()), n=500)
    U_line = U(hs, p) - U(hs, p).min()
    for k, col in enumerate([C["ox"], C["hyp"]][: len(attrs)]):
        m = lab == k
        ax.fill_between(hs[m], U_line[m], Ug.max() + 0.35, color=col, alpha=0.18, linewidth=0)
    ax.plot(hg, Ug, color=C["ink"], lw=2.8)
    labels = {}
    for hr, kind, _ in fps:
        uu = U(hr, p) - U(hg, p).min()
        if kind == "attractor":
            name = "Oxygenated\nbasin" if hr < 0 else "Hypoxic\nbasin"
            col = C["ox"] if hr < 0 else C["hyp"]
            ax.scatter([hr], [uu], s=160, c=col, zorder=5, edgecolors="white", lw=1.0)
            t = ax.text(hr, uu + 0.12 * Ug.max(), name, ha="center", va="bottom", color=col, fontsize=12, fontweight="600")
            stroke(t)
            labels[name] = hr
        else:
            ax.scatter([hr], [uu], s=110, c=C["saddle"], marker="D", zorder=5, edgecolors="white", lw=0.9)
            t = ax.text(hr, uu + 0.08 * Ug.max(), "decision ridge\n(plastic)", ha="center", color=C["saddle"], fontsize=10)
            stroke(t)
    # enter / exit arrows along potential
    if len(attrs) >= 2:
        ax.annotate(
            "enter hypoxia",
            xy=(attrs[1][0] - 0.15, 0.22 * Ug.max()),
            xytext=(attrs[0][0] + 0.2, 0.22 * Ug.max()),
            color=C["enter"],
            fontsize=11,
            fontweight="600",
            arrowprops=dict(arrowstyle="->", color=C["enter"], lw=2.0),
        )
        ax.annotate(
            "revert / exit",
            xy=(attrs[0][0] + 0.15, 0.38 * Ug.max()),
            xytext=(attrs[1][0] - 0.2, 0.38 * Ug.max()),
            color=C["exit"],
            fontsize=11,
            fontweight="600",
            arrowprops=dict(arrowstyle="->", color=C["exit"], lw=2.0),
        )
    ax.set_ylabel("energy landscape  U(h)   ↓ more stable")
    ax.set_xlabel("hypoxia program  h   →")
    ax.set_title("E14S + E15S · hypoxia has two stable homes", loc="left", color=C["ink"], fontsize=15, pad=10)
    ax.set_xlim(hg.min(), hg.max())
    ax.set_ylim(-0.05 * Ug.max(), Ug.max() + 0.45)

    # bottom-left: phase plane with dynamics colors
    ax = fig.add_subplot(gs[1, 0])
    order = [
        "normoxic_stable",
        "reverted_stable",
        "reverted_posthypoxic",
        "exiting_hypoxia",
        "transitional",
        "persistent_hypoxia",
        "entering_hypoxia",
        "entering_deep_hypoxia",
    ]
    for st in order:
        m = df["hypoxia_dynamics"].astype(str).eq(st)
        if m.sum() == 0:
            continue
        ax.scatter(
            df.loc[m, "h"],
            df.loc[m, "v"],
            s=14,
            c=STATE_COL.get(st, C["cell"]),
            alpha=0.75,
            linewidths=0,
            label=st.replace("_", " "),
        )
    hs = np.linspace(h.min() - 0.2, h.max() + 0.2, 24)
    vs = np.linspace(v.min() - 0.2, v.max() + 0.2, 24)
    HH, VV = np.meshgrid(hs, vs)
    ax.streamplot(HH, VV, VV, -dU(HH, p) - p[6] * VV, color="#ffffff55", density=1.05, linewidth=0.8, arrowsize=0.75)
    for hr, kind, _ in fps:
        col = C["saddle"] if kind == "saddle" else (C["ox"] if hr < 0 else C["hyp"])
        ax.scatter([hr], [0], s=140, c=col, marker="*" if kind == "attractor" else "D", edgecolors="white", zorder=6)
    ax.axhline(0, color=C["line"], lw=0.8)
    ax.set_xlabel("hypoxia program  h")
    ax.set_ylabel("velocity  v  (up = entering)")
    ax.set_title("Where cells sit · colored by fate", loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper left", labelcolor=C["mute"], ncols=1)

    # bottom-right: basins with falling paths
    ax = fig.add_subplot(gs[1, 1])
    hs2, lab2, attrs2 = basins_1d(p, (h.min() - 0.3, h.max() + 0.3), n=240)
    vs2 = np.linspace(v.min() - 0.3, v.max() + 0.3, 240)
    HH2, VV2 = np.meshgrid(hs2, vs2)
    LAB = np.tile(lab2, (len(vs2), 1))
    cmap = LinearSegmentedColormap.from_list("b", [C["ox"], C["hyp"]])
    ax.pcolormesh(HH2, VV2, LAB, cmap=cmap, shading="auto", alpha=0.35, vmin=0, vmax=max(1, LAB.max()))
    ax.scatter(h, v, s=5, c="#ffffff33", linewidths=0)
    rng = np.random.default_rng(3)
    starts = np.column_stack(
        [rng.uniform(h.min(), h.max(), 16), rng.uniform(0.55 * v.min(), 0.55 * v.max(), 16)]
    )
    for y in trajs(p, starts):
        ax.plot(y[0], y[1], color=C["ink"], lw=1.0, alpha=0.55)
        ax.scatter(y[0, -1], y[1, -1], s=22, c=C["saddle"], zorder=5, edgecolors="white", lw=0.4)
    for i, fp in enumerate(attrs2):
        col = C["ox"] if i == 0 else C["hyp"]
        name = "Oxygenated" if i == 0 else "Hypoxic"
        ax.scatter([fp[0]], [0], s=180, c=col, marker="*", edgecolors="white", zorder=6)
        t = ax.text(fp[0], 0.08 * (v.max() - v.min()), name, ha="center", color=col, fontsize=11, fontweight="600")
        stroke(t)
    ax.set_xlabel("h")
    ax.set_ylabel("v")
    ax.set_title("Basins · drop a cell anywhere, watch where it settles", loc="left")

    fig.suptitle(
        f"Hypoxia attractor landscape · E14S+E15S tumor cells (n={len(df):,})",
        color=C["ink"],
        fontsize=16,
        y=0.995,
    )
    fig.savefig(path, dpi=210, bbox_inches="tight")
    plt.close(fig)


def plot_libraries_on_shared(df, p, fps, path):
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.6), sharex=True, sharey=True)
    h, v = df["h"].to_numpy(), df["v"].to_numpy()
    hs = np.linspace(h.min() - 0.2, h.max() + 0.2, 22)
    vs = np.linspace(v.min() - 0.2, v.max() + 0.2, 22)
    HH, VV = np.meshgrid(hs, vs)
    FH, FV = VV, -dU(HH, p) - p[6] * VV

    panels = [
        (axes[0], df["sample"].eq("E14S"), C["e14"], "E14S · GFP− / ImageIT−"),
        (axes[1], df["sample"].eq("E15S"), C["e15"], "E15S · GFP+ / ImageIT+"),
        (axes[2], np.ones(len(df), bool), None, "Shared chip landscape"),
    ]
    for ax, m, col, title in panels:
        ax.streamplot(HH, VV, FH, FV, color="#ffffff40", density=0.95, linewidth=0.7, arrowsize=0.7)
        if col is None:
            ax.scatter(df.loc[df["sample"].eq("E14S"), "h"], df.loc[df["sample"].eq("E14S"), "v"], s=10, c=C["e14"], alpha=0.55, lw=0, label="E14S")
            ax.scatter(df.loc[df["sample"].eq("E15S"), "h"], df.loc[df["sample"].eq("E15S"), "v"], s=10, c=C["e15"], alpha=0.45, lw=0, label="E15S")
            ax.legend(frameon=False, fontsize=8, labelcolor=C["mute"])
        else:
            ax.scatter(df.loc[m, "h"], df.loc[m, "v"], s=12, c=col, alpha=0.7, lw=0)
        for hr, kind, _ in fps:
            cc = C["saddle"] if kind == "saddle" else (C["ox"] if hr < 0 else C["hyp"])
            ax.scatter([hr], [0], s=100, c=cc, marker="*" if kind == "attractor" else "D", edgecolors="white", zorder=5)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("h")
    axes[0].set_ylabel("v")
    fig.suptitle("Same basins on both ImageIT gates — more cells, same geometry", color=C["ink"], y=1.03)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_membership_story(df, P, Hent, centers, path):
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.4))
    h, v = df["h"].to_numpy(), df["v"].to_numpy()
    titles = [
        ("Belongs to oxygenated basin", C["ox"], 0),
        ("Stuck near the ridge (plastic)", C["saddle"], 1),
        ("Belongs to hypoxic basin", C["hyp"], 2),
    ]
    for ax, (title, col, k) in zip(axes, titles):
        cmap = LinearSegmentedColormap.from_list("m", [C["panel"], col])
        sc = ax.scatter(h, v, c=P[:, k], s=11, cmap=cmap, lw=0)
        ax.scatter(centers[k, 0], centers[k, 1], s=150, marker="*", c=col, edgecolors="white", zorder=5)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("h")
        ax.set_ylabel("v")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle("Soft membership · how strongly each cell is claimed by a basin", color=C["ink"], y=1.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # entropy intuitive
    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    sc = ax.scatter(h, v, c=Hent, s=14, cmap="magma", lw=0)
    # highlight top plastic
    top = Hent >= np.quantile(Hent, 0.9)
    ax.scatter(h[top], v[top], s=28, facecolors="none", edgecolors=C["saddle"], linewidths=0.9, label="most plastic 10%")
    ax.set_title("Plastic cells live on the ridge between basins", loc="left")
    ax.set_xlabel("h")
    ax.set_ylabel("v")
    ax.legend(frameon=False, labelcolor=C["mute"])
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("membership entropy  (high = undecided)", color=C["mute"])
    fig.savefig(path.parent / "e14e15_plastic_ridge.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_flux(T, path):
    names = ["Oxygenated", "Ridge", "Hypoxic"]
    fig, ax = plt.subplots(figsize=(6.4, 5.5))
    im = ax.imshow(T, cmap=LinearSegmentedColormap.from_list("f", [C["panel"], C["ox"], C["hyp"]]), vmin=0, vmax=max(0.55, T.max()))
    ax.set_xticks(range(3), names)
    ax.set_yticks(range(3), names)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{100*T[i,j]:.0f}%", ha="center", va="center", color="white", fontsize=13, fontweight="600")
    ax.set_xlabel("where they go")
    ax.set_ylabel("where they start")
    ax.set_title("Flow between basins (short step along velocity)", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_howto(path):
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    xs = np.linspace(0.6, 11.4, 400)
    Uline = 0.012 * (xs - 6) ** 4 - 0.22 * (xs - 6) ** 2 + 0.04 * np.exp(-((xs - 6) ** 2)) + 2.35
    ax.fill_between(xs, Uline, 3.6, color="#152033")
    ax.plot(xs, Uline, color=C["ink"], lw=2.6)
    ax.scatter([2.6, 9.4], [Uline[np.argmin(np.abs(xs - 2.6))], Uline[np.argmin(np.abs(xs - 9.4))]],
               s=180, c=[C["ox"], C["hyp"]], edgecolors="white", zorder=5)
    ax.scatter([6.0], [Uline[np.argmin(np.abs(xs - 6.0))]], s=120, c=C["saddle"], marker="D", edgecolors="white", zorder=5)
    ax.text(2.6, 1.05, "Oxygenated\nattractor", ha="center", color=C["ox"], fontsize=12, fontweight="600")
    ax.text(9.4, 1.05, "Hypoxic\nattractor", ha="center", color=C["hyp"], fontsize=12, fontweight="600")
    ax.text(6.0, 2.75, "ridge\n(enter / exit)", ha="center", color=C["saddle"], fontsize=10)
    ax.annotate("", xy=(8.6, 1.55), xytext=(3.4, 1.55), arrowprops=dict(arrowstyle="->", color=C["enter"], lw=2.2))
    ax.text(6, 1.68, "ImageIT+ history can sit anywhere — velocity says which way", ha="center", color=C["enter"], fontsize=9)
    ax.text(6.0, 0.35, "Read left→right as hypoxia program · depth = stability · arrows = reversion vs entry",
            ha="center", color=C["mute"], fontsize=10)
    ax.set_title("How to read this model", color=C["ink"], fontsize=14, loc="left", pad=8)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_counts(df, path):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    # by sample
    ax = axes[0]
    vc = df["sample"].value_counts()
    ax.bar(["E14S\nGFP−", "E15S\nGFP+"], [vc.get("E14S", 0), vc.get("E15S", 0)], color=[C["e14"], C["e15"]])
    ax.set_ylabel("tumor cells in landscape")
    ax.set_title("More cells than E28S · pooled for basins", loc="left")
    for i, s in enumerate(["E14S", "E15S"]):
        ax.text(i, vc.get(s, 0) + 30, str(vc.get(s, 0)), ha="center", color=C["ink"])

    ax = axes[1]
    order = [
        "normoxic_stable",
        "reverted_stable",
        "reverted_posthypoxic",
        "exiting_hypoxia",
        "transitional",
        "persistent_hypoxia",
        "entering_hypoxia",
        "entering_deep_hypoxia",
    ]
    counts = [int((df["hypoxia_dynamics"] == s).sum()) for s in order]
    cols = [STATE_COL[s] for s in order]
    ax.barh([s.replace("_", " ") for s in order[::-1]], counts[::-1], color=cols[::-1])
    ax.set_xlabel("cells")
    ax.set_title("Fate labels on the same landscape", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_site(summary):
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>E14 + E15 · Hypoxia basins</title>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,560;700&family=Source+Sans+3:wght@400;600&display=swap" rel="stylesheet"/>
<style>
:root{{--bg:#0b1320;--panel:#101a2c;--ink:#f2f5fa;--mute:#93a0b8;--ox:#2ec4b6;--hyp:#ff5a36;--sad:#ffc14d;--line:#243049}}
body{{margin:0;background:radial-gradient(900px 500px at 15% -10%,#1b3d3c66,transparent),radial-gradient(800px 480px at 90% 0%,#4a221866,transparent),var(--bg);color:var(--ink);font-family:"Source Sans 3",system-ui,sans-serif}}
.hero{{padding:4rem 6vw 2.5rem;max-width:52rem}}
.brand{{font-family:Fraunces,serif;font-size:clamp(2.3rem,5.5vw,3.8rem);line-height:.96;margin:0 0 .8rem;letter-spacing:-.02em}}
.brand span{{color:var(--hyp)}}
.lede{{color:var(--mute);font-size:1.08rem;line-height:1.5;margin:0 0 1.2rem}}
.pills{{display:flex;flex-wrap:wrap;gap:.5rem}}
.pill{{border:1px solid var(--line);background:var(--panel);padding:.45rem .8rem;border-radius:999px;font-size:.88rem;color:var(--mute)}}
.pill b{{color:var(--ox)}}
main{{padding:1rem 6vw 4rem;max-width:1140px;margin:0 auto}}
h2{{font-family:Fraunces,serif;font-size:1.45rem;margin:2.2rem 0 .7rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1rem}}
figure{{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:.55rem}}
img{{width:100%;border-radius:10px;display:block}}
figcaption{{color:var(--mute);font-size:.86rem;padding:.5rem .25rem .15rem;line-height:1.35}}
.note{{color:var(--mute);line-height:1.5;max-width:46rem}}
.math{{font-family:ui-monospace,Menlo,monospace;background:#090e18;border:1px solid var(--line);border-radius:12px;padding:.9rem 1rem;color:var(--ox);font-size:.86rem;white-space:pre-wrap}}
footer{{padding:1.4rem 6vw 3rem;color:var(--mute);font-size:.85rem;border-top:1px solid var(--line)}}
a{{color:#7ec8ff}}
</style></head><body>
<header class="hero">
  <h1 class="brand">E14 + E15<br/>hypoxia <span>basins</span></h1>
  <p class="lede">Two ImageIT gates, one shared dynamical landscape. Tumor cells fall into an <b style="color:var(--ox)">oxygenated</b> attractor or a <b style="color:var(--hyp)">hypoxic</b> attractor, with a plastic ridge between them where entry and reversion happen.</p>
  <div class="pills">
    <div class="pill"><b>{summary['n']:,}</b> tumor cells</div>
    <div class="pill"><b>{summary['n_e14']:,}</b> E14S · GFP−</div>
    <div class="pill"><b>{summary['n_e15']:,}</b> E15S · GFP+</div>
    <div class="pill">wells at h≈{summary['hN']:.2f} &amp; {summary['hH']:.2f}</div>
  </div>
</header>
<main>
  <h2>Start here</h2>
  <div class="grid">
    <figure><img src="figures/e14e15_howto_read.png"/><figcaption>How to read the landscape in one picture.</figcaption></figure>
    <figure><img src="figures/e14e15_counts.png"/><figcaption>Cell counts by library and fate label.</figcaption></figure>
  </div>

  <h2>The basins</h2>
  <p class="note">We fit a multistable energy <code>U(h)</code> on the hypoxia score axis, then lift it to the phase plane with velocity. Deep wells = stable fates. The ridge = undecided / entering / exiting cells.</p>
  <figure style="margin:1rem 0 1.2rem"><img src="figures/e14e15_hero_landscape.png"/><figcaption>Hero view: potential wells, fate-colored cells, and basins with falling trajectories.</figcaption></figure>

  <h2>E14 and E15 on the same map</h2>
  <figure style="margin:1rem 0 1.2rem"><img src="figures/e14e15_libraries.png"/><figcaption>GFP− and GFP+ both occupy the shared basins — ImageIT marks history, the landscape marks current dynamics.</figcaption></figure>

  <h2>Who is claimed by which basin?</h2>
  <div class="grid">
    <figure><img src="figures/e14e15_membership.png"/><figcaption>Soft membership to oxygenated, ridge, and hypoxic basins.</figcaption></figure>
    <figure><img src="figures/e14e15_plastic_ridge.png"/><figcaption>High membership entropy = plastic cells on the ridge.</figcaption></figure>
    <figure><img src="figures/e14e15_flux.png"/><figcaption>Short-time flow between basins along velocity.</figcaption></figure>
  </div>

  <h2>Model</h2>
  <div class="math">dh/dt = v
dv/dt = −U'(h) − γ v
U(h) = (a/4)h⁴ − (b/2)h² + c h + d exp(−λ(h−hₘ)²)</div>
  <p class="note" style="margin-top:.8rem">Inspired by multistable attractors / transition membership in Zhou et al. <i>Nat Methods</i> 2024 (STT), and attractor–basin / metastable substate thinking in Mason et al. <i>Stem Cell Reports</i> 2025.</p>
</main>
<footer>
  HAL · E14S + E15S pooled tumor cells · <a href="https://www.nature.com/articles/s41592-024-02266-x">STT</a> · <a href="https://doi.org/10.1016/j.stemcr.2025.102532">Substates &amp; attractors</a>
</footer>
</body></html>"""
    (ROOT / "index.html").write_text(html)


def main():
    style()
    print("loading E14+E15…", flush=True)
    df = load_e14_e15()
    print(df.groupby("sample").size().to_dict(), "total", len(df), flush=True)

    h, v = df["h"].to_numpy(), df["v"].to_numpy()
    X = np.column_stack([h, v])
    kde = KernelDensity(bandwidth=0.4).fit(X)
    w = np.exp(kde.score_samples(X))
    w /= w.mean()

    print("fitting U…", flush=True)
    p, meta = fit_U(h, v, w)
    fps = fixed_points(p, (h.min(), h.max()))
    print("meta", meta, "fps", fps, flush=True)

    gmm = GaussianMixture(
        n_components=3,
        covariance_type="full",
        random_state=0,
        means_init=np.array([[np.quantile(h, 0.2), 0], [np.median(h), 0], [np.quantile(h, 0.8), 0]]),
    )
    gmm.fit(X)
    order = np.argsort(gmm.means_[:, 0])
    centers = gmm.means_[order]
    scales = np.sqrt([np.mean(np.linalg.eigvalsh(gmm.covariances_[i])) for i in order])
    P = soft_membership(X, centers, scales * 1.2)
    Hent = stats.entropy(P.T)

    # transition flux
    F = np.column_stack([v, -dU(h, p) - p[6] * v])
    X2 = X + 0.4 * F
    nn = NearestNeighbors(n_neighbors=1).fit(X)
    _, idx = nn.kneighbors(X2)
    src, dst = P.argmax(1), P[idx[:, 0]].argmax(1)
    T = np.zeros((3, 3))
    for i, j in zip(src, dst):
        T[i, j] += 1
    T = T / np.maximum(T.sum(1, keepdims=True), 1)

    print("plotting…", flush=True)
    plot_howto(FIGS / "e14e15_howto_read.png")
    plot_counts(df, FIGS / "e14e15_counts.png")
    plot_hero_landscape(df, p, fps, FIGS / "e14e15_hero_landscape.png")
    plot_libraries_on_shared(df, p, fps, FIGS / "e14e15_libraries.png")
    plot_membership_story(df, P, Hent, centers, FIGS / "e14e15_membership.png")
    plot_flux(T, FIGS / "e14e15_flux.png")

    df_out = df.copy()
    df_out["attractor"] = [["Oxygenated", "Ridge", "Hypoxic"][i] for i in P.argmax(1)]
    df_out["membership_entropy"] = Hent
    for i, name in enumerate(["pi_ox", "pi_ridge", "pi_hyp"]):
        df_out[name] = P[:, i]
    df_out.to_csv(DATA / "e14e15_cells_with_membership.csv", index=False)

    attrs = [fp for fp in fps if fp[1] == "attractor"]
    summary = {
        "n": int(len(df)),
        "n_e14": int((df["sample"] == "E14S").sum()),
        "n_e15": int((df["sample"] == "E15S").sum()),
        "hN": float(attrs[0][0]) if attrs else float(meta["hN"]),
        "hH": float(attrs[-1][0]) if attrs else float(meta["hH"]),
        "hm": float(meta["hm"]),
        "potential": {"a": p[0], "b": p[1], "c": p[2], "d": p[3], "hm": p[4], "lam": p[5], "gamma": p[6]},
        "fixed_points": [{"h": a, "kind": b, "curvature": c} for a, b, c in fps],
        "transition_matrix": T.tolist(),
        "mean_entropy": float(np.mean(Hent)),
        "attractor_counts": df_out["attractor"].value_counts().to_dict(),
    }
    (DATA / "e14e15_summary.json").write_text(json.dumps(summary, indent=2))
    write_site(summary)
    print("done", summary["n"], "cells →", ROOT / "index.html", flush=True)


if __name__ == "__main__":
    main()
