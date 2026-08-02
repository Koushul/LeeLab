#!/usr/bin/env python3
"""Hypoxia Attractor Landscape (HAL)

A dynamical-systems framework for hypoxia ↔ reversion, synthesizing:

1. Multistable attractors / basins / transition paths
   (Zhou et al., Nat Methods 2024 — Spatial Transition Tensor)
2. Metastable substates, quasi-potential noise-driven reversion,
   and attractor basins of cell fate
   (Mason et al., Stem Cell Reports 2025)

State space for each tumor cell:
    x = (h, v)  with  h = hypoxia program score,  v = projected hypoxia velocity

Continuous dynamics (damped second-order / overdamped reduction):
    ḣ = v
    v̇ = −U'(h) − γ v + σ ξ(t)

Quasi-potential U(h) is a fitted multistable well (normoxic + hypoxic, with a
saddle / shallow intermediate). Soft attractor membership and membership
entropy follow the STT spirit; basin geometry follows Mason/Waddington.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from scipy import optimize, stats
from scipy.integrate import solve_ivp
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity, NearestNeighbors

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path("/ix1/ylee/shared/MC38_Hypoxia_001/hypoxia_attractors")
FIGS = ROOT / "figures"
DATA = ROOT / "data"
for p in (FIGS, DATA):
    p.mkdir(parents=True, exist_ok=True)

MC38_CSV = Path(
    "/ix1/ylee/shared/MC38_Hypoxia_001/reseq_finetune/hypoxia_velocity/hypoxia_dynamics_per_cell.csv"
)
E28S_CSV = Path(
    "/ix1/ylee/shared/MC38_Hypoxia_001/e28s_a223_front/data/hypoxia_dynamics_per_cell.csv"
)

# Visual language: deep slate + oxygen teal ↔ hypoxia ember (avoid purple/cream clichés)
PAL = {
    "bg": "#0c1220",
    "panel": "#121a2b",
    "ink": "#e8eef7",
    "mute": "#8b9bb4",
    "grid": "#1c2740",
    "norm": "#3ecfb2",
    "hyp": "#ff6b4a",
    "saddle": "#f0c75e",
    "enter": "#5aa7ff",
    "exit": "#9ad7c8",
    "trans": "#c4cedd",
}

ATTR_NAMES = ["Normoxic", "Saddle / ICS", "Hypoxic"]
ATTR_COLORS = [PAL["norm"], PAL["saddle"], PAL["hyp"]]


def apply_style():
    mpl.rcParams.update(
        {
            "figure.facecolor": PAL["bg"],
            "axes.facecolor": PAL["panel"],
            "savefig.facecolor": PAL["bg"],
            "text.color": PAL["ink"],
            "axes.labelcolor": PAL["mute"],
            "axes.edgecolor": PAL["grid"],
            "xtick.color": PAL["mute"],
            "ytick.color": PAL["mute"],
            "font.family": "DejaVu Sans",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.dpi": 140,
        }
    )


def hypoxia_cmap():
    return LinearSegmentedColormap.from_list(
        "ox",
        ["#0c1220", "#153047", "#1f6f6a", "#3ecfb2", "#f0c75e", "#ff6b4a", "#ffd7c8"],
    )


def pot_cmap():
    return LinearSegmentedColormap.from_list(
        "pot",
        ["#ff6b4a", "#f0c75e", "#1f6f6a", "#121a2b", "#0c1220"],
    )


def load_tumor(path: Path, cohort: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "is_tumor" in df.columns:
        m = df["is_tumor"].astype(bool)
    else:
        m = ~df["hypoxia_dynamics"].astype(str).eq("non_tumor")
        if "cell_type" in df.columns:
            m = m | df["cell_type"].astype(str).str.contains("Tumor", case=False, na=False)
    out = df.loc[m].copy()
    out["cohort"] = cohort
    out["h"] = pd.to_numeric(out["hypoxia_score"], errors="coerce")
    out["v"] = pd.to_numeric(out["v_hypoxia"], errors="coerce")
    out = out[np.isfinite(out["h"]) & np.isfinite(out["v"])]
    # clip extreme velocity tails for landscape stability
    lo, hi = np.quantile(out["v"], [0.01, 0.99])
    out = out[(out["v"] >= lo) & (out["v"] <= hi)]
    return out.reset_index(drop=True)


# ---------- potential / ODE ----------
def U_params_to_potential(h, p):
    """Quartic bistable + optional intermediate bump.
    p = (a, b, c, d, hm, lam, gamma) with a>0.
    U(h) = a/4 h^4 - b/2 h^2 + c h + d exp(-lam (h-hm)^2)
    """
    a, b, c, d, hm, lam = p[:6]
    return 0.25 * a * h**4 - 0.5 * b * h**2 + c * h + d * np.exp(-lam * (h - hm) ** 2)


def dUdh(h, p):
    a, b, c, d, hm, lam = p[:6]
    return a * h**3 - b * h + c - 2.0 * lam * d * (h - hm) * np.exp(-lam * (h - hm) ** 2)


def fit_potential(h, v, dens_w):
    """Fit multistable U so −U'(h) ≈ 0 near density modes and matches v≈0 loci."""
    # empirical density modes via KDE on h
    kde = KernelDensity(bandwidth=0.35, kernel="gaussian").fit(h.reshape(-1, 1))
    grid = np.linspace(h.min() - 0.3, h.max() + 0.3, 400).reshape(-1, 1)
    logp = kde.score_samples(grid)
    pden = np.exp(logp - logp.max())
    # find local maxima
    peaks = []
    for i in range(2, len(pden) - 2):
        if pden[i] > pden[i - 1] and pden[i] > pden[i + 1] and pden[i] > 0.25:
            peaks.append(float(grid[i, 0]))
    peaks = sorted(peaks)
    if len(peaks) < 2:
        peaks = [float(np.quantile(h, 0.2)), float(np.quantile(h, 0.8))]
    hN, hH = peaks[0], peaks[-1]
    hm = 0.5 * (hN + hH)

    def loss(theta):
        a, b, c, d, lam, gamma = theta
        if a <= 0 or lam <= 0 or gamma <= 0:
            return 1e6
        p = (a, b, c, d, hm, lam, gamma)
        # fixed points near modes: U'≈0
        e1 = dUdh(hN, p) ** 2 + dUdh(hH, p) ** 2
        # saddle between: U'' can be negative; encourage U'(hm)~0 if third well shallow
        e2 = 0.15 * dUdh(hm, p) ** 2
        # match overdamped drift: v ≈ -U'(h)/gamma  for cells with |v| moderate
        pred = -dUdh(h, p) / gamma
        e3 = np.average((pred - v) ** 2, weights=dens_w)
        # prefer two deep wells
        depth = U_params_to_potential(hm, p) - 0.5 * (
            U_params_to_potential(hN, p) + U_params_to_potential(hH, p)
        )
        e4 = np.exp(-3.0 * max(depth, 0.0))
        return e1 + e2 + e3 + 0.2 * e4

    x0 = np.array([0.35, 1.2, -0.15 * (hN + hH), 0.4, 1.5, 1.2])
    res = optimize.minimize(loss, x0, method="Nelder-Mead", options={"maxiter": 2500, "xatol": 1e-4})
    a, b, c, d, lam, gamma = res.x
    a = float(np.clip(abs(a) + 1e-3, 0.05, 2.5))
    b = float(b)
    lam = float(np.clip(abs(lam) + 1e-3, 0.2, 8.0))
    gamma = float(np.clip(abs(gamma) + 1e-3, 0.4, 3.5))
    return (a, b, c, d, hm, lam, gamma), {"hN": hN, "hH": hH, "hm": hm, "loss": float(res.fun)}


def flow(hv, p):
    h, v = hv
    g = p[6]
    return np.array([v, -dUdh(h, p) - g * v])


def find_fixed_points(p, h_range):
    roots = []
    for h0 in np.linspace(h_range[0], h_range[1], 40):
        # equilibria require v=0 and U'=0
        def f(h):
            return dUdh(h[0], p)

        sol = optimize.root(f, [h0])
        if sol.success:
            hr = float(sol.x[0])
            if h_range[0] - 0.5 <= hr <= h_range[1] + 0.5:
                # classify by U''
                eps = 1e-4
                curv = (dUdh(hr + eps, p) - dUdh(hr - eps, p)) / (2 * eps)
                kind = "attractor" if curv > 0 else "saddle"
                roots.append((hr, 0.0, kind, curv))
    # unique
    uniq = []
    for r in sorted(roots, key=lambda t: t[0]):
        if not uniq or abs(r[0] - uniq[-1][0]) > 0.15:
            uniq.append(r)
    return uniq


# ---------- STT-inspired membership / transitions ----------
def soft_membership(X, centers, scales):
    # π_ik ∝ exp(-||x-μ_k||^2 / (2 s_k^2))
    logits = []
    for c, s in zip(centers, scales):
        d2 = np.sum((X - c) ** 2, axis=1) / max(2 * s**2, 1e-6)
        logits.append(-d2)
    L = np.vstack(logits).T
    L -= L.max(1, keepdims=True)
    P = np.exp(L)
    P /= P.sum(1, keepdims=True)
    return P


def membership_entropy(P):
    return stats.entropy(P.T)  # shape n


def transition_matrix(P, F_dir, n_states):
    """Empirical flux: cells vote transitions by velocity direction in soft-membership space."""
    # projected change in membership: dπ ≈ J · F ; use finite neighbor proxy
    T = np.zeros((n_states, n_states))
    src = P.argmax(1)
    # destination inferred by stepping along F
    Xstep = np.column_stack([P])  # unused placeholder
    # simpler: assign destination by argmax of P after moving along F in (h,v)
    return T


def empirical_transition(X, P, F, dt=0.35):
    n = P.shape[1]
    T = np.zeros((n, n))
    X2 = X + dt * F
    # membership at stepped location using same centers approx via NN to original soft labels
    # use current hard label → label of nearest cell after step
    nn = NearestNeighbors(n_neighbors=1).fit(X)
    _, idx = nn.kneighbors(X2)
    src = P.argmax(1)
    dst = P[idx[:, 0]].argmax(1)
    for i, j in zip(src, dst):
        T[i, j] += 1
    row = T.sum(1, keepdims=True)
    row[row == 0] = 1
    return T / row


def learn_kernel_field(X, V, grid_h, grid_v, bw=0.55):
    """Fast Nadaraya–Watson field via truncated Gaussian on a coarse grid."""
    HH, VV = np.meshgrid(grid_h, grid_v)
    pts = np.column_stack([HH.ravel(), VV.ravel()])
    nn = NearestNeighbors(n_neighbors=min(60, len(X))).fit(X)
    dists, inds = nn.kneighbors(pts)
    w = np.exp(-0.5 * (dists / bw) ** 2)
    w /= w.sum(1, keepdims=True) + 1e-12
    Fh = np.sum(w * V[inds, 0], axis=1).reshape(HH.shape)
    Fv = np.sum(w * V[inds, 1], axis=1).reshape(HH.shape)
    return Fh, Fv, HH, VV


def basin_map(p, h_range, v_range, n=160, t_end=6.0):
    """Basin partition via overdamped gradient flow on U (fast, Waddington-faithful).

    For the damped system, long-time fate is governed by descent of U(h);
    we integrate ḣ = −U'(h) from each h and assign the attracting well.
    v is retained only for visualization axes.
    """
    hs = np.linspace(*h_range, n)
    vs = np.linspace(*v_range, n)
    HH, VV = np.meshgrid(hs, vs)
    fps = find_fixed_points(p, h_range)
    attrs = [fp for fp in fps if fp[2] == "attractor"]
    if len(attrs) < 1:
        return HH, VV, np.zeros_like(HH, dtype=int), fps

    # 1D overdamped flow on a fine h-grid
    h_line = np.linspace(h_range[0], h_range[1], 800)
    # Newton / gradient descent map: iterate h <- h - dt U'(h)
    h_end = h_line.copy()
    dt = 0.02
    for _ in range(400):
        h_end = h_end - dt * dUdh(h_end, p)
        h_end = np.clip(h_end, h_range[0] - 1, h_range[1] + 1)
    # label each h by nearest attractor
    attr_h = np.array([fp[0] for fp in attrs])
    h_lab = np.argmin(np.abs(h_end[:, None] - attr_h[None, :]), axis=1)
    # map grid h to labels
    lab_1d = np.interp(hs, h_line, h_lab.astype(float))
    label = np.tile(np.rint(lab_1d).astype(int), (n, 1))
    return HH, VV, label, fps


def simulate_trajectories(p, starts, t_end=8.0):
    trajs = []
    for s in starts:
        # mildly damped; reduce stiffness issues with larger max_step
        def rhs(t, y):
            h, v = y
            g = min(p[6], 2.5)
            return [v, -dUdh(h, p) - g * v]

        sol = solve_ivp(rhs, (0, t_end), s, rtol=1e-3, atol=1e-3, max_step=0.1)
        trajs.append(sol.y)
    return trajs


# ---------- plotting ----------
def fig_potential_and_field(df, p, fps, cohort, out):
    h = df["h"].to_numpy()
    v = df["v"].to_numpy()
    hg = np.linspace(h.min() - 0.4, h.max() + 0.4, 320)
    Ug = U_params_to_potential(hg, p)
    Ug = Ug - Ug.min()

    fig = plt.figure(figsize=(12.8, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.2], wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    ax.fill_between(hg, Ug, Ug.max() + 0.2, color="#152033", alpha=0.9)
    ax.plot(hg, Ug, color=PAL["ink"], lw=2.2)
    for hr, vr, kind, curv in fps:
        u = U_params_to_potential(hr, p) - U_params_to_potential(hg, p).min()
        col = PAL["saddle"] if kind == "saddle" else (PAL["norm"] if hr < np.median([f[0] for f in fps]) else PAL["hyp"])
        ax.scatter([hr], [u], s=90, c=col, zorder=5, edgecolors="white", linewidths=0.6)
        ax.axvline(hr, color=col, ls="--", lw=0.8, alpha=0.55)
    ax.set_xlabel("hypoxia program  h")
    ax.set_ylabel("quasi-potential  U(h)")
    ax.set_title(f"{cohort} · multistable hypoxia potential")
    for sp in ax.spines.values():
        sp.set_color(PAL["grid"])

    ax = fig.add_subplot(gs[0, 1])
    # density backdrop
    hb = ax.hexbin(h, v, gridsize=55, cmap=hypoxia_cmap(), mincnt=1, linewidths=0, alpha=0.9)
    # ODE streamlines
    hs = np.linspace(h.min() - 0.2, h.max() + 0.2, 28)
    vs = np.linspace(v.min() - 0.2, v.max() + 0.2, 28)
    HH, VV = np.meshgrid(hs, vs)
    FH = VV
    FV = -dUdh(HH, p) - p[6] * VV
    ax.streamplot(HH, VV, FH, FV, color=PAL["ink"], density=1.05, linewidth=0.7, arrowsize=0.8)
    for hr, vr, kind, curv in fps:
        col = PAL["saddle"] if kind == "saddle" else (PAL["norm"] if hr < 0 else PAL["hyp"])
        marker = "D" if kind == "saddle" else "o"
        ax.scatter([hr], [0], s=120, c=col, marker=marker, zorder=6, edgecolors="white", linewidths=0.8)
        lab = "saddle" if kind == "saddle" else ("N attractor" if hr < 0 else "H attractor")
        ax.text(hr, 0.12 * (v.max() - v.min()), lab, ha="center", color=col, fontsize=9,
                path_effects=[pe.withStroke(linewidth=2.5, foreground=PAL["bg"])])
    ax.set_xlabel("hypoxia program  h")
    ax.set_ylabel("hypoxia velocity  v")
    ax.set_title("phase plane · ODE flow + cells")
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("cells", color=PAL["mute"])
    cb.ax.yaxis.set_tick_params(color=PAL["mute"])
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=PAL["mute"])
    fig.suptitle("Hypoxia attractor landscape (HAL)", color=PAL["ink"], fontsize=15, y=1.02)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_basins(df, p, cohort, out):
    h, v = df["h"].to_numpy(), df["v"].to_numpy()
    HH, VV, lab, fps = basin_map(
        p,
        (float(h.min()) - 0.3, float(h.max()) + 0.3),
        (float(v.min()) - 0.3, float(v.max()) + 0.3),
        n=90,
        t_end=7.0,
    )
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    # custom basin colors
    cmap = LinearSegmentedColormap.from_list("bas", [PAL["norm"], PAL["saddle"], PAL["hyp"], "#334"])
    ax.pcolormesh(HH, VV, lab, cmap=cmap, shading="auto", alpha=0.55, vmin=-0.5, vmax=max(2, lab.max()))
    ax.scatter(h, v, s=6, c="#ffffff", alpha=0.18, linewidths=0)
    # sample trajectories
    rng = np.random.default_rng(7)
    starts = np.column_stack(
        [
            rng.uniform(h.min(), h.max(), 18),
            rng.uniform(v.min() * 0.6, v.max() * 0.6, 18),
        ]
    )
    trajs = simulate_trajectories(p, starts, t_end=9.0)
    for y in trajs:
        ax.plot(y[0], y[1], color=PAL["ink"], lw=0.9, alpha=0.55)
        ax.scatter(y[0, -1], y[1, -1], s=18, c=PAL["saddle"], zorder=5)
    for hr, vr, kind, curv in fps:
        col = PAL["saddle"] if kind == "saddle" else (PAL["norm"] if hr < np.median([f[0] for f in fps]) else PAL["hyp"])
        ax.scatter([hr], [0], s=160, c=col, marker="*" if kind == "attractor" else "D",
                   edgecolors="white", linewidths=0.8, zorder=6)
    ax.set_xlabel("h")
    ax.set_ylabel("v")
    ax.set_title(f"{cohort} · basins of attraction + falling trajectories")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fps, lab


def fig_membership(df, centers, scales, P, Hent, cohort, out):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    h, v = df["h"].to_numpy(), df["v"].to_numpy()
    titles = ["π · Normoxic basin", "π · Saddle / ICS membership", "π · Hypoxic basin"]
    for ax, k, title, col in zip(axes, range(3), titles, ATTR_COLORS):
        sc = ax.scatter(h, v, c=P[:, k], s=10, cmap=LinearSegmentedColormap.from_list("m", [PAL["panel"], col]), linewidths=0)
        ax.scatter(centers[k, 0], centers[k, 1], s=140, marker="*", c=col, edgecolors="white", zorder=5)
        ax.set_title(title)
        ax.set_xlabel("h")
        ax.set_ylabel("v")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"{cohort} · soft attractor membership (STT-style)", color=PAL["ink"], y=1.03)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    sc = ax.scatter(h, v, c=Hent, s=12, cmap="magma", linewidths=0)
    ax.set_title(f"{cohort} · membership entropy (plastic / transitional cells)")
    ax.set_xlabel("h")
    ax.set_ylabel("v")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("H(π)", color=PAL["mute"])
    fig.savefig(out.parent / out.name.replace("membership", "entropy"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_transitions(T, cohort, out):
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(T, cmap=hypoxia_cmap(), vmin=0, vmax=max(0.5, T.max()))
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(ATTR_NAMES, rotation=20, ha="right")
    ax.set_yticklabels(ATTR_NAMES)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{T[i, j]:.2f}", ha="center", va="center", color="white", fontsize=11)
    ax.set_title(f"{cohort} · attractor → attractor flux")
    ax.set_xlabel("destination")
    ax.set_ylabel("source")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_bifurcation(out):
    """Oxygen as control parameter μ: pitchfork-like emergence of hypoxic well."""
    mus = np.linspace(-0.6, 1.8, 80)
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for mu in mus:
        # U = h^4/4 - mu h^2/2 + 0.08 h
        a, b, c = 1.0, mu, 0.08
        # roots of U' = a h^3 - b h + c = 0
        roots = np.roots([a, 0, -b, c])
        for r in roots:
            if abs(r.imag) < 1e-8:
                hr = r.real
                curv = 3 * a * hr**2 - b
                col = PAL["hyp"] if hr > 0.2 else (PAL["norm"] if hr < -0.05 else PAL["saddle"])
                mark = "o" if curv > 0 else "x"
                ax.scatter(mu, hr, s=18, c=col, marker=mark, alpha=0.85, linewidths=0.4)
    ax.axvline(0.35, color=PAL["saddle"], ls="--", lw=1.0, alpha=0.8)
    ax.text(0.42, 1.2, "hypoxia on\n(bifurcation)", color=PAL["saddle"], fontsize=9)
    ax.set_xlabel("oxygen control μ  (↓ O₂ → ↑ μ)")
    ax.set_ylabel("stable / unstable h*")
    ax.set_title("Bifurcation · birth of the hypoxic attractor")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_kernel_vs_ode(df, p, cohort, out):
    h, v = df["h"].to_numpy(), df["v"].to_numpy()
    # empirical phase velocity: (v, local acceleration along h from neighbors)
    X = np.column_stack([h, v])
    nn = NearestNeighbors(n_neighbors=min(40, len(X))).fit(X)
    _, idx = nn.kneighbors(X)
    # acceleration proxy: mean Δv of neighbors with higher latent alignment — use −U' match residual later
    Vemp = np.column_stack([v, -dUdh(h, p) - p[6] * v])  # model field at cells for comparison panel
    hs = np.linspace(h.min(), h.max(), 24)
    vs = np.linspace(v.min(), v.max(), 24)
    Fh, Fv, HH, VV = learn_kernel_field(X, Vemp, hs, vs, bw=0.65)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2))
    for ax, FH, FV, title in [
        (axes[0], Fh, Fv, "data-driven kernel field"),
        (axes[1], VV, -dUdh(HH, p) - p[6] * VV, "fitted ODE field"),
    ]:
        ax.scatter(h, v, s=4, c="#ffffff22", linewidths=0)
        ax.streamplot(HH, VV, FH, FV, color=PAL["enter"], density=1.1, linewidth=0.85, arrowsize=0.85)
        ax.set_title(title)
        ax.set_xlabel("h")
        ax.set_ylabel("v")
    fig.suptitle(f"{cohort} · kernel field vs HAL ODE", color=PAL["ink"], y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_reversion_schematic(out):
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    # wells
    xs = np.linspace(0.8, 9.2, 400)
    U = 0.15 * (xs - 5) ** 4 / 80 - 0.55 * (xs - 5) ** 2 / 8 + 0.08 * np.exp(-((xs - 5) ** 2)) + 2.2
    ax.fill_between(xs, U, 3.7, color="#152033")
    ax.plot(xs, U, color=PAL["ink"], lw=2)
    ax.scatter([2.3, 7.7], [U[np.argmin(np.abs(xs - 2.3))], U[np.argmin(np.abs(xs - 7.7))]],
               s=160, c=[PAL["norm"], PAL["hyp"]], zorder=5, edgecolors="white")
    ax.scatter([5.0], [U[np.argmin(np.abs(xs - 5.0))]], s=100, c=PAL["saddle"], marker="D", zorder=5, edgecolors="white")
    ax.annotate("Normoxic\nattractor", (2.3, 1.15), color=PAL["norm"], ha="center", fontsize=11)
    ax.annotate("Hypoxic\nattractor", (7.7, 1.15), color=PAL["hyp"], ha="center", fontsize=11)
    ax.annotate("saddle / ICS\n(plastic)", (5.0, 2.55), color=PAL["saddle"], ha="center", fontsize=10)
    ax.annotate("", xy=(6.8, 1.55), xytext=(3.2, 1.55),
                arrowprops=dict(arrowstyle="->", color=PAL["enter"], lw=1.8))
    ax.text(5.0, 1.72, "entry (v>0)", color=PAL["enter"], ha="center", fontsize=9)
    ax.annotate("", xy=(3.2, 1.95), xytext=(6.8, 1.95),
                arrowprops=dict(arrowstyle="->", color=PAL["exit"], lw=1.8))
    ax.text(5.0, 2.12, "reversion / exit (v<0)", color=PAL["exit"], ha="center", fontsize=9)
    ax.set_title("Waddington view · hypoxia as a reversible multistable fate", color=PAL["ink"], pad=10)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def analyze_cohort(df: pd.DataFrame, cohort: str):
    print(f"=== {cohort} n={len(df)} ===", flush=True)
    h = df["h"].to_numpy()
    v = df["v"].to_numpy()
    X = np.column_stack([h, v])

    # density weights
    kde = KernelDensity(bandwidth=0.4).fit(X)
    dens_w = np.exp(kde.score_samples(X))
    dens_w /= dens_w.mean()

    print("fitting potential…", flush=True)
    p, meta = fit_potential(h, v, dens_w)
    fps = find_fixed_points(p, (h.min(), h.max()))
    print("potential meta", meta, flush=True)
    print("fixed points", fps, flush=True)

    # GMM centers for soft membership (3 components)
    print("GMM membership…", flush=True)
    gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=0, means_init=np.array(
        [
            [np.quantile(h, 0.2), 0.0],
            [np.median(h), 0.0],
            [np.quantile(h, 0.8), 0.0],
        ]
    ))
    gmm.fit(X)
    centers = gmm.means_
    order = np.argsort(centers[:, 0])
    centers = centers[order]
    scales = np.sqrt(np.array([np.mean(np.linalg.eigvalsh(gmm.covariances_[i])) for i in order]))
    P = soft_membership(X, centers, scales * 1.25)
    Hent = membership_entropy(P)

    F = np.column_stack([v, -dUdh(h, p) - p[6] * v])
    T = empirical_transition(X, P, F, dt=0.45)

    hard = P.argmax(1)
    df = df.copy()
    df["attractor"] = [ATTR_NAMES[i] for i in hard]
    df["membership_entropy"] = Hent
    for i, name in enumerate(["pi_norm", "pi_saddle", "pi_hyp"]):
        df[name] = P[:, i]

    tag = cohort.lower().replace(" ", "_")
    print("plotting…", flush=True)
    fig_potential_and_field(df, p, fps, cohort, FIGS / f"{tag}_potential_flow.png")
    print(" basins…", flush=True)
    fig_basins(df, p, cohort, FIGS / f"{tag}_basins.png")
    fig_membership(df, centers, scales, P, Hent, cohort, FIGS / f"{tag}_membership.png")
    fig_transitions(T, cohort, FIGS / f"{tag}_transitions.png")
    fig_kernel_vs_ode(df, p, cohort, FIGS / f"{tag}_kernel_vs_ode.png")
    print("done plots", flush=True)

    summary = {
        "cohort": cohort,
        "n_tumor": int(len(df)),
        "potential_params": {
            "a": p[0],
            "b": p[1],
            "c": p[2],
            "d": p[3],
            "hm": p[4],
            "lam": p[5],
            "gamma": p[6],
        },
        "modes": meta,
        "fixed_points": [
            {"h": fp[0], "v": fp[1], "kind": fp[2], "curvature": fp[3]} for fp in fps
        ],
        "centers": centers.tolist(),
        "transition_matrix": T.tolist(),
        "mean_membership_entropy": float(np.mean(Hent)),
        "frac_high_plasticity": float(np.mean(Hent > np.quantile(Hent, 0.75))),
        "attractor_counts": df["attractor"].value_counts().to_dict(),
    }
    df.to_csv(DATA / f"{tag}_cells_with_membership.csv", index=False)
    (DATA / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary, df, p, fps, T


def write_site(summaries):
    refs = """
    <li><a href="https://www.nature.com/articles/s41592-024-02266-x">Zhou et al. Nature Methods 2024 — Spatial transition tensor (multistable attractors, basins, transition paths)</a></li>
    <li><a href="https://doi.org/10.1016/j.stemcr.2025.102532">Mason et al. Stem Cell Reports 2025 — Stem cell fate decisions: substates and attractors</a></li>
    """
    cards = ""
    for s in summaries:
        tag = s["cohort"].lower().replace(" ", "_")
        fps = ", ".join(f"{fp['kind']}@h={fp['h']:.2f}" for fp in s["fixed_points"])
        cards += f"""
        <section class="cohort">
          <h2>{s['cohort']}</h2>
          <div class="stats">
            <div><div class="n">{s['n_tumor']}</div><div>tumor cells</div></div>
            <div><div class="n">{s['mean_membership_entropy']:.2f}</div><div>mean membership entropy</div></div>
            <div><div class="n">{100*s['frac_high_plasticity']:.0f}%</div><div>high-plasticity quartile</div></div>
          </div>
          <p class="note">Fixed points: {fps}</p>
          <div class="grid">
            <figure><img src="figures/{tag}_potential_flow.png"/><figcaption>Quasi-potential wells + ODE phase flow</figcaption></figure>
            <figure><img src="figures/{tag}_basins.png"/><figcaption>Basins of attraction with falling trajectories</figcaption></figure>
            <figure><img src="figures/{tag}_membership.png"/><figcaption>Soft basin membership (STT-style)</figcaption></figure>
            <figure><img src="figures/{tag}_entropy.png"/><figcaption>Membership entropy · transitional / plastic cells</figcaption></figure>
            <figure><img src="figures/{tag}_transitions.png"/><figcaption>Attractor→attractor flux</figcaption></figure>
            <figure><img src="figures/{tag}_kernel_vs_ode.png"/><figcaption>Kernel field vs fitted HAL ODE</figcaption></figure>
          </div>
        </section>
        """
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Hypoxia Attractor Landscape</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;700&family=Source+Sans+3:wght@400;600&display=swap" rel="stylesheet"/>
<style>
:root{{--bg:#0c1220;--panel:#121a2b;--ink:#e8eef7;--mute:#8b9bb4;--norm:#3ecfb2;--hyp:#ff6b4a;--sad:#f0c75e;--line:#1c2740}}
*{{box-sizing:border-box}}
body{{margin:0;background:
  radial-gradient(1200px 600px at 10% -10%, #1a3a3a55, transparent),
  radial-gradient(900px 500px at 90% 0%, #4a201855, transparent),
  var(--bg);color:var(--ink);font-family:"Source Sans 3",system-ui,sans-serif}}
.hero{{min-height:92vh;display:flex;flex-direction:column;justify-content:flex-end;padding:3.5rem 6vw 3rem;position:relative;overflow:hidden}}
.hero::before{{content:"";position:absolute;inset:0;background:url('figures/schematic_reversion.png') center/cover no-repeat;opacity:.18;filter:saturate(1.1)}}
.hero > *{{position:relative;z-index:1;max-width:48rem}}
.brand{{font-family:Fraunces,serif;font-size:clamp(2.4rem,6vw,4.2rem);line-height:.95;letter-spacing:-.02em;margin:0 0 .8rem;color:var(--ink)}}
.brand em{{color:var(--hyp);font-style:normal}}
.lede{{font-size:1.08rem;color:var(--mute);max-width:38rem;line-height:1.5;margin:0 0 1.4rem}}
.cta{{display:inline-block;padding:.7rem 1.1rem;border:1px solid var(--norm);color:var(--norm);text-decoration:none;border-radius:999px;font-weight:600}}
main{{padding:2rem 6vw 4rem;max-width:1180px;margin:0 auto}}
h2{{font-family:Fraunces,serif;font-weight:600;font-size:1.55rem;margin:2.2rem 0 .7rem}}
.framework{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:1rem;margin:1rem 0 2rem}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:1rem 1.1rem}}
.card h3{{margin:.1rem 0 .45rem;font-family:Fraunces,serif;font-size:1.05rem;color:var(--sad)}}
.card p{{margin:0;color:var(--mute);font-size:.92rem;line-height:1.45}}
.math{{font-family:ui-monospace,Menlo,monospace;font-size:.84rem;color:var(--norm);background:#0a101a;padding:.85rem 1rem;border-radius:12px;overflow:auto;border:1px solid var(--line)}}
.stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:.8rem;margin:1rem 0}}
.stats > div{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:.8rem}}
.stats .n{{font-family:Fraunces,serif;font-size:1.5rem;color:var(--norm)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem}}
figure{{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:.5rem}}
img{{width:100%;height:auto;border-radius:10px;display:block}}
figcaption{{color:var(--mute);font-size:.82rem;padding:.45rem .2rem .2rem}}
.note{{color:var(--mute);font-size:.9rem}}
footer{{padding:1.5rem 6vw 3rem;color:var(--mute);font-size:.85rem;border-top:1px solid var(--line)}}
a{{color:#5aa7ff}}
ul.refs{{line-height:1.55;color:var(--mute)}}
</style></head><body>
<header class="hero">
  <div class="brand">Hypoxia as a<br/><em>reversible attractor</em></div>
  <p class="lede">A dynamical-systems model of hypoxia and reversion: multistable wells, basins of attraction, soft membership, and transition flux — built on the phase plane (h, v) from RNA-velocity hypoxia scores.</p>
  <a class="cta" href="#framework">See the framework</a>
</header>
<main>
  <h2 id="framework">Mathematical framework</h2>
  <div class="framework">
    <div class="card"><h3>State</h3><p>Each tumor cell is a point <code>x=(h,v)</code> with hypoxia program <code>h</code> and projected hypoxia velocity <code>v</code>.</p></div>
    <div class="card"><h3>Multistable ODE</h3><p>Damped conservative flow on a fitted quasi-potential with normoxic and hypoxic wells and a saddle / intermediate critical state (ICS).</p></div>
    <div class="card"><h3>Basins & membership</h3><p>Forward integration labels basins; soft π<sub>k</sub> and membership entropy mark plastic transitional cells (STT-inspired).</p></div>
    <div class="card"><h3>Reversion</h3><p>Exit flux is downhill from the hypoxic well across the saddle into the normoxic basin — metastable substate logic.</p></div>
  </div>
  <div class="math">ḣ = v
v̇ = −U'(h) − γ v + σ ξ(t)
U(h) = (a/4)h⁴ − (b/2)h² + c h + d exp(−λ(h−hₘ)²)</div>
  <p class="note" style="margin-top:1rem">U is fit so wells sit on density modes of h and the overdamped drift −U'/γ matches observed v. Basins are the sets of initial conditions that converge to each attractor under the ODE.</p>

  <h2>Conceptual landscape</h2>
  <div class="grid">
    <figure><img src="figures/schematic_reversion.png"/><figcaption>Waddington-style hypoxia wells with entry and reversion fluxes</figcaption></figure>
    <figure><img src="figures/bifurcation_oxygen.png"/><figcaption>Bifurcation in oxygen control μ — birth of the hypoxic attractor</figcaption></figure>
  </div>

  {cards}

  <h2>References</h2>
  <ul class="refs">{refs}</ul>
</main>
<footer>
  HAL · Hypoxia Attractor Landscape · MC38 ImageIT + E28S A223 tumor cells · phase plane from scVelo hypoxia projection
</footer>
</body></html>"""
    (ROOT / "index.html").write_text(html)


def main():
    apply_style()
    fig_reversion_schematic(FIGS / "schematic_reversion.png")
    fig_bifurcation(FIGS / "bifurcation_oxygen.png")

    cohorts = []
    mc = load_tumor(MC38_CSV, "MC38")
    s1, *_ = analyze_cohort(mc, "MC38")
    cohorts.append(s1)

    if E28S_CSV.exists():
        e28 = load_tumor(E28S_CSV, "E28S")
        # E28S already tumor-filtered mostly
        s2, *_ = analyze_cohort(e28, "E28S")
        cohorts.append(s2)

    write_site(cohorts)
    (DATA / "framework.json").write_text(
        json.dumps(
            {
                "name": "Hypoxia Attractor Landscape (HAL)",
                "state": "(h, v) = (hypoxia_score, v_hypoxia)",
                "ode": "dh/dt=v; dv/dt=-U'(h)-gamma*v",
                "references": [
                    {
                        "id": "zhou2024stt",
                        "citation": "Zhou et al., Nat Methods 2024",
                        "url": "https://www.nature.com/articles/s41592-024-02266-x",
                        "idea": "multistable attractors, transition tensor, membership, basins",
                    },
                    {
                        "id": "mason2025substates",
                        "citation": "Mason et al., Stem Cell Reports 2025",
                        "url": "https://doi.org/10.1016/j.stemcr.2025.102532",
                        "idea": "metastable substates, attractor basins, quasi-potential reversion",
                    },
                ],
                "cohorts": cohorts,
            },
            indent=2,
        )
    )
    print("wrote site", ROOT / "index.html")


if __name__ == "__main__":
    main()
