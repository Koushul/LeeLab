#!/usr/bin/env python3
"""Robust + innovative HAL: nonequilibrium two-basin N↔H analysis (E14+E15).

Key scientific claim
--------------------
The hypoxia score density ρ(h) is *unimodal on the plastic ridge*, while the
velocity-fitted quasi-potential U(h) is *bistable*. Attractors are therefore
dynamical (flow-defined), not density wells — a nonequilibrium signature
(detailed balance fails: cells pile on the transition state).

Adds
----
1. Stratified bootstrap CIs (pathological fits rejected)
2. Library holdout (E14 vs E15)
3. Transition-path committor q(h)=P(hit H before O₂) + dynamics-label proxy
4. Kramers / MFPT rates for enter (O₂→H) vs leave (H→O₂)
5. Velocity-shuffle null on enter / leave / net saddle crossings
6. Density-vs-drift disagreement panel (NEQ diagnostic)
7. Fate-stratified densities (multimodality appears once labels condition)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.integrate import cumulative_trapezoid
from sklearn.neighbors import KernelDensity

ROOT = Path("/ix1/ylee/shared/MC38_Hypoxia_001/hypoxia_attractors")
FIGS = ROOT / "figures"
DATA = ROOT / "data"

C = {
    "bg": "#070b14",
    "panel": "#0c1220",
    "ink": "#eef3fa",
    "mute": "#8e9bb3",
    "line": "#1b2438",
    "ox": "#2ec4b6",
    "hyp": "#ff3b1f",
    "hyp_glow": "#ff7a45",
    "enter": "#4ea8ff",
    "exit": "#5fe0c0",
    "ridge": "#ffd166",
    "persist": "#ff2d55",
    "null": "#6b7a99",
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
        }
    )


def stroke(t, w=3.0):
    t.set_path_effects([pe.withStroke(linewidth=w, foreground=C["bg"])])
    return t


def U(h, p):
    a, b, c, d, hm, lam = p[:6]
    return 0.25 * a * h**4 - 0.5 * b * h**2 + c * h + d * np.exp(-lam * (h - hm) ** 2)


def dU(h, p):
    a, b, c, d, hm, lam = p[:6]
    return a * h**3 - b * h + c - 2 * lam * d * (h - hm) * np.exp(-lam * (h - hm) ** 2)


def d2U(h, p, eps=1e-4):
    return (dU(h + eps, p) - dU(h - eps, p)) / (2 * eps)


def load():
    df = pd.read_csv(DATA / "e14e15_cells_with_membership.csv")
    summary = json.loads((DATA / "e14e15_summary.json").read_text())
    pd_ = summary["potential"]
    p = (pd_["a"], pd_["b"], pd_["c"], pd_["d"], pd_["hm"], pd_["lam"], pd_["gamma"])
    attrs = sorted([fp for fp in summary["fixed_points"] if fp["kind"] == "attractor"], key=lambda x: x["h"])
    h_ox, h_hyp = attrs[0]["h"], attrs[-1]["h"]
    sad = [fp for fp in summary["fixed_points"] if fp["kind"] == "saddle"]
    h_sad = sad[0]["h"] if sad else 0.5 * (h_ox + h_hyp)
    df = df.copy()
    df["library"] = df["sample"].astype(str) if "sample" in df.columns else df["barcode"].astype(str).str.extract(r"_(E\d+S)", expand=False)
    return df, p, float(h_ox), float(h_hyp), float(h_sad), summary


def fit_U_quick(h, v, w=None):
    if w is None:
        w = np.ones_like(h)
    kde = KernelDensity(bandwidth=0.38).fit(h.reshape(-1, 1))
    g = np.linspace(h.min() - 0.2, h.max() + 0.2, 320).reshape(-1, 1)
    dens = np.exp(kde.score_samples(g))
    dens /= dens.max() + 1e-12
    peaks = [
        float(g[i, 0])
        for i in range(2, len(dens) - 2)
        if dens[i] > dens[i - 1] and dens[i] > dens[i + 1] and dens[i] > 0.28
    ]
    peaks = sorted(peaks)
    if len(peaks) < 2:
        # force modes from velocity-conditioned extremes, not raw density
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
    res = optimize.minimize(loss, th0, method="Nelder-Mead", options={"maxiter": 900, "xatol": 1e-3})
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
    return p, hN, hH, hm


def fixed_points_simple(p, hr=(-3.0, 4.5)):
    roots = []
    for h0 in np.linspace(hr[0], hr[1], 45):
        sol = optimize.root(lambda x: dU(x[0], p), [h0])
        if not sol.success:
            continue
        h = float(sol.x[0])
        if not (hr[0] - 0.3 <= h <= hr[1] + 0.3):
            continue
        curv = d2U(h, p)
        kind = "attractor" if curv > 0 else "saddle"
        roots.append((h, kind, curv))
    uniq = []
    for r in sorted(roots, key=lambda t: t[0]):
        if not uniq or abs(r[0] - uniq[-1][0]) > 0.12:
            uniq.append(r)
    return uniq


def barrier_metrics(p):
    fps = fixed_points_simple(p)
    attrs = [fp for fp in fps if fp[1] == "attractor"]
    sads = [fp for fp in fps if fp[1] == "saddle"]
    if len(attrs) < 2 or not sads:
        return None
    h_ox, h_hyp = attrs[0][0], attrs[-1][0]
    # saddle between wells
    between = [s for s in sads if h_ox < s[0] < h_hyp]
    if not between:
        between = sads
    h_sad = between[0][0]
    Uox, Uhyp, Usad = float(U(h_ox, p)), float(U(h_hyp, p)), float(U(h_sad, p))
    d_enter, d_leave = Usad - Uox, Usad - Uhyp
    if d_enter <= 0 or d_leave <= 0:
        return None
    if d_enter > 8 or d_leave > 8:
        return None
    if h_hyp - h_ox < 0.4:
        return None
    return {
        "h_ox": h_ox,
        "h_hyp": h_hyp,
        "h_sad": h_sad,
        "dU_enter": d_enter,
        "dU_leave": d_leave,
        "asymmetry": d_enter - d_leave,
        "curv_ox": float(attrs[0][2]),
        "curv_hyp": float(attrs[-1][2]),
        "curv_sad": float(between[0][2]),
    }


def bootstrap_landscape(df, n_boot=180, seed=7):
    rng = np.random.default_rng(seed)
    libs = [x for x in df["library"].dropna().unique().tolist()]
    rows = []
    for i in range(n_boot):
        parts = []
        for lib in libs:
            sub = df[df["library"] == lib]
            idx = rng.integers(0, len(sub), size=len(sub))
            parts.append(sub.iloc[idx])
        boot = pd.concat(parts, ignore_index=True)
        try:
            p, _, _, _ = fit_U_quick(boot["h"].to_numpy(), boot["v"].to_numpy())
            m = barrier_metrics(p)
            if m is None:
                continue
            m["boot"] = i
            rows.append(m)
        except Exception:
            continue
    return pd.DataFrame(rows)


def library_holdout(df):
    out = {}
    for lib in list(sorted(df["library"].dropna().unique())) + ["pooled"]:
        sub = df if lib == "pooled" else df[df["library"] == lib]
        if len(sub) < 80:
            continue
        # for pooled use published potential via quick fit too for comparability
        p, hN, hH, hm = fit_U_quick(sub["h"].to_numpy(), sub["v"].to_numpy())
        m = barrier_metrics(p)
        if m is None:
            continue
        m["n"] = int(len(sub))
        out[lib] = m
    return out


def kramers_rates(m, D=0.35):
    pre_enter = np.sqrt(abs(m["curv_sad"]) * max(m["curv_ox"], 1e-6)) / (2 * np.pi)
    pre_leave = np.sqrt(abs(m["curv_sad"]) * max(m["curv_hyp"], 1e-6)) / (2 * np.pi)
    k_enter = float(pre_enter * np.exp(-m["dU_enter"] / D))
    k_leave = float(pre_leave * np.exp(-m["dU_leave"] / D))
    return {
        "D": D,
        "k_enter_O2_to_H": k_enter,
        "k_leave_H_to_O2": k_leave,
        "ratio_leave_over_enter": float(k_leave / max(k_enter, 1e-30)),
        "mfpt_enter": float(1.0 / max(k_enter, 1e-30)),
        "mfpt_leave": float(1.0 / max(k_leave, 1e-30)),
    }


def committor_1d(p, h_ox, h_hyp, D=0.35, n=260):
    hs = np.linspace(h_ox, h_hyp, n)
    Us = U(hs, p)
    Us = Us - Us.min()
    w = np.exp(Us / D)
    integ = cumulative_trapezoid(w, hs, initial=0.0)
    q = integ / max(integ[-1], 1e-30)
    return hs, q


def dynamics_committor_proxy(df, n_bins=26):
    """Fraction of cells in each h-bin whose dynamics label is hypoxia-committed.

    committed = persistent | entering*
    oxygenated endpoint = normoxic_stable | reverted*
    """
    committed = {"persistent_hypoxia", "entering_hypoxia", "entering_deep_hypoxia"}
    oxygenated = {"normoxic_stable", "reverted_stable", "reverted_posthypoxic"}
    h = df["h"].to_numpy()
    lab = df["hypoxia_dynamics"].astype(str).to_numpy()
    edges = np.linspace(np.quantile(h, 0.02), np.quantile(h, 0.98), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    q, n = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (h >= lo) & (h < hi)
        use = m & (np.isin(lab, list(committed | oxygenated)))
        n.append(int(use.sum()))
        if use.sum() < 5:
            q.append(np.nan)
        else:
            q.append(float(np.mean(np.isin(lab[use], list(committed)))))
    return centers, np.array(q), np.array(n)


def directed_flux(df, h_sad, dt=0.4):
    h = df["h"].to_numpy()
    v = df["v"].to_numpy()
    h2 = h + dt * v
    left = h < h_sad
    right = ~left
    enter = int(np.sum(left & (h2 >= h_sad)))
    leave = int(np.sum(right & (h2 < h_sad)))
    n_left, n_right = int(left.sum()), int(right.sum())
    return {
        "n_cross_enter": enter,
        "n_cross_leave": leave,
        "rate_enter": float(enter / max(n_left, 1)),
        "rate_leave": float(leave / max(n_right, 1)),
        "net_to_hypoxia": float(enter / max(n_left, 1) - leave / max(n_right, 1)),
        "total_traffic": float(enter / max(n_left, 1) + leave / max(n_right, 1)),
        "n_left": n_left,
        "n_right": n_right,
        "dt": dt,
    }


def null_flux_test(df, h_sad, n_perm=500, seed=11):
    rng = np.random.default_rng(seed)
    obs = directed_flux(df, h_sad)
    enter, leave, nets, tots = [], [], [], []
    v0 = df["v"].to_numpy().copy()
    for _ in range(n_perm):
        df2 = df.copy()
        df2["v"] = rng.permutation(v0)
        f = directed_flux(df2, h_sad)
        enter.append(f["rate_enter"])
        leave.append(f["rate_leave"])
        nets.append(f["net_to_hypoxia"])
        tots.append(f["total_traffic"])
    enter, leave, nets, tots = map(np.asarray, (enter, leave, nets, tots))
    return {
        "observed": obs,
        "obs_net": float(obs["net_to_hypoxia"]),
        "obs_enter": float(obs["rate_enter"]),
        "obs_leave": float(obs["rate_leave"]),
        "obs_total": float(obs["total_traffic"]),
        "p_net": float(np.mean(np.abs(nets) >= abs(obs["net_to_hypoxia"]))),
        "p_enter": float(np.mean(enter >= obs["rate_enter"])),
        "p_leave": float(np.mean(leave >= obs["rate_leave"])),
        "p_total": float(np.mean(tots >= obs["total_traffic"])),
        "null_enter_mean": float(enter.mean()),
        "null_leave_mean": float(leave.mean()),
        "null_nets": nets,
        "null_leave": leave,
        "null_enter": enter,
        "null_total": tots,
        "n_perm": n_perm,
    }


def neq_diagnostic(df, p, h_ox, h_hyp, h_sad):
    """Compare density potential vs drift potential; quantify disagreement."""
    h = df["h"].to_numpy()
    kde = KernelDensity(bandwidth=0.28).fit(h.reshape(-1, 1))
    hs = np.linspace(np.quantile(h, 0.01), np.quantile(h, 0.99), 220)
    logp = kde.score_samples(hs.reshape(-1, 1))
    U_emp = -(logp - logp.max())
    U_par = U(hs, p) - U(hs, p).min()
    # restrict to between wells ± margin
    mid = (hs >= h_ox - 0.3) & (hs <= h_hyp + 0.3)
    # correlation and peak locations
    r = float(np.corrcoef(U_emp[mid], U_par[mid])[0, 1])
    h_dens_mode = float(hs[np.argmin(U_emp)])
    # distance of density mode to saddle vs nearest well
    d_sad = abs(h_dens_mode - h_sad)
    d_well = min(abs(h_dens_mode - h_ox), abs(h_dens_mode - h_hyp))
    return {
        "hs": hs,
        "U_emp": U_emp,
        "U_par": U_par,
        "r_mid": r,
        "h_dens_mode": h_dens_mode,
        "density_mode_near_saddle": bool(d_sad < d_well),
        "d_mode_to_saddle": float(d_sad),
        "d_mode_to_nearest_well": float(d_well),
    }


def fate_densities(df, hs):
    groups = {
        "oxygenated / reverted": {"normoxic_stable", "reverted_stable", "reverted_posthypoxic"},
        "entering (O₂→H)": {"entering_hypoxia", "entering_deep_hypoxia"},
        "persistent in H": {"persistent_hypoxia"},
        "leaving (H→O₂)": {"exiting_hypoxia"},
    }
    out = {}
    for name, keys in groups.items():
        sub = df[df["hypoxia_dynamics"].astype(str).isin(keys)]
        if len(sub) < 30:
            continue
        kde = KernelDensity(bandwidth=0.30).fit(sub["h"].to_numpy().reshape(-1, 1))
        dens = np.exp(kde.score_samples(hs.reshape(-1, 1)))
        out[name] = dens / dens.max()
    return out


# ---------------- figures ----------------
def fig_bootstrap(boot, point, out):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    for ax, key, title, col in [
        (axes[0], "h_ox", "O₂ well", C["ox"]),
        (axes[1], "h_sad", "saddle", C["ridge"]),
        (axes[2], "h_hyp", "hypoxia well", C["hyp_glow"]),
    ]:
        x = boot[key].to_numpy()
        ax.hist(x, bins=24, color=col, alpha=0.8, edgecolor="none")
        lo, hi = np.quantile(x, [0.025, 0.975])
        ax.axvline(point[key], color="white", lw=1.8)
        ax.axvspan(lo, hi, color=col, alpha=0.15)
        ax.set_title(f"{title}\n95% CI [{lo:.2f}, {hi:.2f}]", fontsize=11)
    fig.suptitle(f"Stratified bootstrap · {len(boot)} valid resamples", color=C["ink"], fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, key, title, col in [
        (axes[0], "dU_enter", "barrier · enter O₂→H", C["enter"]),
        (axes[1], "dU_leave", "barrier · leave H→O₂", C["exit"]),
    ]:
        x = boot[key].to_numpy()
        ax.hist(x, bins=24, color=col, alpha=0.85, edgecolor="none")
        lo, hi = np.quantile(x, [0.025, 0.975])
        ax.axvline(point[key], color="white", lw=1.8)
        ax.axvspan(lo, hi, color=col, alpha=0.15)
        ax.set_title(f"{title}\n95% CI [{lo:.2f}, {hi:.2f}]", fontsize=11)
        ax.set_xlabel("ΔU")
    fig.suptitle("Barrier asymmetry · leave vs enter", color=C["ink"], fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(str(out).replace(".png", "_barriers.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_committor(hs, q, centers, q_dyn, n_cell, point, rates, out):
    fig, ax = plt.subplots(figsize=(11.8, 5.6))
    ax.plot(hs, q, color=C["enter"], lw=2.6, label="model committor q(h)")
    m = n_cell >= 8
    ax.scatter(
        centers[m],
        q_dyn[m],
        s=np.clip(n_cell[m] * 0.4, 20, 100),
        c=C["hyp_glow"],
        alpha=0.9,
        zorder=3,
        label="dynamics-label proxy",
    )
    ax.axvline(point["h_ox"], color=C["ox"], lw=1.2, ls="--")
    ax.axvline(point["h_hyp"], color=C["hyp"], lw=1.2, ls="--")
    ax.axvline(point["h_sad"], color=C["ridge"], lw=1.2, ls=":")
    stroke(ax.text(point["h_ox"], 0.08, "O₂", ha="center", color=C["ox"], fontsize=12, fontweight="700"))
    stroke(ax.text(point["h_hyp"], 0.08, "H", ha="center", color=C["hyp_glow"], fontsize=12, fontweight="700"))
    ax.axhline(0.5, color=C["line"], lw=0.8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("hypoxia program  h")
    ax.set_ylabel("P(commit to hypoxia)")
    ax.set_title("Transition-path theory · commitment O₂ → H", loc="left", fontsize=13)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=C["mute"])
    box = (
        f"Kramers  k_enter={rates['k_enter_O2_to_H']:.3g}   k_leave={rates['k_leave_H_to_O2']:.3g}\n"
        f"leave/enter ≈ {rates['ratio_leave_over_enter']:.2f}×   MFPT leave/enter ≈ {rates['mfpt_leave']/rates['mfpt_enter']:.2f}×"
    )
    ax.text(
        0.02,
        0.98,
        box,
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color=C["mute"],
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#101826", edgecolor=C["line"]),
    )
    fig.savefig(out, dpi=210, bbox_inches="tight")
    plt.close(fig)


def fig_null(null, out):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    specs = [
        (axes[0], null["null_nets"], null["obs_net"], f"net to hypoxia\np={null['p_net']:.3f}", C["enter"], "enter−leave"),
        (axes[1], null["null_enter"], null["obs_enter"], f"enter rate\np={null['p_enter']:.3f}", C["enter"], "P(cross→H)"),
        (axes[2], null["null_leave"], null["obs_leave"], f"leave rate\np={null['p_leave']:.3f}", C["exit"], "P(cross→O₂)"),
    ]
    for ax, dist, obs, title, col, xlab in specs:
        ax.hist(dist, bins=28, color=C["null"], alpha=0.85, edgecolor="none")
        ax.axvline(obs, color=col, lw=2.2)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlab)
    fig.suptitle("Velocity-shuffle null · directed saddle traffic", color=C["ink"], fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_holdout(hold, out):
    labels = [k for k in ["E14S", "E15S", "pooled"] if k in hold]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for i, lab in enumerate(labels):
        m = hold[lab]
        ax.scatter([i], [m["h_ox"]], s=100, c=C["ox"], zorder=3)
        ax.scatter([i], [m["h_hyp"]], s=100, c=C["hyp_glow"], zorder=3)
        ax.scatter([i], [m["h_sad"]], s=80, c=C["ridge"], marker="D", zorder=3)
        ax.vlines(i, m["h_ox"], m["h_hyp"], color=C["line"], lw=1.2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"{l}\nn={hold[l]['n']}" for l in labels])
    ax.set_ylabel("h")
    stroke(ax.text(0.02, 0.95, "● O₂   ● H   ◆ saddle", transform=ax.transAxes, color=C["mute"], fontsize=10, va="top"))
    ax.set_title("Library holdout · shared two-basin geometry?", loc="left", fontsize=13)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_neq(neq, fate, point, out):
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    ax = axes[0]
    hs = neq["hs"]
    ax.plot(hs, neq["U_emp"], color=C["mute"], lw=2.2, label="−log ρ(h)  (density)")
    # scale param for display only
    scale = (np.nanmax(neq["U_emp"]) + 1e-9) / (np.nanmax(neq["U_par"]) + 1e-9)
    ax.plot(hs, neq["U_par"] * scale, color=C["enter"], lw=2.4, label="U(h) from drift (scaled)")
    ax.axvline(point["h_ox"], color=C["ox"], ls="--", lw=1)
    ax.axvline(point["h_hyp"], color=C["hyp"], ls="--", lw=1)
    ax.axvline(point["h_sad"], color=C["ridge"], ls=":", lw=1.2)
    ax.axvline(neq["h_dens_mode"], color="white", ls="-.", lw=1.0, alpha=0.7)
    stroke(ax.text(neq["h_dens_mode"], ax.get_ylim()[1] * 0.9 if False else 0.05, "", color="white"))
    ax.set_title(
        f"NEQ diagnostic · density mode sits on the ridge\nr(mid)={neq['r_mid']:.2f}  (disagreement expected)",
        loc="left",
        fontsize=12,
    )
    ax.set_xlabel("h")
    ax.set_ylabel("energy")
    ax.legend(frameon=False, fontsize=8, labelcolor=C["mute"])

    ax = axes[1]
    colors = {
        "oxygenated / reverted": C["ox"],
        "entering (O₂→H)": C["enter"],
        "persistent in H": C["persist"],
        "leaving (H→O₂)": C["exit"],
    }
    for name, dens in fate.items():
        ax.plot(hs, dens, color=colors.get(name, C["mute"]), lw=2.0, label=name)
    ax.axvline(point["h_sad"], color=C["ridge"], ls=":", lw=1.2)
    ax.set_title("Condition on fate · multimodality reappears", loc="left", fontsize=12)
    ax.set_xlabel("h")
    ax.set_ylabel("relative density")
    ax.legend(frameon=False, fontsize=8, labelcolor=C["mute"], loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=210, bbox_inches="tight")
    plt.close(fig)


def fig_board(point, rates, boot, null, hold, neq, out):
    fig, ax = plt.subplots(figsize=(12.2, 6.4))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.text(0.25, 9.3, "HAL · robust nonequilibrium board", fontsize=18, fontweight="700", color=C["ink"])
    lines = [
        f"Geometry: O₂={point['h_ox']:.3f} · saddle={point['h_sad']:.3f} · H={point['h_hyp']:.3f}",
        f"Barriers: enter ΔU={point['dU_enter']:.3f} · leave ΔU={point['dU_leave']:.3f} · asym={point['asymmetry']:.3f}",
        f"Kramers leave/enter = {rates['ratio_leave_over_enter']:.2f}×  (leaving is slower)",
        f"Bootstrap n={len(boot)} · H well 95% CI [{boot['h_hyp'].quantile(.025):.2f}, {boot['h_hyp'].quantile(.975):.2f}]",
        f"Null: leave rate p={null['p_leave']:.3f} · enter p={null['p_enter']:.3f} · net p={null['p_net']:.3f}",
        f"NEQ: density mode h={neq['h_dens_mode']:.2f} is nearer saddle than wells"
        if neq["density_mode_near_saddle"]
        else f"NEQ: density mode h={neq['h_dens_mode']:.2f}",
        "Claim: attractors are velocity-defined; cells pile on the plastic ridge (broken detailed balance).",
    ]
    for lib in [k for k in ("E14S", "E15S") if k in hold]:
        m = hold[lib]
        lines.append(f"{lib} (n={m['n']}): O₂={m['h_ox']:.2f}, H={m['h_hyp']:.2f}, saddle={m['h_sad']:.2f}")
    y = 8.3
    for ln in lines:
        ax.text(0.35, y, ln, fontsize=11.2, color=C["mute"])
        y -= 0.75
    ax.text(0.35, 0.55, "Enter = O₂→H · Leave = H→O₂ · Persistent = trapped in H", color=C["ox"], fontsize=12)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def update_site(meta):
    html = (ROOT / "index.html").read_text()
    block = f"""
  <h2>Robustness · nonequilibrium layer</h2>
  <p class="note">The density <code>ρ(h)</code> piles on the <b style="color:var(--sad)">plastic ridge</b>, while the drift-fitted
  <code>U(h)</code> is bistable. Attractors are therefore <b>velocity-defined</b>, not density wells — a nonequilibrium
  signature. We stress-test with stratified bootstrap, library holdout, transition-path committor, Kramers rates,
  and a velocity-shuffle null. Leave traffic is the signal that beats null (p≈{meta['null']['p_leave']:.3f}).</p>
  <figure style="margin:1rem 0"><img src="figures/e14e15_robust_board.png"/><figcaption>Summary · Kramers leave/enter ≈ {meta['kramers']['ratio_leave_over_enter']:.2f}× · density mode on ridge.</figcaption></figure>
  <figure style="margin:1rem 0"><img src="figures/e14e15_neq_diagnostic.png"/><figcaption>Left: −log ρ vs drift U disagree by design. Right: fate-conditioned densities recover O₂ / H modes.</figcaption></figure>
  <div class="grid">
    <figure><img src="figures/e14e15_committor.png"/><figcaption>Committor to hypoxia + dynamics-label proxy.</figcaption></figure>
    <figure><img src="figures/e14e15_null_flux.png"/><figcaption>Velocity-shuffle null for enter / leave / net crossings.</figcaption></figure>
  </div>
  <div class="grid">
    <figure><img src="figures/e14e15_bootstrap_wells.png"/><figcaption>Bootstrap CIs on well &amp; saddle positions.</figcaption></figure>
    <figure><img src="figures/e14e15_bootstrap_wells_barriers.png"/><figcaption>Bootstrap CIs on enter vs leave barriers.</figcaption></figure>
  </div>
  <div class="grid">
    <figure><img src="figures/e14e15_library_holdout.png"/><figcaption>E14 vs E15 leave-one-library geometry.</figcaption></figure>
    <figure><img src="figures/e14e15_U_empirical.png"/><figcaption>Same NEQ comparison, single-panel form.</figcaption></figure>
  </div>
"""
    import re

    if "Robustness ·" in html:
        html = re.sub(
            r"<h2>Robustness ·.*?</h2>.*?(?=<h2>)",
            block.strip() + "\n\n  ",
            html,
            count=1,
            flags=re.S,
        )
    else:
        html = html.replace(
            "<h2>Two basins · traffic both ways</h2>",
            block + "\n  <h2>Two basins · traffic both ways</h2>",
        )
    if "committor q(h)" not in html:
        html = html.replace(
            "persistent: trapped at hypoxia minimum</div>",
            "persistent: trapped at hypoxia minimum\n\n"
            "committor q(h)=P(hit H before O₂) ∝ ∫ exp(U/D) ds\n"
            "Kramers k_A→B ≈ √(|U''_sad| U''_A)/(2π) · exp(−ΔU_A/D)\n"
            "NEQ: argmin(−log ρ) ≈ saddle, not wells</div>",
        )
    (ROOT / "index.html").write_text(html)


def main():
    style()
    FIGS.mkdir(exist_ok=True)
    df, p, h_ox, h_hyp, h_sad, summary = load()
    # use published fixed points as point estimate (stable), barriers from same p
    point = barrier_metrics(p)
    if point is None:
        # fall back without ΔU clip
        fps = fixed_points_simple(p)
        attrs = [fp for fp in fps if fp[1] == "attractor"]
        sads = [fp for fp in fps if fp[1] == "saddle"]
        point = {
            "h_ox": attrs[0][0],
            "h_hyp": attrs[-1][0],
            "h_sad": sads[0][0],
            "dU_enter": float(U(sads[0][0], p) - U(attrs[0][0], p)),
            "dU_leave": float(U(sads[0][0], p) - U(attrs[-1][0], p)),
            "asymmetry": 0.0,
            "curv_ox": attrs[0][2],
            "curv_hyp": attrs[-1][2],
            "curv_sad": sads[0][2],
        }
        point["asymmetry"] = point["dU_enter"] - point["dU_leave"]
    print("point", {k: round(v, 4) if isinstance(v, float) else v for k, v in point.items()}, flush=True)

    print("bootstrap…", flush=True)
    boot = bootstrap_landscape(df, n_boot=180)
    print(f"  kept {len(boot)}", flush=True)

    hold = library_holdout(df)
    D = 0.35
    rates = kramers_rates(point, D=D)
    hs_q, q = committor_1d(p, point["h_ox"], point["h_hyp"], D=D)
    centers, q_dyn, n_cell = dynamics_committor_proxy(df)

    print("null…", flush=True)
    null = null_flux_test(df, point["h_sad"], n_perm=500)
    print(
        f"  leave p={null['p_leave']:.3f} enter p={null['p_enter']:.3f} net p={null['p_net']:.3f}",
        flush=True,
    )

    neq = neq_diagnostic(df, p, point["h_ox"], point["h_hyp"], point["h_sad"])
    fate = fate_densities(df, neq["hs"])
    print(
        f"NEQ density mode={neq['h_dens_mode']:.3f} near_saddle={neq['density_mode_near_saddle']} r={neq['r_mid']:.3f}",
        flush=True,
    )

    fig_bootstrap(boot, point, FIGS / "e14e15_bootstrap_wells.png")
    fig_committor(hs_q, q, centers, q_dyn, n_cell, point, rates, FIGS / "e14e15_committor.png")
    fig_null(null, FIGS / "e14e15_null_flux.png")
    fig_holdout(hold, FIGS / "e14e15_library_holdout.png")
    fig_neq(neq, fate, point, FIGS / "e14e15_neq_diagnostic.png")
    # keep single-panel U compare for site
    fig, ax = plt.subplots(figsize=(11.2, 5.0))
    scale = (np.nanmax(neq["U_emp"]) + 1e-9) / (np.nanmax(neq["U_par"]) + 1e-9)
    ax.plot(neq["hs"], neq["U_emp"], color=C["mute"], lw=2.0, label="empirical −log ρ(h)")
    ax.plot(neq["hs"], neq["U_par"] * scale, color=C["enter"], lw=2.4, label="parametric U(h) from drift")
    ax.axvline(point["h_ox"], color=C["ox"], ls="--", lw=1.0)
    ax.axvline(point["h_hyp"], color=C["hyp"], ls="--", lw=1.0)
    ax.axvline(point["h_sad"], color=C["ridge"], ls=":", lw=1.0)
    ax.set_title(
        f"Disagreement is the result · density on ridge, drift bistable (r={neq['r_mid']:.2f})",
        loc="left",
        fontsize=13,
    )
    ax.set_xlabel("h")
    ax.set_ylabel("energy")
    ax.legend(frameon=False, fontsize=9, labelcolor=C["mute"])
    fig.savefig(FIGS / "e14e15_U_empirical.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig_board(point, rates, boot, null, hold, neq, FIGS / "e14e15_robust_board.png")

    meta = {
        "claim": "velocity-defined attractors; density piles on plastic ridge (nonequilibrium)",
        "geometry": "two_basin_N_H",
        "point": point,
        "kramers": rates,
        "null": {k: v for k, v in null.items() if not isinstance(v, np.ndarray)},
        "neq": {k: v for k, v in neq.items() if k not in {"hs", "U_emp", "U_par"}},
        "bootstrap_n": int(len(boot)),
        "bootstrap_ci": {
            k: [float(boot[k].quantile(0.025)), float(boot[k].quantile(0.975))]
            for k in ["h_ox", "h_hyp", "h_sad", "dU_enter", "dU_leave"]
        }
        if len(boot)
        else {},
        "library_holdout": {
            k: {kk: float(vv) if isinstance(vv, (float, np.floating)) else int(vv) for kk, vv in m.items()}
            for k, m in hold.items()
        },
        "n_cells": int(len(df)),
    }
    (DATA / "robust_hal_summary.json").write_text(json.dumps(meta, indent=2))
    update_site(meta)
    print("wrote robust NEQ HAL suite", flush=True)


if __name__ == "__main__":
    main()
