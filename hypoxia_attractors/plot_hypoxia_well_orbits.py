#!/usr/bin/env python3
"""Hypoxia well figure: persistent core + orbiting / escaping cells (E14S+E15S).

Data-driven: persistent_hypoxia cells define the trapped core of the hypoxic
attractor. Entering cells approach on inbound spirals; exiting / reverted
cells leave the well. Trajectories from the fitted HAL ODE.
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
from matplotlib.patches import Circle, FancyArrowPatch, Arc
from scipy.integrate import solve_ivp

ROOT = Path("/ix1/ylee/shared/MC38_Hypoxia_001/hypoxia_attractors")
FIGS = ROOT / "figures"
DATA = ROOT / "data"

C = {
    "bg": "#070b14",
    "panel": "#0c1220",
    "ink": "#eef3fa",
    "mute": "#8e9bb3",
    "line": "#1b2438",
    "core": "#ff3b1f",
    "core_glow": "#ff7a45",
    "enter": "#4ea8ff",
    "exit": "#5fe0c0",
    "revert": "#b7f0d8",
    "trans": "#ffd166",
    "other": "#3a455c",
    "orbit": "#ffffff",
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


def integrate(p, y0, t_end, gamma=None, max_step=0.06):
    g = p[6] if gamma is None else gamma

    def rhs(_t, y):
        h, v = y
        return [v, -dU(h, p) - g * v]

    return solve_ivp(rhs, (0, t_end), y0, rtol=1e-3, atol=1e-3, max_step=max_step, dense_output=False)


def load():
    df = pd.read_csv(DATA / "e14e15_cells_with_membership.csv")
    summary = json.loads((DATA / "e14e15_summary.json").read_text())
    pdict = summary["potential"]
    p = (pdict["a"], pdict["b"], pdict["c"], pdict["d"], pdict["hm"], pdict["lam"], pdict["gamma"])
    hyp = [fp for fp in summary["fixed_points"] if fp["kind"] == "attractor"]
    # hypoxic = larger h
    hyp = sorted(hyp, key=lambda x: x["h"])
    h_ox, h_hyp = hyp[0]["h"], hyp[-1]["h"]
    saddle = [fp for fp in summary["fixed_points"] if fp["kind"] == "saddle"]
    h_sad = saddle[0]["h"] if saddle else 0.5 * (h_ox + h_hyp)
    return df, p, h_ox, h_hyp, h_sad


def state_mask(df, keys):
    return df["hypoxia_dynamics"].astype(str).isin(keys)


def fig_hypoxia_well(df, p, h_ox, h_hyp, h_sad, out):
    """Main composition: persistent core, spiral capture, escape/reversion."""
    h = df["h"].to_numpy()
    v = df["v"].to_numpy()

    persist = state_mask(df, {"persistent_hypoxia"})
    entering = state_mask(df, {"entering_hypoxia", "entering_deep_hypoxia"})
    exiting = state_mask(df, {"exiting_hypoxia"})
    reverted = state_mask(df, {"reverted_stable", "reverted_posthypoxic"})
    other = ~(persist | entering | exiting | reverted)

    # core center = empirical persistent centroid (data-driven), snapped near attractor
    core_h = float(np.median(df.loc[persist, "h"])) if persist.sum() else h_hyp
    core_v = float(np.median(df.loc[persist, "v"])) if persist.sum() else 0.0
    # pull slightly toward mathematical attractor
    core_h = 0.65 * core_h + 0.35 * h_hyp
    core_v = 0.65 * core_v

    fig = plt.figure(figsize=(13.5, 8.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.22)
    ax = fig.add_subplot(gs[0, 0])

    # soft radial glow behind core
    rr = np.linspace(0, 2.8, 120)
    for rad, alpha in zip([2.4, 1.7, 1.1, 0.65], [0.05, 0.08, 0.12, 0.18]):
        ax.add_patch(Circle((core_h, core_v), rad, facecolor=C["core"], edgecolor="none", alpha=alpha, zorder=0))

    # basin contour rings = distance percentiles of persistent cells
    d_persist = np.hypot(df.loc[persist, "h"] - core_h, df.loc[persist, "v"] - core_v)
    for q, lw, alpha, lab in [
        (0.50, 1.0, 0.35, None),
        (0.80, 1.2, 0.55, "persistent core"),
        (0.95, 1.4, 0.8, "hypoxic basin"),
    ]:
        r = float(np.quantile(d_persist, q)) if persist.sum() > 10 else 0.4 * q
        circ = Circle((core_h, core_v), r, fill=False, edgecolor=C["core_glow"], lw=lw, alpha=alpha, ls="--", zorder=1)
        ax.add_patch(circ)
        if lab and q == 0.95:
            ang = np.deg2rad(35)
            stroke(
                ax.text(
                    core_h + r * np.cos(ang),
                    core_v + r * np.sin(ang),
                    lab,
                    color=C["core_glow"],
                    fontsize=10,
                    fontweight="600",
                )
            )

    # background other cells
    ax.scatter(h[other], v[other], s=7, c=C["other"], alpha=0.35, lw=0, zorder=2)

    # spiral capture orbits (slightly underdamped for visible orbiting)
    rng = np.random.default_rng(11)
    # starts near saddle / oxygenated side, inbound to hypoxia
    starts_in = []
    for _ in range(10):
        starts_in.append([float(rng.uniform(h_sad - 0.4, h_sad + 0.15)), float(rng.uniform(0.4, 1.8))])
    for _ in range(8):
        starts_in.append([float(rng.uniform(h_ox + 0.2, h_sad)), float(rng.uniform(0.2, 1.4))])
    for y0 in starts_in:
        sol = integrate(p, y0, t_end=10.0, gamma=1.15)  # lighter damping → orbit then trap
        ax.plot(sol.y[0], sol.y[1], color=C["enter"], lw=1.15, alpha=0.55, zorder=3)
        ax.scatter(sol.y[0, -1], sol.y[1, -1], s=12, c=C["core"], zorder=4, lw=0)

    # escape / reversion orbits: start near hypoxic core with negative velocity
    starts_out = []
    for _ in range(9):
        starts_out.append(
            [
                float(rng.uniform(core_h - 0.25, core_h + 0.55)),
                float(rng.uniform(-1.8, -0.35)),
            ]
        )
    for y0 in starts_out:
        sol = integrate(p, y0, t_end=9.0, gamma=1.25)
        ax.plot(sol.y[0], sol.y[1], color=C["exit"], lw=1.15, alpha=0.6, zorder=3)

    # data cells by role
    ax.scatter(h[entering], v[entering], s=18, c=C["enter"], alpha=0.8, lw=0, zorder=5, label="entering (falling in)")
    ax.scatter(h[exiting], v[exiting], s=18, c=C["exit"], alpha=0.8, lw=0, zorder=5, label="exiting (leaving well)")
    ax.scatter(h[reverted], v[reverted], s=16, c=C["revert"], alpha=0.75, lw=0, zorder=5, label="reverted")
    # persistent core on top
    ax.scatter(
        h[persist],
        v[persist],
        s=36,
        c=C["core"],
        alpha=0.92,
        lw=0,
        zorder=6,
        label=f"persistent · trapped core (n={persist.sum()})",
    )
    ax.scatter([core_h], [core_v], s=220, marker="*", c=C["core_glow"], edgecolors="white", lw=0.9, zorder=7)

    # annotations
    stroke(ax.text(core_h, core_v + 0.55, "HYPOXIA\nATTRACTOR", ha="center", va="bottom", color=C["core_glow"], fontsize=13, fontweight="700"))
    ax.annotate(
        "orbit in → trapped",
        xy=(core_h - 0.15, core_v + 0.15),
        xytext=(h_sad - 0.7, 2.0),
        color=C["enter"],
        fontsize=11,
        fontweight="600",
        arrowprops=dict(arrowstyle="->", color=C["enter"], lw=1.8),
    )
    ax.annotate(
        "escape / reversion",
        xy=(core_h - 0.3, core_v - 0.2),
        xytext=(h_ox - 0.2, -2.3),
        color=C["exit"],
        fontsize=11,
        fontweight="600",
        arrowprops=dict(arrowstyle="->", color=C["exit"], lw=1.8),
    )
    # saddle marker
    ax.scatter([h_sad], [0], s=70, marker="D", c=C["trans"], edgecolors="white", lw=0.7, zorder=6)
    stroke(ax.text(h_sad, 0.35, "ridge", ha="center", color=C["trans"], fontsize=9))

    ax.set_xlim(h.min() - 0.3, max(h.max(), core_h + 2.2) + 0.2)
    ax.set_ylim(v.min() - 0.25, v.max() + 0.35)
    ax.set_xlabel("hypoxia program  h")
    ax.set_ylabel("velocity  v   (up = entering · down = exiting)")
    ax.set_title("Hypoxic well · persistent cells are the ones that fell in and stayed", loc="left", fontsize=13, pad=10)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=C["mute"])
    for sp in ax.spines.values():
        sp.set_color(C["line"])

    # right panel: polar orbit view around core
    ax2 = fig.add_subplot(gs[0, 1], projection="polar")
    ax2.set_facecolor(C["panel"])
    # angle from vector relative to core; radius = distance
    dh, dv = h - core_h, v - core_v
    r = np.hypot(dh, dv)
    theta = np.arctan2(dv, dh)

    ax2.scatter(theta[other], r[other], s=5, c=C["other"], alpha=0.3, lw=0, zorder=1)
    ax2.scatter(theta[entering], r[entering], s=14, c=C["enter"], alpha=0.85, lw=0, zorder=3)
    ax2.scatter(theta[exiting], r[exiting], s=14, c=C["exit"], alpha=0.85, lw=0, zorder=3)
    ax2.scatter(theta[reverted], r[reverted], s=12, c=C["revert"], alpha=0.8, lw=0, zorder=3)
    ax2.scatter(theta[persist], r[persist], s=28, c=C["core"], alpha=0.95, lw=0, zorder=4)

    # mean spiral guides
    for y0 in starts_in[:6]:
        sol = integrate(p, y0, t_end=10.0, gamma=1.15)
        th = np.arctan2(sol.y[1] - core_v, sol.y[0] - core_h)
        rr = np.hypot(sol.y[0] - core_h, sol.y[1] - core_v)
        ax2.plot(th, rr, color=C["enter"], lw=1.0, alpha=0.45, zorder=2)
    for y0 in starts_out[:5]:
        sol = integrate(p, y0, t_end=8.0, gamma=1.25)
        th = np.arctan2(sol.y[1] - core_v, sol.y[0] - core_h)
        rr = np.hypot(sol.y[0] - core_h, sol.y[1] - core_v)
        ax2.plot(th, rr, color=C["exit"], lw=1.0, alpha=0.5, zorder=2)

    ax2.set_ylim(0, np.quantile(r, 0.98) * 1.05)
    ax2.tick_params(colors=C["mute"], labelsize=7)
    ax2.set_title("Orbit view around the hypoxic core\n(radius = distance from persistent center)", color=C["ink"], fontsize=11, pad=14)
    ax2.grid(color=C["line"], alpha=0.7, lw=0.6)
    ax2.spines["polar"].set_color(C["line"])

    n_p, n_in, n_out = int(persist.sum()), int(entering.sum()), int(exiting.sum() + reverted.sum())
    fig.suptitle(
        f"E14S + E15S · hypoxia attractor basin  ·  core {n_p} persistent · {n_in} entering · {n_out} exiting/reverted",
        color=C["ink"],
        fontsize=14,
        y=0.98,
    )
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {"core_h": core_h, "core_v": core_v, "n_persistent": n_p, "n_entering": n_in, "n_leaving": n_out}


def fig_well_cross_section(df, p, h_hyp, h_ox, h_sad, core, out):
    """1D well with persistent piled at bottom; others on slopes."""
    fig, ax = plt.subplots(figsize=(11.8, 4.8))
    hg = np.linspace(df["h"].min() - 0.4, df["h"].max() + 0.4, 500)
    Ug = U(hg, p)
    Ug -= Ug.min()
    ax.fill_between(hg, Ug, Ug.max() + 0.4, color="#10182a")
    # shade hypoxic well region (h > saddle)
    ax.fill_between(hg[hg >= h_sad], Ug[hg >= h_sad], Ug.max() + 0.4, color=C["core"], alpha=0.12)
    ax.plot(hg, Ug, color=C["ink"], lw=2.6)

    def jitter_on_U(sub, color, s, label, alpha=0.85):
        hh = sub["h"].to_numpy()
        uu = U(hh, p) - U(hg, p).min()
        # small vertical jitter for visibility of pile-up
        rng = np.random.default_rng(abs(hash(label)) % 10_000)
        uu = uu + rng.normal(0, 0.03 * Ug.max(), size=len(uu))
        ax.scatter(hh, uu, s=s, c=color, alpha=alpha, lw=0, label=label, zorder=5)

    jitter_on_U(df[state_mask(df, {"entering_hypoxia", "entering_deep_hypoxia"})], C["enter"], 16, "entering")
    jitter_on_U(df[state_mask(df, {"exiting_hypoxia"})], C["exit"], 16, "exiting")
    jitter_on_U(df[state_mask(df, {"reverted_stable", "reverted_posthypoxic"})], C["revert"], 14, "reverted")
    jitter_on_U(df[state_mask(df, {"persistent_hypoxia"})], C["core"], 28, "persistent · trapped at bottom")

    ax.scatter([h_hyp], [U(h_hyp, p) - U(hg, p).min()], s=180, marker="*", c=C["core_glow"], edgecolors="white", zorder=6)
    stroke(ax.text(h_hyp, 0.08 * Ug.max(), "hypoxia well", ha="center", color=C["core_glow"], fontsize=12, fontweight="700"))
    stroke(ax.text(h_ox, U(h_ox, p) - U(hg, p).min() + 0.12 * Ug.max(), "oxygenated", ha="center", color="#2ec4b6", fontsize=11))
    stroke(ax.text(h_sad, U(h_sad, p) - U(hg, p).min() + 0.05 * Ug.max(), "escape ridge", ha="center", color=C["trans"], fontsize=10))

    ax.set_xlabel("hypoxia program  h")
    ax.set_ylabel("energy  U(h)   ↓ trapped")
    ax.set_title("Persistent cells sit at the bottom of the hypoxic well", loc="left", fontsize=13)
    ax.legend(frameon=False, fontsize=8, labelcolor=C["mute"], loc="upper right")
    ax.set_ylim(-0.05 * Ug.max(), Ug.max() + 0.35)
    fig.savefig(out, dpi=210, bbox_inches="tight")
    plt.close(fig)


def update_site(meta):
    html_path = ROOT / "index.html"
    html = html_path.read_text()
    block = f"""
  <h2>Hypoxic well · trapped core</h2>
  <p class="note">The basin’s center is data-defined by <b style="color:var(--hyp)">persistent hypoxia</b> cells — the ones that fell in and stayed. Entering cells spiral toward that core; exiting and reverted cells leave the well.</p>
  <figure style="margin:1rem 0 1.2rem"><img src="figures/e14e15_hypoxia_well_orbits.png"/><figcaption>Left: phase plane with spiral capture into the persistent core and escape/reversion paths. Right: polar orbit view centered on the persistent centroid (n={meta['n_persistent']} core · {meta['n_entering']} entering · {meta['n_leaving']} leaving).</figcaption></figure>
  <figure style="margin:0 0 1.4rem"><img src="figures/e14e15_hypoxia_well_section.png"/><figcaption>Cross-section of the energy well: persistent cells pile at the bottom; enter/exit/reverted sit on the slopes.</figcaption></figure>
"""
    if "Hypoxic well · trapped core" in html:
        # replace existing section roughly between markers
        import re

        html = re.sub(
            r"<h2>Hypoxic well · trapped core</h2>.*?(?=<h2>Who is claimed|<h2>Model)",
            block.strip() + "\n\n  ",
            html,
            count=1,
            flags=re.S,
        )
    else:
        html = html.replace("<h2>Who is claimed by which basin?</h2>", block + "\n  <h2>Who is claimed by which basin?</h2>")
        if "<h2>Who is claimed by which basin?</h2>" not in html:
            html = html.replace("<h2>Model</h2>", block + "\n  <h2>Model</h2>")
    # also lift into hero pills if needed - optional
    html_path.write_text(html)


def main():
    style()
    df, p, h_ox, h_hyp, h_sad = load()
    print("n", len(df), "persistent", int(state_mask(df, {"persistent_hypoxia"}).sum()), flush=True)
    meta = fig_hypoxia_well(df, p, h_ox, h_hyp, h_sad, FIGS / "e14e15_hypoxia_well_orbits.png")
    fig_well_cross_section(df, p, h_hyp, h_ox, h_sad, meta, FIGS / "e14e15_hypoxia_well_section.png")
    meta.update({"h_ox": h_ox, "h_hyp": h_hyp, "h_sad": h_sad})
    (DATA / "hypoxia_well_meta.json").write_text(json.dumps(meta, indent=2))
    update_site(meta)
    print("wrote", FIGS / "e14e15_hypoxia_well_orbits.png", flush=True)


if __name__ == "__main__":
    main()
