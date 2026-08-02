#!/usr/bin/env python3
"""Correct N↔H hypoxia transitions + 3D Waddington landscape (E14S+E15S).

Math fix:
  entering hypoxia  = oxygenated basin  →  hypoxic basin
  leaving / exit    = hypoxic basin     →  oxygenated basin
  persistent        = trapped at hypoxic minimum
  reverted/normoxic = in oxygenated minimum

3D surface:
  U(h, y) = U₁(h) + ½ κ y²
  with y a transverse coordinate (centered prolif), so the two wells
  are valleys connected by a saddle pass — not orbits around one well.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
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
    "ox": "#2ec4b6",
    "hyp": "#ff3b1f",
    "hyp_glow": "#ff7a45",
    "enter": "#4ea8ff",
    "exit": "#5fe0c0",
    "revert": "#b7f0d8",
    "ridge": "#ffd166",
    "persist": "#ff2d55",
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


def stroke(t, w=3.2):
    t.set_path_effects([pe.withStroke(linewidth=w, foreground=C["bg"])])
    return t


def U1(h, p):
    a, b, c, d, hm, lam = p[:6]
    return 0.25 * a * h**4 - 0.5 * b * h**2 + c * h + d * np.exp(-lam * (h - hm) ** 2)


def dU1(h, p):
    a, b, c, d, hm, lam = p[:6]
    return a * h**3 - b * h + c - 2 * lam * d * (h - hm) * np.exp(-lam * (h - hm) ** 2)


def U2(h, y, p, kappa):
    return U1(h, p) + 0.5 * kappa * y**2


def load():
    df = pd.read_csv(DATA / "e14e15_cells_with_membership.csv")
    summary = json.loads((DATA / "e14e15_summary.json").read_text())
    pd_ = summary["potential"]
    p = (pd_["a"], pd_["b"], pd_["c"], pd_["d"], pd_["hm"], pd_["lam"], pd_["gamma"])
    attrs = sorted([fp for fp in summary["fixed_points"] if fp["kind"] == "attractor"], key=lambda x: x["h"])
    h_ox, h_hyp = attrs[0]["h"], attrs[-1]["h"]
    sad = [fp for fp in summary["fixed_points"] if fp["kind"] == "saddle"]
    h_sad = sad[0]["h"] if sad else 0.5 * (h_ox + h_hyp)
    # transverse coord from prolif (centered)
    y = pd.to_numeric(df["prolif_score"], errors="coerce").to_numpy()
    y = np.nan_to_num(y, nan=np.nanmedian(y))
    y = (y - np.median(y)) / (np.std(y) + 1e-9)
    df = df.copy()
    df["y"] = np.clip(y, -2.8, 2.8)
    return df, p, h_ox, h_hyp, h_sad


def msk(df, names):
    return df["hypoxia_dynamics"].astype(str).isin(names)


def integrate_path(p, kappa, y0, t_end=12.0, gamma=None):
    """Overdamped-ish 2D flow on U(h,y) plus mild inertia on h via v."""
    g = 1.35 if gamma is None else gamma

    def rhs(_t, s):
        h, v, y = s
        # ḣ = v ; v̇ = -∂U/∂h - g v ; ẏ = -∂U/∂y = -κ y
        return [v, -dU1(h, p) - g * v, -kappa * y]

    return solve_ivp(rhs, (0, t_end), y0, rtol=1e-3, atol=1e-3, max_step=0.08)


def make_enter_exit_paths(p, kappa, h_ox, h_hyp, h_sad, n_each=9):
    rng = np.random.default_rng(21)
    enter, leave = [], []
    # N → H : start near oxygenated well with +v, small y
    for _ in range(n_each):
        y0 = [
            float(h_ox + rng.normal(0, 0.15)),
            float(rng.uniform(0.35, 1.4)),
            float(rng.normal(0, 0.55)),
        ]
        sol = integrate_path(p, kappa, y0, t_end=14.0, gamma=1.2)
        enter.append(sol.y)
    # H → N : start near hypoxic well with -v
    for _ in range(n_each):
        y0 = [
            float(h_hyp + rng.normal(0, 0.18)),
            float(rng.uniform(-1.5, -0.35)),
            float(rng.normal(0, 0.55)),
        ]
        sol = integrate_path(p, kappa, y0, t_end=14.0, gamma=1.25)
        leave.append(sol.y)
    return enter, leave


def fig_3d_landscape(df, p, h_ox, h_hyp, h_sad, kappa, enter, leave, out):
    fig = plt.figure(figsize=(13.2, 8.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(C["bg"])
    fig.patch.set_facecolor(C["bg"])

    # surface grid
    hg = np.linspace(min(df["h"].min(), h_ox) - 0.6, max(df["h"].max(), h_hyp) + 0.6, 90)
    yg = np.linspace(-2.6, 2.6, 70)
    HH, YY = np.meshgrid(hg, yg)
    ZZ = U2(HH, YY, p, kappa)
    ZZ = ZZ - ZZ.min()
    surf = ax.plot_surface(
        HH,
        YY,
        ZZ,
        cmap=mpl.colors.LinearSegmentedColormap.from_list(
            "wadd", ["#102030", "#1f4d4a", "#2ec4b6", "#ffd166", "#ff5a36", "#3a1010"]
        ),
        linewidth=0,
        antialiased=True,
        alpha=0.78,
        rstride=1,
        cstride=1,
    )
    surf.set_edgecolor("none")

    # valley floors (y≈0 curve)
    z_floor = U2(hg, np.zeros_like(hg), p, kappa) - U2(HH, YY, p, kappa).min()
    ax.plot(hg, np.zeros_like(hg), z_floor, color="white", lw=2.0, alpha=0.85, zorder=10)

    # cells on surface
    def scatter_state(sub, color, s, label, zoff=0.02):
        if len(sub) == 0:
            return
        hh = sub["h"].to_numpy()
        yy = sub["y"].to_numpy()
        zz = U2(hh, yy, p, kappa) - U2(HH, YY, p, kappa).min() + zoff
        ax.scatter(hh, yy, zz, c=color, s=s, alpha=0.85, depthshade=True, label=label, linewidths=0)

    persist = df[msk(df, {"persistent_hypoxia"})]
    entering = df[msk(df, {"entering_hypoxia", "entering_deep_hypoxia"})]
    exiting = df[msk(df, {"exiting_hypoxia"})]
    oxy = df[msk(df, {"normoxic_stable", "reverted_stable", "reverted_posthypoxic"})]

    scatter_state(oxy.sample(min(400, len(oxy)), random_state=0) if len(oxy) else oxy, C["ox"], 10, "oxygenated / reverted")
    scatter_state(entering, C["enter"], 14, "entering  (N → H)")
    scatter_state(exiting, C["exit"], 14, "exiting  (H → N)")
    scatter_state(persist, C["persist"], 28, "persistent · trapped in H")

    z0 = -U2(HH, YY, p, kappa).min()
    # mark wells
    for hx, name, col in [(h_ox, "O₂ well", C["ox"]), (h_hyp, "hypoxia well", C["hyp_glow"])]:
        zz = U2(hx, 0, p, kappa) + z0
        ax.scatter([hx], [0], [zz], s=160, c=col, marker="*", edgecolors="white", linewidths=0.7, zorder=20)
    zz_s = U2(h_sad, 0, p, kappa) + z0
    ax.scatter([h_sad], [0], [zz_s], s=70, c=C["ridge"], marker="D", edgecolors="white", linewidths=0.6, zorder=20)

    # paths on surface
    for Y in enter:
        h, v, y = Y
        z = U2(h, y, p, kappa) + z0 + 0.03
        ax.plot(h, y, z, color=C["enter"], lw=1.6, alpha=0.85)
    for Y in leave:
        h, v, y = Y
        z = U2(h, y, p, kappa) + z0 + 0.03
        ax.plot(h, y, z, color=C["exit"], lw=1.6, alpha=0.85)

    ax.set_xlabel("hypoxia program  h", labelpad=8)
    ax.set_ylabel("transverse  y (prolif)", labelpad=8)
    ax.set_zlabel("energy  U(h,y)", labelpad=8)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(C["line"])
    ax.yaxis.pane.set_edgecolor(C["line"])
    ax.zaxis.pane.set_edgecolor(C["line"])
    ax.tick_params(colors=C["mute"])
    ax.view_init(elev=24, azim=-58)
    ax.set_title(
        "3D Waddington landscape · entering = O₂→hypoxia · exiting = hypoxia→O₂",
        color=C["ink"],
        pad=12,
        fontsize=13,
    )
    leg = ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=C["mute"])
    fig.savefig(out, dpi=210, bbox_inches="tight")
    plt.close(fig)

    # alternate angle
    fig = plt.figure(figsize=(12.5, 7.6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(C["bg"])
    fig.patch.set_facecolor(C["bg"])
    ax.plot_surface(HH, YY, ZZ, cmap=surf.get_cmap(), linewidth=0, antialiased=True, alpha=0.8, rstride=1, cstride=1)
    ax.plot(hg, np.zeros_like(hg), z_floor, color="white", lw=2.0, alpha=0.9)
    scatter_state(persist, C["persist"], 26, "persistent in hypoxia well")
    scatter_state(oxy.sample(min(300, len(oxy)), random_state=1) if len(oxy) else oxy, C["ox"], 9, "oxygenated well")
    for Y in enter[:6]:
        h, v, y = Y
        ax.plot(h, y, U2(h, y, p, kappa) + z0 + 0.03, color=C["enter"], lw=2.0, alpha=0.9)
    for Y in leave[:6]:
        h, v, y = Y
        ax.plot(h, y, U2(h, y, p, kappa) + z0 + 0.03, color=C["exit"], lw=2.0, alpha=0.9)
    # big annotations via text
    ax.text(h_ox, 0, U2(h_ox, 0, p, kappa) + z0 + 0.4, "O₂", color=C["ox"], fontsize=12, fontweight="700")
    ax.text(h_hyp, 0, U2(h_hyp, 0, p, kappa) + z0 + 0.4, "H", color=C["hyp_glow"], fontsize=12, fontweight="700")
    ax.text(h_sad, 0, U2(h_sad, 0, p, kappa) + z0 + 0.55, "pass", color=C["ridge"], fontsize=10)
    ax.set_xlabel("h")
    ax.set_ylabel("y")
    ax.set_zlabel("U")
    ax.view_init(elev=18, azim=28)
    ax.set_title("Side view · blue paths enter hypoxia · teal paths return to O₂", color=C["ink"], fontsize=13)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor(C["line"])
    fig.savefig(str(out).replace(".png", "_side.png"), dpi=210, bbox_inches="tight")
    plt.close(fig)


def fig_corrected_2d(df, p, h_ox, h_hyp, h_sad, enter, leave, out):
    """Phase plane with N↔H paths only — no false 'orbit around hypoxia alone'."""
    fig, ax = plt.subplots(figsize=(12.2, 7.0))
    h, v = df["h"].to_numpy(), df["v"].to_numpy()

    # soft wells
    ax.axvspan(df["h"].min() - 0.4, h_sad, color=C["ox"], alpha=0.07, lw=0)
    ax.axvspan(h_sad, df["h"].max() + 0.4, color=C["hyp"], alpha=0.08, lw=0)

    other = ~msk(df, {"persistent_hypoxia", "entering_hypoxia", "entering_deep_hypoxia", "exiting_hypoxia", "reverted_stable", "reverted_posthypoxic", "normoxic_stable"})
    ax.scatter(h[other], v[other], s=6, c="#2a3348", alpha=0.35, lw=0, zorder=1)

    oxy = msk(df, {"normoxic_stable", "reverted_stable", "reverted_posthypoxic"})
    ent = msk(df, {"entering_hypoxia", "entering_deep_hypoxia"})
    exi = msk(df, {"exiting_hypoxia"})
    per = msk(df, {"persistent_hypoxia"})

    ax.scatter(h[oxy], v[oxy], s=14, c=C["ox"], alpha=0.75, lw=0, zorder=3, label="in O₂ basin (normoxic/reverted)")
    ax.scatter(h[ent], v[ent], s=18, c=C["enter"], alpha=0.85, lw=0, zorder=4, label="entering · O₂ → hypoxia")
    ax.scatter(h[exi], v[exi], s=18, c=C["exit"], alpha=0.85, lw=0, zorder=4, label="exiting · hypoxia → O₂")
    ax.scatter(h[per], v[per], s=32, c=C["persist"], alpha=0.95, lw=0, zorder=5, label="persistent · trapped in hypoxia")

    # N→H and H→N model paths (h,v only)
    for Y in enter:
        ax.plot(Y[0], Y[1], color=C["enter"], lw=1.4, alpha=0.7, zorder=2)
        ax.scatter(Y[0, -1], Y[1, -1], s=18, c=C["persist"], lw=0, zorder=5)
        ax.scatter(Y[0, 0], Y[1, 0], s=16, c=C["ox"], lw=0, zorder=5)
    for Y in leave:
        ax.plot(Y[0], Y[1], color=C["exit"], lw=1.4, alpha=0.75, zorder=2)
        ax.scatter(Y[0, -1], Y[1, -1], s=18, c=C["ox"], lw=0, zorder=5)
        ax.scatter(Y[0, 0], Y[1, 0], s=16, c=C["persist"], lw=0, zorder=5)

    ax.scatter([h_ox], [0], s=180, marker="*", c=C["ox"], edgecolors="white", zorder=6)
    ax.scatter([h_hyp], [0], s=180, marker="*", c=C["hyp_glow"], edgecolors="white", zorder=6)
    ax.scatter([h_sad], [0], s=80, marker="D", c=C["ridge"], edgecolors="white", zorder=6)
    stroke(ax.text(h_ox, 0.45, "O₂ attractor", ha="center", color=C["ox"], fontsize=12, fontweight="700"))
    stroke(ax.text(h_hyp, 0.45, "hypoxia attractor", ha="center", color=C["hyp_glow"], fontsize=12, fontweight="700"))
    stroke(ax.text(h_sad, -0.55, "saddle pass", ha="center", color=C["ridge"], fontsize=10))

    ax.annotate(
        "",
        xy=(h_hyp - 0.15, 0.9),
        xytext=(h_ox + 0.15, 0.9),
        arrowprops=dict(arrowstyle="->", color=C["enter"], lw=2.2),
    )
    stroke(ax.text(0.5 * (h_ox + h_hyp), 1.05, "ENTER hypoxia", ha="center", color=C["enter"], fontsize=11, fontweight="700"))
    ax.annotate(
        "",
        xy=(h_ox + 0.15, -0.95),
        xytext=(h_hyp - 0.15, -0.95),
        arrowprops=dict(arrowstyle="->", color=C["exit"], lw=2.2),
    )
    stroke(ax.text(0.5 * (h_ox + h_hyp), -1.2, "LEAVE → return to O₂", ha="center", color=C["exit"], fontsize=11, fontweight="700"))

    ax.axhline(0, color=C["line"], lw=0.8)
    ax.set_xlabel("hypoxia program  h")
    ax.set_ylabel("velocity  v")
    ax.set_title("Corrected geometry · two basins, traffic both ways over the pass", loc="left", fontsize=13)
    ax.legend(loc="upper right", frameon=False, fontsize=8, labelcolor=C["mute"])
    fig.savefig(out, dpi=210, bbox_inches="tight")
    plt.close(fig)


def fig_schematic_nh(out):
    fig, ax = plt.subplots(figsize=(10.8, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    # two wells
    xs = np.linspace(0.5, 9.5, 400)
    U = 0.015 * (xs - 5) ** 4 - 0.28 * (xs - 5) ** 2 + 2.15
    ax.fill_between(xs, U, 3.3, color="#121a2b")
    ax.fill_between(xs[xs < 5], U[xs < 5], 3.3, color=C["ox"], alpha=0.12)
    ax.fill_between(xs[xs >= 5], U[xs >= 5], 3.3, color=C["hyp"], alpha=0.12)
    ax.plot(xs, U, color="white", lw=2.4)
    ax.scatter([2.5, 7.5], [U[np.argmin(np.abs(xs - 2.5))], U[np.argmin(np.abs(xs - 7.5))]], s=170, c=[C["ox"], C["hyp"]], edgecolors="white", zorder=5)
    ax.text(2.5, 0.85, "Oxygenated\nattractor", ha="center", color=C["ox"], fontsize=12, fontweight="700")
    ax.text(7.5, 0.85, "Hypoxic\nattractor\n(persistent core)", ha="center", color=C["hyp_glow"], fontsize=12, fontweight="700")
    ax.text(5.0, 2.55, "saddle", ha="center", color=C["ridge"], fontsize=10)
    ax.annotate("entering", xy=(6.7, 1.55), xytext=(3.3, 1.55), color=C["enter"], fontsize=12, fontweight="700",
                arrowprops=dict(arrowstyle="->", color=C["enter"], lw=2.2))
    ax.annotate("leaving / reversion", xy=(3.3, 1.95), xytext=(6.7, 1.95), color=C["exit"], fontsize=12, fontweight="700",
                arrowprops=dict(arrowstyle="->", color=C["exit"], lw=2.2))
    ax.set_title("Right math: enter/leave are transitions between two attractors", loc="left", color=C["ink"], fontsize=13)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def update_site():
    html = (ROOT / "index.html").read_text()
    block = """
  <h2>Two basins · traffic both ways</h2>
  <p class="note"><b style="color:var(--ox)">Entering hypoxia</b> means leaving the oxygenated attractor and falling into the hypoxic well.
  <b style="color:#5fe0c0">Leaving / exiting</b> is the reverse trip back to O₂. Persistent cells are trapped at the hypoxic minimum.</p>
  <div class="grid">
    <figure><img src="figures/e14e15_two_basin_schematic.png"/><figcaption>Schematic: enter = O₂→H · leave = H→O₂.</figcaption></figure>
    <figure><img src="figures/e14e15_two_basin_phase.png"/><figcaption>Phase plane with N↔H model paths and data.</figcaption></figure>
  </div>
  <figure style="margin:1rem 0"><img src="figures/e14e15_waddington_3d.png"/><figcaption>3D Waddington surface U(h,y): valleys = attractors, ridge = saddle pass, blue paths enter hypoxia, teal paths return to O₂.</figcaption></figure>
  <figure style="margin:0 0 1.4rem"><img src="figures/e14e15_waddington_3d_side.png"/><figcaption>Side angle on the same landscape.</figcaption></figure>
"""
    if "Two basins · traffic both ways" in html:
        import re
        html = re.sub(
            r"<h2>Two basins · traffic both ways</h2>.*?(?=<h2>)",
            block.strip() + "\n\n  ",
            html,
            count=1,
            flags=re.S,
        )
    else:
        # insert near top after hypoxic attractor or start here
        if "<h2>The hypoxic attractor</h2>" in html:
            html = html.replace("<h2>The hypoxic attractor</h2>", block + "\n  <h2>The hypoxic attractor</h2>")
        else:
            html = html.replace("<h2>Start here</h2>", block + "\n  <h2>Start here</h2>")
    # soften old orbit wording if present
    html = html.replace(
        "Other cells orbit in (entering) or escape (exiting / reverted).",
        "Entering cells move O₂→hypoxia; exiting/reverted cells move hypoxia→O₂; persistents stay trapped in H.",
    )
    (ROOT / "index.html").write_text(html)


def main():
    style()
    df, p, h_ox, h_hyp, h_sad = load()
    kappa = 0.85
    print(f"wells O2={h_ox:.3f}  H={h_hyp:.3f}  saddle={h_sad:.3f}", flush=True)
    enter, leave = make_enter_exit_paths(p, kappa, h_ox, h_hyp, h_sad)
    fig_schematic_nh(FIGS / "e14e15_two_basin_schematic.png")
    fig_corrected_2d(df, p, h_ox, h_hyp, h_sad, enter, leave, FIGS / "e14e15_two_basin_phase.png")
    fig_3d_landscape(df, p, h_ox, h_hyp, h_sad, kappa, enter, leave, FIGS / "e14e15_waddington_3d.png")
    update_site()
    meta = {
        "geometry": "two_attractor_Waddington",
        "enter": "oxygenated -> hypoxic",
        "leave": "hypoxic -> oxygenated",
        "persistent": "trapped at hypoxic minimum",
        "U": "U(h,y)=U1(h)+0.5*kappa*y^2",
        "kappa": kappa,
        "h_ox": h_ox,
        "h_hyp": h_hyp,
        "h_sad": h_sad,
        "n_enter_paths": len(enter),
        "n_leave_paths": len(leave),
    }
    (DATA / "two_basin_geometry.json").write_text(json.dumps(meta, indent=2))
    print("wrote 3D + corrected 2D", flush=True)


if __name__ == "__main__":
    main()
