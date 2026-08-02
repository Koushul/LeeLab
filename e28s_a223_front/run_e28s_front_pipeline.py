#!/usr/bin/env python3
"""E28S A223 hypoxia-front analysis (MC38 E14/E15 analogue).

Design rules:
  - edge and core are separate physical chips — never overlaid across slides
  - within each slide, GFP− / GFP+ share the chip (OVERLAY = GFP±)
  - H≤8 on ADT microwell entropy (H_adt), MAP coords map_row_adt × map_col_adt
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
from scipy import sparse, stats
from sklearn.neighbors import NearestNeighbors

# Avoid umap→tensorflow crash under numpy2
_fake = types.ModuleType("umap.parametric_umap")
_fake.ParametricUMAP = None
_fake.load_ParametricUMAP = None
sys.modules.setdefault("umap.parametric_umap", _fake)

ROOT = Path("/ix1/ylee/shared/MC38_Hypoxia_001/e28s_a223_front")
DATA = ROOT / "data"
FIGS = ROOT / "front_figures"
ANIM = FIGS / "animations"
SITE = ROOT / "site"
for p in (DATA, FIGS, ANIM, SITE):
    p.mkdir(parents=True, exist_ok=True)

FINAL = Path(
    "/ix1/ylee/kor11/tools/af_tutorial/E28S_simpleaf/E28S_master_analysis/data/E28S_final.h5ad"
)
ENTROPY_META = Path(
    "/ix1/ylee/kor11/tools/spac_analysis/out/e28s_adt_entropy5_merci_spp1/cell_metadata_adtH5.csv"
)
REVERSION = Path(
    "/ix1/ylee/kor11/tools/af_tutorial/E28S_simpleaf/E28S_master_analysis/"
    "E28S_hypoxia_reversion_repro/cache/reversion_obs.csv"
)

H_CUT = 8.0
K = 30
N_PERM = 2000
RNG = np.random.default_rng(20260801)

HYPOXIA = [
    "Slc2a1", "Hk2", "Ldha", "Pgk1", "Bnip3", "Vegfa", "Egln3", "Eno1",
    "Pdk1", "Gapdh", "Aldoa", "Pfkl", "Ndrg1", "Fam162a", "P4ha1", "Higd1a",
]
PROLIF = ["Mki67", "Top2a", "Cdk1", "Ccnb1", "Stmn1", "Pcna", "Ube2c"]

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
    "GFP-": "#eb6f92",
    "GFP+": "#9ccfd8",
}


def style_ax(ax):
    ax.set_facecolor(RP["surface"])
    for sp in ax.spines.values():
        sp.set_color(RP["hl"])
    ax.tick_params(colors=RP["subtle"])
    ax.xaxis.label.set_color(RP["subtle"])
    ax.yaxis.label.set_color(RP["subtle"])
    ax.title.set_color(RP["text"])


def _as_csr(X):
    if sparse.issparse(X):
        return X.tocsr()
    return sparse.csr_matrix(X)


def jitter_xy(row_idx, col_idx, keys, radius=0.28):
    xy = np.column_stack([row_idx.astype(float), col_idx.astype(float)])
    groups = {}
    for i, key in enumerate(zip(row_idx.tolist(), col_idx.tolist())):
        groups.setdefault(key, []).append(i)
    for idxs in groups.values():
        n = len(idxs)
        if n <= 1:
            continue
        for rank, i in enumerate(idxs):
            ang = 2 * np.pi * rank / n
            h = abs(hash(str(keys[i]))) % 1000 / 1000.0
            r = radius * (0.55 + 0.45 * h)
            xy[i, 0] += r * np.cos(ang)
            xy[i, 1] += r * np.sin(ang)
    return xy


def score_module(adata, genes, key, layer="Ms"):
    present = [g for g in genes if g in adata.var_names]
    if len(present) < 2:
        adata.obs[key] = 0.0
        return present
    X = adata.layers[layer] if layer in adata.layers else adata.X
    X = _as_csr(X)
    idx = [adata.var_names.get_loc(g) for g in present]
    sub = np.asarray(X[:, idx].todense(), dtype=np.float64)
    mu, sd = sub.mean(0), sub.std(0)
    sd[sd < 1e-8] = 1.0
    adata.obs[key] = ((sub - mu) / sd).mean(1)
    return present


def hypoxia_velocity_delta(adata, genes, key="v_hypoxia"):
    present = [g for g in genes if g in adata.var_names]
    if "velocity" not in adata.layers or not present:
        adata.obs[key] = 0.0
        return
    V = _as_csr(adata.layers["velocity"])
    idx = []
    for g in present:
        j = adata.var_names.get_loc(g)
        col = V[:, j]
        if sparse.issparse(col) and col.nnz == 0:
            continue
        idx.append(j)
    if not idx:
        adata.obs[key] = 0.0
        return
    vsub = np.asarray(V[:, idx].todense(), dtype=np.float64)
    good = ~np.isnan(vsub).all(0)
    adata.obs[key] = np.nanmean(vsub[:, good], axis=1)


def classify_hypoxia_dynamics(adata, tumor_mask):
    s = adata.obs["hypoxia_score"].to_numpy(dtype=float)
    v = adata.obs["v_hypoxia"].to_numpy(dtype=float)
    s_t, v_t = s[tumor_mask], v[tumor_mask]
    s_hi = float(np.quantile(s_t, 0.65))
    s_lo = float(np.quantile(s_t, 0.35))
    v_pos = float(np.quantile(np.abs(v_t), 0.40))
    v_up = max(v_pos, float(np.quantile(v_t, 0.60)))
    v_dn = min(-v_pos, float(np.quantile(v_t, 0.40)))
    states = np.full(adata.n_obs, "non_tumor", dtype=object)
    for i in np.where(tumor_mask)[0]:
        si, vi = s[i], v[i]
        if si >= s_hi and vi >= v_up:
            st = "entering_deep_hypoxia"
        elif si >= s_hi and vi <= v_dn:
            st = "exiting_hypoxia"
        elif si >= s_hi and abs(vi) < v_pos:
            st = "persistent_hypoxia"
        elif si <= s_lo and vi >= v_up:
            st = "entering_hypoxia"
        elif si <= s_lo and vi <= v_dn:
            st = "reverted_stable"
        elif s_lo < si < s_hi and vi >= v_up:
            st = "entering_hypoxia"
        elif s_lo < si < s_hi and vi <= v_dn:
            st = "exiting_hypoxia"
        elif si <= s_lo and abs(vi) < v_pos:
            st = "normoxic_stable"
        else:
            st = "transitional"
        states[i] = st
    adata.obs["hypoxia_dynamics"] = pd.Categorical(states)
    return {"s_hi": s_hi, "s_lo": s_lo, "v_pos": v_pos, "v_up": v_up, "v_dn": v_dn}


def build_entropy_table(adata):
    meta = pd.read_csv(ENTROPY_META)
    # index align by barcode_core / barcodes
    meta["bc"] = meta["barcode_core"].astype(str) if "barcode_core" in meta.columns else meta["barcodes"].astype(str).str.split("-").str[0]
    adata.obs["bc"] = adata.obs["barcode_core"].astype(str) if "barcode_core" in adata.obs.columns else adata.obs_names.astype(str).str.split("-").str[0]
    m = meta.drop_duplicates("bc").set_index("bc")
    rows = []
    for name, o in adata.obs.iterrows():
        bc = str(o["bc"])
        if bc not in m.index:
            continue
        r = m.loc[bc]
        rows.append(
            {
                "barcode": name,
                "barcode_raw": bc,
                "sample": str(o["sample_id"]),
                "slide": str(o["slide"]),
                "gfp_status": str(o["gfp_status"]),
                "entropy_bits": float(r["H_adt"]),
                "entropy_row_bits": float(r.get("H_row_adt", np.nan)),
                "entropy_col_bits": float(r.get("H_col_adt", np.nan)),
                "map_row_idx": int(r["map_row_adt"]),
                "map_col_idx": int(r["map_col_adt"]),
                "map_prob": float(r.get("conf_adt", np.nan)),
                "adt_umi_total": float(r.get("umi_sbc", r.get("adt_counts", np.nan))),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "microwell_entropy_per_cell.csv", index=False)
    summary = {
        "n": int(len(df)),
        "h_cut": H_CUT,
        "frac_pass": float((df["entropy_bits"] <= H_CUT).mean()),
        "n_pass": int((df["entropy_bits"] <= H_CUT).sum()),
        "by_slide": df.groupby("slide")["entropy_bits"].agg(["count", "median"]).to_dict(),
        "pass_by_sample": df[df["entropy_bits"] <= H_CUT]["sample"].value_counts().to_dict(),
        "layout": "48x48 MAP from H_adt (4-BC wells aggregated to row/col ADT halves)",
        "chip_rule": "edge and core are separate chips; never overlay across slides",
    }
    (DATA / "microwell_entropy_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return df, summary


def run_velocity(adata):
    print("scVelo preprocess …")
    scv.settings.verbosity = 2
    # USA convention
    S = _as_csr(adata.layers["spliced"]) + _as_csr(adata.layers["ambiguous"])
    U = _as_csr(adata.layers["unspliced"])
    adata.layers["spliced"] = S
    adata.layers["unspliced"] = U
    adata.X = S.copy()

    retain = sorted(set(HYPOXIA + PROLIF))
    scv.pp.filter_and_normalize(adata, min_shared_counts=10, n_top_genes=3000, retain_genes=retain)
    if "X_pca" not in adata.obsm:
        sc.tl.pca(adata, n_comps=30)
    sc.pp.neighbors(adata, n_neighbors=30, use_rep="X_pca")
    scv.pp.moments(adata, n_pcs=None, n_neighbors=None)
    print("velocity stochastic …")
    scv.tl.velocity(adata, mode="stochastic")
    scv.tl.velocity_graph(adata, n_jobs=8, show_progress_bar=False)

    hyp_genes = score_module(adata, HYPOXIA, "hypoxia_score_ms", layer="Ms")
    score_module(adata, PROLIF, "prolif_score", layer="Ms")
    # prefer module_hypoxia from annotation if present
    if "module_hypoxia" in adata.obs.columns and adata.obs["module_hypoxia"].notna().any():
        adata.obs["hypoxia_score"] = pd.to_numeric(adata.obs["module_hypoxia"], errors="coerce")
    elif "hypoxia_module" in adata.obs.columns:
        adata.obs["hypoxia_score"] = pd.to_numeric(adata.obs["hypoxia_module"], errors="coerce")
    else:
        adata.obs["hypoxia_score"] = adata.obs["hypoxia_score_ms"]

    tumor = (
        adata.obs["lineage"].astype(str).eq("Tumor")
        | adata.obs["celltype"].astype(str).str.contains("Tumor", case=False, na=False)
    ).to_numpy()
    mu = adata.obs.loc[tumor, "hypoxia_score"].mean()
    sd = adata.obs.loc[tumor, "hypoxia_score"].std() or 1.0
    adata.obs["hypoxia_score"] = (adata.obs["hypoxia_score"] - mu) / sd

    hypoxia_velocity_delta(adata, hyp_genes, "v_hypoxia")
    vh = adata.obs["v_hypoxia"].to_numpy(dtype=float)
    mu_v, sd_v = vh[tumor].mean(), vh[tumor].std() or 1.0
    adata.obs["v_hypoxia_raw"] = vh
    adata.obs["v_hypoxia"] = (vh - mu_v) / sd_v

    scv.tl.velocity_pseudotime(adata)
    vpt = adata.obs["velocity_pseudotime"].to_numpy(dtype=float)
    mask = np.isfinite(vpt)
    corr = np.corrcoef(adata.obs["hypoxia_score"].to_numpy()[mask], vpt[mask])[0, 1]
    adata.obs["hypoxia_latent_time"] = (1.0 - vpt) if corr > 0 else vpt

    thr = classify_hypoxia_dynamics(adata, tumor)
    adata.obs["is_tumor"] = tumor
    print("thresholds", thr)
    print(adata.obs["hypoxia_dynamics"].value_counts())
    return thr, hyp_genes


def attach_h8_spatial(adata, ent):
    ent = ent[ent["entropy_bits"] <= H_CUT].copy()
    ent = ent.set_index("barcode")
    keep = adata.obs_names.intersection(ent.index)
    sub = adata[keep].copy()
    for c in ["entropy_bits", "map_prob", "map_row_idx", "map_col_idx", "adt_umi_total"]:
        if c in ent.columns:
            sub.obs[c] = ent.loc[sub.obs_names, c].to_numpy()
    xy = jitter_xy(
        sub.obs["map_row_idx"].to_numpy(),
        sub.obs["map_col_idx"].to_numpy(),
        sub.obs_names.to_numpy(),
    )
    sub.obsm["spatial"] = xy
    sub.obs["spatial_filter"] = f"H_adt<={H_CUT}"
    # analysis sample label within slide: GFP status
    sub.obs["gate"] = sub.obs["gfp_status"].astype(str)
    return sub


def subset_slide(adata, slide, gate=None):
    m = adata.obs["slide"].astype(str).eq(slide)
    if gate in ("GFP+", "GFP-"):
        m = m & adata.obs["gate"].astype(str).eq(gate)
        label = f"{slide}_{gate}"
    else:
        label = f"{slide}_OVERLAY"
    sub = adata[m]
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
    return {
        "enter_f": ef,
        "exit_f": xf,
        "persist_f": persist.astype(float)[idx].mean(1),
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


def permute_metrics(fields, tumor, n_perm=N_PERM):
    enter, exit_, deep = fields["enter"], fields["exit"], fields["deep"]
    knn = fields["idx"]
    front = fields["front"]
    obs = {
        "enr_deep_exit": enrichment(deep, exit_, knn),
        "enr_enter_exit": enrichment(enter, exit_, knn),
        "contact_deep_exit": contact_frac(deep, exit_, knn),
        "front_score": float(np.nanmean(front[tumor])) if tumor.sum() else np.nan,
    }
    # label shuffle among tumor
    t_idx = np.where(tumor)[0]
    labs_base = np.zeros(len(tumor), dtype=object)
    labs_base[enter] = "enter"
    labs_base[exit_] = "exit"
    labs_base[deep] = "deep"
    labs_base[fields["persist"]] = "persist"
    nulls = {k: [] for k in obs}
    for _ in range(n_perm):
        perm = labs_base.copy()
        perm[t_idx] = RNG.permutation(perm[t_idx])
        e = perm == "enter"
        x = perm == "exit"
        d = perm == "deep"
        nulls["enr_deep_exit"].append(enrichment(d, x, knn))
        nulls["enr_enter_exit"].append(enrichment(e, x, knn))
        nulls["contact_deep_exit"].append(contact_frac(d, x, knn))
        ef = e.astype(float)[knn].mean(1)
        xf = x.astype(float)[knn].mean(1)
        fr = np.sqrt(ef * xf)
        nulls["front_score"].append(float(np.nanmean(fr[tumor])))
    rows = []
    for k, v in obs.items():
        arr = np.asarray(nulls[k], float)
        arr = arr[np.isfinite(arr)]
        if not np.isfinite(v) or len(arr) == 0:
            rows.append({"metric": k, "observed": v, "null_mean": np.nan, "null_std": np.nan, "z": np.nan, "p_ge_obs": np.nan, "n_perm": n_perm})
            continue
        z = (v - arr.mean()) / (arr.std() or 1.0)
        p = float((arr >= v).mean())
        rows.append({"metric": k, "observed": v, "null_mean": float(arr.mean()), "null_std": float(arr.std()), "z": float(z), "p_ge_obs": p, "n_perm": n_perm})
    return obs, pd.DataFrame(rows)


def fig_chip_gates(adata, slide):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    fig.patch.set_facecolor(RP["base"])
    for ax, gate, title in [
        (axes[0], "GFP-", f"{slide} GFP−"),
        (axes[1], "GFP+", f"{slide} GFP+"),
        (axes[2], None, f"{slide} GFP± overlay"),
    ]:
        style_ax(ax)
        if gate is None:
            for g, col in [("GFP-", COLORS["GFP-"]), ("GFP+", COLORS["GFP+"])]:
                sub, xy, _ = subset_slide(adata, slide, g)
                ax.scatter(xy[:, 0], xy[:, 1], s=4, c=col, alpha=0.55, linewidths=0, label=g)
            ax.legend(fontsize=8, frameon=False, labelcolor=RP["subtle"])
        else:
            sub, xy, _ = subset_slide(adata, slide, gate)
            ax.scatter(xy[:, 0], xy[:, 1], s=4, c=COLORS[gate], alpha=0.7, linewidths=0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
    fig.suptitle(f"E28S A223 · {slide} chip · H≤{H_CUT} MAP", color=RP["text"], fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIGS / f"{slide}_chip_overlay.png", dpi=180, facecolor=RP["base"])
    plt.close(fig)


def fig_dynamics_bridges(adata, slide, gate=None):
    sub, xy, label = subset_slide(adata, slide, gate)
    labs = sub.obs["hypoxia_dynamics"].astype(str).to_numpy()
    tumor = sub.obs["is_tumor"].to_numpy()
    fields = local_fields(xy, labs, tumor)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4))
    fig.patch.set_facecolor(RP["base"])
    ax = axes[0]
    style_ax(ax)
    ax.scatter(xy[~tumor, 0], xy[~tumor, 1], s=2, c=RP["hl"], alpha=0.35, linewidths=0)
    for st in FRONT_STATES:
        m = (labs == st) & tumor
        if m.sum() == 0:
            continue
        ax.scatter(xy[m, 0], xy[m, 1], s=12, c=COLORS[st], alpha=0.9, linewidths=0, label=st.replace("_", " "))
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{label} · dynamics")
    leg = ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    for t in leg.get_texts():
        t.set_color(RP["subtle"])

    ax = axes[1]
    style_ax(ax)
    enter_m, exit_m = fields["enter"], fields["exit"]
    ax.scatter(xy[~tumor, 0], xy[~tumor, 1], s=2, c=RP["hl"], alpha=0.3, linewidths=0)
    ax.scatter(xy[tumor, 0], xy[tumor, 1], s=3, c=RP["muted"], alpha=0.25, linewidths=0)
    if enter_m.sum() >= 2 and exit_m.sum() >= 2:
        nn = NearestNeighbors(n_neighbors=min(8, int(exit_m.sum()))).fit(xy[exit_m])
        d, idx = nn.kneighbors(xy[enter_m])
        dmed = float(np.median(NearestNeighbors(n_neighbors=2).fit(xy).kneighbors(xy)[0][:, 1]))
        max_d = dmed * 4
        tgt, src = xy[exit_m], xy[enter_m]
        for i in range(len(src)):
            for j, dist in zip(idx[i], d[i]):
                if dist <= max_d:
                    ax.plot([src[i, 0], tgt[j, 0]], [src[i, 1], tgt[j, 1]], color=RP["gold"], lw=0.45, alpha=0.5, zorder=2)
                    break
    ax.scatter(xy[exit_m, 0], xy[exit_m, 1], s=14, c=COLORS["exiting_hypoxia"], linewidths=0, zorder=3)
    ax.scatter(xy[enter_m, 0], xy[enter_m, 1], s=14, c=COLORS["entering_hypoxia"], linewidths=0, zorder=4)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{label} · enter↔exit bridges")
    fig.tight_layout()
    fig.savefig(FIGS / f"{label}_front_bridges.png", dpi=180, facecolor=RP["base"])
    plt.close(fig)

    # ribbons
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.4))
    fig.patch.set_facecolor(RP["base"])
    for ax, key, cmap, title in [
        (axes[0], "front", "YlOrRd", "front score √(enter×exit)"),
        (axes[1], "polarity", "RdBu_r", "polarity exit−enter"),
        (axes[2], "persist_f", "Purples", "persistent density"),
    ]:
        style_ax(ax)
        vals = fields[key]
        sca = ax.scatter(xy[:, 0], xy[:, 1], c=vals, s=6, cmap=cmap, linewidths=0, alpha=0.9)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        cb = fig.colorbar(sca, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.yaxis.set_tick_params(color=RP["subtle"])
        plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=RP["subtle"])
    fig.suptitle(f"{label} · H≤{H_CUT} front fields", color=RP["text"], fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIGS / f"{label}_front_ribbons.png", dpi=180, facecolor=RP["base"])
    plt.close(fig)
    return fields


def fig_enrichment_bars(stats_df):
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    fig.patch.set_facecolor(RP["base"])
    style_ax(ax)
    plot = stats_df[stats_df["metric"].isin(["enr_deep_exit", "enr_enter_exit"])].copy()
    if plot.empty:
        return
    x = np.arange(len(plot))
    ax.bar(x - 0.18, plot["null_mean"], width=0.36, color=RP["hl"], label="null")
    ax.bar(x + 0.18, plot["observed"], width=0.36, color=RP["gold"], label="observed")
    for i, r in enumerate(plot.itertuples()):
        ax.text(i, max(r.observed, r.null_mean) + 0.05, f"p={r.p_ge_obs:.3g}", ha="center", fontsize=7, color=RP["subtle"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.label}\n{r.metric}" for r in plot.itertuples()], fontsize=7)
    ax.axhline(1, color=RP["muted"], ls="--", lw=0.8)
    ax.set_ylabel("enrichment")
    ax.set_title("Deep/enter → exit adjacency · tumor-label shuffle")
    ax.legend(frameon=False, labelcolor=RP["subtle"])
    fig.tight_layout()
    fig.savefig(FIGS / "front_enrichment_bars.png", dpi=180, facecolor=RP["base"])
    plt.close(fig)


def fig_shuffle_summary(stats_df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor(RP["base"])
    for ax, metric, title in [
        (axes[0], "enr_deep_exit", "deep→exit enrichment"),
        (axes[1], "front_score", "mean front score"),
    ]:
        style_ax(ax)
        sub = stats_df[stats_df["metric"] == metric]
        xs = np.arange(len(sub))
        ax.bar(xs - 0.18, sub["null_mean"], width=0.36, color=RP["hl"], label="null")
        ax.bar(xs + 0.18, sub["observed"], width=0.36, color=RP["foam"], label="observed")
        for i, r in enumerate(sub.itertuples()):
            ax.text(i, max(r.observed, r.null_mean) * 1.02, f"p={r.p_ge_obs:.3g}\nz={r.z:.2f}", ha="center", fontsize=7, color=RP["subtle"])
        ax.set_xticks(xs)
        ax.set_xticklabels(sub["label"], rotation=15, ha="right", fontsize=8)
        ax.set_title(title)
        ax.legend(frameon=False, labelcolor=RP["subtle"], fontsize=8)
    fig.suptitle("E28S A223 · H≤8 tumor-label shuffle", color=RP["text"], fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIGS / "coord_shuffle_summary.png", dpi=180, facecolor=RP["base"])
    plt.close(fig)


def fig_persist_vs_front(adata, slide):
    sub, xy, label = subset_slide(adata, slide, None)
    labs = sub.obs["hypoxia_dynamics"].astype(str).to_numpy()
    tumor = sub.obs["is_tumor"].to_numpy()
    fields = local_fields(xy, labs, tumor)
    front = fields["front"]
    thr = np.nanquantile(front[tumor], 0.80) if tumor.sum() else np.nan
    rows = []
    for st in ["entering_hypoxia", "entering_deep_hypoxia", "persistent_hypoxia", "exiting_hypoxia"]:
        m = (labs == st) & tumor
        if m.sum() < 3:
            continue
        rows.append({"state": st, "n": int(m.sum()), "frac_top20_front": float((front[m] >= thr).mean()), "mean_front": float(front[m].mean())})
    df = pd.DataFrame(rows)
    df.to_csv(FIGS / f"{slide}_persistent_vs_front_stats.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.patch.set_facecolor(RP["base"])
    style_ax(ax)
    if len(df):
        ax.bar(df["state"].str.replace("_", "\n"), df["frac_top20_front"], color=[COLORS.get(s, RP["hl"]) for s in df["state"]])
    ax.set_ylabel("fraction on top-20% front ribbon")
    ax.set_title(f"{slide} OVERLAY · persist vs front")
    fig.tight_layout()
    fig.savefig(FIGS / f"{slide}_persistent_vs_front.png", dpi=180, facecolor=RP["base"])
    plt.close(fig)
    return df


def make_simple_animations(adata, slide, gate=None, n_frames=16):
    """Lightweight pulse GIF on front score field."""
    try:
        from PIL import Image
    except ImportError:
        print("PIL missing; skip animations")
        return []
    sub, xy, label = subset_slide(adata, slide, gate)
    labs = sub.obs["hypoxia_dynamics"].astype(str).to_numpy()
    tumor = sub.obs["is_tumor"].to_numpy()
    fields = local_fields(xy, labs, tumor)
    front = fields["front"]
    frames = []
    for fi in range(n_frames):
        phase = 0.55 + 0.45 * np.sin(2 * np.pi * fi / n_frames)
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        fig.patch.set_facecolor(RP["base"])
        style_ax(ax)
        sizes = 4 + 18 * front * phase
        ax.scatter(xy[:, 0], xy[:, 1], c=front, s=sizes, cmap="YlOrRd", linewidths=0, alpha=0.85)
        for st, col in [("exiting_hypoxia", COLORS["exiting_hypoxia"]), ("entering_hypoxia", COLORS["entering_hypoxia"]), ("entering_deep_hypoxia", COLORS["entering_deep_hypoxia"]), ("persistent_hypoxia", COLORS["persistent_hypoxia"])]:
            m = (labs == st) & tumor
            if m.any():
                ax.scatter(xy[m, 0], xy[m, 1], s=10, c=col, linewidths=0, alpha=0.9)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{label} · H≤8 front pulse", color=RP["text"], fontsize=11)
        fig.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(buf[:, :, :3]))
        plt.close(fig)
    out = ANIM / f"{label}_front_pulse.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=90, loop=0, optimize=True)
    return [out.name]


def write_site(summary, stats_df, entropy_summary):
    # gather images
    imgs = sorted(p.name for p in FIGS.glob("*.png"))
    gifs = sorted(p.name for p in ANIM.glob("*.gif"))
    claims = summary.get("headline", {})
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>E28S A223 · Hypoxia front · H≤8</title>
<style>
:root{{--base:#191724;--surf:#1f1d2e;--text:#e0def4;--muted:#908caa;--gold:#f6c177;--foam:#9ccfd8;--love:#eb6f92;--iris:#c4a7e7}}
body{{margin:0;font-family:ui-sans-serif,system-ui;background:var(--base);color:var(--text)}}
header{{padding:1.4rem 1.6rem;border-bottom:1px solid #403d52}}
h1{{margin:0;font-size:1.35rem;color:var(--gold)}}
.sub{{color:var(--muted);margin-top:.35rem;font-size:.92rem}}
main{{padding:1.2rem 1.6rem 3rem;max-width:1100px;margin:0 auto}}
.claim{{background:var(--surf);padding:1rem 1.1rem;border-radius:10px;margin:1rem 0;line-height:1.45}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem}}
figure{{margin:0;background:var(--surf);padding:.6rem;border-radius:10px}}
img{{width:100%;height:auto;border-radius:6px;display:block}}
figcaption{{color:var(--muted);font-size:.82rem;margin-top:.4rem}}
.stat{{background:var(--surf);padding:.8rem;border-radius:10px}}
.stat .n{{font-size:1.4rem;color:var(--foam);font-weight:600}}
table{{width:100%;border-collapse:collapse;font-size:.85rem}}
td,th{{border-bottom:1px solid #403d52;padding:.4rem .3rem;text-align:left}}
.sig{{color:var(--gold)}} .ns{{color:var(--muted)}}
</style></head><body>
<header>
  <h1>E28S A223 · Hypoxia front</h1>
  <div class="sub">Same geometry tests as MC38 E14/E15 · per-slide GFP± overlay · H≤8 MAP · edge ≠ core chips</div>
</header>
<main>
  <div class="claim"><strong>Design.</strong> Edge and core are separate physical chips — never overlaid.
  Within each slide, GFP− and GFP+ share the array (analogue of E14S/E15S ImageIT gates).
  ADT microwell entropy filter <code>H_adt ≤ 8</code> ({entropy_summary.get('frac_pass',0)*100:.1f}% retained).
  Dynamics from scVelo hypoxia phase plane (enter / persist / exit).</div>
  <div class="grid">
    <div class="stat"><div class="n">{summary.get('n_h8', '?')}</div><div>Cells after H≤8</div></div>
    <div class="stat"><div class="n">{summary.get('n_tumor_h8', '?')}</div><div>Tumor cells after H≤8</div></div>
    <div class="stat"><div class="n">{claims.get('edge_enr', '—')}</div><div>Edge OVERLAY deep→exit enr</div></div>
    <div class="stat"><div class="n">{claims.get('core_enr', '—')}</div><div>Core OVERLAY deep→exit enr</div></div>
  </div>
  <h2>Chip placement</h2>
  <div class="grid">
    <figure><img src="../front_figures/edge_chip_overlay.png"/><figcaption>Edge chip · GFP±</figcaption></figure>
    <figure><img src="../front_figures/core_chip_overlay.png"/><figcaption>Core chip · GFP±</figcaption></figure>
  </div>
  <h2>Front bridges & ribbons</h2>
  <div class="grid">
"""
    for name in imgs:
        if "front_bridges" in name or "front_ribbons" in name or "persistent_vs_front" in name:
            html += f'<figure><img src="../front_figures/{name}"/><figcaption>{name}</figcaption></figure>\n'
    html += """</div>
  <h2>Shuffle tests</h2>
  <div class="grid">
    <figure><img src="../front_figures/front_enrichment_bars.png"/><figcaption>Enrichment vs null</figcaption></figure>
    <figure><img src="../front_figures/coord_shuffle_summary.png"/><figcaption>Shuffle summary</figcaption></figure>
  </div>
  <h2>Animations</h2>
  <div class="grid">
"""
    for g in gifs:
        html += f'<figure><img src="../front_figures/animations/{g}"/><figcaption>{g}</figcaption></figure>\n'
    html += """</div>
  <h2>Statistics</h2>
  <table><tr><th>label</th><th>metric</th><th>obs</th><th>null</th><th>z</th><th>p</th></tr>
"""
    for r in stats_df.itertuples():
        cls = "sig" if (isinstance(r.p_ge_obs, float) and r.p_ge_obs < 0.05) else "ns"
        html += f"<tr><td>{r.label}</td><td>{r.metric}</td><td>{r.observed:.3g}</td><td>{r.null_mean:.3g}</td><td>{r.z:.2f}</td><td class='{cls}'>{r.p_ge_obs:.3g}</td></tr>\n"
    html += """</table>
</main></body></html>"""
    (SITE / "index.html").write_text(html)
    # also a self-contained copy with relative assets by copying? site uses ../front_figures
    # publish root = e28s_a223_front with index at site/ — better write root index that embeds
    root_html = html.replace("../front_figures/", "front_figures/")
    (ROOT / "index.html").write_text(root_html)


def main():
    print("loading E28S_final …")
    adata = sc.read_h5ad(FINAL)
    # ensure gene symbols
    if adata.var_names.astype(str).str.startswith("ENSMUSG").mean() > 0.5:
        raise SystemExit("E28S_final should already use gene symbols")

    print("entropy table …")
    ent, ent_sum = build_entropy_table(adata)
    print(ent_sum)

    print("velocity + dynamics …")
    thr, hyp_genes = run_velocity(adata)

    # dynamics CSV
    dyn = adata.obs[
        [
            c
            for c in [
                "sample_id",
                "slide",
                "gfp_status",
                "celltype",
                "lineage",
                "hypoxia_dynamics",
                "hypoxia_score",
                "v_hypoxia",
                "prolif_score",
                "hypoxia_latent_time",
                "velocity_pseudotime",
                "is_tumor",
            ]
            if c in adata.obs.columns
        ]
    ].copy()
    dyn.insert(0, "barcode", adata.obs_names.astype(str))
    dyn.to_csv(DATA / "hypoxia_dynamics_per_cell.csv", index=False)

    print("H≤8 spatial …")
    h8 = attach_h8_spatial(adata, ent)
    h8.write_h5ad(DATA / "e28s_h8_front.h5ad")
    print("H8 n", h8.n_obs, h8.obs["slide"].value_counts().to_dict())
    print("tumor H8", int(h8.obs["is_tumor"].sum()))

    all_stats = []
    for slide in ["edge", "core"]:
        fig_chip_gates(h8, slide)
        for gate in ["GFP-", "GFP+", None]:
            fields = fig_dynamics_bridges(h8, slide, gate)
            sub, xy, label = subset_slide(h8, slide, gate)
            labs = sub.obs["hypoxia_dynamics"].astype(str).to_numpy()
            tumor = sub.obs["is_tumor"].to_numpy()
            fields = local_fields(xy, labs, tumor)
            obs, sdf = permute_metrics(fields, tumor)
            sdf["label"] = label
            sdf["slide"] = slide
            sdf["gate"] = gate or "OVERLAY"
            all_stats.append(sdf)
            print(label, obs)
            make_simple_animations(h8, slide, gate)
        fig_persist_vs_front(h8, slide)

    stats_df = pd.concat(all_stats, ignore_index=True)
    stats_df.to_csv(FIGS / "coord_shuffle_front_test.csv", index=False)
    fig_enrichment_bars(stats_df)
    fig_shuffle_summary(stats_df)

    def headline(slide):
        row = stats_df[(stats_df["slide"] == slide) & (stats_df["gate"] == "OVERLAY") & (stats_df["metric"] == "enr_deep_exit")]
        if len(row) == 0 or not np.isfinite(row.iloc[0]["observed"]):
            return "n/a"
        r = row.iloc[0]
        return f"{r.observed:.2f}× (p={r.p_ge_obs:.3g})"

    summary = {
        "experiment": "E28S A223 hypoxia",
        "filter": f"H_adt<={H_CUT}",
        "n_h8": int(h8.n_obs),
        "n_tumor_h8": int(h8.obs["is_tumor"].sum()),
        "n_by_sample": h8.obs["sample_id"].value_counts().to_dict(),
        "velocity_thresholds": thr,
        "hypoxia_genes": hyp_genes,
        "chip_rule": "edge and core are separate chips; GFP± overlay only within slide",
        "headline": {"edge_enr": headline("edge"), "core_enr": headline("core")},
        "dynamics_counts": h8.obs.loc[h8.obs["is_tumor"], "hypoxia_dynamics"].astype(str).value_counts().to_dict(),
    }
    (DATA / "front_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    (FIGS / "animations" / "manifest.json").write_text(
        json.dumps(
            {
                "spatial": "H_adt<=8 MAP microwells",
                "chip_rule": summary["chip_rule"],
                "animations": [{"file": p.name, "label": p.stem} for p in sorted(ANIM.glob("*.gif"))],
            },
            indent=2,
        )
    )
    write_site(summary, stats_df, ent_sum)
    print("DONE", json.dumps(summary["headline"], indent=2))
    print("site", ROOT / "index.html")


if __name__ == "__main__":
    main()
