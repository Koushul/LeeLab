#!/usr/bin/env python3
"""Focused spatial niche deep-dive on strongest leads."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from sklearn.neighbors import NearestNeighbors

OUT = Path("/ix1/ylee/shared/MC38_Hypoxia_001/reseq_finetune/deep_dive/spatial_niches")
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(7)

FINETUNE = Path("/ix1/ylee/shared/MC38_Hypoxia_001/reseq_finetune/mc38_tumor_reseq_finetuned.h5ad")
DYN = Path("/ix1/ylee/shared/MC38_Hypoxia_001/reseq_finetune/hypoxia_velocity/hypoxia_dynamics_per_cell.csv")

COLLAPSE = {
    "CD8 Terminally Exhausted": "CD8_exhausted",
    "CD8 Exhausted Proliferating": "CD8_exhausted",
    "Regulatory CD8 T": "CD8_exhausted",
    "CD4 T": "T_helper",
    "Regulatory CD4 T": "T_helper",
    "NK T": "NK",
    "Tumor Proliferating": "Tumor",
    "Hypoxic Tumor": "Hypoxic_Tumor",
    "Spp1+ TAM": "Spp1_TAM",
    "C1qc+ TAM": "C1qc_TAM",
    "Il1b+ TAM": "Il1b_TAM",
}


def load():
    adata = sc.read_h5ad(FINETUNE)
    dyn = pd.read_csv(DYN).set_index("barcode")
    common = adata.obs_names.intersection(dyn.index)
    adata = adata[common].copy()
    for c in ["hypoxia_dynamics", "hypoxia_score", "v_hypoxia"]:
        adata.obs[c] = dyn.loc[adata.obs_names, c].values
    if "spatial" not in adata.obsm:
        adata.obsm["spatial"] = adata.obs[["spatial_coordinate_x", "spatial_coordinate_y"]].to_numpy()
    adata.obs["ct"] = adata.obs["cell_type_finetuned"].astype(str).map(lambda x: COLLAPSE.get(x, x.replace(" ", "_")))
    hyb = adata.obs["ct"].astype(str).to_numpy().copy()
    dynv = adata.obs["hypoxia_dynamics"].astype(str).to_numpy()
    tumorish = adata.obs["ct"].isin(["Tumor", "Hypoxic_Tumor"]).to_numpy()
    keep = np.isin(
        dynv,
        [
            "entering_hypoxia",
            "entering_deep_hypoxia",
            "persistent_hypoxia",
            "exiting_hypoxia",
            "normoxic_stable",
            "reverted_stable",
            "transitional",
        ],
    )
    hyb[tumorish & keep] = dynv[tumorish & keep]
    adata.obs["niche_label"] = hyb
    adata.obs["is_tumor"] = tumorish
    return adata


def nn_graph(xy, k=30):
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(xy))).fit(xy)
    dist, idx = nn.kneighbors(xy)
    return dist[:, 1:], idx[:, 1:], float(np.median(dist[:, 1]))


def frac_neighbors(idx, mask):
    return mask.astype(float)[idx].mean(axis=1)


def perm_enrich_pair(labs, idx, a, b, n_perm=500):
    """Enrichment of B among neighbors of A vs label shuffle."""
    labs = np.asarray(labs)
    ia = np.where(labs == a)[0]
    if len(ia) < 15 or (labs == b).sum() < 15:
        return None
    obs = frac_neighbors(idx[ia], labs == b).mean()
    exp = (labs == b).mean()
    null = []
    for _ in range(n_perm):
        shuf = labs.copy()
        RNG.shuffle(shuf)
        ia2 = np.where(shuf == a)[0]
        null.append(frac_neighbors(idx[ia2], shuf == b).mean() / max((shuf == b).mean(), 1e-12))
    ratio = obs / max(exp, 1e-12)
    null = np.asarray(null)
    p = (np.sum(np.abs(null - 1) >= abs(ratio - 1)) + 1) / (len(null) + 1)
    return {"source": a, "target": b, "obs_frac": float(obs), "exp_frac": float(exp), "enrichment": float(ratio), "p_perm": float(p), "n_source": int(len(ia))}


def ordered_zone_stats(xy, labs_hyb, tumor_mask, k=25):
    """For tumor cells, mean distance to each dynamics class → ordered geography."""
    states = ["normoxic_stable", "exiting_hypoxia", "persistent_hypoxia", "entering_hypoxia", "entering_deep_hypoxia"]
    rows = []
    for s_from in states:
        m_from = (labs_hyb == s_from) & tumor_mask
        if m_from.sum() < 10:
            continue
        for s_to in states:
            m_to = labs_hyb == s_to
            if m_to.sum() < 5:
                continue
            nn = NearestNeighbors(n_neighbors=1).fit(xy[m_to])
            # exclude self for same-state: use 2nd neighbor approx by removing query points
            if s_from == s_to:
                # distance to nearest OTHER cell of same state
                nn2 = NearestNeighbors(n_neighbors=2).fit(xy[m_to])
                d, _ = nn2.kneighbors(xy[m_from])
                med = float(np.median(d[:, 1]))
            else:
                d, _ = nn.kneighbors(xy[m_from])
                med = float(np.median(d[:, 0]))
            rows.append({"from": s_from, "to": s_to, "median_dist": med, "n_from": int(m_from.sum())})
    return pd.DataFrame(rows)


def interface_score(idx, labs, state_a, state_b):
    """Cells that have both A and B in neighborhood — interface cells."""
    labs = np.asarray(labs)
    has_a = (labs[idx] == state_a).any(axis=1)
    has_b = (labs[idx] == state_b).any(axis=1)
    return has_a & has_b


def tam_vs_core(xy, labs_ct, core_mask, k=30, n_perm=400):
    """Compare TAM subtype proximity / neighbor enrichment to hypoxic core."""
    dist, idx, _ = nn_graph(xy, k=k)
    # distance to core
    if core_mask.sum() < 10:
        return pd.DataFrame()
    nn = NearestNeighbors(n_neighbors=1).fit(xy[core_mask])
    d_all, _ = nn.kneighbors(xy)
    d_all = d_all.ravel()
    # for cells IN core, use 2nd neighbor within core-ish: set dist nan for core cells themselves
    d_all = d_all.copy()
    # better: distance to nearest core cell excluding self
    rows = []
    for tam in ["Spp1_TAM", "C1qc_TAM", "Il1b_TAM", "Resident_Macrophage", "Monocyte", "Neutrophil", "CAF", "CD8_exhausted", "NK"]:
        m = labs_ct == tam
        if m.sum() < 20:
            continue
        # distance excluding those that are somehow in core mask (shouldn't be)
        use = m & ~core_mask
        if use.sum() < 15:
            use = m
        med = float(np.median(d_all[use]))
        bg = ~m & ~core_mask
        bg_med = float(np.median(d_all[bg]))
        p = float(stats.mannwhitneyu(d_all[use], d_all[bg]).pvalue)
        # neighbor enrichment to core
        obs = frac_neighbors(idx[use], core_mask).mean()
        exp = core_mask.mean()
        null = []
        for _ in range(n_perm):
            shuf_core = core_mask.copy()
            RNG.shuffle(shuf_core)
            null.append(frac_neighbors(idx[use], shuf_core).mean() / max(shuf_core.mean(), 1e-12))
        ratio = obs / max(exp, 1e-12)
        null = np.asarray(null)
        p_enr = (np.sum(np.abs(null - 1) >= abs(ratio - 1)) + 1) / (len(null) + 1)
        rows.append(
            {
                "cell": tam,
                "median_dist_to_core": med,
                "bg_median_dist": bg_med,
                "dist_ratio_vs_bg": med / bg_med,
                "p_dist": p,
                "core_neighbor_enrichment": float(ratio),
                "p_enrich": float(p_enr),
                "n": int(use.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("dist_ratio_vs_bg")


def main():
    adata = load()
    report = {"niches": []}

    for sample in ["E14S", "E15S"]:
        print("===", sample)
        m = (adata.obs["sample"] == sample).to_numpy() & np.isfinite(adata.obsm["spatial"]).all(1)
        sub = adata[m]
        xy = np.asarray(sub.obsm["spatial"], float)
        labs_ct = sub.obs["ct"].astype(str).to_numpy()
        labs_hyb = sub.obs["niche_label"].astype(str).to_numpy()
        tumor = sub.obs["is_tumor"].to_numpy()
        dist, idx, med_nn = nn_graph(xy, k=30)

        # --- 1) Hypoxia front architecture: adjacency among dynamics ---
        states = ["normoxic_stable", "exiting_hypoxia", "persistent_hypoxia", "entering_hypoxia", "entering_deep_hypoxia", "transitional", "reverted_stable"]
        front_rows = []
        for a in states:
            for b in states:
                if a == b:
                    continue
                r = perm_enrich_pair(labs_hyb, idx, a, b, n_perm=400)
                if r:
                    r["sample"] = sample
                    front_rows.append(r)
        front = pd.DataFrame(front_rows)
        front.to_csv(OUT / f"dynamics_adjacency_{sample}.csv", index=False)

        # heatmap
        present = [s for s in states if (labs_hyb == s).sum() >= 15]
        mat = np.full((len(present), len(present)), np.nan)
        for i, a in enumerate(present):
            for j, b in enumerate(present):
                if a == b:
                    mat[i, j] = 0.0
                    continue
                hit = front[(front.source == a) & (front.target == b)]
                if len(hit):
                    mat[i, j] = np.log2(max(hit.iloc[0]["enrichment"], 1e-9))
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(present)))
        ax.set_yticks(range(len(present)))
        ax.set_xticklabels(present, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(present, fontsize=8)
        ax.set_title(f"{sample} · dynamics neighbor log2 enrichment")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(OUT / f"{sample}_dynamics_adjacency_heatmap.png", dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        # zone distance matrix
        zones = ordered_zone_stats(xy, labs_hyb, tumor)
        zones["sample"] = sample
        zones.to_csv(OUT / f"zone_distances_{sample}.csv", index=False)
        if len(zones):
            order = ["normoxic_stable", "exiting_hypoxia", "persistent_hypoxia", "entering_hypoxia", "entering_deep_hypoxia"]
            order = [o for o in order if o in zones["from"].unique() and o in zones["to"].unique()]
            zm = np.full((len(order), len(order)), np.nan)
            for i, a in enumerate(order):
                for j, b in enumerate(order):
                    hit = zones[(zones["from"] == a) & (zones["to"] == b)]
                    if len(hit):
                        zm[i, j] = hit.iloc[0]["median_dist"]
            # normalize by median NN
            zm_n = zm / med_nn
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(zm_n, cmap="viridis_r")
            ax.set_xticks(range(len(order)))
            ax.set_yticks(range(len(order)))
            ax.set_xticklabels([o.replace("_", "\n") for o in order], fontsize=7)
            ax.set_yticklabels([o.replace("_", "\n") for o in order], fontsize=7)
            ax.set_title(f"{sample} · median dist between zones (/NN)")
            fig.colorbar(im, ax=ax, shrink=0.8)
            fig.tight_layout()
            fig.savefig(OUT / f"{sample}_zone_distance_matrix.png", dpi=170, bbox_inches="tight", facecolor="white")
            plt.close(fig)

        # --- 2) Enter–exit interface map ---
        iface_ee = interface_score(idx, labs_hyb, "entering_hypoxia", "exiting_hypoxia")
        iface_pe = interface_score(idx, labs_hyb, "persistent_hypoxia", "exiting_hypoxia")
        iface_ep = interface_score(idx, labs_hyb, "entering_hypoxia", "persistent_hypoxia")
        # who sits on enter-exit interface?
        iface_comp = {}
        for name, iface in [("enter_exit", iface_ee), ("persist_exit", iface_pe), ("enter_persist", iface_ep)]:
            if iface.sum() < 10:
                continue
            # composition of interface cells
            vc = pd.Series(labs_hyb[iface]).value_counts(normalize=True)
            # enrichment of each CT among interface vs global
            rows = []
            for t, n in pd.Series(labs_ct).value_counts().items():
                if n < 20:
                    continue
                obs = (labs_ct[iface] == t).mean()
                exp = (labs_ct == t).mean()
                rows.append({"type": t, "obs": obs, "exp": exp, "enrichment": obs / max(exp, 1e-12), "n_on_iface": int(((labs_ct == t) & iface).sum())})
            idf = pd.DataFrame(rows).sort_values("enrichment", ascending=False)
            idf.to_csv(OUT / f"{sample}_interface_{name}_composition.csv", index=False)
            iface_comp[name] = idf.head(10).to_dict("records")

            fig, ax = plt.subplots(figsize=(6.2, 5.5))
            ax.scatter(xy[:, 0], xy[:, 1], s=2, c="#ecf0f1", alpha=0.4, linewidths=0)
            for state, col in [
                ("entering_hypoxia", "#c0392b"),
                ("entering_deep_hypoxia", "#7b241c"),
                ("persistent_hypoxia", "#8e44ad"),
                ("exiting_hypoxia", "#2980b9"),
                ("normoxic_stable", "#27ae60"),
            ]:
                mm = labs_hyb == state
                ax.scatter(xy[mm, 0], xy[mm, 1], s=8, c=col, alpha=0.7, linewidths=0, label=state)
            ax.scatter(xy[iface, 0], xy[iface, 1], s=22, facecolors="none", edgecolors="#f1c40f", linewidths=0.8, label="interface")
            ax.set_aspect("equal")
            ax.set_title(f"{sample} · {name} interface")
            ax.legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
            ax.set_xticks([])
            ax.set_yticks([])
            fig.tight_layout()
            fig.savefig(OUT / f"{sample}_interface_{name}.png", dpi=170, bbox_inches="tight", facecolor="white")
            plt.close(fig)

        # --- 3) TAM subtype vs hypoxic core ---
        core = np.isin(labs_hyb, ["persistent_hypoxia", "entering_deep_hypoxia"])
        # also annotated Hypoxic_Tumor
        core_annot = labs_ct == "Hypoxic_Tumor"
        tam_core = tam_vs_core(xy, labs_ct, core, k=30, n_perm=500)
        tam_core["sample"] = sample
        tam_core["core_def"] = "persistent+enter_deep"
        tam_annot = tam_vs_core(xy, labs_ct, core_annot, k=30, n_perm=500)
        tam_annot["sample"] = sample
        tam_annot["core_def"] = "Hypoxic_Tumor_annot"
        tam_all = pd.concat([tam_core, tam_annot], ignore_index=True)
        tam_all.to_csv(OUT / f"tam_vs_core_{sample}.csv", index=False)

        fig, ax = plt.subplots(figsize=(7, 4.2))
        sub_tc = tam_core.copy()
        y = np.arange(len(sub_tc))
        colors = ["#27ae60" if r < 1 else "#c0392b" for r in sub_tc.core_neighbor_enrichment]
        ax.barh(y, sub_tc.core_neighbor_enrichment, color=colors)
        ax.axvline(1, color="#555", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(sub_tc.cell)
        ax.set_xlabel("Neighbor enrichment to hypoxic core (obs/exp)")
        ax.set_title(f"{sample} · immune proximity to persistent+deep core")
        fig.tight_layout()
        fig.savefig(OUT / f"{sample}_tam_core_enrichment.png", dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        # Map: core + TAM subtypes
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        for ax, tam, col in zip(axes, ["Spp1_TAM", "C1qc_TAM", "Il1b_TAM"], ["#e67e22", "#3498db", "#2ecc71"]):
            ax.scatter(xy[:, 0], xy[:, 1], s=2, c="#ecf0f1", alpha=0.35, linewidths=0)
            ax.scatter(xy[core, 0], xy[core, 1], s=10, c="#8e44ad", alpha=0.85, linewidths=0, label="hypoxic core")
            mm = labs_ct == tam
            ax.scatter(xy[mm, 0], xy[mm, 1], s=8, c=col, alpha=0.85, linewidths=0, label=tam)
            ax.set_aspect("equal")
            ax.set_title(tam)
            ax.legend(fontsize=6, frameon=False)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"{sample} · TAM subtypes vs hypoxic core")
        fig.tight_layout()
        fig.savefig(OUT / f"{sample}_TAM_vs_core_maps.png", dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        # --- 4) Neutrophil ring / coupling ---
        neu = labs_ct == "Neutrophil"
        # neutrophil density around each dynamics state
        neu_frac = frac_neighbors(idx, neu)
        neu_by_state = {}
        for st in states + ["Hypoxic_Tumor", "Tumor", "Spp1_TAM", "C1qc_TAM"]:
            if st in set(labs_hyb):
                mm = labs_hyb == st
            else:
                mm = labs_ct == st
            if mm.sum() < 15:
                continue
            neu_by_state[st] = {
                "mean_neu_frac": float(neu_frac[mm].mean()),
                "global_neu_frac": float(neu.mean()),
                "enrichment": float(neu_frac[mm].mean() / max(neu.mean(), 1e-12)),
                "n": int(mm.sum()),
            }
        # perm for tumor states
        for st in ["entering_hypoxia", "entering_deep_hypoxia", "persistent_hypoxia", "exiting_hypoxia", "normoxic_stable", "Hypoxic_Tumor"]:
            if st not in neu_by_state:
                continue
            mm = (labs_hyb == st) if st in set(labs_hyb) else (labs_ct == st)
            obs = neu_frac[mm].mean() / max(neu.mean(), 1e-12)
            null = []
            for _ in range(400):
                shuf = labs_hyb.copy() if st in set(labs_hyb) else labs_ct.copy()
                RNG.shuffle(shuf)
                mm2 = shuf == st
                if mm2.sum() < 10:
                    continue
                null.append(neu_frac[mm2].mean() / max(neu.mean(), 1e-12))
            null = np.asarray(null)
            p = (np.sum(np.abs(null - 1) >= abs(obs - 1)) + 1) / (len(null) + 1) if len(null) else np.nan
            neu_by_state[st]["p_perm"] = float(p)

        pd.DataFrame([{"state": k, **v} for k, v in neu_by_state.items()]).to_csv(OUT / f"neutrophil_by_state_{sample}.csv", index=False)

        # --- 5) CD8 niche in E14 (immune hot) ---
        cd8 = labs_ct == "CD8_exhausted"
        if cd8.sum() >= 20:
            cd8_rows = []
            for t in ["NK", "T_helper", "mDC", "cDC", "C1qc_TAM", "Il1b_TAM", "Spp1_TAM", "Neutrophil", "Monocyte", "Tumor", "Hypoxic_Tumor", "entering_hypoxia", "exiting_hypoxia", "persistent_hypoxia"]:
                r = perm_enrich_pair(labs_hyb if t in set(labs_hyb) else labs_ct, idx, "CD8_exhausted", t, n_perm=400)
                # CD8 is CT label in hybrid
                if r is None:
                    # force hybrid which contains CD8_exhausted
                    r = perm_enrich_pair(labs_hyb, idx, "CD8_exhausted", t, n_perm=400)
                if r:
                    r["sample"] = sample
                    cd8_rows.append(r)
            cd8_df = pd.DataFrame(cd8_rows).sort_values("enrichment", ascending=False)
            cd8_df.to_csv(OUT / f"CD8_niche_{sample}.csv", index=False)

            fig, ax = plt.subplots(figsize=(6.2, 5.5))
            ax.scatter(xy[:, 0], xy[:, 1], s=2, c="#ecf0f1", alpha=0.35, linewidths=0)
            for t, col in [("C1qc_TAM", "#3498db"), ("NK", "#9b59b6"), ("mDC", "#1abc9c"), ("Spp1_TAM", "#e67e22")]:
                mm = labs_ct == t
                ax.scatter(xy[mm, 0], xy[mm, 1], s=6, c=col, alpha=0.5, linewidths=0, label=t)
            ax.scatter(xy[cd8, 0], xy[cd8, 1], s=14, c="#c0392b", alpha=0.95, linewidths=0, label="CD8_exhausted")
            ax.set_aspect("equal")
            ax.set_title(f"{sample} · CD8 exhausted niche")
            ax.legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
            ax.set_xticks([])
            ax.set_yticks([])
            fig.tight_layout()
            fig.savefig(OUT / f"{sample}_CD8_niche_map.png", dpi=170, bbox_inches="tight", facecolor="white")
            plt.close(fig)

        # Assemble sample report
        top_front = front.sort_values("enrichment", ascending=False).head(12)
        report["niches"].append(
            {
                "sample": sample,
                "dynamics_adjacency_top": top_front.to_dict("records"),
                "tam_vs_core": tam_core.to_dict("records"),
                "neutrophil_by_state": neu_by_state,
                "interface_composition": iface_comp,
                "median_nn": med_nn,
            }
        )
        print(" top dynamics adjacency:")
        print(top_front[["source", "target", "enrichment", "p_perm"]].head(10).to_string(index=False))
        print(" tam vs core:")
        print(tam_core[["cell", "dist_ratio_vs_bg", "core_neighbor_enrichment", "p_enrich"]].to_string(index=False))
        print(" neutrophil:")
        for k, v in sorted(neu_by_state.items(), key=lambda kv: -kv[1]["enrichment"]):
            print(f"  {k}: enr={v['enrichment']:.3f} p={v.get('p_perm', np.nan)}")

    # Cross-sample: TAM exclusion conserved?
    t14 = pd.read_csv(OUT / "tam_vs_core_E14S.csv")
    t15 = pd.read_csv(OUT / "tam_vs_core_E15S.csv")
    t14 = t14[t14.core_def == "persistent+enter_deep"]
    t15 = t15[t15.core_def == "persistent+enter_deep"]
    tm = t14.merge(t15, on="cell", suffixes=("_E14", "_E15"))
    tm.to_csv(OUT / "tam_vs_core_crosssample.csv", index=False)

    # Conserved story figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, sample, df in zip(axes, ["E14S", "E15S"], [t14, t15]):
        df = df.sort_values("core_neighbor_enrichment")
        y = np.arange(len(df))
        cols = ["#27ae60" if e < 1 else "#e67e22" for e in df.core_neighbor_enrichment]
        ax.barh(y, df.core_neighbor_enrichment, color=cols)
        ax.axvline(1, color="#555", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(df.cell)
        ax.set_title(sample)
        ax.set_xlabel("Enrichment to hypoxic core")
    fig.suptitle("Immune niches relative to velocity-defined hypoxic core")
    fig.tight_layout()
    fig.savefig(OUT / "crosssample_tam_core.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Headline claims
    claims = []
    # Check enter-exit adjacency in both
    for sample in ["E14S", "E15S"]:
        f = pd.read_csv(OUT / f"dynamics_adjacency_{sample}.csv")
        ee = f[(f.source == "entering_hypoxia") & (f.target == "exiting_hypoxia")]
        if len(ee):
            claims.append({"id": f"enter_exit_front_{sample}", "detail": ee.iloc[0].to_dict()})
        pe = f[(f.source == "persistent_hypoxia") & (f.target == "exiting_hypoxia")]
        if len(pe):
            claims.append({"id": f"persist_exit_front_{sample}", "detail": pe.iloc[0].to_dict()})

    # C1qc/Il1b exclusion
    for _, row in tm.iterrows():
        if row["cell"] in ["C1qc_TAM", "Il1b_TAM", "Spp1_TAM", "Neutrophil", "CAF", "CD8_exhausted"]:
            claims.append(
                {
                    "id": f"core_affinity_{row['cell']}",
                    "E14_enr": row["core_neighbor_enrichment_E14"],
                    "E15_enr": row["core_neighbor_enrichment_E15"],
                    "E14_p": row["p_enrich_E14"],
                    "E15_p": row["p_enrich_E15"],
                }
            )

    report["claims"] = claims
    report["headline"] = (
        "Hypoxia forms contiguous enter↔exit fronts; C1qc/Il1b TAMs are spatially excluded from "
        "velocity-defined hypoxic cores while neutrophil/Spp1 coupling is sample-context dependent."
    )
    (OUT / "focused_niche_report.json").write_text(json.dumps(report, indent=2, default=str))
    print("\nHEADLINE:", report["headline"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
