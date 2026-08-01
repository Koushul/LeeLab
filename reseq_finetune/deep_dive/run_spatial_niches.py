#!/usr/bin/env python3
"""Systematic spatial niche discovery for E14S / E15S MC38."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse, stats
from sklearn.neighbors import NearestNeighbors, radius_neighbors_graph

warnings.filterwarnings("ignore", category=FutureWarning)

OUT = Path("/ix1/ylee/shared/MC38_Hypoxia_001/reseq_finetune/deep_dive/spatial_niches")
OUT.mkdir(parents=True, exist_ok=True)
FINETUNE = Path("/ix1/ylee/shared/MC38_Hypoxia_001/reseq_finetune/mc38_tumor_reseq_finetuned.h5ad")
DYN = Path("/ix1/ylee/shared/MC38_Hypoxia_001/reseq_finetune/hypoxia_velocity/hypoxia_dynamics_per_cell.csv")

K = 30
N_PERM = 400
RNG = np.random.default_rng(42)

# Collapse rare / related labels for power
COLLAPSE = {
    "CD8 Terminally Exhausted": "CD8_exhausted",
    "CD8 Exhausted Proliferating": "CD8_exhausted",
    "Regulatory CD8 T": "CD8_exhausted",
    "CD4 T": "T_helper",
    "Regulatory CD4 T": "T_helper",
    "NK T": "NK",
    "Tumor Proliferating": "Tumor",
    "Hypoxic Tumor": "Hypoxic_Tumor",
}


def collapse_label(x: str) -> str:
    return COLLAPSE.get(x, x.replace("+", "pos").replace(" ", "_"))


def load():
    adata = sc.read_h5ad(FINETUNE)
    dyn = pd.read_csv(DYN).set_index("barcode")
    common = adata.obs_names.intersection(dyn.index)
    adata = adata[common].copy()
    for c in ["hypoxia_dynamics", "hypoxia_score", "v_hypoxia", "prolif_score"]:
        adata.obs[c] = dyn.loc[adata.obs_names, c].values
    if "spatial" not in adata.obsm:
        adata.obsm["spatial"] = adata.obs[["spatial_coordinate_x", "spatial_coordinate_y"]].to_numpy()
    adata.obs["ct"] = adata.obs["cell_type_finetuned"].astype(str).map(collapse_label)
    # hybrid label: tumor hypoxia dynamics override tumor labels
    hyb = adata.obs["ct"].astype(str).to_numpy().copy()
    dynv = adata.obs["hypoxia_dynamics"].astype(str).to_numpy()
    tumorish = adata.obs["ct"].isin(["Tumor", "Hypoxic_Tumor"]).to_numpy()
    keep_dyn = np.isin(
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
    hyb[tumorish & keep_dyn] = dynv[tumorish & keep_dyn]
    # merge enter deep into entering for niche power optionally kept separate
    adata.obs["niche_label"] = hyb
    adata.obs["is_tumor"] = tumorish
    adata.var["mt"] = adata.var_names.str.startswith("mt-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    adata.obs["hq"] = (adata.obs["n_genes_by_counts"] >= 800) & (adata.obs["pct_counts_mt"] < 20)
    return adata


def pairwise_enrichment(labels, knn_idx, types, n_perm=N_PERM):
    """Enrichment of type B among neighbors of type A: obs/exp with permutation p."""
    labs = np.asarray(labels)
    n = len(labs)
    # neighbor count matrix
    def counts(lab_arr):
        mat = {a: {b: 0 for b in types} for a in types}
        tot = {a: 0 for a in types}
        for i, neigh in enumerate(knn_idx):
            a = lab_arr[i]
            if a not in mat:
                continue
            for j in neigh:
                b = lab_arr[j]
                if b in mat[a]:
                    mat[a][b] += 1
                    tot[a] += 1
        return mat, tot

    obs, tot = counts(labs)
    # global frequencies
    freq = {t: (labs == t).mean() for t in types}
    rows = []
    # permutations of labels (preserve positions)
    null_ratios = {(a, b): [] for a in types for b in types if a != b}
    # Only permute among cells; store ratio obs/exp
    for p in range(n_perm):
        shuf = labs.copy()
        RNG.shuffle(shuf)
        mat_p, tot_p = counts(shuf)
        freq_p = {t: (shuf == t).mean() for t in types}
        for a in types:
            if tot_p[a] == 0:
                continue
            for b in types:
                if a == b:
                    continue
                exp = freq_p[b]
                if exp < 1e-12:
                    continue
                ratio = (mat_p[a][b] / tot_p[a]) / exp
                null_ratios[(a, b)].append(ratio)

    for a in types:
        if tot[a] == 0:
            continue
        for b in types:
            if a == b:
                continue
            frac = obs[a][b] / tot[a]
            exp = freq[b]
            if exp < 1e-12:
                continue
            ratio = frac / exp
            null = np.asarray(null_ratios[(a, b)], float)
            if len(null) == 0:
                p = np.nan
            else:
                # two-sided
                p = (np.sum(np.abs(null - 1) >= abs(ratio - 1)) + 1) / (len(null) + 1)
            rows.append(
                {
                    "source": a,
                    "target": b,
                    "obs_frac": frac,
                    "exp_frac": exp,
                    "enrichment": ratio,
                    "log2_enrichment": float(np.log2(max(ratio, 1e-12))),
                    "n_source": int((labs == a).sum()),
                    "n_target": int((labs == b).sum()),
                    "p_perm": float(p),
                }
            )
    return pd.DataFrame(rows)


def radius_cooccurrence(xy, labels, types, radius, n_perm=200):
    """Fraction of sources with ≥1 target within radius; enrichment vs label shuffle."""
    xy = np.asarray(xy, float)
    labs = np.asarray(labels)
    # precompute all pairs within radius via sklearn
    nn = NearestNeighbors(radius=radius).fit(xy)
    neigh = nn.radius_neighbors(xy, return_distance=False)

    def hit_rate(lab_arr):
        out = {}
        for a in types:
            ia = np.where(lab_arr == a)[0]
            if len(ia) == 0:
                continue
            for b in types:
                if a == b:
                    continue
                hits = 0
                for i in ia:
                    nb = lab_arr[neigh[i]]
                    # exclude self
                    if np.any(nb == b):
                        hits += 1
                out[(a, b)] = hits / len(ia)
        return out

    obs = hit_rate(labs)
    nulls = {(a, b): [] for a in types for b in types if a != b}
    for _ in range(n_perm):
        shuf = labs.copy()
        RNG.shuffle(shuf)
        h = hit_rate(shuf)
        for k, v in h.items():
            nulls[k].append(v)
    rows = []
    for (a, b), v in obs.items():
        null = np.asarray(nulls[(a, b)], float)
        exp = float(null.mean()) if len(null) else np.nan
        if len(null):
            p = (np.sum(null >= v) + 1) / (len(null) + 1) if v >= exp else (np.sum(null <= v) + 1) / (len(null) + 1)
            # two-sided-ish
            p2 = (np.sum(np.abs(null - exp) >= abs(v - exp)) + 1) / (len(null) + 1)
        else:
            p2 = np.nan
            exp = np.nan
        rows.append(
            {
                "source": a,
                "target": b,
                "hit_rate": v,
                "null_mean": exp,
                "enrichment": v / exp if exp and exp > 1e-12 else np.nan,
                "p_perm": float(p2),
                "radius": radius,
            }
        )
    return pd.DataFrame(rows)


def local_density_field(xy, mask, k=40):
    """For each cell, fraction of kNN that are in mask."""
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(xy))).fit(xy)
    idx = nn.kneighbors(xy, return_distance=False)[:, 1:]
    m = mask.astype(float)
    return m[idx].mean(axis=1)


def distance_to_set(xy, dest_mask):
    if dest_mask.sum() < 3:
        return np.full(len(xy), np.nan)
    nn = NearestNeighbors(n_neighbors=1).fit(xy[dest_mask])
    d, _ = nn.kneighbors(xy)
    return d.ravel()


def triad_enrichment(labels, knn_idx, center, partner_a, partner_b, n_perm=300):
    """Among neighbors of center cells, co-presence of A and B vs independence."""
    labs = np.asarray(labels)
    ic = np.where(labs == center)[0]
    if len(ic) < 20:
        return None
    both = ab = bb = 0
    for i in ic:
        nb = labs[knn_idx[i]]
        has_a = np.any(nb == partner_a)
        has_b = np.any(nb == partner_b)
        ab += int(has_a)
        bb += int(has_b)
        both += int(has_a and has_b)
    n = len(ic)
    p_a, p_b = ab / n, bb / n
    obs = both / n
    exp = p_a * p_b
    # permute partners by shuffling all labels
    null = []
    for _ in range(n_perm):
        shuf = labs.copy()
        RNG.shuffle(shuf)
        ic2 = np.where(shuf == center)[0]
        if len(ic2) < 10:
            continue
        both2 = ab2 = bb2 = 0
        for i in ic2:
            nb = shuf[knn_idx[i]]
            ha = np.any(nb == partner_a)
            hb = np.any(nb == partner_b)
            ab2 += int(ha)
            bb2 += int(hb)
            both2 += int(ha and hb)
        n2 = len(ic2)
        null.append(both2 / n2 - (ab2 / n2) * (bb2 / n2))
    excess = obs - exp
    null = np.asarray(null, float)
    p = (np.sum(np.abs(null) >= abs(excess)) + 1) / (len(null) + 1) if len(null) else np.nan
    return {
        "center": center,
        "partner_a": partner_a,
        "partner_b": partner_b,
        "obs_cooccur": obs,
        "exp_indep": exp,
        "excess": excess,
        "p_perm": float(p),
        "n_center": n,
        "enrichment": obs / exp if exp > 1e-12 else np.nan,
    }


def plot_map(xy, color, title, path, cmap="viridis", s=4, categorical=False, legend_labels=None):
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    if categorical:
        cats = pd.Categorical(color)
        # build discrete colors
        uniq = list(cats.categories)
        palette = plt.cm.tab20(np.linspace(0, 1, max(len(uniq), 1)))
        for i, u in enumerate(uniq):
            m = cats == u
            ax.scatter(xy[m, 0], xy[m, 1], s=s, c=[palette[i]], label=str(u), alpha=0.85, linewidths=0)
        ax.legend(fontsize=5, markerscale=2, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    else:
        sca = ax.scatter(xy[:, 0], xy[:, 1], c=color, s=s, cmap=cmap, alpha=0.85, linewidths=0)
        fig.colorbar(sca, ax=ax, shrink=0.7)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    print("loading…")
    adata = load()
    findings = []
    all_enrich = []
    all_dist = []
    all_triads = []

    for sample in ["E14S", "E15S"]:
        print(f"=== {sample} ===")
        m = (adata.obs["sample"] == sample).to_numpy() & np.isfinite(adata.obsm["spatial"]).all(axis=1)
        # keep HQ for gene overlays later but use all spatial-ok for niche structure
        sub = adata[m].copy()
        xy = np.asarray(sub.obsm["spatial"], float)
        # standardize coords per sample for comparable radii
        xy_z = (xy - xy.mean(0)) / (xy.std(0) + 1e-9)

        labs_ct = sub.obs["ct"].astype(str).to_numpy()
        labs_hyb = sub.obs["niche_label"].astype(str).to_numpy()

        nn = NearestNeighbors(n_neighbors=min(K + 1, len(xy))).fit(xy)
        knn = nn.kneighbors(xy, return_distance=False)[:, 1:]

        # types with enough cells
        vc = pd.Series(labs_ct).value_counts()
        types_ct = [t for t, n in vc.items() if n >= 40]
        vc_h = pd.Series(labs_hyb).value_counts()
        types_hyb = [t for t, n in vc_h.items() if n >= 30]

        print(" pairwise CT enrichment…", len(types_ct), "types")
        enr = pairwise_enrichment(labs_ct, knn, types_ct, n_perm=N_PERM)
        enr["sample"] = sample
        enr["label_space"] = "cell_type"
        all_enrich.append(enr)

        print(" pairwise hybrid (dynamics) enrichment…")
        enr_h = pairwise_enrichment(labs_hyb, knn, types_hyb, n_perm=min(N_PERM, 300))
        enr_h["sample"] = sample
        enr_h["label_space"] = "hybrid_dynamics"
        all_enrich.append(enr_h)

        # radius based on median kNN distance * factor
        d_nn, _ = nn.kneighbors(xy)
        med_nn = float(np.median(d_nn[:, 1]))
        radius = med_nn * 3.5
        print(f" radius co-occurrence (r={radius:.3f})…")
        # focus on interesting sources/targets
        focus = [
            t
            for t in [
                "Spp1pos_TAM",
                "C1qcpos_TAM",
                "Il1bpos_TAM",
                "Neutrophil",
                "CAF",
                "Stromal",
                "CD8_exhausted",
                "NK",
                "Monocyte",
                "Hypoxic_Tumor",
                "Tumor",
                "entering_hypoxia",
                "entering_deep_hypoxia",
                "persistent_hypoxia",
                "exiting_hypoxia",
                "normoxic_stable",
            ]
            if t in set(types_hyb) or t in set(types_ct)
        ]
        # use hybrid labels for radius
        types_r = [t for t in types_hyb if t in focus or True]
        # limit types for speed
        types_r = [t for t, n in vc_h.items() if n >= 50][:18]
        rad = radius_cooccurrence(xy, labs_hyb, types_r, radius=radius, n_perm=150)
        rad["sample"] = sample
        rad.to_csv(OUT / f"radius_cooccur_{sample}.csv", index=False)

        # Distance of each CT / dynamics to niche anchors
        print(" distance-to-anchor…")
        anchors = {
            "Spp1_TAM": labs_ct == "Spp1pos_TAM",
            "C1qc_TAM": labs_ct == "C1qcpos_TAM",
            "Il1b_TAM": labs_ct == "Il1bpos_TAM",
            "Neutrophil": labs_ct == "Neutrophil",
            "CAF": labs_ct == "CAF",
            "Stromal": labs_ct == "Stromal",
            "CD8_exh": labs_ct == "CD8_exhausted",
            "NK": labs_ct == "NK",
            "Monocyte": labs_ct == "Monocyte",
            "Hypoxic_Tumor": labs_ct == "Hypoxic_Tumor",
            "Tumor": labs_ct == "Tumor",
        }
        # also dynamics-defined hypoxic core: persistent + entering_deep
        anchors["hypoxic_core_dyn"] = np.isin(labs_hyb, ["persistent_hypoxia", "entering_deep_hypoxia"])
        anchors["exit_front"] = labs_hyb == "exiting_hypoxia"
        anchors["enter_front"] = np.isin(labs_hyb, ["entering_hypoxia", "entering_deep_hypoxia"])

        query_groups = {
            "entering": np.isin(labs_hyb, ["entering_hypoxia", "entering_deep_hypoxia"]),
            "entering_deep": labs_hyb == "entering_deep_hypoxia",
            "persistent": labs_hyb == "persistent_hypoxia",
            "exiting": labs_hyb == "exiting_hypoxia",
            "normoxic": labs_hyb == "normoxic_stable",
            "Spp1_TAM": labs_ct == "Spp1pos_TAM",
            "C1qc_TAM": labs_ct == "C1qcpos_TAM",
            "Il1b_TAM": labs_ct == "Il1bpos_TAM",
            "CD8_exh": labs_ct == "CD8_exhausted",
            "Neutrophil": labs_ct == "Neutrophil",
            "CAF": labs_ct == "CAF",
            "NK": labs_ct == "NK",
            "Monocyte": labs_ct == "Monocyte",
        }
        for qname, qmask in query_groups.items():
            if qmask.sum() < 15:
                continue
            for aname, amask in anchors.items():
                if amask.sum() < 10:
                    continue
                # don't self-compare identical sets
                d = distance_to_set(xy, amask)
                # for cells in query that are NOT in anchor (avoid 0 self-dist domination)
                use = qmask & ~amask
                if use.sum() < 10:
                    use = qmask
                med = float(np.nanmedian(d[use]))
                # background: all other cells
                bg = ~qmask & ~amask
                bg_med = float(np.nanmedian(d[bg])) if bg.sum() > 20 else np.nan
                try:
                    p = float(stats.mannwhitneyu(d[use], d[bg], alternative="two-sided").pvalue) if bg.sum() > 20 else np.nan
                except Exception:
                    p = np.nan
                all_dist.append(
                    {
                        "sample": sample,
                        "query": qname,
                        "anchor": aname,
                        "median_dist": med,
                        "bg_median_dist": bg_med,
                        "ratio_vs_bg": med / bg_med if bg_med and bg_med > 0 else np.nan,
                        "p_vs_bg": p,
                        "n_query": int(use.sum()),
                        "n_anchor": int(amask.sum()),
                    }
                )

        # Local density fields
        print(" density fields…")
        fields = {
            "Spp1_density": local_density_field(xy, labs_ct == "Spp1pos_TAM", k=40),
            "C1qc_density": local_density_field(xy, labs_ct == "C1qcpos_TAM", k=40),
            "Neutrophil_density": local_density_field(xy, labs_ct == "Neutrophil", k=40),
            "CAF_density": local_density_field(xy, labs_ct == "CAF", k=40),
            "CD8_density": local_density_field(xy, labs_ct == "CD8_exhausted", k=40),
            "hypoxic_core_density": local_density_field(xy, anchors["hypoxic_core_dyn"], k=40),
            "exit_density": local_density_field(xy, labs_hyb == "exiting_hypoxia", k=40),
            "enter_density": local_density_field(xy, np.isin(labs_hyb, ["entering_hypoxia", "entering_deep_hypoxia"]), k=40),
        }
        for fname, fval in fields.items():
            plot_map(xy, fval, f"{sample} · {fname}", OUT / f"{sample}_{fname}.png", cmap="magma", s=3)

        # Hypoxia score spatial
        plot_map(xy, sub.obs["hypoxia_score"].to_numpy(), f"{sample} · hypoxia score", OUT / f"{sample}_hypoxia_score.png", cmap="Reds", s=3)

        # Dynamics categorical map (tumor only highlighted)
        dyn_cols = {
            "entering_hypoxia": "#c0392b",
            "entering_deep_hypoxia": "#7b241c",
            "persistent_hypoxia": "#8e44ad",
            "exiting_hypoxia": "#2980b9",
            "normoxic_stable": "#27ae60",
            "transitional": "#f39c12",
            "reverted_stable": "#1abc9c",
        }
        fig, ax = plt.subplots(figsize=(6.5, 5.8))
        ax.scatter(xy[:, 0], xy[:, 1], s=2, c="#dfe6e9", alpha=0.5, linewidths=0)
        for state, col in dyn_cols.items():
            mm = labs_hyb == state
            if mm.sum() == 0:
                continue
            ax.scatter(xy[mm, 0], xy[mm, 1], s=10, c=col, label=state, alpha=0.9, linewidths=0)
        ax.set_aspect("equal")
        ax.set_title(f"{sample} · tumor hypoxia dynamics")
        ax.legend(fontsize=6, markerscale=1.5, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(OUT / f"{sample}_dynamics_map.png", dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        # Cell type map (major)
        major = ["Tumor", "Hypoxic_Tumor", "Spp1pos_TAM", "C1qcpos_TAM", "Il1bpos_TAM", "Neutrophil", "CAF", "Stromal", "CD8_exhausted", "Monocyte", "NK"]
        fig, ax = plt.subplots(figsize=(6.5, 5.8))
        ax.scatter(xy[:, 0], xy[:, 1], s=2, c="#eceff1", alpha=0.4, linewidths=0)
        cmap = plt.cm.tab20(np.linspace(0, 1, len(major)))
        for i, t in enumerate(major):
            mm = labs_ct == t
            if mm.sum() == 0:
                continue
            ax.scatter(xy[mm, 0], xy[mm, 1], s=6, c=[cmap[i]], label=t, alpha=0.85, linewidths=0)
        ax.set_aspect("equal")
        ax.set_title(f"{sample} · major cell types")
        ax.legend(fontsize=6, markerscale=1.8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(OUT / f"{sample}_celltype_map.png", dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        # Triad niches: interesting centers
        print(" triad niches…")
        triad_specs = [
            ("persistent_hypoxia", "Spp1pos_TAM", "Neutrophil"),
            ("persistent_hypoxia", "Spp1pos_TAM", "CAF"),
            ("exiting_hypoxia", "Spp1pos_TAM", "C1qcpos_TAM"),
            ("entering_hypoxia", "Spp1pos_TAM", "Neutrophil"),
            ("entering_deep_hypoxia", "Spp1pos_TAM", "Neutrophil"),
            ("Hypoxic_Tumor", "Spp1pos_TAM", "Neutrophil"),
            ("Hypoxic_Tumor", "CAF", "Spp1pos_TAM"),
            ("CD8_exhausted", "C1qcpos_TAM", "Il1bpos_TAM"),
            ("CD8_exhausted", "Spp1pos_TAM", "Neutrophil"),
            ("CD8_exhausted", "NK", "C1qcpos_TAM"),
            ("Spp1pos_TAM", "Neutrophil", "Hypoxic_Tumor"),
            ("Spp1pos_TAM", "CAF", "Hypoxic_Tumor"),
            ("CAF", "Spp1pos_TAM", "persistent_hypoxia"),
            ("Neutrophil", "Spp1pos_TAM", "entering_deep_hypoxia"),
            ("Monocyte", "Neutrophil", "Il1bpos_TAM"),
            ("exiting_hypoxia", "CAF", "Stromal"),
            ("entering_hypoxia", "Monocyte", "Il1bpos_TAM"),
            ("normoxic_stable", "CD8_exhausted", "NK"),
        ]
        dyn_centers = {
            "entering_hypoxia",
            "entering_deep_hypoxia",
            "persistent_hypoxia",
            "exiting_hypoxia",
            "normoxic_stable",
            "transitional",
            "reverted_stable",
        }
        for center, a, b in triad_specs:
            use_labs = labs_hyb if (center in dyn_centers or np.any(labs_hyb == center)) else labs_ct
            if not np.any(use_labs == center):
                use_labs = labs_hyb
            if not np.any(use_labs == center):
                continue
            # partners must exist in same label space; prefer hybrid (has CT + dynamics)
            res = triad_enrichment(labs_hyb, knn, center, a, b, n_perm=200)
            if res is None:
                continue
            res["sample"] = sample
            all_triads.append(res)

        # Cross-density correlations among tumor cells: which niches predict v_hypoxia / score
        print(" niche predictors of velocity…")
        tumor_m = sub.obs["is_tumor"].to_numpy() & sub.obs["hq"].to_numpy()
        if tumor_m.sum() > 80:
            dfp = pd.DataFrame({k: fields[k][tumor_m] for k in fields})
            dfp["hypoxia_score"] = sub.obs["hypoxia_score"].to_numpy()[tumor_m]
            dfp["v_hypoxia"] = sub.obs["v_hypoxia"].to_numpy()[tumor_m]
            corr_rows = []
            for k in fields:
                for y in ["hypoxia_score", "v_hypoxia"]:
                    r, p = stats.spearmanr(dfp[k], dfp[y], nan_policy="omit")
                    corr_rows.append({"sample": sample, "field": k, "response": y, "spearman_r": float(r), "p": float(p)})
            pd.DataFrame(corr_rows).to_csv(OUT / f"field_velocity_corr_{sample}.csv", index=False)
            findings.append(
                {
                    "id": f"field_corr_{sample}",
                    "title": f"Spatial niche densities predicting hypoxia score/velocity ({sample})",
                    "top": sorted(corr_rows, key=lambda d: abs(d["spearman_r"]), reverse=True)[:12],
                }
            )

        # Top enrichments summary
        top_enr = enr.query("p_perm < 0.05").sort_values("enrichment", ascending=False).head(15)
        top_dep = enr.query("p_perm < 0.05").sort_values("enrichment", ascending=True).head(10)
        findings.append(
            {
                "id": f"top_enrich_{sample}",
                "title": f"Top spatial enrichments ({sample})",
                "enriched": top_enr.to_dict("records"),
                "depleted": top_dep.to_dict("records"),
            }
        )
        top_h = enr_h.query("p_perm < 0.05").sort_values("enrichment", ascending=False).head(20)
        findings.append({"id": f"top_hybrid_enrich_{sample}", "title": f"Dynamics-aware niche enrichments ({sample})", "enriched": top_h.to_dict("records")})

        # Heatmap of enrichment among key types
        key = [t for t in ["Hypoxic_Tumor", "Tumor", "Spp1pos_TAM", "C1qcpos_TAM", "Il1bpos_TAM", "Neutrophil", "CAF", "Stromal", "CD8_exhausted", "Monocyte", "NK", "Resident_Macrophage"] if t in types_ct]
        if len(key) >= 4:
            mat = np.full((len(key), len(key)), np.nan)
            for i, a in enumerate(key):
                for j, b in enumerate(key):
                    if a == b:
                        mat[i, j] = 0.0
                        continue
                    hit = enr[(enr.source == a) & (enr.target == b)]
                    if len(hit):
                        mat[i, j] = float(hit.iloc[0]["log2_enrichment"])
            fig, ax = plt.subplots(figsize=(7, 6))
            im = ax.imshow(mat, cmap="RdBu_r", vmin=-1.5, vmax=1.5)
            ax.set_xticks(range(len(key)))
            ax.set_yticks(range(len(key)))
            ax.set_xticklabels(key, rotation=60, ha="right", fontsize=8)
            ax.set_yticklabels(key, fontsize=8)
            ax.set_title(f"{sample} · log2 neighbor enrichment")
            fig.colorbar(im, ax=ax, shrink=0.8, label="log2(obs/exp)")
            fig.tight_layout()
            fig.savefig(OUT / f"{sample}_enrichment_heatmap.png", dpi=160, bbox_inches="tight", facecolor="white")
            plt.close(fig)

    enrich_df = pd.concat(all_enrich, ignore_index=True)
    enrich_df.to_csv(OUT / "pairwise_enrichment_all.csv", index=False)
    dist_df = pd.DataFrame(all_dist)
    dist_df.to_csv(OUT / "distance_to_anchors.csv", index=False)
    triad_df = pd.DataFrame(all_triads)
    if len(triad_df):
        triad_df.to_csv(OUT / "triad_niches.csv", index=False)

    # Cross-sample conserved enrichments
    print(" conserved niches…")
    e14 = enrich_df[(enrich_df.sample == "E14S") & (enrich_df.label_space == "cell_type")]
    e15 = enrich_df[(enrich_df.sample == "E15S") & (enrich_df.label_space == "cell_type")]
    merged = e14.merge(e15, on=["source", "target"], suffixes=("_E14", "_E15"))
    cons = merged[
        (merged.p_perm_E14 < 0.05)
        & (merged.p_perm_E15 < 0.05)
        & (((merged.enrichment_E14 > 1.15) & (merged.enrichment_E15 > 1.15)) | ((merged.enrichment_E14 < 0.87) & (merged.enrichment_E15 < 0.87)))
    ].copy()
    cons["mean_enrichment"] = (cons.enrichment_E14 + cons.enrichment_E15) / 2
    cons = cons.sort_values("mean_enrichment", ascending=False)
    cons.to_csv(OUT / "conserved_enrichments.csv", index=False)

    # Interesting distance asymmetries conserved or strong in E15
    dist_sig = dist_df.dropna(subset=["ratio_vs_bg"]).copy()
    dist_sig = dist_sig[dist_sig.p_vs_bg < 0.01]
    dist_sig = dist_sig.sort_values("ratio_vs_bg")
    dist_sig.to_csv(OUT / "significant_distances.csv", index=False)

    # Rank triad hits
    if len(triad_df):
        triad_hits = triad_df[(triad_df.p_perm < 0.05) & (triad_df.excess > 0)].sort_values("enrichment", ascending=False)
        triad_hits.to_csv(OUT / "triad_hits.csv", index=False)
    else:
        triad_hits = pd.DataFrame()

    # Build headline niches
    headlines = []
    if len(cons):
        headlines.append(
            {
                "id": "conserved_ct_niches",
                "title": "Cell-type niches conserved across E14S and E15S",
                "top_enriched": cons.head(20)[["source", "target", "enrichment_E14", "enrichment_E15", "p_perm_E14", "p_perm_E15"]].to_dict("records"),
                "top_depleted": cons.sort_values("mean_enrichment").head(15)[["source", "target", "enrichment_E14", "enrichment_E15", "p_perm_E14", "p_perm_E15"]].to_dict("records"),
            }
        )

    # Dynamics-specific: who neighbors enter vs exit vs persist
    dyn_enr = enrich_df[enrich_df.label_space == "hybrid_dynamics"]
    for state in ["entering_hypoxia", "entering_deep_hypoxia", "persistent_hypoxia", "exiting_hypoxia"]:
        sube = dyn_enr[(dyn_enr.source == state) & (dyn_enr.p_perm < 0.05)].sort_values("enrichment", ascending=False)
        if len(sube):
            headlines.append({"id": f"neighbors_of_{state}", "title": f"Enriched neighbors of {state}", "rows": sube.head(12).to_dict("records")})

    if len(triad_hits):
        headlines.append({"id": "triad_niches", "title": "Multi-cell triad niches (co-occurrence excess)", "rows": triad_hits.head(20).to_dict("records")})

    # Strong approach-to-anchor (ratio << 1)
    close = dist_sig[dist_sig.ratio_vs_bg < 0.85].head(30)
    far = dist_sig[dist_sig.ratio_vs_bg > 1.15].sort_values("ratio_vs_bg", ascending=False).head(20)
    headlines.append({"id": "spatial_proximity", "title": "Queries closer/farther than background to anchors", "closer": close.to_dict("records"), "farther": far.to_dict("records")})

    summary = {
        "headline": "Spatial niche atlas of E14S/E15S hypoxia continuum",
        "n_conserved_ct_pairs": int(len(cons)),
        "n_triad_hits": int(len(triad_hits)) if len(triad_df) else 0,
        "headlines": headlines,
        "findings": findings,
    }
    (OUT / "niche_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # Overview figure: top conserved enrichments
    if len(cons):
        top = cons.head(12)
        fig, ax = plt.subplots(figsize=(8, 5))
        y = np.arange(len(top))
        ax.barh(y - 0.15, top.enrichment_E14, height=0.3, label="E14S", color="#c0392b")
        ax.barh(y + 0.15, top.enrichment_E15, height=0.3, label="E15S", color="#2980b9")
        ax.axvline(1, color="#888", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{s}→{t}" for s, t in zip(top.source, top.target)], fontsize=8)
        ax.set_xlabel("Neighbor enrichment (obs/exp)")
        ax.set_title("Conserved spatial niches (E14S & E15S)")
        ax.legend(frameon=False)
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(OUT / "conserved_niches_bar.png", dpi=170, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    # Print digest
    print("\n===== DIGEST =====")
    print("conserved enriched pairs:", len(cons))
    if len(cons):
        print(cons.head(15)[["source", "target", "enrichment_E14", "enrichment_E15"]].to_string(index=False))
    if len(triad_hits):
        print("\ntriad hits:")
        print(triad_hits.head(12)[["sample", "center", "partner_a", "partner_b", "enrichment", "excess", "p_perm"]].to_string(index=False))
    print("\nclosest proximities:")
    print(close.head(12)[["sample", "query", "anchor", "ratio_vs_bg", "p_vs_bg"]].to_string(index=False) if len(close) else "none")
    print("done →", OUT)


if __name__ == "__main__":
    main()
