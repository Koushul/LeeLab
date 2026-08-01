#!/usr/bin/env python3
"""Iterative discovery analysis on MC38 E14S/E15S."""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse, stats
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
OUT = Path('/ix1/ylee/shared/MC38_Hypoxia_001/reseq_finetune/deep_dive')
OUT.mkdir(parents=True, exist_ok=True)
ANNOT = Path('/ix1/ylee/shared/MC38_Hypoxia_001/reseq_finetune/mc38_tumor_reseq_finetuned.h5ad')
HYPO = Path('/ix1/ylee/shared/MC38_Hypoxia_001/reseq_finetune/hypoxia_velocity/hypoxia_dynamics_per_cell.csv')

TUMOR = {'Tumor','Hypoxic Tumor','Tumor Proliferating'}
HYPOXIA_GENES = ['Slc2a1','Hk2','Ldha','Pgk1','Bnip3','Vegfa','Egln3','Eno1','Pdk1','Gapdh','Aldoa','Ndrg1','Fam162a','P4ha1','Higd1a','Car9','Anxa2','Mif','Pgam1']
IFN = ['Stat1','Irf7','Irf9','Isg15','Ifit1','Ifit3','Mx1','Rsad2','Oasl2','Cxcl10','Cxcl9','Cd274']
EMT = ['Vim','Zeb1','Zeb2','Snai1','Snai2','Twist1','Cdh2','Fn1','Tagln','Acta2']
OXPHOS = ['Ndufa1','Ndufb3','Cox5a','Cox7a2','Atp5g1','Atp5e','Uqcr11','Sdhd']
GLYCOLYSIS = ['Hk2','Pfkl','Aldoa','Gapdh','Pgk1','Eno1','Pkm','Ldha','Slc2a1']
SPP1_PROG = ['Spp1','Arg1','Mrc1','Cd163','Vegfa','Fn1','Hilpda','Hmox1','Il1b']
TREM2_PROG = ['Trem2','Apoe','C1qa','C1qb','C1qc','Lyz2','Ms4a7']
CD8_EXH = ['Pdcd1','Tox','Lag3','Havcr2','Tigit','Entpd1','Cd244','Ctla4']
CD8_EFF = ['Gzmb','Gzma','Prf1','Ifng','Tnf','Nkg7','Ccl5']

def dens(X):
    return X.toarray() if sparse.issparse(X) else np.asarray(X)

def score(adata, genes, key):
    present=[g for g in genes if g in adata.var_names]
    if len(present)<2:
        adata.obs[key]=0.0; return present
    idx=[adata.var_names.get_loc(g) for g in present]
    sub=dens(adata.X[:,idx]).astype(float)
    mu,sd=sub.mean(0),sub.std(0); sd[sd<1e-8]=1
    adata.obs[key]=((sub-mu)/sd).mean(1)
    return present

def lognorm_counts(adata):
    a=adata.copy()
    if 'counts' in a.layers:
        a.X=a.layers['counts'].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a

def wilcox_de(adata, mask_a, mask_b, min_frac=0.1, top=100):
    """Simple Wilcoxon DE between two groups on log-normalized X."""
    X=dens(adata.X)
    xa, xb = X[mask_a], X[mask_b]
    if xa.shape[0]<20 or xb.shape[0]<20:
        return pd.DataFrame()
    mean_a, mean_b = xa.mean(0), xb.mean(0)
    frac_a=(xa>0).mean(0); frac_b=(xb>0).mean(0)
    # subsample genes expressed enough
    keep=(frac_a>=min_frac)|(frac_b>=min_frac)
    idxs=np.where(keep)[0]
    rows=[]
    # further limit to speed: top variance genes among kept
    var=X[:,idxs].var(0)
    take=idxs[np.argsort(-var)[:min(4000,len(idxs))]]
    for j in take:
        # approximate p via mannwhitney on subsample for speed
        n=min(800, xa.shape[0], xb.shape[0])
        ra=np.random.default_rng(j).choice(xa.shape[0], n, replace=False)
        rb=np.random.default_rng(j+1).choice(xb.shape[0], n, replace=False)
        try:
            u,p=stats.mannwhitneyu(xa[ra,j], xb[rb,j], alternative='two-sided')
        except Exception:
            continue
        lfc=float(mean_a[j]-mean_b[j])
        rows.append({
            'gene': adata.var_names[j],
            'lfc_a_minus_b': lfc,
            'mean_a': float(mean_a[j]),
            'mean_b': float(mean_b[j]),
            'frac_a': float(frac_a[j]),
            'frac_b': float(frac_b[j]),
            'pval': float(p),
        })
    df=pd.DataFrame(rows)
    if df.empty: return df
    df['padj']=np.minimum(df['pval']*len(df), 1.0)  # Bonferroni rough
    return df.sort_values(['padj','lfc_a_minus_b'], ascending=[True, False])

def spatial_enrichment(xy, labels, focus_label, n_perm=200, radius_quantile=0.05):
    """Permutation test: are focus cells closer to each other / to partners than random?"""
    ok=np.isfinite(xy).all(1)
    xy=xy[ok]; labels=np.asarray(labels)[ok]
    if xy.shape[0]<50: return {}
    # kNN radius from quantile of nearest neighbor dists
    nn=NearestNeighbors(n_neighbors=2).fit(xy)
    d,_=nn.kneighbors(xy)
    rad=np.quantile(d[:,1], 0.9)*2.5
    nn=NearestNeighbors(radius=rad).fit(xy)
    ind=nn.radius_neighbors(xy, return_distance=False)
    focus=labels==focus_label
    if focus.sum()<10: return {'n':int(focus.sum())}
    # mean fraction of neighbors that are also focus
    frac=[]
    for i in np.where(focus)[0]:
        nbrs=ind[i]; nbrs=nbrs[nbrs!=i]
        if len(nbrs)==0: continue
        frac.append((labels[nbrs]==focus_label).mean())
    obs=float(np.mean(frac)) if frac else np.nan
    null=[]
    rng=np.random.default_rng(0)
    for _ in range(n_perm):
        lab=labels.copy(); rng.shuffle(lab)
        f=lab==focus_label
        fr=[]
        for i in np.where(f)[0][:min(200,f.sum())]:
            nbrs=ind[i]; nbrs=nbrs[nbrs!=i]
            if len(nbrs)==0: continue
            fr.append((lab[nbrs]==focus_label).mean())
        if fr: null.append(np.mean(fr))
    null=np.array(null)
    p=float((null>=obs).mean()) if len(null) else np.nan
    return {'n':int(focus.sum()), 'obs_self_neighbor_frac':obs, 'null_mean':float(null.mean()) if len(null) else None,
            'enrichment': float(obs/(null.mean()+1e-9)) if len(null) else None, 'p_perm':p, 'radius':float(rad)}

def neighbor_type_affinity(xy, labels, source, targets, radius=None):
    ok=np.isfinite(xy).all(1)
    xy=xy[ok]; labels=np.asarray(labels)[ok]
    nn=NearestNeighbors(n_neighbors=2).fit(xy)
    d,_=nn.kneighbors(xy)
    rad=radius or float(np.quantile(d[:,1],0.9)*2.5)
    nn=NearestNeighbors(radius=rad).fit(xy)
    ind=nn.radius_neighbors(xy, return_distance=False)
    src=np.where(labels==source)[0]
    out={}
    for t in targets:
        fr=[]
        for i in src:
            nbrs=ind[i]; nbrs=nbrs[nbrs!=i]
            if len(nbrs)==0: continue
            fr.append((labels[nbrs]==t).mean())
        out[t]={'mean_frac': float(np.mean(fr)) if fr else np.nan, 'n_src': int(len(src))}
    # background expected = global freq
    for t in targets:
        out[t]['global_freq']=float((labels==t).mean())
        out[t]['affinity']= out[t]['mean_frac']/(out[t]['global_freq']+1e-9)
    return out

def main():
    print('loading…')
    raw=sc.read_h5ad(ANNOT)
    hypo=pd.read_csv(HYPO, index_col=0)
    # align hypoxia columns
    shared=raw.obs_names.intersection(hypo.index)
    raw=raw[shared].copy()
    for c in ['hypoxia_dynamics','hypoxia_score','v_hypoxia','hypoxia_latent_time','prolif_score']:
        if c in hypo.columns:
            raw.obs[c]=hypo.loc[shared, c].to_numpy()
    adata=lognorm_counts(raw)
    # use spatial from obsm if present
    if 'spatial' in raw.obsm:
        adata.obsm['spatial']=raw.obsm['spatial'].copy()
    else:
        adata.obsm['spatial']=adata.obs[['spatial_coordinate_x','spatial_coordinate_y']].to_numpy()

    findings=[]

    # ---- 1. Composition asymmetry ----
    ct=pd.crosstab(adata.obs['sample'], adata.obs['cell_type'], normalize='index')
    ct.to_csv(OUT/'composition_fraction_by_sample.csv')
    tumor_frac=adata.obs.assign(is_tumor=adata.obs['cell_type'].isin(TUMOR)).groupby('sample')['is_tumor'].mean()
    findings.append({
        'id':'composition_asymmetry',
        'title':'E14S vs E15S sample radically different TME/tumor mix',
        'detail': f"Tumor fraction E14S={tumor_frac.get('E14S',np.nan):.3f}, E15S={tumor_frac.get('E15S',np.nan):.3f}. "
                  f"E14S CD8-exhausted rich; E15S stroma/CAF/tumor rich.",
        'metrics':{'tumor_frac':tumor_frac.to_dict(),
                   'cd8_exh':adata.obs.groupby('sample').apply(lambda d:(d['cell_type']=='CD8 Terminally Exhausted').mean()).to_dict(),
                   'spp1':adata.obs.groupby('sample').apply(lambda d:(d['cell_type']=='Spp1+ TAM').mean()).to_dict()},
        'novelty_score': 6,
    })

    # ---- 2. Program scores ----
    for name,genes in [('hypoxia_mod',HYPOXIA_GENES),('ifn',IFN),('emt',EMT),('oxphos',OXPHOS),
                       ('glycolysis',GLYCOLYSIS),('spp1_prog',SPP1_PROG),('trem2_prog',TREM2_PROG),
                       ('cd8_exh',CD8_EXH),('cd8_eff',CD8_EFF)]:
        score(adata, genes, name)

    # ---- 3. Hypoxia dynamics DE: exiting vs entering ----
    print('DE exiting vs entering…')
    tum=adata.obs['cell_type'].isin(TUMOR).to_numpy()
    exit_m=tum & (adata.obs['hypoxia_dynamics'].astype(str)=='exiting_hypoxia').to_numpy()
    enter_m=tum & (adata.obs['hypoxia_dynamics'].astype(str).isin(['entering_hypoxia','entering_deep_hypoxia'])).to_numpy()
    pers_m=tum & (adata.obs['hypoxia_dynamics'].astype(str)=='persistent_hypoxia').to_numpy()
    rev_m=tum & (adata.obs['hypoxia_dynamics'].astype(str).isin(['reverted_posthypoxic','reverted_stable'])).to_numpy()
    de_ex_en=wilcox_de(adata, exit_m, enter_m)
    de_ex_en.to_csv(OUT/'DE_exiting_vs_entering_tumor.csv', index=False)
    de_rev_pers=wilcox_de(adata, rev_m, pers_m)
    de_rev_pers.to_csv(OUT/'DE_reverted_vs_persistent_tumor.csv', index=False)
    de_pers_enter=wilcox_de(adata, pers_m, enter_m)
    de_pers_enter.to_csv(OUT/'DE_persistent_vs_entering_tumor.csv', index=False)

    # highlight unusual programs in exiting cells
    if not de_ex_en.empty:
        up_exit=de_ex_en[(de_ex_en.lfc_a_minus_b>0.25)&(de_ex_en.padj<0.05)].head(40)
        up_enter=de_ex_en[(de_ex_en.lfc_a_minus_b<-0.25)&(de_ex_en.padj<0.05)].head(40)
        findings.append({
            'id':'exiting_vs_entering_de',
            'title':'Transcriptional divergence of exiting vs entering hypoxia tumor cells',
            'detail':'Top genes elevated in exiting (reverting) vs entering hypoxic tumor cells.',
            'up_exiting': up_exit.gene.tolist(),
            'up_entering': up_enter.gene.tolist(),
            'n_exit':int(exit_m.sum()), 'n_enter':int(enter_m.sum()),
            'novelty_score': 8,
        })

    # ---- 4. Reverted cells: residual IFN / EMT / antigen presentation? ----
    print('reverted programs…')
    prog_cols=['hypoxia_mod','ifn','emt','oxphos','glycolysis','prolif_score' if 'prolif_score' in adata.obs else 'emt']
    # ensure prolif
    if 'prolif_score' not in adata.obs: score(adata,['Mki67','Top2a','Cdk1','Ccnb1'],'prolif_score')
    rows=[]
    for st in ['entering_hypoxia','entering_deep_hypoxia','persistent_hypoxia','exiting_hypoxia','reverted_posthypoxic','reverted_stable','normoxic_stable','transitional']:
        m=tum & (adata.obs['hypoxia_dynamics'].astype(str)==st).to_numpy()
        if m.sum()<10: continue
        row={'state':st,'n':int(m.sum())}
        for p in ['hypoxia_mod','ifn','emt','oxphos','glycolysis','prolif_score']:
            row[p]=float(adata.obs.loc[m,p].mean())
        rows.append(row)
    prog_df=pd.DataFrame(rows).set_index('state')
    prog_df.to_csv(OUT/'tumor_state_program_means.csv')

    # ---- 5. Spatial: do exiting/reverted cells occupy distinct niches? ----
    print('spatial affinities…')
    spatial_results={}
    for sample in ['E14S','E15S']:
        m=(adata.obs['sample'].astype(str)==sample).to_numpy()
        sub=adata[m]
        xy=np.asarray(sub.obsm['spatial'], dtype=float)
        # dynamics labels for tumor; others as cell_type
        lab=sub.obs['cell_type'].astype(str).to_numpy()
        dyn=sub.obs['hypoxia_dynamics'].astype(str).to_numpy()
        # replace tumor labels with dynamics
        lab2=lab.copy()
        for i,ct_ in enumerate(lab):
            if ct_ in TUMOR:
                lab2[i]=dyn[i]
        spatial_results[sample] = {}
        for focus in ['exiting_hypoxia', 'persistent_hypoxia', 'entering_hypoxia', 'reverted_posthypoxic', 'Spp1+ TAM', 'Hypoxic Tumor']:
            flab = lab if focus == 'Spp1+ TAM' else lab2
            if (flab == focus).sum() >= 10:
                spatial_results[sample][focus] = spatial_enrichment(xy, flab, focus, n_perm=300)
        comb = lab.copy()
        for i, ct_ in enumerate(lab):
            if ct_ in TUMOR:
                comb[i] = dyn[i]
        for src in ['persistent_hypoxia', 'exiting_hypoxia', 'entering_hypoxia', 'reverted_posthypoxic']:
            targets = [
                t for t in [
                    'Spp1+ TAM', 'C1qc+ TAM', 'CAF', 'Stromal',
                    'CD8 Terminally Exhausted', 'Neutrophil', 'Il1b+ TAM', 'Resident Macrophage',
                ]
                if (comb == t).any()
            ]
            if (comb == src).sum() >= 10 and targets:
                spatial_results[sample][f'affinity_{src}'] = neighbor_type_affinity(xy, comb, src, targets)

    (OUT/'spatial_results.json').write_text(json.dumps(spatial_results, indent=2, default=str))

    # ---- 6. Spp1 TAM vs hypoxia: which tumor dynamic do they hug? ----
    print('Spp1–hypoxia coupling…')
    spp1_coupling={}
    for sample in ['E14S','E15S']:
        m=(adata.obs['sample'].astype(str)==sample).to_numpy()
        sub=adata[m]
        xy=np.asarray(sub.obsm['spatial'], float)
        lab=sub.obs['cell_type'].astype(str).to_numpy()
        dyn=sub.obs['hypoxia_dynamics'].astype(str).to_numpy()
        comb=lab.copy()
        for i,ct_ in enumerate(lab):
            if ct_ in TUMOR: comb[i]=dyn[i]
        spp1_coupling[sample]=neighbor_type_affinity(
            xy, comb, 'Spp1+ TAM',
            [t for t in ['persistent_hypoxia','exiting_hypoxia','entering_hypoxia','entering_deep_hypoxia','reverted_posthypoxic','reverted_stable','normoxic_stable','transitional'] if (comb==t).any()]
        )
    (OUT/'spp1_hypoxia_coupling.json').write_text(json.dumps(spp1_coupling, indent=2))

    # ---- 7. E14 vs E15 within shared cell types (batch biology) ----
    print('E14 vs E15 within-type DE…')
    within={}
    for ct_ in ['Spp1+ TAM','C1qc+ TAM','Tumor','Hypoxic Tumor','Neutrophil','Resident Macrophage']:
        a=(adata.obs['cell_type'].astype(str)==ct_) & (adata.obs['sample'].astype(str)=='E14S')
        b=(adata.obs['cell_type'].astype(str)==ct_) & (adata.obs['sample'].astype(str)=='E15S')
        if a.sum()<30 or b.sum()<30: continue
        df=wilcox_de(adata, a.to_numpy(), b.to_numpy())
        df.to_csv(OUT/f'DE_{ct_.replace(" ","_").replace("+","pos")}_E14_vs_E15.csv', index=False)
        within[ct_]={
            'n_e14':int(a.sum()),'n_e15':int(b.sum()),
            'top_E14': df[(df.lfc_a_minus_b>0.3)&(df.padj<0.05)].head(15).gene.tolist() if not df.empty else [],
            'top_E15': df[(df.lfc_a_minus_b<-0.3)&(df.padj<0.05)].head(15).gene.tolist() if not df.empty else [],
        }
    (OUT/'within_type_E14_vs_E15.json').write_text(json.dumps(within, indent=2))

    # ---- 8. Novel axis hunt: genes correlating with v_hypoxia independent of hypoxia_score ----
    print('velocity-residual gene screen…')
    tum_idx=np.where(tum)[0]
    rng=np.random.default_rng(0)
    # residualize X for hypoxia_score, correlate with v_hypoxia
    hs=adata.obs['hypoxia_score'].to_numpy(float)[tum_idx]
    vh=adata.obs['v_hypoxia'].to_numpy(float)[tum_idx]
    # pick variable genes
    X=dens(adata.X[tum_idx])
    var=X.var(0)
    genes_idx=np.argsort(-var)[:2500]
    # residualize vh ~ hs
    coef=np.polyfit(hs, vh, 1)
    vh_res=vh - (coef[0]*hs + coef[1])
    corrs=[]
    for j in genes_idx:
        x=X[:,j]
        # residualize gene on hypoxia score
        c2=np.polyfit(hs, x, 1)
        x_res=x-(c2[0]*hs+c2[1])
        if x_res.std()<1e-6: continue
        r=float(np.corrcoef(x_res, vh_res)[0,1])
        corrs.append((adata.var_names[j], r))
    corrs=sorted(corrs, key=lambda z: -abs(z[1]))
    pd.DataFrame(corrs, columns=['gene','corr_v_hypoxia_residual']).to_csv(OUT/'genes_corr_v_hypoxia_independent_of_score.csv', index=False)
    findings.append({
        'id':'velocity_residual_genes',
        'title':'Genes tracking hypoxia velocity beyond hypoxia score',
        'top_pos':[g for g,r in corrs[:25] if r>0],
        'top_neg':[g for g,r in corrs[:50] if r<0][:25],
        'novelty_score': 9,
    })

    # ---- 9. CD8 exhaustion niche in E14S near which hypoxia state? ----
    print('CD8 niche…')
    cd8_aff={}
    for sample in ['E14S','E15S']:
        m=(adata.obs['sample'].astype(str)==sample).to_numpy()
        sub=adata[m]
        xy=np.asarray(sub.obsm['spatial'], float)
        lab=sub.obs['cell_type'].astype(str).to_numpy()
        dyn=sub.obs['hypoxia_dynamics'].astype(str).to_numpy()
        comb=lab.copy()
        for i,ct_ in enumerate(lab):
            if ct_ in TUMOR: comb[i]=dyn[i]
        if (lab=='CD8 Terminally Exhausted').sum()>=10:
            targets=[t for t in np.unique(comb) if t!='CD8 Terminally Exhausted']
            # limit targets
            allow = set(TUMOR) | {
                'persistent_hypoxia', 'exiting_hypoxia', 'entering_hypoxia', 'entering_deep_hypoxia',
                'reverted_posthypoxic', 'Spp1+ TAM', 'C1qc+ TAM', 'Neutrophil', 'CAF', 'Stromal',
                'Tumor', 'Hypoxic Tumor', 'Tumor Proliferating',
            }
            targets = [t for t in targets if t in allow]
            cd8_aff[sample] = neighbor_type_affinity(xy, comb, 'CD8 Terminally Exhausted', targets)
    (OUT/'cd8_spatial_affinity.json').write_text(json.dumps(cd8_aff, indent=2))

    # ---- 10. Plots ----
    print('plots…')
    fig, axes=plt.subplots(2,2, figsize=(11,9), dpi=140)
    # composition
    ax=axes[0,0]
    ct.T.plot(kind='bar', ax=ax, width=0.8)
    ax.set_ylabel('fraction'); ax.set_title('Cell-type composition'); ax.legend(fontsize=6); ax.tick_params(axis='x', labelsize=6)
    # program heatmap-ish
    ax=axes[0,1]
    if not prog_df.empty:
        mat=prog_df[['hypoxia_mod','glycolysis','oxphos','ifn','emt','prolif_score']]
        im=ax.imshow(mat.to_numpy(), aspect='auto', cmap='coolwarm')
        ax.set_yticks(range(len(mat.index))); ax.set_yticklabels(mat.index, fontsize=7)
        ax.set_xticks(range(len(mat.columns))); ax.set_xticklabels(mat.columns, rotation=45, ha='right', fontsize=7)
        ax.set_title('Tumor hypoxia-state programs')
        fig.colorbar(im, ax=ax, fraction=0.04)
    # spatial E15S hypoxia dynamics
    ax=axes[1,0]
    m=adata.obs['sample'].astype(str)=='E15S'
    sub=adata[m]
    xy=np.asarray(sub.obsm['spatial'], float)
    dyn=sub.obs['hypoxia_dynamics'].astype(str).to_numpy()
    is_t=sub.obs['cell_type'].isin(TUMOR).to_numpy()
    ax.scatter(xy[~is_t,0], xy[~is_t,1], s=1, c='#333333', alpha=0.15, linewidths=0)
    pal={'entering_hypoxia':'#2a9d8f','entering_deep_hypoxia':'#1d3557','persistent_hypoxia':'#e76f51',
         'exiting_hypoxia':'#f4a261','reverted_posthypoxic':'#9b5de5','reverted_stable':'#bdb2ff',
         'normoxic_stable':'#8d99ae','transitional':'#cad2c5'}
    for st,c in pal.items():
        sel=is_t & (dyn==st)
        if sel.any():
            ax.scatter(xy[sel,0], xy[sel,1], s=4, c=c, label=st, linewidths=0, alpha=0.85)
    ax.set_aspect('equal'); ax.set_title('E15S spatial · tumor hypoxia dynamics'); ax.legend(fontsize=5, markerscale=2, frameon=False); ax.set_xticks([]); ax.set_yticks([])
    # v_hypoxia residual top genes bar
    ax=axes[1,1]
    top=corrs[:15]
    ax.barh([g for g,_ in top][::-1], [r for _,r in top][::-1], color=['#e76f51' if r>0 else '#2a9d8f' for _,r in top][::-1])
    ax.set_title('corr(gene residual, v_hypoxia residual)'); ax.tick_params(axis='y', labelsize=7)
    fig.tight_layout(); fig.savefig(OUT/'discovery_overview.png', bbox_inches='tight', facecolor='white'); plt.close(fig)

    # E14 spatial CD8 + hypoxia
    fig, ax=plt.subplots(figsize=(7,6), dpi=140)
    m=adata.obs['sample'].astype(str)=='E14S'
    sub=adata[m]; xy=np.asarray(sub.obsm['spatial'], float)
    lab=sub.obs['cell_type'].astype(str).to_numpy()
    dyn=sub.obs['hypoxia_dynamics'].astype(str).to_numpy()
    ax.scatter(xy[:,0], xy[:,1], s=2, c='#2a2a2a', alpha=0.2, linewidths=0)
    for st,c in pal.items():
        sel=(np.isin(lab, list(TUMOR))) & (dyn==st)
        if sel.any(): ax.scatter(xy[sel,0], xy[sel,1], s=6, c=c, linewidths=0, alpha=0.9)
    sel=lab=='CD8 Terminally Exhausted'
    ax.scatter(xy[sel,0], xy[sel,1], s=14, c='#00e5ff', linewidths=0, alpha=0.95, label='CD8 exhausted')
    sel=lab=='Spp1+ TAM'
    ax.scatter(xy[sel,0], xy[sel,1], s=8, c='#ffd166', linewidths=0, alpha=0.8, label='Spp1+ TAM')
    ax.set_aspect('equal'); ax.set_title('E14S · CD8 exhausted (cyan) over tumor hypoxia dynamics'); ax.legend(fontsize=7, frameon=False); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(OUT/'E14S_cd8_hypoxia_spatial.png', bbox_inches='tight', facecolor='white'); plt.close(fig)

    # Save scored obs
    adata.obs[['sample','cell_type','hypoxia_dynamics','hypoxia_score','v_hypoxia','hypoxia_mod','ifn','emt','oxphos','glycolysis','spp1_prog','trem2_prog','cd8_exh','cd8_eff','prolif_score']].to_csv(OUT/'cells_with_programs.csv')

    # Rank findings heuristically from spatial+DE
    # Boost novelty if Spp1 preferentially near exiting or persistent
    for sample,coup in spp1_coupling.items():
        if not coup: continue
        ranked=sorted(coup.items(), key=lambda kv: -kv[1].get('affinity',0))
        findings.append({
            'id':f'spp1_affinity_{sample}',
            'title':f'Spp1+ TAM spatial affinity to hypoxia dynamic states ({sample})',
            'ranked':[{'state':k, **v} for k,v in ranked[:6]],
            'novelty_score': 8 if ranked and ranked[0][0] in ('exiting_hypoxia','reverted_posthypoxic','persistent_hypoxia') else 5,
        })

    (OUT/'findings_raw.json').write_text(json.dumps(findings, indent=2, default=str))
    print(json.dumps(findings, indent=2, default=str)[:4000])
    print('DONE', OUT)

if __name__=='__main__':
    main()
