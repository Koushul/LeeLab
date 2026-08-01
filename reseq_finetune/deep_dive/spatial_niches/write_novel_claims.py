#!/usr/bin/env python3
"""Write novel spatial biology synthesis from H≤8 dig + literature framing."""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(
    "/ix1/ylee/shared/MC38_Hypoxia_001/reseq_finetune/deep_dive/spatial_niches/front_figures"
)

claims = {
    "filter": "entropy_bits<=8 MAP microwells; E14S+E15S same-chip overlay",
    "novel_claims": [
        {
            "id": "velocity_front",
            "claim": "Hypoxia dynamics form a contiguous spatial front (enter abut exit), not isolated HIF islands.",
            "evidence": "H≤8 overlay: deep→exit enrichment 1.25× (p≈0.06); enter↔exit contact 64–72%; Moran's I(front)≈0.66, Moran's I(polarity)≈0.64.",
            "literature_context": "Classical models describe diffusion-limited O2 gradients (~100–200 µm from vessels) and discrete hypoxic niches; recent ST work maps hypoxia gene gradients and myeloid niches in GBM. Velocity-resolved enter vs exit as opposite faces of one ribbon is a finer dynamical framing than static hypoxia scores.",
            "novelty": "high",
            "citations": [
                "https://ecancer.org/en/news/4081-scientists-define-a-new-mechanism-leading-to-tumour-hypoxia",
                "https://pmc.ncbi.nlm.nih.gov/articles/PMC11100569/",
                "https://www.nature.com/articles/s41467-024-50904-x",
            ],
        },
        {
            "id": "marker_vs_transcription",
            "claim": "ImageIT/GFP fate mark ≠ ongoing transcriptional hypoxia; most marked cells have exited or reverted.",
            "evidence": "E15S=GFP+/ImageIT+ by design, yet only a minority of tumor dynamics cells are velocity-persistent (~12% in earlier census); H≤8 overlay still shows abundant normoxic/exit states in GFP+ pool.",
            "literature_context": "Hypoxia fate-mapping (DsRed→GFP) shows post-hypoxic cells invade oxygenated regions and retain hypoxia memory / ROS resistance after reoxygenation — marker persistence without current hypoxia.",
            "novelty": "high_in_this_system",
            "citations": [
                "https://www.nature.com/articles/s41467-019-12412-1",
                "https://pubmed.ncbi.nlm.nih.gov/39341835/",
            ],
        },
        {
            "id": "entry_neq_exit",
            "claim": "Entering and exiting are molecularly asymmetric and spatially polarized sides of the same front.",
            "evidence": "Prior DE: entry↑ ISR/IEG/AA (Ppp1r15a, Jun/Jund, Slc38a2, Chka, Neat1); exit↑ dual OXPHOS+glycolysis reboot + ribosomes — not a simple glycolysis↔OXPHOS flip. Spatial polarity (local exit−enter) is strongly autocorrelated (Moran's I≈0.64).",
            "literature_context": "Standard hypoxia metabolism emphasizes HIF-driven glycolysis and OXPHOS repression; reoxygenation can restore OXPHOS. Dual metabolic reboot on exit is a sharper, velocity-defined observation than the textbook one-way Warburg shift.",
            "novelty": "high",
            "citations": [
                "https://www.mdpi.com/2073-4409/9/12/2598",
                "https://pmc.ncbi.nlm.nih.gov/articles/PMC6362613/",
            ],
        },
        {
            "id": "persist_off_ribbon",
            "claim": "Persistent hypoxia occupies the dwell/core off the enter×exit ribbon.",
            "evidence": "H≤8 overlay: persist on top-20% front ribbon 18% vs enter/exit ~33–35%; lower front score than enter (MW p≪1e-4).",
            "literature_context": "Hypoxic/necrotic cores and SPP1+ TAM enrichment in necrosis are established; here the core is defined by velocity-persistent state rather than histology alone.",
            "novelty": "medium",
            "citations": [
                "https://doi.org/10.1158/2159-8290.cd-23-1300",
                "https://doi.org/10.3390/biomedicines14020294",
            ],
        },
        {
            "id": "immune_front_architecture",
            "claim": "The oxygen interface has a stereotyped immune skirt: CAF+neutrophil enriched, CD8 and C1qc-TAM depleted; Spp1-TAM mildly front/exit-biased.",
            "evidence": "H≤8 overlay zone enrichments (perm p): front CAF 1.50 (p=0), Neutrophil 1.29 (p=0), CD8 0.64 (p=0.022), C1qc-TAM 0.77 (p=0); Spp1-TAM front 1.14 (p=0.04). Shared-well Fisher: E15 hypoxic tumor wells co-occur with E14 neutrophils OR=1.86 p=4.6e-5.",
            "literature_context": "CD8 often peaks away from hypoxic cores; SPP1+ TAMs mark hypoxic/necrotic niches; neutrophils reprogram into hypoxic-core states (dcTRAIL-R1+/T3) and can drive perinecrotic biology. Cross-gate neutrophil–hypoxia well co-occurrence on one chip is a new experimental angle.",
            "novelty": "high",
            "citations": [
                "https://www.nature.com/articles/s41467-024-50904-x",
                "https://doi.org/10.1158/2159-8290.cd-23-1300",
                "https://doi.org/10.1126/science.adf6493",
                "https://www.nature.com/articles/s41586-025-09278-3",
            ],
        },
    ],
    "falsifiable_next": [
        "smFISH for entry (Ppp1r15a/Slc38a2) vs exit (OXPHOS+glycolysis) genes along the front ribbon",
        "Neutrophil state scoring (T3/hypoxic program) on the ribbon vs parenchyma",
        "Recompute fronts after dropping wells with MAP col/row edge residuals",
    ],
}

(OUT / "h8_novel_spatial_claims.json").write_text(json.dumps(claims, indent=2))
print("wrote", OUT / "h8_novel_spatial_claims.json")
