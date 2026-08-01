# MC38 Hypoxia (E14S + E15S)

XYZeq / SPAC-seq style MC38 tumor libraries:

| Library | Role |
|---------|------|
| **E14S** | Sample 1 (batch 0) — original GEX+ADT Cell Ranger |
| **E15S** | Sample 2 (batch 1) — **resequenced** GEX (Jul 2026 S2) |

Combined curated object (original depth): `mc38_tumor.h5ad`, `finer_annot.h5ad`, `annotations.csv`.

## Reseq + fine-tuned annotations

See [`reseq_finetune/README.md`](reseq_finetune/README.md).

```bash
python3 incorporate_e15s_reseq_and_finetune.py
```

Produces `reseq_finetune/mc38_tumor_reseq_finetuned.h5ad` (14,459 cells = E14S 5,292 + reseq E15S 9,167).

## Legacy MALT workflow

Scripts restored from git history:

1. `prepare_mouse_colon_reference_gse193342.py`
2. `export_mc38_query_for_malt.py`
3. `marker_label_transfer.py`
4. `annotate_mc38_tumor_compartment.py`
5. `malt_annotation_workflow.ipynb`
