# E15S reseq incorporation + annotation fine-tune

## What changed

Only **E15S** was resequenced (`/ix1/ylee/shared/ResequencedE15S_E27_E28S/…/E15S_GEX_S2_*`). E14S is unchanged.

| Matrix | Cells | Median UMI |
|--------|------:|-----------:|
| E15S original (Apr CR, in `E15S/`) | 8,669 | ~3,977 |
| E15S reseq (Palak `cellranger multi`, S2) | 9,167 | ~12,542 |

All original E15S barcodes are contained in the reseq call; **+498** new cells.

## Outputs

```
reseq_finetune/
  mc38_tumor_reseq_finetuned.h5ad   # E14S + reseq E15S, full GEX
  annotations_reseq_finetuned.csv
  summary.json
  run.log
```

Obs columns of interest:

- `cell_type` — curated labels transferred by barcode + kNN for new cells (sticky)
- `cell_type_finetuned` — conservative tumor subtype tweaks from hypoxia/prolif scores on deeper counts
- `annotation_source` — `transferred` | `knn` | `leiden_majority`
- `score_hypoxia`, `score_prolif`, `score_epi`, …

## Reproduce

```bash
cd /ix1/ylee/shared/MC38_Hypoxia_001
python3 incorporate_e15s_reseq_and_finetune.py
```

## Notes

- Palak CR used **S2 only** (not original S3). simpleleaf already merged S2+S3 under `organized_experiments/simpleleaf_runs/E15S/`.
- Original `mc38_tumor.h5ad` / `finer_annot.h5ad` are left untouched.
