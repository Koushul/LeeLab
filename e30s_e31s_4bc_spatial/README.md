# E30S / E31S 4-barcode ADT entropy spatial placement

MC38 spatial-hash chips (same 48×48 / 4-barcode layout as E28S).

## Inputs

| Dataset | Annotated object | Spatial lanes |
|---------|------------------|---------------|
| E30S (OCM1) | `spatial_crispr/.../E30S_gex_adt_guide_ocm.h5ad` | All OB1–OB4 |
| E31S (OCM2) | `spatial_crispr/.../E31S_gex_adt_guide_ocm.h5ad` | OB1–OB2 only (`spatial=True`) |

E31S OB3/OB4 are pooled non-spatial perturb-seq and are **excluded**.

Chips (never overlay across chips):

- `v11_spatial_chip`: E30S OB1/OB2 + E31S OB1
- `v70_spatial_chip`: E30S OB3/OB4 + E31S OB2

## Algorithm

Same as E28S front / `spac_analysis` script 16:

1. Take ADT `sbc1…sbc192`
2. `row = sbc1–48 + sbc145–192`, `col = sbc49–96 + sbc97–144`
3. Shannon entropy `H_adt = H(row) + H(col)` (bits)
4. MAP well = `(argmax(row), argmax(col))` (1-indexed)
5. Default place filter: **H_adt ≤ 8**

Also reports the soft multinomial 4-BC posterior (`export_cells.py` / lucid-crystal site) as `H_4bc` / `map_*_4bc` for comparison.

## Run

```bash
PY=/ix1/ylee/kor11/tools/af_tutorial/conda_env/bin/python
$PY run_place_spatial.py
```

## Outputs

- `data/*_assignments.csv` — per-cell MAP + entropy
- `data/placement_summary.json`
- `figures/` — entropy histograms + MAP scatters by chip
- `data/*_placed.h5ad` — spatial subset with `obsm['spatial']` (gitignored)
