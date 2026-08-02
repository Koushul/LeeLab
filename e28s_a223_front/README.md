# E28S A223 · Hypoxia front (MC38 E14/E15 analogue)

Same geometric tests as the E14S/E15S ImageIT front analysis, applied to **E28S A223** hypoxia GFP reporter tumors.

## Design

| Rule | Detail |
|------|--------|
| Chips | **Edge** and **core** are separate physical chips — never overlaid |
| Gates | Within each slide, **GFP− / GFP+** share the array (E14/E15 analogue) |
| Filter | ADT microwell entropy `H_adt ≤ 8` (~86% cells) |
| Coords | MAP `map_row_adt × map_col_adt` (48×48) |
| Dynamics | scVelo stochastic velocity → hypoxia phase-plane states |

## Inputs

- Annotated object: `…/E28S_master_analysis/data/E28S_final.h5ad`
- Entropy: `…/spac_analysis/out/e28s_adt_entropy5_merci_spp1/cell_metadata_adtH5.csv`

## Run

```bash
cd /ix1/ylee/shared/MC38_Hypoxia_001/e28s_a223_front
python3 run_e28s_front_pipeline.py
```

## Outputs

- `data/hypoxia_dynamics_per_cell.csv`, `microwell_entropy_per_cell.csv`, `e28s_h8_front.h5ad`, `front_summary.json`
- `front_figures/` bridges, ribbons, shuffle panels, pulse GIFs
- `index.html` report site

## Headline (H≤8 overlay)

See `data/front_summary.json`. With only ~430 tumor cells after H≤8, shuffle power is limited versus E14/E15.
