# Hypoxia Attractor Landscape (HAL)

Dynamical-systems model of hypoxia and reversion on the phase plane `(h, v)`.

## References

- Zhou et al., *Nat Methods* 2024 — Spatial transition tensor (multistable attractors, basins, transition paths): https://www.nature.com/articles/s41592-024-02266-x
- Mason et al., *Stem Cell Reports* 2025 — Substates and attractors: https://doi.org/10.1016/j.stemcr.2025.102532

## Model

```
dh/dt = v
dv/dt = -U'(h) - γ v
U(h) = (a/4)h⁴ - (b/2)h² + c h + d exp(-λ(h−hₘ)²)
```

Basins from overdamped descent on `U`; soft membership + entropy (STT-style); attractor→attractor flux from velocity-stepped walks.

## Run

```bash
python3 run_hypoxia_attractor_landscape.py
```
