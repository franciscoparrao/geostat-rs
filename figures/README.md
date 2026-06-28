# Paper figures

Publication-quality figures for the geostat-rs paper (publishable methods only —
no transport/warped kriging). Two-step pipeline:

```sh
# 1. gstat references (once) — needed by the parity figure
Rscript validation/gstat_reference.R
Rscript validation/idw_gstat.R

# 2. generate figure data, then render
PYTHONPATH=<dir with geostat_rs.so> python3 figures/make_figure_data.py
Rscript figures/figures.R          # writes figures/*.pdf
```

Figures:
- `fig_parity.pdf` — geostat-rs vs gstat (OK and IDW) on the Meuse grid;
  machine-precision agreement (the validation centrepiece).
- `fig_compare.pdf` — leave-one-out VEcv by method on Meuse log-zinc.
- `fig_idw_tune.pdf` — IDW power tuned by predictive accuracy (VEcv).
- `fig_anisotropy.pdf` — Meuse log-zinc anisotropy: the 2-D variogram map plus
  directional variograms along the fitted major/minor axes with the fitted
  anisotropic model (recovers the NE-SW river-floodplain continuity). Needs the
  `variogram_map` and `fit_anisotropic` Python bindings (rebuild the module).
- `fig_multielement.pdf` — rare-earth grade prediction by element × method
  (covariates + ML-geostat hybrid). Uses the public geochemical registry of
  tailings deposits in Chile (SERNAGEOMIN, 2023).
- `fig_wasm.png` — screenshot of the WebAssembly build running in the browser
  (a committed static asset, not produced by the pipeline). To regenerate:
  `wasm-pack build crates/geostat-wasm --target web --release`, serve the repo
  root (`python3 -m http.server`), open `/examples/wasm-demo/` and screenshot.

Generated `data/*.csv`, `*.pdf` and other `*.png` are git-ignored; rerun to
rebuild. `fig_wasm.png` is the committed exception.
