# geostat-rs (Python bindings)

Python bindings for [geostat-rs](https://github.com/franciscoparrao/geostat-rs), a
geostatistics engine written in Rust: variography, kriging (simple, ordinary,
universal, external drift, block, lognormal), co-kriging, indicator kriging,
Vecchia-approximated large-scale fitting/prediction, and sequential
(Gaussian/indicator) simulation.

```python
import geostat_rs as gr

model = gr.fit_variogram(x, y, values, n_lags=15)
predictions, variances = gr.krige_grid(
    x, y, values, model, bbox=(xmin, ymin, xmax, ymax), nx=100, ny=100,
)
```

Array-shaped results (grid predictions, simulation realizations,
cross-validation arrays) are returned as numpy arrays. See the project
repository for the full API and validation against `gstat`.
