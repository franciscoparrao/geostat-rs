# geostat-rs — Motor de geoestadística en Rust ("GSLIB moderno")

> **Estado:** MVP v0.1 implementado y **validado contra gstat** (2026-06-10):
> paridad a precisión de máquina en variograma/OK/CV sobre **Meuse y Walker Lake**,
> y SGS estadísticamente indistinguible (1000 realizaciones, ensemble vs ensemble).
> Ver `validation/README.md`. Pendiente: PyO3, WASM, paper.
> Familia de motores Rust del autor: SurtGIS, Hydroflux, Smelt, Anvil, Cantus, Criterium.
> Doc madre: `~/proyectos/ideas-motores-rust.md` (idea A1).

## Qué es
Librería + CLI en Rust para geoestadística completa: variografía, kriging y
simulación estocástica. Single-binary, sin dependencias pesadas.

## El gap que llena
SurtGIS tiene kriging básico pero **sin variografía ni simulación**. El campo
hoy se reparte entre **gstat** (R, lento) y **GSLIB** (Fortran, arcaico, sin
bindings modernos). No hay un motor geoestadístico Rust con WASM/Python.

## Alcance MVP (v0.1)
- [x] Variograma experimental (isotrópico/anisotrópico) + ajuste de modelos
      (esférico, exponencial, gaussiano, Matérn ν=3/2 y 5/2).
- [x] Kriging simple, ordinario y universal (drift lineal/cuadrático).
- [x] Cross-validation (LOO) y mapas de varianza de kriging.
- [x] Simulación secuencial gaussiana (SGS) → realizaciones + incertidumbre.
      RNG determinista (xoshiro256++), reproducible cross-platform.
- [ ] (v0.2) Co-kriging, kriging con deriva externa, SIS, anisotropía en
      modelos, kd-tree para búsqueda de vecinos, benches con criterion.

## Arquitectura tentativa
- `geostat-core`: variograma, sistemas de kriging, RNG determinista.
- Targets: native (Rayon) + Python (PyO3) + CLI; WASM como demo.
- I/O: reusar el lector raster/punto de SurtGIS o CSV/GeoPackage.

## Validación / paridad numérica
Cross-check contra **gstat** (R) y datasets clásicos (Meuse, Walker Lake).

## Venue objetivo
**Mathematical Geosciences** (IAMG) — encaja con perfil geomatemático. Alt:
Computers & Geosciences.

## Conexiones con tu ecosistema
- **SurtGIS**: reemplaza/expande su módulo de interpolación.
- **Smelt**: cuantificación de incertidumbre espacial para ML.

## Próximos pasos al retomar
1. ~~Decidir si es crate standalone o módulo `surtgis geostat`~~ → **standalone** (2026-06-10).
2. ~~Exportar Meuse y validar contra gstat~~ → **hecho** (2026-06-10): paridad
   1e-12 en OK/CV, bins exactos, ajuste coincide a 1e-6. `validation/compare.py`.
3. ~~git init + primer commit~~ → repo en https://github.com/franciscoparrao/geostat-rs (privado).
4. ~~Validación Walker Lake + SGS distribucional~~ → **hecho** (2026-06-10):
   paridad 1e-12 en OK/CV; SGS a 1000 realizaciones pasa todos los chequeos
   distribucionales (`validation/compare_walker.py`). geostat-rs 7× más
   rápido que gstat en SGS. Caveat documentado: despiking de empates antes
   de SGS es responsabilidad del usuario (práctica GSLIB).
5. v0.2: co-kriging, KED, SIS, anisotropía en modelos, kd-tree, benches criterion.
6. Bindings PyO3 y demo WASM; luego draft para Mathematical Geosciences.
