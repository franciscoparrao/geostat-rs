# geostat-rs — Motor de geoestadística en Rust ("GSLIB moderno")

> **Estado:** v0.5 completa y **validada contra gstat** (2026-06-14).
> v0.1 (variografía/OK/UK/SK/CV/SGS) + v0.2 (co-kriging LMC, KED, SIS,
> anisotropía, kd-tree, benches) + v0.3 (PyO3 bit-idéntico al CLI, WASM +
> demo, block kriging) + v0.4 (core genérico 2-D/3-D, co-kriging heterotópico,
> indicator kriging) + v0.5 (kriging lognormal/trans-gaussiano, block
> co-kriging) — paridad a precisión de máquina en todo lo determinista
> (Meuse + Walker Lake + sintético 3-D); SGS validado distribucionalmente.
> Ver `validation/README.md`. Pendiente: draft del paper.
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
- [x] (v0.2) Co-kriging ordinario con LMC (+ fit Goulard-style con proyección
      PSD), kriging con deriva externa, SIS (sisim-style), anisotropía
      geométrica en modelos, kd-tree (kriging) + bucket grid (simulación),
      benches con criterion. Todo lo determinista validado contra gstat.
- [x] (v0.3) PyO3 (`import geostat_rs`, abi3≥3.9, maturin), WASM
      (wasm-bindgen + demo en `examples/wasm-demo/`), kriging por bloques
      (validado vs gstat; C̄(B,B) sin nugget en puntos coincidentes),
      feature `parallel` para compilar sin rayon en wasm32.
- [x] (v0.4) Core genérico sobre la dimensión (`PointSet<const D>`, kd-tree
      y bucket grid D-dim, anisotropía con `ratio_z`, dirección con dip/cono
      en 3-D, drift polinomial 3-D, `Grid3D`); co-kriging heterotópico (datos
      no colocalizados); indicator kriging standalone (ccdf local + E-type +
      varianza condicional). 3-D e IK expuestos en PyO3. Validado vs gstat a
      precisión de máquina (3-D OK/CV, hetero co-kriging, IK).
- [x] (v0.5) Kriging lognormal/trans-gaussiano (back-transform J&H, multiplicador
      de Lagrange expuesto en `KrigingEstimate`); block co-kriging. Block
      co-kriging valida a 1e-14; lognormal SK a 1e-9 vs gstat (krigeTg usa un
      estimador GLS distinto — ver notas). Lognormal expuesto en PyO3.
- [ ] (futuro) Draft paper; opcional GeoPackage/raster I/O para integrar SurtGIS.

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
5. ~~v0.2: co-kriging, KED, SIS, anisotropía, kd-tree, benches~~ → **hecho**
   (2026-06-11), validado contra gstat a 1e-12 (KED/aniso/co-kriging);
   SIS con tests internos (sills de indicador ≈ p(1-p)).
6. ~~Bindings PyO3 y demo WASM~~ → **hecho** (2026-06-11) + block kriging
   validado a 1e-14. Python: paridad bit a bit con el CLI.
7. ~~v0.4: 3-D, co-kriging heterotópico, indicator kriging~~ → **hecho**
   (2026-06-13). Core genérico sobre la dimensión sin tocar el camino 2-D;
   3-D OK/CV + hetero co-kriging a 1e-14, IK a 7e-10 (clamp de probabilidad
   correcto donde gstat sale de [0,1]). `validation/compare_v04.py`.
8. Draft para Mathematical Geosciences. Material listo: paridad ~10 métodos
   × (Meuse + Walker Lake + sintético 3-D), benchmark 7× vs gstat en SGS,
   reproducibilidad determinista cross-platform, bindings Python/WASM, 2-D y
   3-D desde un solo motor const-genérico.
