# geostat-rs — Motor de geoestadística en Rust ("GSLIB moderno")

> **Estado:** v0.6 completa (2026-06-15). v0.5 y anteriores validadas contra gstat.
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
- [x] (v0.6) **Puente con TGPY**: kriging con transporte (warped kriging) — núcleo
      marginal de Transport Gaussian Processes (Rios & Tobar). Transportes
      marginales aprendibles (Box-Cox, Yeo-Johnson, sinh-arcsinh) ajustados por ML;
      kriging latente + back-transform Monte Carlo → E-type + std + cuantiles.
      Anclado al lognormal analítico (gstat-validado) a <1%. Nelder-Mead extraído
      a `optim`. **NOTA (2026-06):** el módulo `tgp`, el CLI `tgp` y
      `warped_kriging` fueron **extraídos a un crate privado separado**
      (commit `88ed567`); en este repo solo queda `optim` como legado de v0.6.
- [ ] (futuro) Draft paper; transportes compuestos/SVGD; GeoPackage I/O para SurtGIS;
      aplicar warped kriging a los datos de relaves de TGPY (Dulcinea, El Mauro).

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
9. Auditoría técnica completa (2026-07-02): `docs/AUDIT-2026-07.md` — roadmap
   en 4 fases hacia estado del arte. **Fase 0 ejecutada** (fix block kriging
   3-D, duplicados, metadata/CI, unificación ridge + helpers al core) y
   **Fase 1 ejecutada** (variograma de residuos UK/KED, colas GSLIB
   ltail/utail en SGS/SIS/IK, Goulard–Voltz iterativo, cell declustering +
   nscore ponderado, octantes/ndmin, error de medición estilo gstat Err).
   Paridad gstat extendida: `validation/compare_v07.py` (residuos 3e-15,
   Err 2.5e-14). **Fase 2 ejecutada** (escalabilidad): plan Vecchia
   O(n log n) exacto (maxmin lazy-heap + predecesores incrementales; n=2e5
   en 23 s, antes ~5 min), shell walk por superficie en BucketGrid (H13),
   Cholesky SPD + workspaces en hot loops (loglik 1.74×, MLE 1.6×, mismo
   resultado), predicción Vecchia K&G (`vecchia_predict`, == SK exacto a
   condicionamiento completo; CLI `krige --vecchia`, Python
   `vecchia_krige`), SGS con cuotas ndmax/nodmax y camino multigrid
   (`--nodmax`, `--multigrid`). Walker re-validado distribucionalmente
   (1000 realizaciones en 2.6 s). **Fase 2 completada** (2026-07-02):
   log-parametrización (nugget=x², sill/range=exp(x), sin penalización
   discontinua) + multi-start en los 4 fits Nelder-Mead (`fit_model`,
   `fit_anisotropic_kind`, `vecchia_mle`, `vecchia_reml*`) vía
   `optim::nelder_mead_multistart`; grouping de Guinness (2018) en la
   verosimilitud Vecchia (`vecchia_mle_grouped`, `vecchia_reml_grouped`,
   `vecchia_reml_drift_grouped`, `vecchia_loglik_grouped` — bloques
   adaptativos `guinness_blocks` que comparten una factorización Cholesky
   solo cuando el solape real de vecindarios lo justifica, `group_size<=1`
   reproduce el camino sin agrupar bit a bit, exacto bajo full conditioning
   para cualquier `group_size`); limpieza de covarianzas en los hot loops
   de Vecchia/kriging/SIS/SGS/co-kriging (evitar recomputar `total_sill()`
   por par vía `sill - gamma_dh(..)`, y eliminar la asignación de
   `VariogramModel`+`Vec` por par en `Lmc::gamma_dh`). **Nota honesta**: el
   grouping da una mejora modesta y dependiente de los datos (~1.1–1.4×
   típico en campos uniformes 2-D), no el 2-5× de la cita de Guinness —el
   orden maxmin dispersa deliberadamente los puntos consecutivos por todo
   el dominio, así que el solape real de vecindarios consecutivos suele ser
   bajo; el diseño adaptativo garantiza que nunca sea más lento que el
   camino sin agrupar. Paridad gstat re-validada sin regresiones (Meuse +
   Walker Lake, incl. SGS distribucional). Pendiente (menor): SIMD
   explícito (`std::simd`/crate vectorial) más allá de la limpieza de
   covarianzas escalares hecha ahora.
9. **Fase 3 en curso** (2026-07-02): Matérn ν continuo hecho —
   `ModelKind::Matern(f64)` evalúa la correlación general vía `K_ν` de
   Bessel (integral `∫exp(-x cosh t)cosh(νt)dt`, cuadratura Gauss-Legendre
   compuesta) + Γ (Lanczos), ambos implementados in-house (sin dependencia
   nueva, WASM-friendly) en `variogram::bessel`, validados contra
   `besselK`/`gamma` de R a ≤1e-9 relativo. `Matern15`/`Matern25` (formas
   cerradas) sin cambios; `Matern(1.5)`/`Matern(2.5)` coinciden con ellas a
   ~1e-9. **Hallazgo de paridad no trivial**: gstat's `"Ste"` usa la
   parametrización de Stein (escala `2√ν`) en vez de R&W (`√(2ν)` — la que
   ya usaban `Matern15`/`Matern25` desde siempre, nunca antes validada
   contra gstat); las dos convenciones de `range` difieren por un factor
   constante `√2` independiente de ν (`range_rw = range_ste/√2`, verificado
   analítica y numéricamente a 2e-15). Con esa conversión, el fit WLS
   coincide con gstat a ~5e-7 (mismo orden que el resto de la paridad de
   "optimizadores independientes"). Documentado en el docstring de
   `ModelKind` y en `validation/matern_gstat.R`/`compare_matern.py`. CLI:
   `--fit "matern:1.2"`.

   Mismo día, resto del ítem #16: **familias nuevas** `Circular`
   (`ModelKind::ALL` pasa a 6), `Stable(α)` (power-exponential, α∈(0,2],
   generaliza exponential/gaussian igual que Matérn generaliza vía ν),
   `Hole` y `Wave` (hole-effect cardinal-sine, oscilan más allá del sill —
   excluidas de `ALL` a propósito). Las 4 fórmulas se **derivaron
   numéricamente** contra `variogramLine()` de gstat (`Cir`/`Hol`/`Wav`/`Exc`)
   en vez de confiar en fórmulas de memoria — enfoque forzado por el
   hallazgo Matérn/Ste de la sesión anterior — y coinciden exactas (resid
   ~1e-11 a 1e-15): `Cir` = fórmula clásica del disco; `Hol` = `1-sin(d)/d`;
   `Wav` = `1-sin(πd)/(πd)` (cruce por cero exactamente en `range`); `Exc` =
   `1-exp(-d^κ)`. Tests: `new_families_match_gstat_reference_values`,
   `circular_and_stable_bounded_and_monotone`,
   `hole_and_wave_oscillate_and_are_valid_covariances`. **Anisotropía
   zonal**: se relajó `ratio ≤ 1` a `ratio > 0` (cualquier positivo finito)
   — `effective_h` ya era válido para `ratio > 1` sin cambios (solo
   documentación/validación); `ratio > 1` = el eje ortogonal a `azimuth_deg`
   es el más largo, sin el "truco" de rotar 90° a mano
   (`zonal_anisotropy_ratio_above_one`, verifica simetría espejo exacta con
   azimuth+90°/ratio inverso). **Pesos WLS seleccionables**: `FitWeights`
   {`NPairs`, `Cressie` (N/γ_modelo², autoconsistente, recalculado en cada
   iteración del optimizador), `Ols`, `NOverHSquared` (default, sin cambio)}
   + `fit_model_weighted`; `NPairs` coincide con gstat `fit.method=1` a
   ~0.07% relativo, `Cressie` cae dentro del mismo rango de variabilidad que
   gstat muestra entre corridas con `fit.method=2` (~1-2%, el propio gstat
   no es perfectamente autoconsistente en este esquema no lineal).
   **Rotación 3-D completa**: `Anisotropy` gana `dip_deg`/`rake_deg`
   (GSLIB `ang2`/`ang3`, default 0, ignorados en 2-D) +
   `rotation_matrix_3d` (matriz `setrot` de GSLIB/Deutsch & Journel 1998) +
   `Structure::with_rotation`; cierra la asimetría que señalaba el audit
   (el variograma experimental ya aceptaba dip vía `DirectionConfig`, pero
   el modelo ajustado no podía representarlo). Fórmula verificada contra
   `gstat::variogramLine()` con `anis=c(ang1,ang2,ang3,anis1,anis2)` a
   precisión de máquina (resid ~1e-16) en azimuth-solo/dip-solo/rake-solo/
   combinado, incl. casos límite del branch `ang1≥270°` y ángulos
   negativos-equivalentes (`rotation_3d_matches_gstat_setrot`). Con
   `dip=rake=0` se probó analítica y numéricamente que reduce exacto a la
   fórmula 2-D anterior (`rotation_3d_reduces_to_2d_style...`) — cero
   riesgo para el código 2-D/3-D-sin-dip ya validado.

   **Ajuste conjunto de ν/α por WLS y MLE**: `fit_matern`/`fit_stable`
   (WLS, nugget+sill+range+ν/α optimizados juntos vía Nelder-Mead
   multistart) y `vecchia_mle_matern` (MLE Vecchia, mismo conjunto de
   parámetros). Ambos con la advertencia honesta de la confusión ν-range
   bien conocida en Matérn (un modelo más liso y de rango largo puede
   ajustar casi tan bien como uno más rugoso de rango corto — por eso ν se
   multi-arranca junto al rango, no se trata como parámetro fijo).
   Recuperan ν/α verdaderos desde curvas sintéticas sin ruido
   (`fit_matern_recovers_true_nu_and_range`,
   `fit_stable_recovers_true_alpha_and_range`). **Nota de rendimiento**: el
   primer `vecchia_mle_matern` (9 arranques × 2000 iters, cada evaluación
   de verosimilitud llamando `K_ν` de Bessel por cada par de covarianza)
   tardaba >5 min en debug para n=60/m=12 — impracticable. Se redujo la
   cuadratura de Bessel de 120+80 a 60+40 puntos (error sube de ~1e-9 a
   ~2e-8, todavía muy por debajo de lo que necesita el fitting) y el
   multistart de `vecchia_mle_matern` de 9 a 4 arranques (2000→1200 iters);
   con eso el test pasa en ~50s en debug. Un test de casos especiales tuvo
   que cambiar de tolerancia relativa pura a relativa-o-absoluta (a
   d=0.001 gamma es ~1e-6, así que el error relativo se infla con
   denominador casi cero — artefacto de la métrica, no una regresión
   real). Paridad gstat (Meuse+Matérn fijo) re-confirmada sin cambios tras
   la reducción de cuadratura.

   **Anidamiento multi-estructura**: `fit_nested(exp_v, kinds: &[ModelKind])`
   — nugget + una estructura por elemento de `kinds` (p.ej.
   `&[Spherical, Spherical]` para corto+largo alcance), ajustadas juntas
   por WLS (parametrización log/multistart igual que el resto);
   `kinds.len()==1` es exactamente `fit_model`. Nota honesta igual que
   ν/α: el anidamiento es propenso a óptimos casi-degenerados cuando el
   variograma no necesita en realidad más de una escala (una estructura
   absorbe casi todo el sill) — confirmado que gstat's `fit.variogram`
   tiene el mismo problema (warning "No convergence") al anidar 2
   esféricas sobre el variograma de Meuse, que ya es bien explicado por 1
   sola estructura. Validado por autoconsistencia (recupera 2 estructuras
   verdaderas con escalas bien separadas — corto alcance 50 + largo
   alcance 400 — desde una curva sintética sin ruido:
   `fit_nested_recovers_two_structures`), no por paridad gstat dado ese
   problema de identificabilidad compartido.

   **Con esto el ítem #16 queda cerrado salvo `Power`** (no estacionaria —
   requeriría krigear en forma-γ en vez de forma-covarianza, cambio de
   arquitectura real que tocaría kriging/vecchia/sis/simulation, no solo
   una fórmula nueva; documentado, deliberadamente no implementado esta
   sesión).

10. **Ítem #17 en curso** (2026-07-03): **collocated cokriging MM1/MM2**
    hecho — nuevo módulo `crates/geostat-core/src/collocated.rs`,
    `CollocatedCokriging<D>` (forma simple-kriging, media conocida; la
    forma ordinaria del sistema colocalizado es notoriamente inestable —
    pesos negativos por la ecuación extra de la secundaria — así que GSLIB
    y SGeMS también usan SK por defecto ahí). Resuelve el gap "más
    citable" del audit frente a GSLIB2/SGeMS: predicción condicionada al
    vecindario móvil de la primaria + **un solo** valor de la secundaria
    colocalizado con el target (no necesita la secundaria en cada dato
    primario, viable con secundaria exhaustiva tipo ráster/sísmica).
    `MarkovModel::Mm1` (`C12(h)=ρ12(σ2/σ1)C1(h)`, solo necesita el
    variograma de la primaria) / `Mm2{secondary_model}`
    (`C12(h)=ρ12(σ1/σ2)C2(h)`, necesita el variograma propio de la
    secundaria) — Journel (1999). `estimate_collocated_stats` calcula
    ρ12/σ1/σ2 desde pares colocalizados. Validado por: reduce a SK puro
    con ρ12=0; exactitud en un dato primario propio (con ρ12/σ no
    degenerados); MM1 y MM2 coinciden exactamente cuando C2(h)=k·C1(h) con
    σ2²=k·σ1² (chequeo de consistencia interna entre los dos caminos de
    código); varianza decrece monótonamente con ρ12 creciente.
    **Hallazgo documentado, no bug**: con ρ12=1 y σ1==σ2 exactos, MM1 hace
    C12(h)≡C1(h) — la secundaria se vuelve un duplicado informacional
    exacto de la primaria, el sistema queda casi-singular, y el LU con
    pivoteo parcial da una solución inesperada (todo el peso a la
    secundaria) en vez de fallar; fijado como test de regresión
    (`mm1_perfectly_collinear_secondary_can_be_singular`) — en la práctica
    ρ12 es una correlación muestral y nunca es exactamente 1.0. Sin
    paridad gstat (gstat no tiene collocated cokriging nativo) — validado
    por autoconsistencia y propiedades teóricas, igual que el resto de
    fitting con problemas de identificabilidad conocidos de esta sesión.
    168 tests, clippy limpio. **Sin exponer aún en CLI/Python** (alcance
    de esta sesión: el núcleo del motor).

    **Median/ordinary IK** hecho — `sis.rs`/`ik.rs` compartían ya
    `indicator_sk` (kriging simple de indicador); se dividió en
    `indicator_weights` (la parte cara: arma + factoriza + resuelve el
    sistema) + `indicator_estimate` (barato: producto punto), unificadas en
    `indicator_ccdf`. **Median IK** (GSLIB `mik=1`): `SisConfig::models`/
    `IkConfig::models` ahora aceptan `len()==1` (un solo modelo compartido
    para todos los cutoffs, adivinado automáticamente por longitud, sin
    campo nuevo) en vez de exigir `len()==nc` — cuando hay un solo modelo,
    `indicator_ccdf` factoriza el sistema **una sola vez** por nodo/target
    y reusa los pesos para los `nc` cutoffs, el ahorro ~nc× del hot loop
    que señalaba el audit. Nuevo `fit_median_indicator_model` (fit.rs) para
    ajustar ese único modelo en el cutoff mediano. **Ordinary IK**: nuevo
    campo `ordinary: bool` en ambos configs (default `false`, sin cambiar
    el comportamiento SK previo) — agrega la fila/columna de Lagrange
    (Σw=1) al sistema existente, mismo patrón que `Kriging<D>` ya usa para
    OK. Validado por: **igualdad exacta** median-vs-full IK cuando los
    modelos coinciden (`median_ik_matches_full_ik_when_models_coincide` en
    ik.rs, y el mismo test en sis.rs comparando **realizaciones byte a
    byte** con la misma semilla — prueba fuerte de que el refactor no
    cambia el resultado, solo cómo se calcula); ordinary IK acotado/
    monótono/exacto en datos propios. Expuesto en CLI (`sis`/`ik --mik
    --ordinary`) y Python (`sis(mik=, ordinary=)`, `indicator_kriging(mik=,
    ordinary=)`), probado end-to-end con Meuse. 175 tests, clippy limpio,
    paridad gstat re-confirmada sin regresiones.

    **Markov-Bayes** (mismo día) hecho — la clave fue notar que la
    calibración de Zhu & Journel (1993) es *exactamente* MM1 de collocated
    cokriging (`crate::collocated`) aplicada por-cutoff a un canal de
    probabilidad blanda en vez de una vez a una secundaria continua, así
    que se reusó la misma matemática en vez de inventar una nueva:
    `sis::MarkovBayesCalibration{rho, sigma_soft}` +
    `calibrate_markov_bayes(hard, soft)` (estima ambos por cutoff desde
    pares colocalizados duro+blando, reusando literalmente
    `collocated::estimate_collocated_stats` columna por columna) +
    `ik::indicator_kriging_soft(data, targets, soft, calib, cfg)` — extiende
    el sistema n×n del indicador duro con una fila/columna para el dato
    blando colocalizado en el target, cruzada vía
    `C_IY(h)=ρ·(σ_soft/σ_I)·C_I(h)`. Solo IK estándar (no SIS — llevar
    datos blandos al hot loop secuencial necesitaría muestrear un ráster en
    cada nodo simulado, una abstracción que no existe todavía) y solo
    simple (no ordinary — mismo motivo que collocated cokriging). El ahorro
    ~nc× de median IK NO se hereda aquí: la calibración es propia de cada
    cutoff aunque el modelo duro sea compartido, así que los pesos se
    resuelven cutoff por cutoff igual que full IK. Validado por: ρ=0
    coincide exacto con IK solo-duro; un canal blando informativo (pero
    imperfecto) baja la varianza condicional promedio vs solo-duro;
    `calibrate_markov_bayes` recupera una correlación conocida desde pares
    sintéticos; rechaza `ordinary=true` y dimensiones inconsistentes.
    **Sin exponer en CLI/Python** (igual que collocated cokriging — alcance
    de esta sesión: el núcleo). 179 tests, clippy limpio, paridad gstat
    re-confirmada.

    **Con esto el ítem #17 queda completo.**

11. **`Power` (2026-07-03)**, el único pendiente del ítem #16, hecho —
    `ModelKind::Power(θ)` (θ∈(0,2), `γ(h)=sill·h^θ`, matchea la convención
    de gstat `vgm(psill,"Pow",range)` exactamente, donde su "range" dobla
    como exponente; el campo `range` de `Structure` queda **ignorado** para
    Power, ya que no hay escala de longitud que fijar — sin meseta, sin
    covarianza). La resolución arquitectónica: Power es un IRF-0
    (intrinsic random function de orden 0), y OK/UK se pueden escribir
    **directamente en forma-γ** (`Σw_jγ(i,j)+μ=γ(i,0)`, restricción
    Σw=1) sin necesitar nunca C(0) — la derivación clásica de Cressie
    §3.4.5 / GSLIB `kt3d`. La sustitución resultó más simple de lo
    esperado: **`build_lhs`/`predict_inner` en `kriging.rs` usan la MISMA
    plantilla de código** con un closure `entry(h)` que devuelve `γ(h)`
    (Power) o `c0-γ(h)` (resto) — y la varianza en forma-γ resulta ser
    exactamente `reduction` (sin `c0 -`), derivado y verificado a mano
    contra la fórmula de kriging variance estándar. Kriging simple queda
    **rechazado** para Power (necesita covarianza real, no solo γ; Σw=1 es
    lo que hace funcionar la sustitución) — solo Ordinary/Universal/
    ExternalDrift lo soportan. Block kriging con Power también rechazado
    (necesitaría γ̄(B,B) block-averaged, no implementado). **Validado
    exacto contra gstat**: `krige(z~1, d, target, model=vgm(2.0,"Pow",1.2))`
    en 5 puntos sintéticos reproducido a <1e-6 en valor y varianza
    (`validation/power_gstat.R`, embebido como referencia hardcodeada en
    el test — no hay script `compare_power.py` separado dado lo acotado
    del caso). **Guards explícitos** (rechazo con error claro, no NaN
    silencioso) en cada camino basado en covarianza que es irreconciliable
    con un modelo sin meseta: Vecchia (los 10 puntos de entrada públicos:
    `vecchia_predict/loglik(_grouped)/mle(_grouped)/reml(_grouped)/
    reml_drift(_grouped)/param_se` — Vecchia necesita factorizar Cholesky
    una covarianza real, imposible con varianza infinita), SIS/SGS
    (`sis_at`, `sgs_at_with_levels`), IK (`indicator_kriging`,
    `indicator_kriging_soft`), co-kriging LMC (`Lmc::new`) y collocated
    cokriging (`CollocatedCokriging::new`, ambos MM1/MM2). El ajuste WLS
    (`fit_model`) funciona con Power **sin ningún cambio** — solo toca
    `gamma()`, nunca covarianza — verificado recuperando nugget/pendiente
    desde una curva sintética. **Con esto el ítem #16 queda 100%
    completo.** 189 tests, clippy limpio, paridad gstat re-confirmada.

    Quedan los ítems #18 (block CV espacial, accuracy plots de Deutsch),
    #19 (trait de covarianza, rust-numpy, proptest, publicación
    crates.io/PyPI), y exponer collocated cokriging/Markov-Bayes/block IRF-0
    en CLI/Python (todo deliberadamente diferido, alcance de esta sesión: el
    núcleo del motor).
