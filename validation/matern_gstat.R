#!/usr/bin/env Rscript
# Numerical parity check for the continuous-nu Matern model
# (ModelKind::Matern) against gstat's "Ste" (M. Stein's parameterization).
#
# gstat's Ste uses a DIFFERENT Matern "range" convention than this engine:
#   - geostat-rs (Rasmussen & Williams): corr(h) = f(sqrt(2*nu) * h/range)
#   - gstat Ste (Stein 1999):            corr(h) = f(2*sqrt(nu) * h/range)
# where f(s) = 2^(1-nu)/Gamma(nu) * s^nu * K_nu(s). Both are valid, widely
# used parameterizations of the same Matern family; they just disagree on
# what "range" means. The two are related by a *nu-independent* constant:
#   range_rw = range_ste / sqrt(2)
# (verified analytically: sqrt(2*nu) / (2*sqrt(nu)) = 1/sqrt(2) for all nu).
# This script fits gstat's Ste model (kappa fixed, not part of the WLS
# optimization -- matching how this engine's WLS fit only ever estimates
# nugget/sill/range, never nu) and exports both the raw Ste range and the
# already-converted R&W range for compare_matern.py to check directly.
#
# Run from the repo root: Rscript validation/matern_gstat.R

suppressPackageStartupMessages({
  library(sp)
  library(gstat)
})

out <- "validation/out"
dir.create(out, recursive = TRUE, showWarnings = FALSE)
options(digits = 15)

data(meuse)
coordinates(meuse) <- ~ x + y

cutoff <- 1500
width <- 100
kappa <- 1.2

v <- variogram(log(zinc) ~ 1, meuse, cutoff = cutoff, width = width)
vm <- fit.variogram(v, vgm(0.6, "Ste", 900, 0.05, kappa = kappa), fit.kappa = FALSE)

nug <- vm$psill[as.character(vm$model) == "Nug"]
if (length(nug) == 0) nug <- 0
ste <- vm[as.character(vm$model) == "Ste", ]

range_rw <- ste$range / sqrt(2)

write.csv(
  data.frame(
    nugget = nug,
    psill = ste$psill,
    range_ste = ste$range,
    range_rw = range_rw,
    kappa = kappa
  ),
  file.path(out, "gstat_matern_model.csv"),
  row.names = FALSE
)

cat("gstat Ste fit (kappa fixed at", kappa, "):\n")
print(vm)
cat(sprintf("range (Rasmussen & Williams convention) = range_ste / sqrt(2) = %.6f\n", range_rw))
