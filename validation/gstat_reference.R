#!/usr/bin/env Rscript
# Reference results from gstat (R) for numerical parity validation.
#
# Exports, for log(zinc) of the classic Meuse dataset:
#   - the dataset itself (meuse_lzinc.csv)
#   - experimental variogram with cutoff 1500 / width 100 (15 lags)
#   - a spherical model fitted by gstat (also as geostat-rs JSON)
#   - ordinary kriging predictions + variances on meuse.grid (global nbhd)
#   - leave-one-out cross-validation results
#
# Run from the repo root: Rscript validation/gstat_reference.R

suppressPackageStartupMessages({
  library(sp)
  library(gstat)
})

out <- "validation/out"
dir.create(out, recursive = TRUE, showWarnings = FALSE)
options(digits = 15)

data(meuse)
data(meuse.grid)

meuse_df <- data.frame(x = meuse$x, y = meuse$y, lzinc = log(meuse$zinc))
write.csv(meuse_df, file.path(out, "meuse_lzinc.csv"), row.names = FALSE)

coordinates(meuse) <- ~ x + y

# --- 1. Experimental variogram -------------------------------------------
cutoff <- 1500
width <- 100
v <- variogram(log(zinc) ~ 1, meuse, cutoff = cutoff, width = width)
write.csv(
  data.frame(np = v$np, dist = v$dist, gamma = v$gamma),
  file.path(out, "gstat_vario.csv"),
  row.names = FALSE
)

# --- 2. Model fit (gstat WLS, default fit.method = 7: N_j/h_j^2) ---------
vm <- fit.variogram(v, vgm(0.6, "Sph", 900, 0.05))
write.csv(
  data.frame(model = as.character(vm$model), psill = vm$psill, range = vm$range),
  file.path(out, "gstat_model.csv"),
  row.names = FALSE
)
nug <- vm$psill[as.character(vm$model) == "Nug"]
if (length(nug) == 0) nug <- 0
sph <- vm[as.character(vm$model) == "Sph", ]
writeLines(
  sprintf(
    '{"nugget": %.12f, "structures": [{"kind": "spherical", "sill": %.12f, "range": %.12f}]}',
    nug, sph$psill, sph$range
  ),
  file.path(out, "gstat_model.json")
)

# --- 3. Ordinary kriging on meuse.grid, global neighborhood --------------
coordinates(meuse.grid) <- ~ x + y
k <- krige(log(zinc) ~ 1, meuse, meuse.grid, model = vm, debug.level = 0)
write.csv(
  data.frame(
    x = coordinates(meuse.grid)[, 1],
    y = coordinates(meuse.grid)[, 2],
    pred = k$var1.pred,
    var = k$var1.var
  ),
  file.path(out, "gstat_krige.csv"),
  row.names = FALSE
)

# --- 4. Leave-one-out cross-validation ------------------------------------
cv <- krige.cv(log(zinc) ~ 1, meuse, model = vm, verbose = FALSE)
write.csv(
  data.frame(
    x = coordinates(cv)[, 1],
    y = coordinates(cv)[, 2],
    observed = cv$observed,
    pred = cv$var1.pred,
    var = cv$var1.var
  ),
  file.path(out, "gstat_cv.csv"),
  row.names = FALSE
)

cat("gstat reference written to", out, "\n")
cat("fitted model:\n")
print(vm)
