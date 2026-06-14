#!/usr/bin/env Rscript
# gstat reference for v0.4: 3-D kriging/CV, heterotopic co-kriging, and
# indicator kriging — all on synthetic / Meuse data with shared models so
# the comparison is engine-vs-engine (machine precision).
#
# Run from the repo root: Rscript validation/v04_gstat.R

suppressPackageStartupMessages({
  library(sp)
  library(gstat)
})

out <- "validation/out"
dir.create(out, recursive = TRUE, showWarnings = FALSE)
options(digits = 15)

model_json_1 <- function(nug, psill, range) {
  sprintf(
    '{"nugget": %.12f, "structures": [{"kind": "spherical", "sill": %.12f, "range": %.12f}]}',
    nug, psill, range
  )
}

## ---- 1. 3-D ordinary kriging + CV --------------------------------------
# Deterministic synthetic 3-D dataset (mirrors the Rust test field).
set.seed(404)
n <- 200
d3 <- data.frame(
  x = runif(n, 0, 100),
  y = runif(n, 0, 100),
  z = runif(n, 0, 40)
)
d3$v <- sin(d3$x / 15) + cos(d3$y / 20) + sin(d3$z / 8) + rnorm(n, sd = 0.05)
write.csv(d3, file.path(out, "synth3d.csv"), row.names = FALSE)

# Fixed isotropic 3-D model, shared with Rust.
nug <- 0.05; psill <- 1.0; rng <- 40.0
writeLines(model_json_1(nug, psill, rng), file.path(out, "model3d.json"))
vm3 <- vgm(psill, "Sph", rng, nug)

d3sp <- d3
coordinates(d3sp) <- ~ x + y + z

# Target grid (8 x 8 x 4 cell centers over the bbox 0..100,0..100,0..40).
gx <- seq(0, 100, length.out = 9); gx <- (head(gx, -1) + tail(gx, -1)) / 2
gy <- gx
gz <- seq(0, 40, length.out = 5); gz <- (head(gz, -1) + tail(gz, -1)) / 2
tg <- expand.grid(x = gx, y = gy, z = gz)
write.csv(tg, file.path(out, "targets3d.csv"), row.names = FALSE)
tgsp <- tg
coordinates(tgsp) <- ~ x + y + z

k3 <- krige(v ~ 1, d3sp, tgsp, model = vm3, debug.level = 0)
write.csv(
  data.frame(x = tg$x, y = tg$y, z = tg$z, pred = k3$var1.pred, var = k3$var1.var),
  file.path(out, "gstat_krige3d.csv"),
  row.names = FALSE
)

cv3 <- krige.cv(v ~ 1, d3sp, model = vm3, verbose = FALSE)
cat(sprintf(
  "3-D CV: ME %.6f RMSE %.6f\n",
  mean(cv3$residual), sqrt(mean(cv3$residual^2))
))
write.csv(
  data.frame(
    x = coordinates(cv3)[, 1], y = coordinates(cv3)[, 2], z = coordinates(cv3)[, 3],
    observed = cv3$observed, pred = cv3$var1.pred, var = cv3$var1.var
  ),
  file.path(out, "gstat_cv3d.csv"),
  row.names = FALSE
)

## ---- 2. Heterotopic ordinary co-kriging --------------------------------
# Primary = log(zinc) at all Meuse points; secondary = log(lead) at a SUBSET
# (heterotopic). Shared LMC (single spherical structure, range 900).
data(meuse)
data(meuse.grid)
meuse$lzinc <- log(meuse$zinc)
meuse$llead <- log(meuse$lead)
prim <- data.frame(x = meuse$x, y = meuse$y, lzinc = meuse$lzinc)
# Secondary subset: every other point.
sec_idx <- seq(1, nrow(meuse), by = 2)
sec <- data.frame(x = meuse$x[sec_idx], y = meuse$y[sec_idx], llead = meuse$llead[sec_idx])
write.csv(prim, file.path(out, "meuse_primary.csv"), row.names = FALSE)
write.csv(sec, file.path(out, "meuse_secondary.csv"), row.names = FALSE)

# Hand-built PSD LMC: direct sills 0.6/0.5, cross 0.5; nugget 0.05/0.04, cross 0.0.
lmc_json <- paste0(
  '{"nugget": [[0.05, 0.0], [0.0, 0.04]], ',
  '"structures": [{"kind": "spherical", "range": 900.0, ',
  '"sills": [[0.6, 0.5], [0.5, 0.5]]}]}'
)
writeLines(lmc_json, file.path(out, "lmc_hetero.json"))

prim_sp <- prim; coordinates(prim_sp) <- ~ x + y
sec_sp <- sec; coordinates(sec_sp) <- ~ x + y
g <- gstat(NULL, id = "lz", formula = lzinc ~ 1, data = prim_sp)
g <- gstat(g, id = "ll", formula = llead ~ 1, data = sec_sp)
# Inject the shared LMC explicitly (no fitting).
g$model$lz <- vgm(0.6, "Sph", 900, 0.05)
g$model$ll <- vgm(0.5, "Sph", 900, 0.04)
g$model$lz.ll <- vgm(0.5, "Sph", 900, 0.0)

gm <- as.matrix(meuse.grid[, c("x", "y")])
mg_sp <- data.frame(x = meuse.grid$x, y = meuse.grid$y)
coordinates(mg_sp) <- ~ x + y
ck <- predict(g, mg_sp, debug.level = 0)
write.csv(
  data.frame(x = gm[, 1], y = gm[, 2], pred = ck$lz.pred, var = ck$lz.var),
  file.path(out, "gstat_cokrige_hetero.csv"),
  row.names = FALSE
)
cat("heterotopic co-kriging written\n")

## ---- 3. Indicator kriging ----------------------------------------------
# Single cutoff (median of log(zinc)); compare F(cutoff) at grid nodes.
zc <- median(meuse$lzinc)
writeLines(sprintf("%.12f", zc), file.path(out, "ik_cutoff.txt"))
ind <- as.numeric(meuse$lzinc <= zc)
ik_df <- data.frame(x = meuse$x, y = meuse$y, ind = ind, lzinc = meuse$lzinc)
write.csv(
  data.frame(x = meuse$x, y = meuse$y, lzinc = meuse$lzinc),
  file.path(out, "meuse_ik.csv"),
  row.names = FALSE
)
# Indicator variogram model (shared); use simple kriging with the global
# proportion as the mean (matches geostat-rs indicator SK).
p <- mean(ind)
writeLines(model_json_1(0.05, 0.18, 700.0), file.path(out, "ik_model.json"))
vm_ind <- vgm(0.18, "Sph", 700, 0.05)
ik_sp <- ik_df; coordinates(ik_sp) <- ~ x + y
ik <- krige(ind ~ 1, ik_sp, mg_sp, model = vm_ind, beta = p, debug.level = 0)
write.csv(
  data.frame(x = gm[, 1], y = gm[, 2], F = ik$var1.pred),
  file.path(out, "gstat_ik.csv"),
  row.names = FALSE
)
cat(sprintf("indicator kriging written (cutoff %.4f, global p %.4f)\n", zc, p))
