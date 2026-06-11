#!/usr/bin/env Rscript
# gstat reference for the v0.2 features, on Meuse:
#   1. Kriging with external drift: log(zinc) ~ sqrt(dist).
#   2. Ordinary kriging with a geometrically anisotropic model
#      (fixed model on both sides: 0.05 Nug + 0.55 Sph(900), anis 30°/0.5).
#   3. Ordinary co-kriging log(zinc) + log(lead) with an LMC from fit.lmc
#      (the fitted LMC is exported as JSON and used by both engines).
#
# Run from the repo root: Rscript validation/v02_gstat.R

suppressPackageStartupMessages({
  library(sp)
  library(gstat)
})

out <- "validation/out"
dir.create(out, recursive = TRUE, showWarnings = FALSE)
options(digits = 15)

data(meuse)
data(meuse.grid)

meuse$lzinc <- log(meuse$zinc)
meuse$llead <- log(meuse$lead)
meuse$sdist <- sqrt(meuse$dist)
write.csv(
  data.frame(
    x = meuse$x, y = meuse$y,
    lzinc = meuse$lzinc, llead = meuse$llead, sdist = meuse$sdist
  ),
  file.path(out, "meuse_multi.csv"),
  row.names = FALSE
)
meuse.grid$sdist <- sqrt(meuse.grid$dist)
write.csv(
  data.frame(x = meuse.grid$x, y = meuse.grid$y, sdist = meuse.grid$sdist),
  file.path(out, "grid_targets.csv"),
  row.names = FALSE
)

coordinates(meuse) <- ~ x + y
coordinates(meuse.grid) <- ~ x + y
gxy <- coordinates(meuse.grid)

model_json <- function(vm) {
  nug <- vm$psill[as.character(vm$model) == "Nug"]
  if (length(nug) == 0) nug <- 0
  sph <- vm[as.character(vm$model) == "Sph", ]
  sprintf(
    '{"nugget": %.12f, "structures": [{"kind": "spherical", "sill": %.12f, "range": %.12f}]}',
    nug, sph$psill, sph$range
  )
}

# --- 1. Kriging with external drift --------------------------------------
v <- variogram(lzinc ~ sdist, meuse, cutoff = 1500, width = 100)
vmk <- fit.variogram(v, vgm(0.5, "Sph", 900, 0.05))
writeLines(model_json(vmk), file.path(out, "gstat_ked_model.json"))
k <- krige(lzinc ~ sdist, meuse, meuse.grid, model = vmk, debug.level = 0)
write.csv(
  data.frame(x = gxy[, 1], y = gxy[, 2], pred = k$var1.pred, var = k$var1.var),
  file.path(out, "gstat_ked.csv"),
  row.names = FALSE
)
cat("KED model:\n"); print(vmk)

# --- 2. Anisotropic ordinary kriging (fixed model) ------------------------
vma <- vgm(0.55, "Sph", 900, 0.05, anis = c(30, 0.5))
ka <- krige(lzinc ~ 1, meuse, meuse.grid, model = vma, debug.level = 0)
write.csv(
  data.frame(x = gxy[, 1], y = gxy[, 2], pred = ka$var1.pred, var = ka$var1.var),
  file.path(out, "gstat_aniso.csv"),
  row.names = FALSE
)
writeLines(
  paste0(
    '{"nugget": 0.05, "structures": [{"kind": "spherical", "sill": 0.55, ',
    '"range": 900.0, "anis": {"azimuth_deg": 30.0, "ratio": 0.5}}]}'
  ),
  file.path(out, "aniso_model.json")
)

# --- 3. Ordinary co-kriging with fit.lmc -----------------------------------
g <- gstat(NULL, id = "lz", formula = lzinc ~ 1, data = meuse)
g <- gstat(g, id = "ll", formula = llead ~ 1, data = meuse)
vg <- variogram(g, cutoff = 1500, width = 100)
g <- gstat(g, model = vgm(0.5, "Sph", 900, 0.05), fill.all = TRUE)
gf <- fit.lmc(vg, g)
cat("LMC models:\n"); print(gf$model)

psill_of <- function(m, kind) m$psill[as.character(m$model) == kind]
n_lz <- psill_of(gf$model$lz, "Nug")
s_lz <- psill_of(gf$model$lz, "Sph")
n_ll <- psill_of(gf$model$ll, "Nug")
s_ll <- psill_of(gf$model$ll, "Sph")
n_x <- psill_of(gf$model$lz.ll, "Nug")
s_x <- psill_of(gf$model$lz.ll, "Sph")
rng <- gf$model$lz$range[as.character(gf$model$lz$model) == "Sph"]
writeLines(
  sprintf(
    paste0(
      '{"nugget": [[%.12f, %.12f], [%.12f, %.12f]], ',
      '"structures": [{"kind": "spherical", "range": %.12f, ',
      '"sills": [[%.12f, %.12f], [%.12f, %.12f]]}]}'
    ),
    n_lz, n_x, n_x, n_ll, rng, s_lz, s_x, s_x, s_ll
  ),
  file.path(out, "gstat_lmc.json")
)

ck <- predict(gf, meuse.grid, debug.level = 0)
write.csv(
  data.frame(x = gxy[, 1], y = gxy[, 2], pred = ck$lz.pred, var = ck$lz.var),
  file.path(out, "gstat_cokrige.csv"),
  row.names = FALSE
)
cat("v0.2 gstat reference written to", out, "\n")
