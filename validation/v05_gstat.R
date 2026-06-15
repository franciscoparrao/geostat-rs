#!/usr/bin/env Rscript
# gstat reference for v0.5: ordinary lognormal kriging and block co-kriging,
# both on Meuse with shared models so the comparison is engine-vs-engine.
#
# Run from the repo root: Rscript validation/v05_gstat.R

suppressPackageStartupMessages({
  library(sp)
  library(gstat)
})

out <- "validation/out"
dir.create(out, recursive = TRUE, showWarnings = FALSE)
options(digits = 15)

data(meuse)
data(meuse.grid)
gm <- as.matrix(meuse.grid[, c("x", "y")])
mg_sp <- data.frame(x = meuse.grid$x, y = meuse.grid$y)
coordinates(mg_sp) <- ~ x + y

## ---- 1. Simple lognormal kriging (clean back-transform oracle) ----------
# SK of log(zinc) with a known mean beta; the lognormal back-transform is
# exp(y + sigma2/2) with no Lagrange term, so this is an unambiguous oracle
# for the back-transform machinery. The model of log(zinc) is shared.
write.csv(
  data.frame(x = meuse$x, y = meuse$y, zinc = meuse$zinc),
  file.path(out, "meuse_zinc.csv"),
  row.names = FALSE
)
m_sp <- meuse
coordinates(m_sp) <- ~ x + y
v <- variogram(log(zinc) ~ 1, m_sp, cutoff = 1500, width = 100)
vm <- fit.variogram(v, vgm(0.6, "Sph", 900, 0.05))
nug <- vm$psill[as.character(vm$model) == "Nug"]; if (length(nug) == 0) nug <- 0
sph <- vm[as.character(vm$model) == "Sph", ]
writeLines(
  sprintf(
    '{"nugget": %.12f, "structures": [{"kind": "spherical", "sill": %.12f, "range": %.12f}]}',
    nug, sph$psill, sph$range
  ),
  file.path(out, "logzinc_model.json")
)

beta <- mean(log(meuse$zinc))
writeLines(sprintf("%.12f", beta), file.path(out, "logzinc_beta.txt"))
sk <- krige(log(zinc) ~ 1, m_sp, mg_sp, model = vm, beta = beta, debug.level = 0)
write.csv(
  data.frame(
    x = gm[, 1], y = gm[, 2],
    pred = exp(sk$var1.pred + sk$var1.var / 2),  # analytic SK lognormal
    log_pred = sk$var1.pred,
    log_var = sk$var1.var
  ),
  file.path(out, "gstat_lognormal.csv"),
  row.names = FALSE
)
cat(sprintf("simple lognormal kriging written (beta = %.4f)\n", beta))

## ---- 2. Block co-kriging -------------------------------------------------
# Collocated log(zinc) + log(lead), shared LMC, predicted over 40 m blocks
# discretised as a 4x4 grid (offsets -15,-5,5,15) — same as Rust.
meuse$lzinc <- log(meuse$zinc)
meuse$llead <- log(meuse$lead)
write.csv(
  data.frame(x = meuse$x, y = meuse$y, lzinc = meuse$lzinc, llead = meuse$llead),
  file.path(out, "meuse_multi2.csv"),
  row.names = FALSE
)
ms <- meuse
coordinates(ms) <- ~ x + y
g <- gstat(NULL, id = "lz", formula = lzinc ~ 1, data = ms)
g <- gstat(g, id = "ll", formula = llead ~ 1, data = ms)
g$model$lz <- vgm(0.6, "Sph", 900, 0.05)
g$model$ll <- vgm(0.5, "Sph", 900, 0.04)
g$model$lz.ll <- vgm(0.5, "Sph", 900, 0.0)
writeLines(
  paste0(
    '{"nugget": [[0.05, 0.0], [0.0, 0.04]], ',
    '"structures": [{"kind": "spherical", "range": 900.0, ',
    '"sills": [[0.6, 0.5], [0.5, 0.5]]}]}'
  ),
  file.path(out, "lmc_block.json")
)

offs <- ((seq_len(4) - 0.5) / 4 - 0.5) * 40
block <- expand.grid(x = offs, y = offs)
ck <- predict(g, mg_sp, block = block, debug.level = 0)
write.csv(
  data.frame(x = gm[, 1], y = gm[, 2], pred = ck$lz.pred, var = ck$lz.var),
  file.path(out, "gstat_block_cokrige.csv"),
  row.names = FALSE
)
cat("block co-kriging written\n")
