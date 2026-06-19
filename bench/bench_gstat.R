#!/usr/bin/env Rscript
# Timing benchmark for gstat, matching bench_rust.py: same Meuse data, same
# fitted spherical model, the same regular grids (global neighbourhood), and
# leave-one-out cross-validation on the 155 points. Only the operation is
# timed via system.time (data preloaded), best of N_REP elapsed seconds.
#
# Run from the repo root, AFTER bench_rust.py (which writes the grids):
#   Rscript bench/bench_gstat.R

suppressPackageStartupMessages({
  library(sp)
  library(gstat)
})
options(digits = 6)
N_REP <- 3
sizes <- c(50, 100, 200)

d <- read.csv("validation/out/meuse_lzinc.csv")
meuse <- d
coordinates(meuse) <- ~ x + y

# Same model as the parity work (gstat WLS fit; matches gstat_model.json).
v <- variogram(lzinc ~ 1, meuse, cutoff = 1500, width = 100)
vm <- fit.variogram(v, vgm(0.6, "Sph", 900, 0.05))

# Re-evaluate the expression fresh each repetition (avoid promise caching).
best <- function(expr) {
  e <- substitute(expr)
  t <- Inf
  for (i in seq_len(N_REP)) {
    t <- min(t, system.time(eval(e, parent.frame()))[["elapsed"]])
  }
  t
}

res <- data.frame(task = character(), seconds = numeric())
for (n in sizes) {
  g <- read.csv(sprintf("bench/grid_%d.csv", n))
  coordinates(g) <- ~ x + y
  t <- best(krige(lzinc ~ 1, meuse, g, model = vm, nmax = 32, debug.level = 0))
  res <- rbind(res, data.frame(task = sprintf("OK grid %d", n * n), seconds = t))
  cat(sprintf("  OK grid %6d cells: %8.1f ms\n", n * n, t * 1000))
}

t_cv <- best(krige.cv(lzinc ~ 1, meuse, model = vm, nmax = 32, nfold = nrow(d)))
res <- rbind(res, data.frame(task = "LOO CV 155", seconds = t_cv))
cat(sprintf("  LOO CV (155 points): %8.1f ms\n", t_cv * 1000))

write.csv(res, "bench/results_gstat.csv", row.names = FALSE)
cat("wrote bench/results_gstat.csv\n")
