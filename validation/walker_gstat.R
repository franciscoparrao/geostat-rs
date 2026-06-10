#!/usr/bin/env Rscript
# Reference results from gstat (R) on Walker Lake (V variable), plus a
# conditional Gaussian simulation reference for the distributional SGS check.
#
# Parity outputs (deterministic):
#   - walker_v.csv               sample data (x, y, v), duplicates removed
#   - gstat_walker_vario.csv     experimental variogram, cutoff 120 / 15 lags
#   - gstat_walker_model.{csv,json}  spherical fit
#   - gstat_walker_krige.csv     OK on a 26x30 grid (res 10), global nbhd
#   - gstat_walker_cv.csv        leave-one-out CV
#
# SGS reference (stochastic, distributional comparison only):
#   - walker_scores.csv          normal scores of V (computed once, shared)
#   - gstat_ns_model.json        spherical fit to the score variogram
#   - gstat_sgs_nodes.csv        per-node ensemble mean/std of 200 conditional
#                                Gaussian simulations (SK, beta = 0, nmax = 16)
#                                + SK prediction/variance at the same nodes
#   - gstat_sgs_pooled.csv       pooled moments and quantiles of all draws
#
# Run from the repo root: Rscript validation/walker_gstat.R

suppressPackageStartupMessages({
  library(sp)
  library(gstat)
})

out <- "validation/out"
dir.create(out, recursive = TRUE, showWarnings = FALSE)
options(digits = 15)

data(walker)
w <- as.data.frame(walker)[, c("X", "Y", "V")]
names(w) <- c("x", "y", "v")

# Remove exactly coincident sample locations (singular kriging systems).
dup <- duplicated(w[, c("x", "y")])
if (any(dup)) {
  cat("removing", sum(dup), "duplicate locations\n")
  w <- w[!dup, ]
}
write.csv(w, file.path(out, "walker_v.csv"), row.names = FALSE)
n <- nrow(w)
cat("walker sample:", n, "points\n")

wsp <- w
coordinates(wsp) <- ~ x + y

# --- 1. Experimental variogram + model fit (parity) -----------------------
cutoff <- 120
width <- 8
v <- variogram(v ~ 1, wsp, cutoff = cutoff, width = width)
write.csv(
  data.frame(np = v$np, dist = v$dist, gamma = v$gamma),
  file.path(out, "gstat_walker_vario.csv"),
  row.names = FALSE
)
sv <- var(w$v)
vm <- fit.variogram(v, vgm(0.75 * sv, "Sph", 50, 0.25 * sv))
write.csv(
  data.frame(model = as.character(vm$model), psill = vm$psill, range = vm$range),
  file.path(out, "gstat_walker_model.csv"),
  row.names = FALSE
)
model_json <- function(vm) {
  nug <- vm$psill[as.character(vm$model) == "Nug"]
  if (length(nug) == 0) nug <- 0
  sph <- vm[as.character(vm$model) == "Sph", ]
  sprintf(
    '{"nugget": %.12f, "structures": [{"kind": "spherical", "sill": %.12f, "range": %.12f}]}',
    nug, sph$psill, sph$range
  )
}
writeLines(model_json(vm), file.path(out, "gstat_walker_model.json"))
cat("V model:\n"); print(vm)

# --- 2. OK + CV (parity) ---------------------------------------------------
grid <- expand.grid(x = seq(5, 255, by = 10), y = seq(5, 295, by = 10))
gridsp <- grid
coordinates(gridsp) <- ~ x + y
k <- krige(v ~ 1, wsp, gridsp, model = vm, debug.level = 0)
write.csv(
  data.frame(x = grid$x, y = grid$y, pred = k$var1.pred, var = k$var1.var),
  file.path(out, "gstat_walker_krige.csv"),
  row.names = FALSE
)
cv <- krige.cv(v ~ 1, wsp, model = vm, verbose = FALSE)
write.csv(
  data.frame(
    x = coordinates(cv)[, 1], y = coordinates(cv)[, 2],
    observed = cv$observed, pred = cv$var1.pred, var = cv$var1.var
  ),
  file.path(out, "gstat_walker_cv.csv"),
  row.names = FALSE
)

# --- 3. SGS reference (distributional) -------------------------------------
# Normal scores computed once and shared with geostat-rs, so both engines
# simulate the same Gaussian-space dataset. Rank convention (r - 0.5)/n
# matches geostat-rs's NormalScore. ties.method = "first" keeps all scores
# distinct: geostat-rs's internal transform then reduces to the identity and
# its back-transform clamp sits at +-3.07 (0.2% of N(0,1) mass), so the
# comparison isolates the simulator. With averaged ties, Walker's many V = 0
# values would collapse into one knot at -1.85 and clamp the whole lower tail.
score <- qnorm((rank(w$v, ties.method = "first") - 0.5) / n)
write.csv(
  data.frame(x = w$x, y = w$y, score = score),
  file.path(out, "walker_scores.csv"),
  row.names = FALSE
)

ssp <- data.frame(x = w$x, y = w$y, score = score)
coordinates(ssp) <- ~ x + y
vs <- variogram(score ~ 1, ssp, cutoff = cutoff, width = width)
vms <- fit.variogram(vs, vgm(0.8, "Sph", 50, 0.2))
writeLines(model_json(vms), file.path(out, "gstat_ns_model.json"))
cat("score model:\n"); print(vms)

nsim <- 1000
set.seed(42)
sims <- krige(
  score ~ 1, ssp, gridsp,
  model = vms, beta = 0, nmax = 16, nsim = nsim, debug.level = 0
)
m <- as.matrix(as.data.frame(sims)[, paste0("sim", 1:nsim)])

# SK prediction at the same nodes (the ensemble mean must converge to it).
sk <- krige(score ~ 1, ssp, gridsp, model = vms, beta = 0, nmax = 16,
            debug.level = 0)

write.csv(
  data.frame(
    x = grid$x, y = grid$y,
    mean = rowMeans(m),
    std = apply(m, 1, sd),
    sk_pred = sk$var1.pred,
    sk_var = sk$var1.var
  ),
  file.path(out, "gstat_sgs_nodes.csv"),
  row.names = FALSE
)
pooled <- as.vector(m)
q <- quantile(pooled, c(0.1, 0.25, 0.5, 0.75, 0.9))
write.csv(
  data.frame(
    mean = mean(pooled), std = sd(pooled),
    q10 = q[1], q25 = q[2], q50 = q[3], q75 = q[4], q90 = q[5]
  ),
  file.path(out, "gstat_sgs_pooled.csv"),
  row.names = FALSE
)
cat("SGS reference:", nsim, "realizations on", nrow(grid), "nodes\n")
