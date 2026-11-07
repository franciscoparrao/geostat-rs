#!/usr/bin/env Rscript
# gstat references for MOVING-neighbourhood kriging on Meuse log(zinc), the
# regime the benchmarks and applications use (the original parity scripts use
# a global neighbourhood). Same data and model as gstat_reference.R.
#
# Writes validation/out/gstat_moving.csv (one row per case x grid cell) and
# validation/out/gstat_moving_cv.csv (LOO with nmax = 32).
# Run from the repo root:  Rscript validation/moving_gstat.R
suppressPackageStartupMessages({ library(sp); library(gstat); library(jsonlite) })
options(digits = 15)
out <- "validation/out"
md <- fromJSON(file.path(out, "gstat_model.json"))
vm <- vgm(md$structures$sill[1], "Sph", md$structures$range[1], md$nugget)
d <- read.csv(file.path(out, "meuse_lzinc.csv")); coordinates(d) <- ~ x + y
g <- read.csv(file.path(out, "grid_targets.csv"))[, c("x", "y")]; coordinates(g) <- ~ x + y

cases <- list(
  list(name = "ok_nmax8",            f = lzinc ~ 1,     nmax = 8,   maxdist = Inf, beta = NULL),
  list(name = "ok_nmax16",           f = lzinc ~ 1,     nmax = 16,  maxdist = Inf, beta = NULL),
  list(name = "ok_nmax32",           f = lzinc ~ 1,     nmax = 32,  maxdist = Inf, beta = NULL),
  list(name = "ok_nmax32_radius600", f = lzinc ~ 1,     nmax = 32,  maxdist = 600, beta = NULL),
  list(name = "sk_nmax16",           f = lzinc ~ 1,     nmax = 16,  maxdist = Inf, beta = mean(d$lzinc)),
  list(name = "uk1_nmax32",          f = lzinc ~ x + y, nmax = 32,  maxdist = Inf, beta = NULL)
)
res <- list()
for (cs in cases) {
  k <- if (is.null(cs$beta)) {
    krige(cs$f, d, g, model = vm, nmax = cs$nmax, maxdist = cs$maxdist, debug.level = 0)
  } else {
    krige(cs$f, d, g, model = vm, nmax = cs$nmax, maxdist = cs$maxdist, beta = cs$beta, debug.level = 0)
  }
  res[[cs$name]] <- data.frame(case = cs$name, x = coordinates(g)[, 1], y = coordinates(g)[, 2],
                               pred = k$var1.pred, var = k$var1.var)
}
write.csv(do.call(rbind, res), file.path(out, "gstat_moving.csv"), row.names = FALSE)
cv <- krige.cv(lzinc ~ 1, d, model = vm, nmax = 32, verbose = FALSE)
write.csv(data.frame(pred = cv$var1.pred, var = cv$var1.var), file.path(out, "gstat_moving_cv.csv"),
          row.names = FALSE)
cat("wrote gstat_moving.csv (", length(cases), "cases) and gstat_moving_cv.csv\n")
