#!/usr/bin/env Rscript
# gstat side of bench/bench_v2.py: same data, the same fixed variogram models
# and the same targets. Only the operation is timed (system.time elapsed);
# every repetition is written to bench/results_v2.csv (long format).
#
#   python3 bench/bench_v2.py --write-data   # first (writes synth_*.csv)
#   Rscript bench/bench_gstat_v2.R
suppressPackageStartupMessages({ library(sp); library(gstat); library(jsonlite) })
REPS <- as.integer(Sys.getenv("REPS", "10"))
out <- "bench/results_v2.csv"
rows <- list()
rec <- function(task, times) {
  rows[[length(rows) + 1]] <<- data.frame(tool = "gstat", task = task, threads = "1",
                                          rep = seq_along(times) - 1, seconds = times)
  cat(sprintf("  gstat       %-22s median %9.1f ms\n", task, median(times) * 1000))
}
time_reps <- function(expr, reps) {
  e <- substitute(expr)
  env <- parent.frame()
  vapply(seq_len(reps), function(i) system.time(eval(e, env))[["elapsed"]], 0.0)
}
mk_model <- function(nugget, psill, range) vgm(psill, "Sph", range, nugget)

# --- Meuse grids, the parity model (validation/out/gstat_model.json) ---
md <- fromJSON("validation/out/gstat_model.json")
vm <- mk_model(md$nugget, md$structures$sill[1], md$structures$range[1])
d <- read.csv("validation/out/meuse_lzinc.csv"); coordinates(d) <- ~ x + y
for (n in c(50, 100, 200)) {
  g <- read.csv(sprintf("bench/grid_%d.csv", n)); coordinates(g) <- ~ x + y
  reps <- if (n < 200) REPS else max(3, REPS %/% 2)
  rec(sprintf("meuse_grid_%d", n * n),
      time_reps(krige(lzinc ~ 1, d, g, model = vm, nmax = 32, debug.level = 0), reps))
}

# --- Synthetic, moving and global neighbourhood ---
sm <- fromJSON("bench/synth_model.json")
vs <- mk_model(sm$nugget, sm$psill, sm$range)
tg <- read.csv("bench/synth_targets.csv"); coordinates(tg) <- ~ x + y
for (n in c(1000, 10000, 100000)) {
  s <- read.csv(sprintf("bench/synth_%d.csv", n)); coordinates(s) <- ~ x + y
  rec(sprintf("synth_nmax_%d", n),
      time_reps(krige(v ~ 1, s, tg, model = vs, nmax = 32, debug.level = 0), REPS))
}
s <- read.csv("bench/synth_1000.csv"); coordinates(s) <- ~ x + y
rec("synth_global_1000", time_reps(krige(v ~ 1, s, tg, model = vs, debug.level = 0), REPS))

# --- SGS: 50 realizations on a 50x50 grid, 16 neighbours, simple kriging (beta = 0) ---
c50 <- (seq_len(50) - 0.5) * 10000 / 50
g50 <- expand.grid(x = c50, y = c50); coordinates(g50) <- ~ x + y
rec("sgs_synth_1000", time_reps(krige(v ~ 1, s, g50, model = vs, beta = 0, nmax = 16,
                                      nsim = 50, debug.level = 0), max(3, REPS %/% 2)))

res <- do.call(rbind, rows)
write.table(res, out, sep = ",", row.names = FALSE, col.names = !file.exists(out),
            append = file.exists(out), quote = FALSE)
cat("appended to", out, "\n")
