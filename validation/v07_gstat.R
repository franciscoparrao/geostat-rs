# gstat reference for v0.7: residual variograms (UK / KED variography).
#
# gstat computes formula-based variograms on OLS residuals; geostat-rs
# implements the same convention (`--detrend`, `--detrend-cols`).
# Uses the same meuse_multi.csv exported by gstat_reference.R so both
# engines read identical numbers.
suppressMessages({
  library(gstat)
  library(sp)
})

out <- file.path("validation", "out")
meuse <- read.csv(file.path(out, "meuse_multi.csv"))
coordinates(meuse) <- ~ x + y

dump_vario <- function(v, name) {
  write.csv(
    data.frame(np = v$np, dist = v$dist, gamma = v$gamma),
    file.path(out, name),
    row.names = FALSE
  )
}

# 1. Residuals of an external drift: log(zinc) ~ sqrt(dist)  (KED variography).
v_drift <- variogram(lzinc ~ sdist, meuse, cutoff = 1500, width = 100)
dump_vario(v_drift, "gstat_resid_drift_vario.csv")

# 2. Residuals of a linear coordinate trend: log(zinc) ~ x + y  (UK variography).
v_poly <- variogram(lzinc ~ x + y, meuse, cutoff = 1500, width = 100)
dump_vario(v_poly, "gstat_resid_poly_vario.csv")

cat("v0.7 gstat references written\n")
