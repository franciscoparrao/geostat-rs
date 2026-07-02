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

# 3. Ordinary kriging with a measurement-error component (gstat "Err"):
#    same fitted model as the v0.1 check plus Err = 0.05, on meuse.grid.
model_json <- jsonlite::fromJSON(file.path(out, "gstat_model.json"))
v_err <- vgm(
  model_json$structures$sill[1], "Sph", model_json$structures$range[1],
  add.to = vgm(model_json$nugget, "Nug", 0, add.to = vgm(0.05, "Err", 0))
)
grid <- read.csv(file.path(out, "grid_targets.csv"))
coordinates(grid) <- ~ x + y
k_err <- krige(lzinc ~ 1, meuse, grid, model = v_err)
write.csv(
  data.frame(
    x = coordinates(k_err)[, 1], y = coordinates(k_err)[, 2],
    pred = k_err$var1.pred, var = k_err$var1.var
  ),
  file.path(out, "gstat_err_krige.csv"),
  row.names = FALSE
)

cat("v0.7 gstat references written\n")
