#!/usr/bin/env Rscript
# Numerical parity check for the Power (IRF-0) variogram model
# (ModelKind::Power) against gstat's "Pow", using ordinary kriging.
#
# gstat's Pow has NO length-scale: gamma(h) = psill * h^range, where "range"
# doubles as the exponent theta. geostat-rs matches this exactly via
# Structure{kind: Power(theta), sill, range} where the Structure's `range`
# field is ignored (ModelKind::Power has no plateau, so no scale is needed)
# and `sill` is the slope coefficient c directly -- gamma(h) = sill * h^theta.
#
# This model has no covariance function (infinite variance), so it can only
# be kriged directly in semivariogram form (ordinary/universal kriging, the
# classical IRF-0 generalization) -- see kriging.rs's `has_power` branch.
# Small hand-picked dataset (no dependency on the Meuse loader) so the
# reference values are easy to eyeball and hardcode as a Rust unit test
# (kriging::tests::power_model_matches_gstat_ordinary_kriging).
#
# Run from the repo root: Rscript validation/power_gstat.R

suppressPackageStartupMessages({
  library(sp)
  library(gstat)
})

options(digits = 15)

d <- data.frame(
  x = c(0, 10, 0, 10, 5),
  y = c(0, 0, 10, 10, 5),
  z = c(1.0, 2.0, 1.5, 2.5, 1.8)
)
coordinates(d) <- ~ x + y

m <- vgm(2.0, "Pow", 1.2) # gamma(h) = 2.0 * h^1.2

targets <- data.frame(x = c(3, 7, 5), y = c(4, 8, 5))
coordinates(targets) <- ~ x + y

k <- krige(z ~ 1, d, targets, model = m, debug.level = 0)
res <- as.data.frame(k)
cat("gstat ordinary kriging with vgm(2.0, \"Pow\", 1.2):\n")
print(res)
