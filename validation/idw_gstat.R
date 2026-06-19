#!/usr/bin/env Rscript
# gstat reference for inverse-distance weighting (IDW), to validate the
# geostat-rs IDW interpolator. Exports gstat's IDW of log(zinc) on meuse.grid
# with power 2 and a global neighbourhood (every datum contributes), plus the
# grid coordinates, so the Python side can predict at the same points.
#
# Run from the repo root: Rscript validation/idw_gstat.R

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
coordinates(meuse) <- ~ x + y
coordinates(meuse.grid) <- ~ x + y

# IDW, power 2, global neighbourhood (default: all points).
k <- idw(lzinc ~ 1, meuse, meuse.grid, idp = 2.0, debug.level = 0)
g <- as.data.frame(meuse.grid)
write.csv(
  data.frame(x = g$x, y = g$y, idw = k$var1.pred),
  file.path(out, "gstat_idw.csv"),
  row.names = FALSE
)
cat(sprintf("Wrote gstat IDW for %d grid cells\n", nrow(g)))
