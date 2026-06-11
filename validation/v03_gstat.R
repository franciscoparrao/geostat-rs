#!/usr/bin/env Rscript
# gstat reference for v0.3: block kriging on Meuse.
#
# The block discretization is passed to gstat EXPLICITLY (as a data.frame of
# offsets) so both engines average over the same points: a regular 4x4 grid
# of cell centers covering a 40 m x 40 m block, i.e. offsets -15, -5, 5, 15.
#
# Run from the repo root after validation/gstat_reference.R (it reuses
# meuse_lzinc.csv and gstat_model.json): Rscript validation/v03_gstat.R

suppressPackageStartupMessages({
  library(sp)
  library(gstat)
})

out <- "validation/out"
options(digits = 15)

data(meuse)
data(meuse.grid)
meuse$lzinc <- log(meuse$zinc)
coordinates(meuse) <- ~ x + y
coordinates(meuse.grid) <- ~ x + y
gxy <- coordinates(meuse.grid)

# Same model the v0.1 harness exported (gstat's own fit).
vm_json <- jsonlite_minimal <- readLines(file.path(out, "gstat_model.json"))
# Parse the three numbers without a JSON package.
nums <- as.numeric(regmatches(vm_json, gregexpr("[0-9]+\\.[0-9]+", vm_json))[[1]])
vm <- vgm(psill = nums[2], model = "Sph", range = nums[3], nugget = nums[1])

offs <- ((seq_len(4) - 0.5) / 4 - 0.5) * 40
block <- expand.grid(x = offs, y = offs)
k <- krige(lzinc ~ 1, meuse, meuse.grid, model = vm, block = block,
           debug.level = 0)
write.csv(
  data.frame(x = gxy[, 1], y = gxy[, 2], pred = k$var1.pred, var = k$var1.var),
  file.path(out, "gstat_block.csv"),
  row.names = FALSE
)
cat("block kriging reference written (model:",
    sprintf("%.4f Nug + %.4f Sph(%.1f)", nums[1], nums[2], nums[3]), ")\n")
