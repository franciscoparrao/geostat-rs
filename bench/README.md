# Timing benchmark: geostat-rs vs gstat

Algorithm-level wall-clock comparison on the Meuse dataset: ordinary kriging on
regular grids (32-point moving neighbourhood) and leave-one-out cross-validation,
same data and fitted model on both sides. Only the operation is timed (data
preloaded, best of 3).

```sh
Rscript validation/gstat_reference.R            # writes meuse_lzinc.csv + model
PYTHONPATH=<dir with geostat_rs.so> python3 bench/bench_rust.py   # writes grids + results_rust.csv
RAYON_NUM_THREADS=1 python3 bench/bench_rust.py  # single-threaded geostat-rs
Rscript bench/bench_gstat.R                       # writes results_gstat.csv
```

Generated `grid_*.csv` and `results_*.csv` are git-ignored. See the paper
(Performance section) for the reported table; numbers are hardware-dependent
(measured on a 16-core i7-1270P, gstat 2.1.4).
