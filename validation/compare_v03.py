#!/usr/bin/env python3
"""v0.3 parity check vs gstat on Meuse: block kriging.

Both engines average over the same explicit 4x4 discretization of a
40 m x 40 m block (offsets -15, -5, 5, 15), so the expected agreement is
machine precision. Note the C-bar(B,B) convention: coincident
discretization points exclude the nugget (measure-zero discontinuity in
the block integral) — matching gstat/GSLIB.

Run after `Rscript validation/v03_gstat.R` and:
  geostat krige -i validation/out/meuse_lzinc.csv --value-col lzinc \
      -m validation/out/gstat_model.json --block 40,40 --block-discr 4,4 \
      --bbox 178440,329600,181560,333760 --nx 78 --ny 104 \
      -o validation/out/rust_block.csv
"""

import csv
import sys
from pathlib import Path

OUT = Path(__file__).parent / "out"


def read_csv(name):
    with open(OUT / name) as f:
        return list(csv.DictReader(f))


def main():
    rust = {
        (round(float(r["x"]), 3), round(float(r["y"]), 3)): r
        for r in read_csv("rust_block.csv")
    }
    pd = vd = 0.0
    n = 0
    for g in read_csv("gstat_block.csv"):
        r = rust[(round(float(g["x"]), 3), round(float(g["y"]), 3))]
        pd = max(pd, abs(float(g["pred"]) - float(r["prediction"])))
        vd = max(vd, abs(float(g["var"]) - float(r["variance"])))
        n += 1
    print(f"Block kriging 40x40 (4x4 discretization), {n} cells:")
    ok_p = pd <= 1e-6
    ok_v = vd <= 1e-6
    print(f"  [{'OK ' if ok_p else 'FAIL'}] predictions, max abs diff: {pd:.3e} (tol 1e-06)")
    print(f"  [{'OK ' if ok_v else 'FAIL'}] block variances, max abs diff: {vd:.3e} (tol 1e-06)")
    if not (ok_p and ok_v):
        print("PARITY FAILED")
        sys.exit(1)
    print("\nV0.3 PARITY OK: block kriging matches gstat.")


if __name__ == "__main__":
    main()
