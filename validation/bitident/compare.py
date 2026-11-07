#!/usr/bin/env python3
"""Bit-for-bit comparison of conditional SGS between the native build (Python
module) and the WebAssembly build (output of wasm_sgs.js), on Meuse log-zinc
with the configuration in config.json (spherical, exponential and
Gaussian normal-score models). Exits non-zero on any differing bit.

    python3 validation/bitident/compare.py <wasm_out.json>
"""
import csv
import json
import struct
import sys
from pathlib import Path

import geostat_rs as gs

HERE = Path(__file__).parent


def main():
    cfg = json.loads((HERE / "config.json").read_text())
    with open(HERE / "meuse_lzinc.csv") as f:
        rows = list(csv.DictReader(f))
    x, y, v = ([float(r[c]) for r in rows] for c in ("x", "y", "lzinc"))
    native = []
    for m in cfg["models_ns"]:
        model = gs.VariogramModel.from_json(json.dumps(m))
        sims = gs.sgs(x, y, v, model, tuple(cfg["bbox"]), cfg["nx"], cfg["ny"],
                      n_realizations=cfg["n_realizations"], seed=cfg["seed"],
                      max_neighbors=cfg["max_neighbors"])
        native += [struct.pack(">d", float(val)).hex() for val in sims.ravel()]
    wasm = json.loads(Path(sys.argv[1]).read_text())["bits"]
    if len(native) != len(wasm):
        sys.exit(f"length differs: native {len(native)} vs wasm {len(wasm)}")
    diff = sum(a != b for a, b in zip(native, wasm))
    print(f"native vs wasm: {len(native)} simulated values, {diff} differ in any bit")
    if diff:
        sys.exit(1)
    print("BIT-IDENTICAL")


if __name__ == "__main__":
    main()
