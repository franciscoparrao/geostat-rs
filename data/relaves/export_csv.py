#!/usr/bin/env python3
"""Exports the subset of the SERNAGEOMIN tailings registry used by the paper.

Reads data/relaves/tierras_raras.pkl (the preprocessed registry, see
provenance.json) and writes data/relaves/coquimbo_ree.csv with exactly the
rows (Region == COQUIMBO, positive coordinates) and columns the application
uses: identifiers, UTM coordinates, the five rare-earth targets and the five
host-mineral covariates, as numbers (decimal commas and '<'/'>' flags already
absent in this file). figures/make_figure_data.py reads the CSV when present.

    python3 data/relaves/export_csv.py
"""
import hashlib
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PKL, CSV = HERE / "tierras_raras.pkl", HERE / "coquimbo_ree.csv"
IDS = ["IDQ", "Faena", "Deposito", "Region"]
XY = ["Coord. E", "Coord. N"]
VALUES = ["La(g/t)", "Ce(g/t)", "Nd(g/t)", "Dy(g/t)", "Y(g/t)",
          "P2O5(%)", "Th(g/t)", "TiO2(%)", "Fe2O3(%)", "Zr(g/t)"]


def num(s):
    s = s.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(s.str.replace("<", "", regex=False).str.replace(">", "", regex=False),
                         errors="coerce")


def main():
    d = pd.read_pickle(PKL)
    out = d[IDS].copy()
    for c in XY + VALUES:
        out[c] = num(d[c])
    out = out[(out["Coord. E"] > 0) & (out["Coord. N"] > 0) & (out["Region"] == "COQUIMBO")]
    out = out.dropna(subset=XY + VALUES).sort_values("IDQ").reset_index(drop=True)
    out.to_csv(CSV, index=False)
    print(f"wrote {CSV} ({len(out)} rows), sha256 {hashlib.sha256(CSV.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
