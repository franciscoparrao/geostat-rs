// Runs conditional SGS in the WebAssembly build (wasm-pack --target nodejs)
// and writes every simulated value as its IEEE-754 bit pattern (hex), so the
// native run can be compared bit for bit by validation/bitident/compare.py.
//
//   wasm-pack build crates/geostat-wasm --target nodejs --release --out-dir <pkg>
//   node validation/bitident/wasm_sgs.js <pkg> <out.json>
const fs = require("fs");
const path = require("path");
const [pkg, outFile] = process.argv.slice(2);
const wasm = require(path.resolve(pkg, "geostat_wasm.js"));
const cfg = JSON.parse(fs.readFileSync(path.join(__dirname, "config.json"), "utf8"));
const rows = fs.readFileSync(path.join(__dirname, "meuse_lzinc.csv"), "utf8")
  .trim().split("\n").slice(1).map((l) => l.split(",").map(Number));
const x = Float64Array.from(rows.map((r) => r[0]));
const y = Float64Array.from(rows.map((r) => r[1]));
const v = Float64Array.from(rows.map((r) => r[2]));
const buf = Buffer.alloc(8);
const hex = [];
for (const model of cfg.models_ns) {
  const sims = wasm.sgs_grid(x, y, v, JSON.stringify(model), ...cfg.bbox,
    cfg.nx, cfg.ny, cfg.n_realizations, BigInt(cfg.seed), cfg.max_neighbors);
  for (const val of sims) { buf.writeDoubleBE(val); hex.push(buf.toString("hex")); }
}
fs.writeFileSync(outFile, JSON.stringify({ n: hex.length, bits: hex }));
console.log(`wasm: wrote ${hex.length} values`);
