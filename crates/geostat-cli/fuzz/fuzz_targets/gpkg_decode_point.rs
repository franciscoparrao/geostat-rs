#![no_main]

use libfuzzer_sys::fuzz_target;

// `decode_point` parses the GeoPackageBinary (GPB) header + WKB body of a
// point geometry blob read straight from a `.gpkg` SQLite file -- an
// adversarial or merely corrupt GeoPackage must never panic the reader
// (AUDIT-2026-07-v2.md §6 Fase 5: this binary-blob parser was the fuzz
// target candidate identified but not yet covered), only ever return Ok or
// an error.
fuzz_target!(|data: &[u8]| {
    let _ = geostat_cli::gpkg::decode_point(data);
});
