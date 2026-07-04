//! Library half of the `geostat` CLI: I/O helpers (`io_utils`) and
//! GeoPackage support (`gpkg`), split out from the binary so they can be
//! exercised independently — currently by `fuzz/` (binary-blob parsing:
//! [`gpkg::decode_point`], [`gpkg::decode_png_u16`]), and by any future
//! integration test that wants the parsing logic without spawning the CLI
//! process (see `tests/cli.rs` for the process-level tests).

pub mod gpkg;
pub mod io_utils;
