#![no_main]

use libfuzzer_sys::fuzz_target;

// `decode_png_u16` decodes a GeoPackage 2D-gridded-coverage raster tile (a
// 16-bit grayscale PNG blob read from a `.gpkg` SQLite file) against the
// tile-matrix-declared width/height -- adversarial or corrupt tile bytes, or
// a width/height that disagrees with the PNG's own header, must never
// panic, only ever return Ok or an error. The first 8 bytes of the fuzz
// input become `tw`/`th` (each clamped to a small range so most inputs
// exercise the width/height-mismatch path rather than allocating a huge
// buffer); the rest is the candidate PNG blob.
fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    let tw = (u32::from_le_bytes([data[0], data[1], data[2], data[3]]) % 4096) as usize;
    let th = (u32::from_le_bytes([data[4], data[5], data[6], data[7]]) % 4096) as usize;
    let _ = geostat_cli::gpkg::decode_png_u16(&data[8..], tw, th);
});
