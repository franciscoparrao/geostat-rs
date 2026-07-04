#![no_main]

use libfuzzer_sys::fuzz_target;

// `ModelKind::from_str` parses user-facing CLI/Python strings ("spherical",
// "matern:1.2", "power:0.8", ...); it must never panic on arbitrary input --
// only ever return Ok or a GeostatError::InvalidParameter.
fuzz_target!(|data: &str| {
    let _ = data.parse::<geostat_core::ModelKind>();
    let _ = geostat_core::ModelKind::parse_list(data);
});
