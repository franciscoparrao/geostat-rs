#![no_main]

use libfuzzer_sys::fuzz_target;

// The CLI/Python bindings load VariogramModel from arbitrary user-supplied
// JSON files (`geostat krige -m model.json`, `VariogramModel.from_json`);
// deserializing and re-validating garbage/adversarial JSON must never panic,
// only ever return Ok or an error.
fuzz_target!(|data: &str| {
    if let Ok(m) = serde_json::from_str::<geostat_core::VariogramModel>(data) {
        let _ = geostat_core::VariogramModel::new(m.nugget, m.structures);
    }
});
