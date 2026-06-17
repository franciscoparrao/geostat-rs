//! Minimal GeoPackage (OGC `.gpkg`) point reader, in pure Rust.
//!
//! A GeoPackage is a SQLite database with an OGC-defined schema. This module
//! reads point feature layers via `rusqlite` (bundled SQLite) — no GDAL, no
//! system dependency — so geostat-rs can exchange data with SurtGIS and the
//! wider GIS ecosystem without going through CSV (which loses the CRS and the
//! geometry typing).
//!
//! Scope: 2-D point layers. Geometry is decoded from the GeoPackageBinary
//! (GPB) blob — a small header plus standard WKB — taking only X and Y (any Z
//! or M is skipped). Layer discovery uses the `gpkg_contents` and
//! `gpkg_geometry_columns` tables.

use std::path::Path;

use anyhow::{Context, Result, bail};
use geostat_core::PointSet;
use rusqlite::{Connection, OpenFlags};

/// Summary of a vector feature layer in a GeoPackage.
pub struct LayerInfo {
    /// Feature-table (layer) name.
    pub name: String,
    /// Name of the geometry column.
    pub geometry_column: String,
    /// Declared geometry type (e.g. `POINT`).
    pub geometry_type: String,
    /// Spatial reference system id (as recorded in `gpkg_contents`).
    pub srs_id: i64,
    /// Number of features in the layer.
    pub n_features: i64,
}

fn open(path: &Path) -> Result<Connection> {
    Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .with_context(|| format!("opening GeoPackage {}", path.display()))
}

/// Lists the vector feature layers in a GeoPackage.
pub fn list_feature_layers(path: &Path) -> Result<Vec<LayerInfo>> {
    let conn = open(path)?;
    let mut stmt = conn
        .prepare(
            "SELECT c.table_name, g.column_name, g.geometry_type_name, c.srs_id \
             FROM gpkg_contents c \
             JOIN gpkg_geometry_columns g ON c.table_name = g.table_name \
             WHERE c.data_type = 'features' \
             ORDER BY c.table_name",
        )
        .context("querying gpkg_contents (is this a GeoPackage?)")?;
    let rows: Vec<(String, String, String, i64)> = stmt
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?
        .collect::<rusqlite::Result<_>>()?;

    let mut out = Vec::with_capacity(rows.len());
    for (name, geometry_column, geometry_type, srs_id) in rows {
        let n_features: i64 = conn
            .query_row(
                &format!("SELECT COUNT(*) FROM {}", quote_ident(&name)),
                [],
                |r| r.get(0),
            )
            .unwrap_or(-1);
        out.push(LayerInfo {
            name,
            geometry_column,
            geometry_type,
            srs_id,
            n_features,
        });
    }
    Ok(out)
}

/// Reads a 2-D point layer into a [`PointSet`], taking `value_col` as the
/// variable. If `layer` is `None`, the single feature layer is used (an error
/// is raised when the file holds more than one and none was named). Rows with a
/// NULL geometry or NULL value are skipped.
pub fn read_points(path: &Path, layer: Option<&str>, value_col: &str) -> Result<PointSet> {
    let layers = list_feature_layers(path)?;
    if layers.is_empty() {
        bail!("no vector feature layers found in {}", path.display());
    }
    let info = match layer {
        Some(name) => layers
            .iter()
            .find(|l| l.name == name)
            .with_context(|| format!("layer '{name}' not found in {}", path.display()))?,
        None => {
            if layers.len() > 1 {
                let names: Vec<&str> = layers.iter().map(|l| l.name.as_str()).collect();
                bail!(
                    "{} has multiple layers; choose one with --layer (available: {})",
                    path.display(),
                    names.join(", ")
                );
            }
            &layers[0]
        }
    };
    if !info.geometry_type.to_uppercase().contains("POINT") {
        bail!(
            "layer '{}' is {} — only point layers are supported",
            info.name,
            info.geometry_type
        );
    }

    let conn = open(path)?;
    let sql = format!(
        "SELECT {}, {} FROM {}",
        quote_ident(&info.geometry_column),
        quote_ident(value_col),
        quote_ident(&info.name),
    );
    let mut stmt = conn
        .prepare(&sql)
        .with_context(|| format!("reading column '{value_col}' from layer '{}'", info.name))?;
    let rows = stmt.query_map([], |r| {
        let geom: Option<Vec<u8>> = r.get(0)?;
        let value: Option<f64> = r.get(1)?;
        Ok((geom, value))
    })?;

    let mut coords = Vec::new();
    let mut values = Vec::new();
    for row in rows {
        let (geom, value) = row?;
        let (Some(blob), Some(v)) = (geom, value) else {
            continue; // skip NULL geometry or value
        };
        let (x, y) = decode_point(&blob)?;
        coords.push([x, y]);
        values.push(v);
    }
    if coords.is_empty() {
        bail!("layer '{}' yielded no usable point/value rows", info.name);
    }
    Ok(PointSet::new(coords, values)?)
}

/// Quotes a SQL identifier (table/column) by doubling embedded double quotes.
fn quote_ident(ident: &str) -> String {
    format!("\"{}\"", ident.replace('"', "\"\""))
}

fn rd_u32(b: &[u8], le: bool) -> u32 {
    let a = [b[0], b[1], b[2], b[3]];
    if le {
        u32::from_le_bytes(a)
    } else {
        u32::from_be_bytes(a)
    }
}

fn rd_f64(b: &[u8], le: bool) -> f64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[..8]);
    if le {
        f64::from_le_bytes(a)
    } else {
        f64::from_be_bytes(a)
    }
}

/// Decodes the (x, y) of a point from a GeoPackageBinary blob: a `GP` header
/// (magic, version, flags, srs_id, optional envelope) followed by standard WKB.
fn decode_point(blob: &[u8]) -> Result<(f64, f64)> {
    if blob.len() < 8 || &blob[0..2] != b"GP" {
        bail!("not a GeoPackage geometry blob");
    }
    let flags = blob[3];
    if (flags >> 4) & 0x01 == 1 {
        bail!("empty geometry");
    }
    // Envelope size from the flag code (bits 1-3): 0/X Y/X Y Z/X Y M/X Y Z M.
    let envelope_len = match (flags >> 1) & 0x07 {
        0 => 0,
        1 => 32,
        2 | 3 => 48,
        4 => 64,
        other => bail!("invalid GPB envelope code {other}"),
    };
    let wkb = blob
        .get(8 + envelope_len..)
        .context("GPB blob truncated before WKB")?;
    if wkb.len() < 21 {
        bail!("WKB too short for a point");
    }
    let le = wkb[0] == 1;
    // Base geometry type, robust to ISO Z/M (1001, 2001, 3001) and EWKB flags.
    let base = (rd_u32(&wkb[1..5], le) & 0x1FFF_FFFF) % 1000;
    if base != 1 {
        bail!("geometry is not a POINT (WKB base type {base})");
    }
    let x = rd_f64(&wkb[5..13], le);
    let y = rd_f64(&wkb[13..21], le);
    Ok((x, y))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a StandardGeoPackageBinary point blob (no envelope, little-endian
    /// WKB) for testing the decoder without a real file.
    fn gpb_point(x: f64, y: f64) -> Vec<u8> {
        let mut b = vec![b'G', b'P', 0x00, 0x01]; // magic, version 0, flags: LE header, no envelope
        b.extend_from_slice(&4326i32.to_le_bytes()); // srs_id
        b.push(0x01); // WKB byte order: little-endian
        b.extend_from_slice(&1u32.to_le_bytes()); // WKB type: Point
        b.extend_from_slice(&x.to_le_bytes());
        b.extend_from_slice(&y.to_le_bytes());
        b
    }

    #[test]
    fn decodes_a_point_blob() {
        let (x, y) = decode_point(&gpb_point(123.5, -45.25)).unwrap();
        assert_eq!((x, y), (123.5, -45.25));
    }

    #[test]
    fn decodes_point_with_envelope() {
        // Same point but advertise an X/Y envelope (code 1, 32 bytes) before WKB.
        let mut b = vec![b'G', b'P', 0x00, 0x03]; // flags: LE header + envelope code 1
        b.extend_from_slice(&4326i32.to_le_bytes());
        for v in [10.0f64, 20.0, 30.0, 40.0] {
            b.extend_from_slice(&v.to_le_bytes()); // minx,maxx,miny,maxy
        }
        b.push(0x01);
        b.extend_from_slice(&1u32.to_le_bytes());
        b.extend_from_slice(&15.0f64.to_le_bytes());
        b.extend_from_slice(&25.0f64.to_le_bytes());
        assert_eq!(decode_point(&b).unwrap(), (15.0, 25.0));
    }

    #[test]
    fn rejects_non_gpb() {
        assert!(decode_point(b"not a blob at all").is_err());
    }

    #[test]
    fn quotes_identifiers_safely() {
        assert_eq!(quote_ident("zinc"), "\"zinc\"");
        assert_eq!(quote_ident("a\"b"), "\"a\"\"b\"");
    }
}
