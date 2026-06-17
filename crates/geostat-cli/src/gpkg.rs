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

/// Encodes a 2-D point as a StandardGeoPackageBinary blob: a `GP` header
/// (little-endian, no envelope) followed by little-endian WKB.
fn encode_point(x: f64, y: f64, srs_id: i32) -> Vec<u8> {
    let mut b = Vec::with_capacity(8 + 21);
    b.extend_from_slice(b"GP"); // magic
    b.push(0x00); // version
    b.push(0x01); // flags: little-endian header, no envelope
    b.extend_from_slice(&srs_id.to_le_bytes());
    b.push(0x01); // WKB byte order: little-endian
    b.extend_from_slice(&1u32.to_le_bytes()); // WKB type: Point
    b.extend_from_slice(&x.to_le_bytes());
    b.extend_from_slice(&y.to_le_bytes());
    b
}

/// Writes 2-D points to a new GeoPackage as a point feature layer, with one
/// REAL attribute column per `(name, values)` pair. The file is created (or
/// replaced); `srs_id` is recorded as the layer's CRS.
///
/// Scope: a pure-Rust point-vector writer (the inverse of [`read_points`]).
/// Raster/tile output is not handled. If `srs_id` is not one of the mandatory
/// GeoPackage entries (-1, 0, 4326), a placeholder `gpkg_spatial_ref_sys` row
/// is inserted so the file stays valid; supply a known EPSG id for a fully
/// defined CRS.
pub fn write_points(
    path: &Path,
    layer: &str,
    srs_id: i32,
    coords: &[[f64; 2]],
    columns: &[(&str, &[f64])],
) -> Result<()> {
    for (name, values) in columns {
        if values.len() != coords.len() {
            bail!(
                "column '{name}' has {} values but there are {} points",
                values.len(),
                coords.len()
            );
        }
    }
    if path.exists() {
        std::fs::remove_file(path)
            .with_context(|| format!("replacing existing {}", path.display()))?;
    }
    let mut conn = Connection::open(path)
        .with_context(|| format!("creating GeoPackage {}", path.display()))?;

    // GeoPackage magic: application_id "GPKG" and user_version 10300.
    conn.pragma_update(None, "application_id", 0x4750_4B47_i64)?;
    conn.pragma_update(None, "user_version", 10300_i64)?;

    conn.execute_batch(
        "CREATE TABLE gpkg_spatial_ref_sys (
            srs_name TEXT NOT NULL, srs_id INTEGER PRIMARY KEY,
            organization TEXT NOT NULL, organization_coordsys_id INTEGER NOT NULL,
            definition TEXT NOT NULL, description TEXT);
         CREATE TABLE gpkg_contents (
            table_name TEXT NOT NULL PRIMARY KEY, data_type TEXT NOT NULL,
            identifier TEXT UNIQUE, description TEXT DEFAULT '',
            last_change TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
            min_x DOUBLE, min_y DOUBLE, max_x DOUBLE, max_y DOUBLE, srs_id INTEGER);
         CREATE TABLE gpkg_geometry_columns (
            table_name TEXT NOT NULL, column_name TEXT NOT NULL,
            geometry_type_name TEXT NOT NULL, srs_id INTEGER NOT NULL,
            z TINYINT NOT NULL, m TINYINT NOT NULL,
            CONSTRAINT pk_geom_cols PRIMARY KEY (table_name, column_name));",
    )?;

    // Mandatory SRS rows, plus the requested one if it is something else.
    conn.execute_batch(
        "INSERT INTO gpkg_spatial_ref_sys VALUES
            ('Undefined cartesian SRS', -1, 'NONE', -1, 'undefined', NULL),
            ('Undefined geographic SRS', 0, 'NONE', 0, 'undefined', NULL),
            ('WGS 84 geodetic', 4326, 'EPSG', 4326,
             'GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563]],PRIMEM[\"Greenwich\",0],UNIT[\"degree\",0.0174532925199433]]',
             NULL);",
    )?;
    if !matches!(srs_id, -1 | 0 | 4326) {
        conn.execute(
            "INSERT INTO gpkg_spatial_ref_sys VALUES (?1, ?2, 'EPSG', ?2, 'undefined', NULL)",
            rusqlite::params![format!("SRS {srs_id}"), srs_id],
        )?;
    }

    // Feature table: integer pk + geometry + REAL attribute columns.
    let attr_defs: String = columns
        .iter()
        .map(|(name, _)| format!(", {} REAL", quote_ident(name)))
        .collect();
    conn.execute(
        &format!(
            "CREATE TABLE {} (fid INTEGER PRIMARY KEY AUTOINCREMENT, geom BLOB{attr_defs})",
            quote_ident(layer)
        ),
        [],
    )?;

    // Register the layer.
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );
    for c in coords {
        min_x = min_x.min(c[0]);
        min_y = min_y.min(c[1]);
        max_x = max_x.max(c[0]);
        max_y = max_y.max(c[1]);
    }
    conn.execute(
        "INSERT INTO gpkg_contents
            (table_name, data_type, identifier, min_x, min_y, max_x, max_y, srs_id)
         VALUES (?1, 'features', ?1, ?2, ?3, ?4, ?5, ?6)",
        rusqlite::params![layer, min_x, min_y, max_x, max_y, srs_id],
    )?;
    conn.execute(
        "INSERT INTO gpkg_geometry_columns VALUES (?1, 'geom', 'POINT', ?2, 0, 0)",
        rusqlite::params![layer, srs_id],
    )?;

    // Bulk-insert the features in one transaction.
    let col_names: String = columns
        .iter()
        .map(|(name, _)| format!(", {}", quote_ident(name)))
        .collect();
    let placeholders: String = (0..columns.len())
        .map(|i| format!(", ?{}", i + 2))
        .collect();
    let insert_sql = format!(
        "INSERT INTO {} (geom{col_names}) VALUES (?1{placeholders})",
        quote_ident(layer)
    );
    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare(&insert_sql)?;
        for (i, c) in coords.iter().enumerate() {
            let blob = encode_point(c[0], c[1], srs_id);
            let mut params: Vec<rusqlite::types::Value> = Vec::with_capacity(1 + columns.len());
            params.push(blob.into());
            for (_, values) in columns {
                params.push(values[i].into());
            }
            stmt.execute(rusqlite::params_from_iter(params.iter()))?;
        }
    }
    tx.commit()?;
    Ok(())
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

    #[test]
    fn write_then_read_round_trips() {
        let path = std::env::temp_dir().join("geostat_rs_gpkg_roundtrip.gpkg");
        let _ = std::fs::remove_file(&path);
        let coords = [[10.0, 20.0], [30.5, 40.25], [-5.0, 0.0]];
        let vals = [1.5, 2.5, 3.5];
        write_points(&path, "kriging", 28992, &coords, &[("pred", &vals)]).unwrap();

        // The writer's layer is discoverable and decodes back to the inputs.
        let layers = list_feature_layers(&path).unwrap();
        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].name, "kriging");
        assert_eq!(layers[0].geometry_type, "POINT");
        assert_eq!(layers[0].srs_id, 28992);
        assert_eq!(layers[0].n_features, 3);

        let ps = read_points(&path, None, "pred").unwrap();
        assert_eq!(ps.len(), 3);
        for (i, c) in coords.iter().enumerate() {
            assert_eq!(ps.coord(i), *c);
            assert_eq!(ps.value(i), vals[i]);
        }
        let _ = std::fs::remove_file(&path);
    }
}
