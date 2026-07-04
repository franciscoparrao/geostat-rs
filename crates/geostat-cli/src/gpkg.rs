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
    // A valid GeoPackage may hold only raster coverages, with no geometry-column
    // table at all; that is "no feature layers", not an error.
    let has_geom_cols: bool = conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='gpkg_geometry_columns'",
            [],
            |_| Ok(true),
        )
        .unwrap_or(false);
    if !has_geom_cols {
        return Ok(Vec::new());
    }
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

/// Encodes a grid of `f64` values (row-major, `iy*nx+ix`, y increasing) as a
/// 16-bit grayscale PNG plus the linear `value = pixel*scale + offset` mapping
/// and the data statistics. Image rows run north→south (row 0 = max y), so the
/// grid's top row (`iy = ny-1`) is written first. NaN cells map to pixel 0
/// (the nodata value); finite values map to `1..=65535`.
struct RasterEncoding {
    png: Vec<u8>,
    scale: f64,
    offset: f64,
    min: f64,
    max: f64,
    mean: f64,
    std_dev: f64,
}

fn encode_raster(nx: usize, ny: usize, values: &[f64]) -> Result<RasterEncoding> {
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        bail!("raster has no finite values");
    }
    let min = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let max = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let n = finite.len() as f64;
    let mean = finite.iter().sum::<f64>() / n;
    let std_dev = (finite.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();

    // Linear map: pixel 1 -> min, pixel 65535 -> max; pixel 0 = nodata.
    let span = (max - min).max(f64::MIN_POSITIVE);
    let scale = span / 65534.0;
    let offset = min - scale; // value = pixel*scale + offset
    let to_pixel = |v: f64| -> u16 {
        if !v.is_finite() {
            0
        } else {
            (1.0 + (v - min) / span * 65534.0)
                .round()
                .clamp(1.0, 65535.0) as u16
        }
    };

    // 16-bit samples, big-endian (PNG order), rows north→south.
    let mut bytes = Vec::with_capacity(nx * ny * 2);
    for r in 0..ny {
        let iy = ny - 1 - r;
        for ix in 0..nx {
            let px = to_pixel(values[iy * nx + ix]);
            bytes.extend_from_slice(&px.to_be_bytes());
        }
    }
    let mut png = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut png, nx as u32, ny as u32);
        enc.set_color(png::ColorType::Grayscale);
        enc.set_depth(png::BitDepth::Sixteen);
        let mut writer = enc
            .write_header()
            .map_err(|e| anyhow::anyhow!("PNG header: {e}"))?;
        writer
            .write_image_data(&bytes)
            .map_err(|e| anyhow::anyhow!("PNG data: {e}"))?;
    }
    Ok(RasterEncoding {
        png,
        scale,
        offset,
        min,
        max,
        mean,
        std_dev,
    })
}

/// Writes a kriged grid to a new GeoPackage as a single-band raster, using the
/// OGC **2D Gridded Coverage** extension (16-bit PNG tile with a linear
/// scale/offset), so the actual prediction values — not just an image — are
/// preserved and read back by GDAL/QGIS as a continuous raster.
///
/// `bbox` is `[min_x, min_y, max_x, max_y]` of the grid extent; `values` are in
/// grid storage order (`iy*nx + ix`, y increasing). A single zoom level holds
/// the whole grid as one tile.
pub fn write_raster(
    path: &Path,
    layer: &str,
    srs_id: i32,
    nx: usize,
    ny: usize,
    bbox: [f64; 4],
    values: &[f64],
) -> Result<()> {
    if values.len() != nx * ny {
        bail!("{} values for an {nx}x{ny} grid", values.len());
    }
    let enc = encode_raster(nx, ny, values)?;
    let [min_x, min_y, max_x, max_y] = bbox;
    let pixel_x = (max_x - min_x) / nx as f64;
    let pixel_y = (max_y - min_y) / ny as f64;

    if path.exists() {
        std::fs::remove_file(path)
            .with_context(|| format!("replacing existing {}", path.display()))?;
    }
    let conn = Connection::open(path)
        .with_context(|| format!("creating GeoPackage {}", path.display()))?;
    conn.pragma_update(None, "application_id", 0x4750_4B47_i64)?;
    conn.pragma_update(None, "user_version", 10300_i64)?;

    // Core + tiles + gridded-coverage schema.
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
         CREATE TABLE gpkg_tile_matrix_set (
            table_name TEXT NOT NULL PRIMARY KEY, srs_id INTEGER NOT NULL,
            min_x DOUBLE NOT NULL, min_y DOUBLE NOT NULL,
            max_x DOUBLE NOT NULL, max_y DOUBLE NOT NULL);
         CREATE TABLE gpkg_tile_matrix (
            table_name TEXT NOT NULL, zoom_level INTEGER NOT NULL,
            matrix_width INTEGER NOT NULL, matrix_height INTEGER NOT NULL,
            tile_width INTEGER NOT NULL, tile_height INTEGER NOT NULL,
            pixel_x_size DOUBLE NOT NULL, pixel_y_size DOUBLE NOT NULL,
            CONSTRAINT pk_ttm PRIMARY KEY (table_name, zoom_level));
         CREATE TABLE gpkg_extensions (
            table_name TEXT, column_name TEXT, extension_name TEXT NOT NULL,
            definition TEXT NOT NULL, scope TEXT NOT NULL,
            CONSTRAINT ge_tce UNIQUE (table_name, column_name, extension_name));
         CREATE TABLE gpkg_2d_gridded_coverage_ancillary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tile_matrix_set_name TEXT NOT NULL UNIQUE,
            datatype TEXT NOT NULL DEFAULT 'integer',
            scale REAL NOT NULL DEFAULT 1.0, offset REAL NOT NULL DEFAULT 0.0,
            precision REAL DEFAULT 1.0, data_null REAL,
            grid_cell_encoding TEXT DEFAULT 'grid-value-is-center',
            uom TEXT, field_name TEXT DEFAULT 'Height',
            quantity_definition TEXT DEFAULT 'Height');
         CREATE TABLE gpkg_2d_gridded_tile_ancillary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tpudt_name TEXT NOT NULL, tpudt_id INTEGER NOT NULL,
            scale REAL NOT NULL DEFAULT 1.0, offset REAL NOT NULL DEFAULT 0.0,
            min REAL, max REAL, mean REAL, std_dev REAL,
            UNIQUE (tpudt_name, tpudt_id));",
    )?;

    conn.execute_batch(
        "INSERT INTO gpkg_spatial_ref_sys VALUES
            ('Undefined cartesian SRS', -1, 'NONE', -1, 'undefined', NULL),
            ('Undefined geographic SRS', 0, 'NONE', 0, 'undefined', NULL),
            ('WGS 84 geodetic', 4326, 'EPSG', 4326, 'undefined', NULL);",
    )?;
    if !matches!(srs_id, -1 | 0 | 4326) {
        conn.execute(
            "INSERT INTO gpkg_spatial_ref_sys VALUES (?1, ?2, 'EPSG', ?2, 'undefined', NULL)",
            rusqlite::params![format!("SRS {srs_id}"), srs_id],
        )?;
    }

    // Tile pyramid user-data table (one tile).
    conn.execute(
        &format!(
            "CREATE TABLE {} (id INTEGER PRIMARY KEY AUTOINCREMENT,
                zoom_level INTEGER NOT NULL, tile_column INTEGER NOT NULL,
                tile_row INTEGER NOT NULL, tile_data BLOB NOT NULL,
                UNIQUE (zoom_level, tile_column, tile_row))",
            quote_ident(layer)
        ),
        [],
    )?;

    conn.execute(
        "INSERT INTO gpkg_contents
            (table_name, data_type, identifier, min_x, min_y, max_x, max_y, srs_id)
         VALUES (?1, '2d-gridded-coverage', ?1, ?2, ?3, ?4, ?5, ?6)",
        rusqlite::params![layer, min_x, min_y, max_x, max_y, srs_id],
    )?;
    conn.execute(
        "INSERT INTO gpkg_tile_matrix_set VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        rusqlite::params![layer, srs_id, min_x, min_y, max_x, max_y],
    )?;
    conn.execute(
        "INSERT INTO gpkg_tile_matrix VALUES (?1, 0, 1, 1, ?2, ?3, ?4, ?5)",
        rusqlite::params![layer, nx as i64, ny as i64, pixel_x, pixel_y],
    )?;

    let def = "http://docs.opengeospatial.org/is/17-066r1/17-066r1.html";
    conn.execute(
        "INSERT INTO gpkg_extensions VALUES
            ('gpkg_2d_gridded_coverage_ancillary', NULL, 'gpkg_2d_gridded_coverage', ?1, 'read-write')",
        rusqlite::params![def],
    )?;
    conn.execute(
        "INSERT INTO gpkg_extensions VALUES
            ('gpkg_2d_gridded_tile_ancillary', NULL, 'gpkg_2d_gridded_coverage', ?1, 'read-write')",
        rusqlite::params![def],
    )?;
    conn.execute(
        "INSERT INTO gpkg_extensions VALUES (?1, 'tile_data', 'gpkg_2d_gridded_coverage', ?2, 'read-write')",
        rusqlite::params![layer, def],
    )?;
    conn.execute(
        "INSERT INTO gpkg_2d_gridded_coverage_ancillary
            (tile_matrix_set_name, datatype, scale, offset, precision, data_null, field_name, quantity_definition)
         VALUES (?1, 'integer', ?2, ?3, ?4, 0.0, 'prediction', 'prediction')",
        rusqlite::params![layer, enc.scale, enc.offset, enc.scale],
    )?;

    // The single tile, plus its value statistics.
    conn.execute(
        &format!(
            "INSERT INTO {} (zoom_level, tile_column, tile_row, tile_data) VALUES (0, 0, 0, ?1)",
            quote_ident(layer)
        ),
        rusqlite::params![enc.png],
    )?;
    let tile_id: i64 = conn.query_row(
        &format!("SELECT id FROM {} WHERE zoom_level=0", quote_ident(layer)),
        [],
        |r| r.get(0),
    )?;
    conn.execute(
        "INSERT INTO gpkg_2d_gridded_tile_ancillary
            (tpudt_name, tpudt_id, scale, offset, min, max, mean, std_dev)
         VALUES (?1, ?2, 1.0, 0.0, ?3, ?4, ?5, ?6)",
        rusqlite::params![layer, tile_id, enc.min, enc.max, enc.mean, enc.std_dev],
    )?;
    Ok(())
}

/// A single-band raster read back from a GeoPackage 2D Gridded Coverage.
///
/// Values are in grid storage order (`iy*nx + ix`, y increasing, so row 0 is the
/// southernmost). Cells with no data are `NaN`.
pub struct RasterGrid {
    /// Coverage (tile-pyramid table) name.
    pub name: String,
    /// Spatial reference system id (from `gpkg_contents`).
    pub srs_id: i64,
    /// Number of columns.
    pub nx: usize,
    /// Number of rows.
    pub ny: usize,
    /// `[min_x, min_y, max_x, max_y]` of the coverage extent.
    pub bbox: [f64; 4],
    /// Cell values, grid storage order (`iy*nx + ix`); `NaN` where no data.
    pub values: Vec<f64>,
}

impl RasterGrid {
    /// Samples the value at `(x, y)` by nearest cell (centres on
    /// `grid-value-is-center`). Returns `None` outside the extent or on a
    /// no-data cell, so it can feed covariates straight into kriging.
    pub fn sample(&self, x: f64, y: f64) -> Option<f64> {
        let [min_x, min_y, max_x, max_y] = self.bbox;
        if x < min_x || x > max_x || y < min_y || y > max_y {
            return None;
        }
        let px = (max_x - min_x) / self.nx as f64;
        let py = (max_y - min_y) / self.ny as f64;
        let ix = (((x - min_x) / px) as usize).min(self.nx - 1);
        let iy = (((y - min_y) / py) as usize).min(self.ny - 1);
        let v = self.values[iy * self.nx + ix];
        v.is_finite().then_some(v)
    }
}

/// Lists the 2D-gridded-coverage (raster) layers in a GeoPackage.
pub fn list_raster_layers(path: &Path) -> Result<Vec<String>> {
    let conn = open(path)?;
    let mut stmt = conn
        .prepare(
            "SELECT table_name FROM gpkg_contents \
             WHERE data_type = '2d-gridded-coverage' ORDER BY table_name",
        )
        .context("querying gpkg_contents (is this a GeoPackage?)")?;
    let names = stmt
        .query_map([], |r| r.get::<_, String>(0))?
        .collect::<rusqlite::Result<_>>()?;
    Ok(names)
}

/// Reads a single-band raster from a GeoPackage 2D Gridded Coverage (the inverse
/// of [`write_raster`]). If `layer` is `None`, the single coverage is used.
///
/// Scope: the `integer` datatype (16-bit PNG tiles) with the coverage-level
/// `scale`/`offset` mapping `value = pixel*scale + offset` and `data_null` as
/// the no-data pixel — the layout geostat-rs writes and the common GDAL/QGIS
/// integer coverage. The `float` datatype (TIFF tiles) is not handled. Tiles at
/// the finest zoom level are mosaicked; the row order is flipped from
/// image (north→south) to grid (y increasing).
pub fn read_raster(path: &Path, layer: Option<&str>) -> Result<RasterGrid> {
    let names = list_raster_layers(path)?;
    if names.is_empty() {
        bail!("no 2d-gridded-coverage layers found in {}", path.display());
    }
    let name = match layer {
        Some(l) => names
            .iter()
            .find(|n| n.as_str() == l)
            .cloned()
            .with_context(|| format!("raster layer '{l}' not found in {}", path.display()))?,
        None => {
            if names.len() > 1 {
                bail!(
                    "{} has multiple raster layers; choose one with --layer (available: {})",
                    path.display(),
                    names.join(", ")
                );
            }
            names[0].clone()
        }
    };

    let conn = open(path)?;
    let datatype: String = conn
        .query_row(
            "SELECT datatype FROM gpkg_2d_gridded_coverage_ancillary WHERE tile_matrix_set_name = ?1",
            rusqlite::params![name],
            |r| r.get(0),
        )
        .with_context(|| format!("reading coverage ancillary for '{name}'"))?;
    if datatype != "integer" {
        bail!("coverage '{name}' has datatype '{datatype}'; only 'integer' (PNG) is supported");
    }
    let (scale, offset, data_null): (f64, f64, Option<f64>) = conn.query_row(
        "SELECT scale, offset, data_null FROM gpkg_2d_gridded_coverage_ancillary \
         WHERE tile_matrix_set_name = ?1",
        rusqlite::params![name],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    )?;
    let null_px = data_null.unwrap_or(0.0);

    let (srs_id, min_x, min_y, max_x, max_y): (i64, f64, f64, f64, f64) = conn.query_row(
        "SELECT srs_id, min_x, min_y, max_x, max_y FROM gpkg_tile_matrix_set WHERE table_name = ?1",
        rusqlite::params![name],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
    )?;

    // Finest zoom level holds the full-resolution grid.
    let (zoom, mw, mh, tw, th): (i64, i64, i64, i64, i64) = conn.query_row(
        "SELECT zoom_level, matrix_width, matrix_height, tile_width, tile_height \
         FROM gpkg_tile_matrix WHERE table_name = ?1 ORDER BY zoom_level DESC LIMIT 1",
        rusqlite::params![name],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
    )?;
    let (mw, mh, tw, th) = (mw as usize, mh as usize, tw as usize, th as usize);
    let nx = mw * tw;
    let ny = mh * th;

    // Mosaic the tiles into a north-up image, then flip rows into grid order.
    let mut img = vec![f64::NAN; nx * ny];
    let mut stmt = conn.prepare(&format!(
        "SELECT tile_column, tile_row, tile_data FROM {} WHERE zoom_level = ?1",
        quote_ident(&name)
    ))?;
    let tiles = stmt.query_map(rusqlite::params![zoom], |r| {
        Ok((
            r.get::<_, i64>(0)? as usize,
            r.get::<_, i64>(1)? as usize,
            r.get::<_, Vec<u8>>(2)?,
        ))
    })?;
    for tile in tiles {
        let (tcol, trow, png_blob) = tile?;
        let pixels = decode_png_u16(&png_blob, tw, th)?;
        for ty in 0..th {
            let img_row = trow * th + ty;
            if img_row >= ny {
                continue;
            }
            for tx in 0..tw {
                let img_col = tcol * tw + tx;
                if img_col >= nx {
                    continue;
                }
                let px = pixels[ty * tw + tx] as f64;
                img[img_row * nx + img_col] = if px == null_px {
                    f64::NAN
                } else {
                    px * scale + offset
                };
            }
        }
    }

    // Image rows run north→south; grid storage is y increasing.
    let mut values = vec![f64::NAN; nx * ny];
    for img_row in 0..ny {
        let iy = ny - 1 - img_row;
        values[iy * nx..iy * nx + nx].copy_from_slice(&img[img_row * nx..img_row * nx + nx]);
    }

    Ok(RasterGrid {
        name,
        srs_id,
        nx,
        ny,
        bbox: [min_x, min_y, max_x, max_y],
        values,
    })
}

/// Decodes a 16-bit grayscale PNG tile into a `tw*th` row-major `u16` buffer.
/// `pub` so `fuzz/fuzz_targets/gpkg_decode_png.rs` can drive it directly on
/// arbitrary bytes without a real GeoPackage.
pub fn decode_png_u16(blob: &[u8], tw: usize, th: usize) -> Result<Vec<u16>> {
    let dec = png::Decoder::new(std::io::Cursor::new(blob));
    let mut reader = dec
        .read_info()
        .map_err(|e| anyhow::anyhow!("PNG header: {e}"))?;
    let mut buf = vec![0u8; reader.output_buffer_size()];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| anyhow::anyhow!("PNG data: {e}"))?;
    if info.bit_depth != png::BitDepth::Sixteen || info.color_type != png::ColorType::Grayscale {
        bail!(
            "tile is {:?}/{:?}; expected 16-bit grayscale",
            info.bit_depth,
            info.color_type
        );
    }
    if info.width as usize != tw || info.height as usize != th {
        bail!(
            "tile is {}x{} but the tile matrix declares {tw}x{th}",
            info.width,
            info.height
        );
    }
    let pixels = (0..tw * th)
        .map(|i| u16::from_be_bytes([buf[i * 2], buf[i * 2 + 1]]))
        .collect();
    Ok(pixels)
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
/// `pub` so `fuzz/fuzz_targets/gpkg_decode_point.rs` can drive it directly on
/// arbitrary bytes without a real GeoPackage.
pub fn decode_point(blob: &[u8]) -> Result<(f64, f64)> {
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
    fn encode_raster_reconstructs_values() {
        // 2x3 grid (nx=2, ny=3), storage order iy*nx+ix, y increasing.
        let nx = 2;
        let ny = 3;
        let values = [1.0, 2.0, 3.0, 4.0, f64::NAN, 6.0];
        let enc = encode_raster(nx, ny, &values).unwrap();
        assert_eq!(enc.min, 1.0);
        assert_eq!(enc.max, 6.0);

        // Decode the PNG and check value = pixel*scale + offset, with the
        // north-up row flip (image row 0 = grid top row iy = ny-1).
        let dec = png::Decoder::new(std::io::Cursor::new(&enc.png));
        let mut reader = dec.read_info().unwrap();
        let mut buf = vec![0u8; reader.output_buffer_size()];
        let info = reader.next_frame(&mut buf).unwrap();
        assert_eq!((info.width, info.height), (nx as u32, ny as u32));
        let pixel = |r: usize, c: usize| -> u16 {
            let i = (r * nx + c) * 2;
            u16::from_be_bytes([buf[i], buf[i + 1]])
        };
        for r in 0..ny {
            let iy = ny - 1 - r; // image row r -> grid row iy
            for c in 0..nx {
                let v = values[iy * nx + c];
                let px = pixel(r, c);
                if v.is_finite() {
                    let recon = px as f64 * enc.scale + enc.offset;
                    assert!((recon - v).abs() <= enc.scale, "{v} -> {recon}");
                } else {
                    assert_eq!(px, 0, "NaN must map to nodata pixel 0");
                }
            }
        }
    }

    #[test]
    fn write_raster_builds_gridded_coverage() {
        let path = std::env::temp_dir().join("geostat_rs_raster_test.gpkg");
        let _ = std::fs::remove_file(&path);
        let (nx, ny) = (4, 3);
        let values: Vec<f64> = (0..nx * ny).map(|i| i as f64).collect();
        write_raster(
            &path,
            "kriging",
            4326,
            nx,
            ny,
            [0.0, 0.0, 4.0, 3.0],
            &values,
        )
        .unwrap();

        let conn = open(&path).unwrap();
        let dtype: String = conn
            .query_row(
                "SELECT data_type FROM gpkg_contents WHERE table_name='kriging'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(dtype, "2d-gridded-coverage");
        let n_tiles: i64 = conn
            .query_row("SELECT COUNT(*) FROM \"kriging\"", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n_tiles, 1);
        let dt: String = conn
            .query_row(
                "SELECT datatype FROM gpkg_2d_gridded_coverage_ancillary",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(dt, "integer");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn raster_write_then_read_round_trips() {
        let path = std::env::temp_dir().join("geostat_rs_raster_roundtrip.gpkg");
        let _ = std::fs::remove_file(&path);
        let (nx, ny) = (5, 4);
        // A smooth ramp plus one no-data cell.
        let mut values: Vec<f64> = (0..nx * ny).map(|i| 2.0 + 0.5 * i as f64).collect();
        values[7] = f64::NAN;
        let bbox = [100.0, 200.0, 150.0, 240.0]; // 10x10 m cells
        write_raster(&path, "dem", 32719, nx, ny, bbox, &values).unwrap();

        assert_eq!(list_raster_layers(&path).unwrap(), vec!["dem".to_string()]);
        // A raster-only GeoPackage has no feature layers (and must not error).
        assert!(list_feature_layers(&path).unwrap().is_empty());
        let g = read_raster(&path, None).unwrap();
        assert_eq!((g.nx, g.ny), (nx, ny));
        assert_eq!(g.srs_id, 32719);
        assert_eq!(g.bbox, bbox);

        // 16-bit quantisation: reconstructed values within one scale step, and
        // the no-data cell stays NaN.
        let span = (values
            .iter()
            .copied()
            .filter(|v| v.is_finite())
            .fold(f64::NEG_INFINITY, f64::max)
            - values
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .fold(f64::INFINITY, f64::min))
        .max(f64::MIN_POSITIVE);
        let step = span / 65534.0;
        for (i, &v) in values.iter().enumerate() {
            if v.is_finite() {
                assert!(
                    (g.values[i] - v).abs() <= step + 1e-9,
                    "cell {i}: {} vs {v}",
                    g.values[i]
                );
            } else {
                assert!(g.values[i].is_nan(), "cell {i} should be NaN");
            }
        }

        // Sampling: a point inside cell (ix=0, iy=0) returns that cell; outside
        // the extent returns None; the no-data cell returns None.
        let px = (bbox[2] - bbox[0]) / nx as f64;
        let py = (bbox[3] - bbox[1]) / ny as f64;
        let inside = g.sample(bbox[0] + 0.5 * px, bbox[1] + 0.5 * py).unwrap();
        assert!((inside - g.values[0]).abs() <= step + 1e-9);
        assert!(g.sample(bbox[0] - 1.0, bbox[1]).is_none());
        // Cell index 7 is (ix=2, iy=1); centre it and expect no-data -> None.
        let nd = g.sample(bbox[0] + 2.5 * px, bbox[1] + 1.5 * py);
        assert!(nd.is_none(), "no-data cell must sample to None, got {nd:?}");
        let _ = std::fs::remove_file(&path);
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
