//! Arrow `<->` device buffer conversion.
//!
//! The engine moves columnar data across the host/device boundary in a single
//! canonical form: dense value vectors plus a `u8` validity mask (`1` valid,
//! `0` null). This crate converts Polars [`Series`] values into that form
//!
//! `missing_const_for_fn` is expected crate-wide: several simple methods
//! could be `const fn` in stable Rust, but the ones that match on enums
//! (`DType::from_polars`, `DType::to_polars`, `HostValues::len`) require the
//! unstable `const_convert` feature on this toolchain.

#![expect(clippy::missing_const_for_fn)]
//! ([`series_to_host`]), uploads it ([`upload`]), downloads it back
//! ([`download`]), and rebuilds [`Series`] values ([`host_to_series`]).
//!
//! Multi-chunk inputs are flattened on ingest so kernels always observe one
//! contiguous buffer; outputs are a single chunk. Chunk boundaries are
//! semantically invisible, so this never changes query results.

use cubecl::prelude::{Client, CubeElement};
use cubecl_runtime::server::Handle;
use polars_arrow::{
    array::{Array, BooleanArray, PrimitiveArray},
    bitmap::Bitmap,
    datatypes::ArrowDataType,
};
use polars_buffer::Buffer;
use polars_core::{
    datatypes::{CompatLevel, DataType},
    frame::DataFrame,
    prelude::{Column, PlSmallStr, Series},
};
use thiserror::Error;

/// Error returned when a column cannot cross the host/device boundary.
#[derive(Debug, Error)]
pub enum BufferError {
    /// The dtype is outside the engine's supported subset.
    #[error("unsupported dtype for device transfer: {0:?}")]
    UnsupportedDtype(DataType),
    /// The Arrow array behind a [`Series`] chunk has an unexpected layout.
    #[error("unexpected arrow layout for dtype {0:?}")]
    ArrowLayout(DataType),
    /// Reading a device buffer back failed.
    #[error("device readback failed: {0}")]
    Readback(String),
    /// Rebuilding a [`Series`] from downloaded data failed.
    #[error("failed to rebuild series: {0}")]
    Rebuild(String),
}

/// A dtype in the engine's supported subset.
///
/// # Examples
///
/// ```
/// use polars_core::datatypes::DataType;
/// use polars_cubecl_buffer::DType;
///
/// assert_eq!(DType::from_polars(&DataType::Int32), Some(DType::I32));
/// assert_eq!(DType::from_polars(&DataType::String), None);
/// assert_eq!(DType::I64.to_polars(), DataType::Int64);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    /// 32-bit signed integer.
    I32,
    /// 64-bit signed integer.
    I64,
    /// 32-bit unsigned integer.
    U32,
    /// 64-bit unsigned integer.
    U64,
    /// 32-bit float.
    F32,
    /// 64-bit float.
    F64,
    /// Boolean, stored as `u8` (`1` true, `0` false) on both sides.
    Bool,
}

impl DType {
    /// Map a Polars dtype to the engine subset, or `None` when unsupported.
    #[must_use]
    pub fn from_polars(dtype: &DataType) -> Option<Self> {
        match dtype {
            DataType::Int32 => Some(Self::I32),
            DataType::Int64 => Some(Self::I64),
            DataType::UInt32 => Some(Self::U32),
            DataType::UInt64 => Some(Self::U64),
            DataType::Float32 => Some(Self::F32),
            DataType::Float64 => Some(Self::F64),
            DataType::Boolean => Some(Self::Bool),
            _ => None,
        }
    }

    /// Map back to the Polars dtype.
    #[must_use]
    pub fn to_polars(self) -> DataType {
        match self {
            Self::I32 => DataType::Int32,
            Self::I64 => DataType::Int64,
            Self::U32 => DataType::UInt32,
            Self::U64 => DataType::UInt64,
            Self::F32 => DataType::Float32,
            Self::F64 => DataType::Float64,
            Self::Bool => DataType::Boolean,
        }
    }

    /// Size of one element in bytes.
    #[must_use]
    pub const fn size(self) -> usize {
        match self {
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
            Self::Bool => 1,
        }
    }
}

/// Dense host-side values for one column. Boolean values are `u8`.
///
/// # Examples
///
/// ```
/// use polars_cubecl_buffer::HostValues;
///
/// let values = HostValues::I32(vec![1, 2, 3]);
/// assert_eq!(values.len(), 3);
/// assert!(!values.is_empty());
///
/// let empty = HostValues::F64(Vec::new());
/// assert!(empty.is_empty());
/// ```
#[derive(Clone, Debug, PartialEq)]
pub enum HostValues {
    /// `i32` values.
    I32(Vec<i32>),
    /// `i64` values.
    I64(Vec<i64>),
    /// `u32` values.
    U32(Vec<u32>),
    /// `u64` values.
    U64(Vec<u64>),
    /// `f32` values.
    F32(Vec<f32>),
    /// `f64` values.
    F64(Vec<f64>),
    /// Boolean values as `u8` (`1`/`0`).
    Bool(Vec<u8>),
}

impl HostValues {
    /// Number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::I32(values) => values.len(),
            Self::I64(values) => values.len(),
            Self::U32(values) => values.len(),
            Self::U64(values) => values.len(),
            Self::F32(values) => values.len(),
            Self::F64(values) => values.len(),
            Self::Bool(values) => values.len(),
        }
    }

    /// Whether the column has no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// View the values as a `&[u8]` of their packed little-endian representation.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        match self {
            Self::I32(v) => bytemuck::cast_slice(v).to_vec(),
            Self::I64(v) => bytemuck::cast_slice(v).to_vec(),
            Self::U32(v) => bytemuck::cast_slice(v).to_vec(),
            Self::U64(v) => bytemuck::cast_slice(v).to_vec(),
            Self::F32(v) => bytemuck::cast_slice(v).to_vec(),
            Self::F64(v) => bytemuck::cast_slice(v).to_vec(),
            Self::Bool(v) => v.clone(),
        }
    }
}

/// One column in canonical host form: dense values plus a `u8` validity mask.
#[derive(Clone, Debug, PartialEq)]
pub struct HostColumn {
    name: PlSmallStr,
    dtype: DType,
    values: HostValues,
    validity: Vec<u8>,
}

impl HostColumn {
    /// Column name.
    #[must_use]
    pub fn name(&self) -> &PlSmallStr {
        &self.name
    }

    /// Engine dtype.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Dense values.
    #[must_use]
    pub fn values(&self) -> &HostValues {
        &self.values
    }

    /// Validity mask (`1` valid, `0` null), one entry per row.
    #[must_use]
    pub fn validity(&self) -> &[u8] {
        &self.validity
    }

    /// Number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the column has no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// One column resident on the device: value and validity handles plus length.
#[derive(Clone, Debug)]
pub struct DeviceColumn {
    name: PlSmallStr,
    dtype: DType,
    values: Handle,
    validity: Handle,
    len: usize,
}

impl DeviceColumn {
    /// Column name.
    #[must_use]
    pub fn name(&self) -> &PlSmallStr {
        &self.name
    }

    /// Engine dtype.
    #[must_use]
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Device handle holding the dense values.
    #[must_use]
    pub const fn values(&self) -> &Handle {
        &self.values
    }

    /// Device handle holding the `u8` validity mask.
    #[must_use]
    pub const fn validity(&self) -> &Handle {
        &self.validity
    }

    /// Number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the column has no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Wrap already-resident handles (used by the executor for kernel outputs).
    #[must_use]
    pub fn from_handles(
        name: PlSmallStr,
        dtype: DType,
        values: Handle,
        validity: Handle,
        len: usize,
    ) -> Self {
        Self { name, dtype, values, validity, len }
    }
}

/// Convert a [`Series`] into canonical host form, flattening chunks.
pub fn series_to_host(series: &Series) -> Result<HostColumn, BufferError> {
    let name = series.name().clone();
    let dtype = DType::from_polars(series.dtype())
        .ok_or_else(|| BufferError::UnsupportedDtype(series.dtype().clone()))?;
    let mut values = empty_values(dtype, series.len());
    let mut validity = Vec::with_capacity(series.len());
    for chunk_idx in 0..series.n_chunks() {
        let array = series.to_arrow(chunk_idx, CompatLevel::newest());
        push_chunk(dtype, array.as_ref(), &mut values, &mut validity)?;
    }
    debug_assert_eq!(values.len(), series.len());
    debug_assert_eq!(validity.len(), series.len());
    Ok(HostColumn { name, dtype, values, validity })
}

/// Upload one canonical host column to the device.
///
/// Empty columns get one padding byte: some backends refuse zero-byte
/// allocations, and lengths are tracked separately in [`DeviceColumn::len`]
/// (see [`download`]), so the padding is never observed.
pub fn upload(client: &Client, column: &HostColumn) -> DeviceColumn {
    let values = match &column.values {
        HostValues::I32(values) => {
            client.create_from_slice(non_empty(bytemuck::cast_slice(values)))
        }
        HostValues::I64(values) => {
            client.create_from_slice(non_empty(bytemuck::cast_slice(values)))
        }
        HostValues::U32(values) => {
            client.create_from_slice(non_empty(bytemuck::cast_slice(values)))
        }
        HostValues::U64(values) => {
            client.create_from_slice(non_empty(bytemuck::cast_slice(values)))
        }
        HostValues::F32(values) => {
            client.create_from_slice(non_empty(bytemuck::cast_slice(values)))
        }
        HostValues::F64(values) => {
            client.create_from_slice(non_empty(bytemuck::cast_slice(values)))
        }
        HostValues::Bool(values) => client.create_from_slice(non_empty(values)),
    };
    let validity = client.create_from_slice(non_empty(&column.validity));
    DeviceColumn {
        name: column.name.clone(),
        dtype: column.dtype,
        values,
        validity,
        len: column.len(),
    }
}

fn non_empty(bytes: &[u8]) -> &[u8] {
    if bytes.is_empty() { &[0] } else { bytes }
}

/// Upload every column of a [`DataFrame`], preserving order.
pub fn upload_frame(client: &Client, frame: &DataFrame) -> Result<Vec<DeviceColumn>, BufferError> {
    frame
        .columns()
        .iter()
        .map(|column| {
            series_to_host(column.as_materialized_series()).map(|host| upload(client, &host))
        })
        .collect()
}

/// Download one device column back into canonical host form.
///
/// Empty columns skip the readback: their handles may be padding (see
/// [`upload`]), and there is nothing to read.
pub fn download(client: &Client, column: &DeviceColumn) -> Result<HostColumn, BufferError> {
    let len = column.len();
    if len == 0 {
        return Ok(HostColumn {
            name: column.name.clone(),
            dtype: column.dtype,
            values: empty_values(column.dtype, 0),
            validity: Vec::new(),
        });
    }
    let value_bytes = client
        .read_one(column.values.clone())
        .map_err(|error| BufferError::Readback(error.to_string()))?;
    let validity_bytes = client
        .read_one(column.validity.clone())
        .map_err(|error| BufferError::Readback(error.to_string()))?;
    // Buffer pooling may return a handle larger than `len * type_size`. The
    // kernel only writes `len` elements, so truncate any trailing padding
    // to avoid reading stale data from a recycled allocation.
    let type_size = column.dtype.size();
    let value_len = len.saturating_mul(type_size);
    let values = match column.dtype {
        DType::I32 => {
            let bytes = &value_bytes[..value_len.min(value_bytes.len())];
            HostValues::I32(i32::from_bytes(bytes).to_vec())
        }
        DType::I64 => {
            let bytes = &value_bytes[..value_len.min(value_bytes.len())];
            HostValues::I64(i64::from_bytes(bytes).to_vec())
        }
        DType::U32 => {
            let bytes = &value_bytes[..value_len.min(value_bytes.len())];
            HostValues::U32(u32::from_bytes(bytes).to_vec())
        }
        DType::U64 => {
            let bytes = &value_bytes[..value_len.min(value_bytes.len())];
            HostValues::U64(u64::from_bytes(bytes).to_vec())
        }
        DType::F32 => {
            let bytes = &value_bytes[..value_len.min(value_bytes.len())];
            HostValues::F32(f32::from_bytes(bytes).to_vec())
        }
        DType::F64 => {
            let bytes = &value_bytes[..value_len.min(value_bytes.len())];
            HostValues::F64(f64::from_bytes(bytes).to_vec())
        }
        DType::Bool => {
            let bytes = &value_bytes[..value_len.min(value_bytes.len())];
            HostValues::Bool(bytes.to_vec())
        }
    };
    let validity = validity_bytes[..len.min(validity_bytes.len())].to_vec();
    Ok(HostColumn { name: column.name.clone(), dtype: column.dtype, values, validity })
}

/// Download multiple device columns to host in a single batched readback.
///
/// Batches all value + validity handle reads into one [`Client::read`] call,
/// collapsing `2 × columns` sequential syncs into a single round-trip.
pub fn download_columns(
    client: &Client,
    columns: &[DeviceColumn],
) -> Result<Vec<HostColumn>, BufferError> {
    if columns.is_empty() {
        return Ok(Vec::new());
    }
    let mut handles = Vec::with_capacity(columns.len() * 2);
    for column in columns {
        handles.push(column.values.clone());
        handles.push(column.validity.clone());
    }
    let batch = client.read(handles).map_err(|error| BufferError::Readback(error.to_string()))?;
    let mut out = Vec::with_capacity(columns.len());
    for (column, chunks) in columns.iter().zip(batch.as_chunks::<2>().0) {
        out.push(build_host_column(column, &chunks[0], &chunks[1])?);
    }
    Ok(out)
}

/// Rebuild a [`Series`] from canonical host form.
pub fn host_to_series(column: &HostColumn) -> Result<Series, BufferError> {
    let dtype = column.dtype.to_polars();
    let arrow_dtype = dtype.to_arrow(CompatLevel::newest());
    let validity = validity_bitmap(&column.validity);
    let array = match (&column.values, &arrow_dtype) {
        (HostValues::I32(values), ArrowDataType::Int32) => {
            primitive_array(values.clone(), validity, arrow_dtype)
        }
        (HostValues::I64(values), ArrowDataType::Int64) => {
            primitive_array(values.clone(), validity, arrow_dtype)
        }
        (HostValues::U32(values), ArrowDataType::UInt32) => {
            primitive_array(values.clone(), validity, arrow_dtype)
        }
        (HostValues::U64(values), ArrowDataType::UInt64) => {
            primitive_array(values.clone(), validity, arrow_dtype)
        }
        (HostValues::F32(values), ArrowDataType::Float32) => {
            primitive_array(values.clone(), validity, arrow_dtype)
        }
        (HostValues::F64(values), ArrowDataType::Float64) => {
            primitive_array(values.clone(), validity, arrow_dtype)
        }
        (HostValues::Bool(values), ArrowDataType::Boolean) => {
            let bits: Bitmap = values.iter().map(|value| *value != 0).collect();
            BooleanArray::new(arrow_dtype, bits, validity).boxed()
        }
        _ => return Err(BufferError::ArrowLayout(dtype)),
    };
    Series::from_arrow(column.name.clone(), array)
        .map_err(|error| BufferError::Rebuild(error.to_string()))
}

/// Download a full set of device columns into a [`DataFrame`].
///
/// Batched readback: submits all value + validity handles via
/// [`Client::read`] (a single `read_async` round-trip) instead of calling
/// `read_one` per buffer. On CUDA this collapses `2×columns` syncs into
/// one stream-level submit, reducing driver overhead.
pub fn download_frame(client: &Client, columns: &[DeviceColumn]) -> Result<DataFrame, BufferError> {
    let mut output = Vec::with_capacity(columns.len());

    if columns.is_empty() {
        return DataFrame::new_infer_height(output)
            .map_err(|error| BufferError::Rebuild(error.to_string()));
    }

    // Build one handle list: [val_0, valid_0, val_1, valid_1, ...].
    let mut handles = Vec::with_capacity(columns.len() * 2);
    for column in columns {
        handles.push(column.values.clone());
        handles.push(column.validity.clone());
    }

    // Single readback round-trip.
    let batch = client.read(handles).map_err(|error| BufferError::Readback(error.to_string()))?;

    // Rebuild host columns interleaving value / validity chunks from the
    // batch result.
    for (column, chunks) in columns.iter().zip(batch.as_chunks::<2>().0) {
        let value_bytes = &chunks[0];
        let validity_bytes = &chunks[1];
        let host = build_host_column(column, value_bytes, validity_bytes)?;
        let series = host_to_series(&host)?;
        output.push(Column::from(series));
    }

    DataFrame::new_infer_height(output).map_err(|error| BufferError::Rebuild(error.to_string()))
}

/// Build a [`HostColumn`] from raw value + validity bytes, mirroring the
/// per-field dispatch in [`download`] but without re-reading the device.
fn build_host_column(
    column: &DeviceColumn,
    value_bytes: &[u8],
    validity_bytes: &[u8],
) -> Result<HostColumn, BufferError> {
    let (name, dtype, len) = (column.name.clone(), column.dtype, column.len());
    if len == 0 {
        return Ok(HostColumn { name, dtype, values: empty_values(dtype, 0), validity: Vec::new() });
    }
    // Truncate to `len` elements in case the pooled handle is larger.
    let type_size = dtype.size();
    let value_len = len.saturating_mul(type_size);
    let values = match dtype {
        DType::I32 => HostValues::I32(
            i32::from_bytes(&value_bytes[..value_len.min(value_bytes.len())]).to_vec(),
        ),
        DType::I64 => HostValues::I64(
            i64::from_bytes(&value_bytes[..value_len.min(value_bytes.len())]).to_vec(),
        ),
        DType::U32 => HostValues::U32(
            u32::from_bytes(&value_bytes[..value_len.min(value_bytes.len())]).to_vec(),
        ),
        DType::U64 => HostValues::U64(
            u64::from_bytes(&value_bytes[..value_len.min(value_bytes.len())]).to_vec(),
        ),
        DType::F32 => HostValues::F32(
            f32::from_bytes(&value_bytes[..value_len.min(value_bytes.len())]).to_vec(),
        ),
        DType::F64 => HostValues::F64(
            f64::from_bytes(&value_bytes[..value_len.min(value_bytes.len())]).to_vec(),
        ),
        DType::Bool => HostValues::Bool(value_bytes[..value_len.min(value_bytes.len())].to_vec()),
    };
    let validity = validity_bytes[..len.min(validity_bytes.len())].to_vec();
    Ok(HostColumn { name, dtype, values, validity })
}

fn empty_values(dtype: DType, capacity: usize) -> HostValues {
    match dtype {
        DType::I32 => HostValues::I32(Vec::with_capacity(capacity)),
        DType::I64 => HostValues::I64(Vec::with_capacity(capacity)),
        DType::U32 => HostValues::U32(Vec::with_capacity(capacity)),
        DType::U64 => HostValues::U64(Vec::with_capacity(capacity)),
        DType::F32 => HostValues::F32(Vec::with_capacity(capacity)),
        DType::F64 => HostValues::F64(Vec::with_capacity(capacity)),
        DType::Bool => HostValues::Bool(Vec::with_capacity(capacity)),
    }
}

fn push_chunk(
    dtype: DType,
    array: &dyn Array,
    values: &mut HostValues,
    validity: &mut Vec<u8>,
) -> Result<(), BufferError> {
    match (dtype, values) {
        (DType::I32, HostValues::I32(target)) => {
            push_primitive::<i32>(array, target, validity, &DataType::Int32)?;
        }
        (DType::I64, HostValues::I64(target)) => {
            push_primitive::<i64>(array, target, validity, &DataType::Int64)?;
        }
        (DType::U32, HostValues::U32(target)) => {
            push_primitive::<u32>(array, target, validity, &DataType::UInt32)?;
        }
        (DType::U64, HostValues::U64(target)) => {
            push_primitive::<u64>(array, target, validity, &DataType::UInt64)?;
        }
        (DType::F32, HostValues::F32(target)) => {
            push_primitive::<f32>(array, target, validity, &DataType::Float32)?;
        }
        (DType::F64, HostValues::F64(target)) => {
            push_primitive::<f64>(array, target, validity, &DataType::Float64)?;
        }
        (DType::Bool, HostValues::Bool(target)) => {
            let source = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| BufferError::ArrowLayout(DataType::Boolean))?;
            target.extend(source.values().iter().map(u8::from));
            push_bitmap_validity(source.validity(), source.len(), validity);
        }
        _ => return Err(BufferError::ArrowLayout(dtype.to_polars())),
    }
    Ok(())
}

fn push_primitive<T>(
    array: &dyn Array,
    target: &mut Vec<T>,
    validity: &mut Vec<u8>,
    dtype: &DataType,
) -> Result<(), BufferError>
where
    T: polars_arrow::types::NativeType,
{
    let source = array
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .ok_or_else(|| BufferError::ArrowLayout(dtype.clone()))?;
    target.extend_from_slice(source.values().as_slice());
    push_bitmap_validity(source.validity(), source.len(), validity);
    Ok(())
}

fn push_bitmap_validity(bitmap: Option<&Bitmap>, length: usize, validity: &mut Vec<u8>) {
    match bitmap {
        None => validity.extend(std::iter::repeat_n(1, length)),
        Some(bits) => {
            validity.reserve(length);
            for index in 0..length {
                validity.push(u8::from(bits.get_bit(index)));
            }
        }
    }
}

fn validity_bitmap(validity: &[u8]) -> Option<Bitmap> {
    if validity.iter().all(|value| *value != 0) {
        None
    } else {
        Some(validity.iter().map(|value| *value != 0).collect())
    }
}

fn primitive_array<T>(
    values: Vec<T>,
    validity: Option<Bitmap>,
    dtype: ArrowDataType,
) -> Box<dyn Array>
where
    T: polars_arrow::types::NativeType,
{
    PrimitiveArray::new(dtype, Buffer::from(values), validity).boxed()
}

#[cfg(test)]
mod tests {
    use polars_core::prelude::{NamedFrom, Series};
    use proptest::{prop_assert, prop_assert_eq};

    use super::*;

    fn column_error(error: polars_error::PolarsError) -> BufferError {
        BufferError::Rebuild(error.to_string())
    }

    fn int_frame() -> Result<DataFrame, BufferError> {
        DataFrame::new_infer_height(vec![
            Series::new("a".into(), &[1_i32, 2, 3]).into(),
            Series::new("b".into(), &[Some(4_i64), None, Some(6)]).into(),
        ])
        .map_err(|error| BufferError::Rebuild(error.to_string()))
    }

    #[test]
    fn series_to_host_reads_values_and_validity() -> Result<(), BufferError> {
        let frame = int_frame()?;
        let series = frame.column("a").map_err(column_error)?;
        let host = series_to_host(series.as_materialized_series())?;
        assert_eq!(host.values, HostValues::I32(vec![1, 2, 3]));
        assert_eq!(host.validity, vec![1, 1, 1]);

        let series = frame.column("b").map_err(column_error)?;
        let host = series_to_host(series.as_materialized_series())?;
        assert_eq!(host.values.len(), 3);
        assert_eq!(host.validity, vec![1, 0, 1]);
        Ok(())
    }

    #[test]
    fn unsupported_dtype_is_rejected() -> Result<(), BufferError> {
        let series = Series::new("s".into(), &["x", "y"]);
        let result = series_to_host(&series);
        assert!(matches!(result, Err(BufferError::UnsupportedDtype(DataType::String))));
        Ok(())
    }

    /// `host -> Series -> host` must preserve every value and null, for
    /// every supported dtype and any number of rows (including empty).
    ///
    /// This is the invariant the whole engine rests on: a round trip that
    /// dropped or reordered a row would surface as a wrong GPU result, not
    /// as a conversion error. Values are dense — one slot per row, with a
    /// placeholder where the row is null — so the null slots are checked
    /// too, not skipped.
    fn assert_round_trip(values: HostValues, validity: Vec<u8>) -> Result<(), BufferError> {
        let dtype = match &values {
            HostValues::I32(_) => DType::I32,
            HostValues::I64(_) => DType::I64,
            HostValues::U32(_) => DType::U32,
            HostValues::U64(_) => DType::U64,
            HostValues::F32(_) => DType::F32,
            HostValues::F64(_) => DType::F64,
            HostValues::Bool(_) => DType::Bool,
        };
        let height = validity.len();
        assert_eq!(values.len(), height, "dense values must have one slot per row");
        let column = HostColumn { name: PlSmallStr::from_static("c"), dtype, values, validity };
        assert_eq!(column.is_empty(), height == 0);
        // Accessor methods are the only other read path besides the fields;
        // asserting both keeps a mutant in either from surviving silently.
        assert_eq!(column.name(), &PlSmallStr::from_static("c"));
        assert_eq!(column.dtype(), dtype);
        assert_eq!(column.values(), &column.values);
        assert_eq!(column.validity(), column.validity.as_slice());

        let series = host_to_series(&column)?;
        let host = series_to_host(&series)?;
        assert_eq!(host.len(), height, "row count changed across the round trip");
        assert_eq!(host.validity, column.validity, "validity changed across the round trip");
        assert_eq!(host.values, column.values, "values changed across the round trip");
        Ok(())
    }

    /// Split `Option` rows into dense values plus a validity mask, keeping a
    /// zero slot where the row is null.
    fn dense_from_options<T: Default>(values: Vec<Option<T>>) -> (Vec<T>, Vec<u8>) {
        let validity = values.iter().map(|value| u8::from(value.is_some())).collect();
        let dense = values.into_iter().map(Option::unwrap_or_default).collect();
        (dense, validity)
    }

    #[test]
    fn non_empty_pads_only_empty_slices() -> Result<(), BufferError> {
        assert_eq!(non_empty(&[]), &[0]);
        assert_eq!(non_empty(&[1, 2]), &[1, 2]);
        Ok(())
    }

    #[test]
    fn empty_column_round_trips() -> Result<(), BufferError> {
        assert_round_trip(HostValues::I32(Vec::new()), Vec::new())
    }

    /// Upload/download round trip on the CPU device: every value, null, name,
    /// and row count must survive the host→device→host crossing.
    ///
    /// This is the only test that touches real device buffers, so it pins
    /// `upload_frame`, `download_frame`, and the `DeviceColumn` accessors.
    #[cfg(feature = "cpu")]
    #[test]
    fn device_round_trip_preserves_frame() -> Result<(), BufferError> {
        let client = cubecl::Device::Cpu(cubecl::device::CpuDevice).client();
        let frame = int_frame()?;
        let device_columns = upload_frame(&client, &frame)?;
        assert_eq!(device_columns.len(), 2);
        assert_eq!(device_columns[0].len(), 3);
        assert!(!device_columns[0].is_empty());
        assert_eq!(device_columns[0].name(), &PlSmallStr::from("a"));
        let downloaded = download_frame(&client, &device_columns)?;
        assert_eq!(downloaded.height(), frame.height());
        assert_eq!(downloaded.get_column_names(), frame.get_column_names());
        for name in frame.get_column_names() {
            let expected =
                series_to_host(frame.column(name).map_err(column_error)?.as_materialized_series())?;
            let actual = series_to_host(
                downloaded.column(name).map_err(column_error)?.as_materialized_series(),
            )?;
            assert_eq!(actual.values, expected.values, "values for column {name}");
            assert_eq!(actual.validity, expected.validity, "validity for column {name}");
        }
        Ok(())
    }

    proptest::proptest! {
        #[test]
        fn round_trip_preserves_i32(values: Vec<Option<i32>>) {
            let (dense, validity) = dense_from_options(values);
            prop_assert!(assert_round_trip(HostValues::I32(dense), validity).is_ok());
        }

        #[test]
        fn round_trip_preserves_i64(values: Vec<Option<i64>>) {
            let (dense, validity) = dense_from_options(values);
            prop_assert!(assert_round_trip(HostValues::I64(dense), validity).is_ok());
        }

        #[test]
        fn round_trip_preserves_u32(values: Vec<Option<u32>>) {
            let (dense, validity) = dense_from_options(values);
            prop_assert!(assert_round_trip(HostValues::U32(dense), validity).is_ok());
        }

        #[test]
        fn round_trip_preserves_u64(values: Vec<Option<u64>>) {
            let (dense, validity) = dense_from_options(values);
            prop_assert!(assert_round_trip(HostValues::U64(dense), validity).is_ok());
        }

        #[test]
        fn round_trip_preserves_f32(values: Vec<Option<f32>>) {
            let (dense, validity) = dense_from_options(values);
            prop_assert!(assert_round_trip(HostValues::F32(dense), validity).is_ok());
        }

        #[test]
        fn round_trip_preserves_f64(values: Vec<Option<f64>>) {
            let (dense, validity) = dense_from_options(values);
            prop_assert!(assert_round_trip(HostValues::F64(dense), validity).is_ok());
        }

        #[test]
        fn round_trip_preserves_bool(values: Vec<Option<bool>>) {
            let (dense, validity) = dense_from_options(values);
            let dense = dense.into_iter().map(u8::from).collect();
            prop_assert!(assert_round_trip(HostValues::Bool(dense), validity).is_ok());
        }

        /// `validity_bitmap` returns `None` iff every entry is non-zero
        /// (all-valid), and `Some(bitmap)` reproducing the mask otherwise.
        #[test]
        fn validity_bitmap_is_none_iff_all_valid(values: Vec<u8>) {
            let bitmap = validity_bitmap(&values);
            if values.iter().all(|value| *value != 0) {
                prop_assert!(bitmap.is_none(), "all-valid validity should yield None");
            } else {
                let bitmap = if let Some(bitmap) = bitmap {
                    bitmap
                } else {
                    prop_assert!(false, "any-zero validity should yield Some");
                    unreachable!()
                };
                for (index, value) in values.iter().enumerate() {
                    prop_assert_eq!(bitmap.get_bit(index), *value != 0);
                }
            }
        }

        /// `non_empty` only replaces empty slices; any non-empty slice is
        /// returned unchanged (so real column bytes are never padded).
        #[test]
        fn non_empty_leaves_nonempty_unchanged(bytes: Vec<u8>) {
            let result = non_empty(&bytes);
            if bytes.is_empty() {
                prop_assert_eq!(result, &[0]);
            } else {
                prop_assert_eq!(result, bytes.as_slice());
            }
        }
    }

    #[test]
    fn validity_bitmap_all_valid_yields_none() {
        assert!(validity_bitmap(&[1, 1, 1]).is_none());
        assert!(validity_bitmap(&[255, 42, 1]).is_none());
        assert!(validity_bitmap(&[]).is_none(), "empty validity is trivially all-valid");
    }

    #[test]
    fn validity_bitmap_any_null_yields_some() {
        assert!(validity_bitmap(&[1, 0, 1]).is_some());
        assert!(validity_bitmap(&[0, 0, 0]).is_some());
        assert!(validity_bitmap(&[0]).is_some());
    }
}
