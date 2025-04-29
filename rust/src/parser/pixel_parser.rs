use crate::types::header::Header;
use crate::types::offset_table::OffsetLookupTable;
use crate::types::pixel_data::{PixelData, PulseEvent, TriggerEvent};
use crate::utils::misc::Coordinate;
use polars::prelude::*;
use rayon::iter::ParallelIterator;
use rayon::prelude::*;
use std::error::Error;
use std::fmt;
use winnow::{
    Parser,
    binary::{le_u8, le_u16, le_u32},
    combinator::repeat,
    error::ContextError,
};

/// Parses a single pulse event record (5 bytes):
/// - 2 bytes: pulse time (u16, little‑endian)
/// - 1 byte: pulse width (u8, little‑endian)
/// - 2 bytes: pulse intensity (u16, little‑endian)
pub fn parse_pulse_event<'a>(input: &mut &'a [u8]) -> Result<PulseEvent, ContextError> {
    let (time, width, intensity) = (le_u16, le_u8, le_u16).parse_next(input)?;

    Ok(PulseEvent::builder()
        .time(time)
        .width(width)
        .intensity(intensity)
        .build())
}

/// Parses a trigger event for a pixel.
/// Trigger record format (8 bytes):
/// - Bytes 0-3: Trigger ID (sequential counter within pixel)
/// - Bytes 4-5: Buffer configuration value (typically 100, 0xFFFF for first trigger)
/// - Bytes 6-7: Number of pulse events that follow
/// First, read an 8‑byte trigger record and extract the number of pulse events from bytes 6..8.
/// Then, parse exactly that many pulse events.
pub fn parse_trigger_event<'a>(input: &mut &'a [u8]) -> Result<TriggerEvent, ContextError> {
    // Read the 8-byte trigger record directly into fields
    let (_trigger_id, _metadata_field, num_pulses) = (le_u32, le_u16, le_u16).parse_next(input)?;

    // Conditionally parse pulse events
    if num_pulses == 0 {
        // If no pulses, return an empty TriggerEvent immediately
        Ok(TriggerEvent::builder()
            .num_pulses(0)
            .pulses(Vec::new())
            .build())
    } else {
        // If there are pulses, parse them
        repeat(num_pulses as usize, parse_pulse_event)
            .map(move |pulses| {
                TriggerEvent::builder()
                    .num_pulses(num_pulses)
                    .pulses(pulses)
                    .build()
            })
            .parse_next(input) // Pass the remaining input to the pulse parser
    }
}

pub fn parse_pixel_data(input: &[u8], header: &Header) -> Result<PixelData, String> {
    let mut input = input;

    repeat(header.triggers_per_pixel as usize, parse_trigger_event)
        .map(|trigger_events: Vec<TriggerEvent>| {
            // The filter step here is redundant because parse_trigger_event
            // already handles num_pulses == 0 correctly.
            PixelData::builder().trigger_events(trigger_events).build()
        })
        .parse_next(&mut input)
        .map_err(|e| format!("Error parsing pixel data: {:?}", e))
}

/// Custom error type for batch_parse_region_to_dataframe.
#[derive(Debug)]
pub enum PixelParseRegionError {
    InvalidDimensions,
    InvalidCoordinates {
        start: Coordinate,
        end: Coordinate,
    },
    CoordinatesOutOfBounds {
        coord: Coordinate,
        dims: (usize, usize),
    },
    InvalidOffset {
        pixel: Coordinate,
        start_offset: u64,
        end_offset: u64,
        reason: String,
    },
    OffsetOverflow {
        pixel: Coordinate,
    },
    PixelParsingFailed {
        pixel: Coordinate,
        source: String,
    },
    DataFrameCreationFailed(PolarsError),
}

impl fmt::Display for PixelParseRegionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PixelParseRegionError::InvalidDimensions => {
                write!(f, "Image rows and cols cannot be zero")
            }
            PixelParseRegionError::InvalidCoordinates { start, end } => {
                write!(
                    f,
                    "Invalid coordinates: start {:?} must be <= end {:?}",
                    start, end
                )
            }
            PixelParseRegionError::CoordinatesOutOfBounds { coord, dims } => {
                write!(
                    f,
                    "End coordinate {:?} out of bounds for image dimensions (rows={}, cols={})",
                    coord, dims.0, dims.1
                )
            }
            PixelParseRegionError::InvalidOffset {
                pixel,
                start_offset,
                end_offset,
                reason,
            } => {
                write!(
                    f,
                    "Invalid offset range for pixel {:?}: start={}, end={}. Reason: {}",
                    pixel, start_offset, end_offset, reason
                )
            }
            PixelParseRegionError::OffsetOverflow { pixel } => {
                write!(
                    f,
                    "Offset value overflow converting u64 to usize for pixel {:?}",
                    pixel
                )
            }
            PixelParseRegionError::PixelParsingFailed { pixel, source } => {
                write!(f, "Error parsing pixel data at {:?}: {}", pixel, source)
            }
            PixelParseRegionError::DataFrameCreationFailed(e) => {
                write!(f, "Failed to create Polars DataFrame: {}", e)
            }
        }
    }
}

impl Error for PixelParseRegionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            PixelParseRegionError::DataFrameCreationFailed(e) => Some(e),
            _ => None,
        }
    }
}

/// Parses pixel data within a specified rectangular region and returns a Polars DataFrame.
/// Version using map, collect, flatten, and manual unzip.
/// Keep this version as it performs better
pub fn batch_parse_region_to_dataframe(
    data: &[u8],
    offset_table: &OffsetLookupTable,
    header: &Header,
    start_coord: Coordinate,
    end_coord: Coordinate,
) -> Result<DataFrame, PixelParseRegionError> {
    let start_row = start_coord.y;
    let start_col = start_coord.x;
    let end_row = end_coord.y;
    let end_col = end_coord.x;
    let img_rows = offset_table.size_y_pixels;
    let img_cols = offset_table.size_x_pixels;

    // --- Input Validation ---
    if img_rows == 0 || img_cols == 0 {
        return Err(PixelParseRegionError::InvalidDimensions);
    }
    if start_row > end_row || start_col > end_col {
        return Err(PixelParseRegionError::InvalidCoordinates {
            start: start_coord,
            end: end_coord,
        });
    }
    if end_row >= img_rows || end_col >= img_cols {
        return Err(PixelParseRegionError::CoordinatesOutOfBounds {
            coord: end_coord,
            dims: (img_rows, img_cols),
        });
    }
    // --- End Validation ---

    // Generate coordinates for the region
    let coords: Vec<Coordinate> = (start_row..=end_row)
        .flat_map(|row| (start_col..=end_col).map(move |col| Coordinate { y: row, x: col }))
        .collect();

    // Define the type for the tuple representing a single pulse event with coordinates
    // Correct order: col, row, trigger_idx, time (u16), width (u8), intensity (u16)
    type PulseTuple = (u32, u32, u16, u16, u8, u16);

    // Reverting to map/collect/flatten approach as try_flat_map is not suitable here
    let results: Vec<Vec<PulseTuple>> = coords
        .into_par_iter()
        // Map each coordinate to a Result<Vec<PulseTuple>, Error>
        .map(|coord| -> Result<Vec<PulseTuple>, PixelParseRegionError> {
            _process_pixel(coord, data, offset_table, header, img_rows, img_cols)
        })
        .collect::<Result<Vec<_>, _>>()?; // Collect results into Result<Vec<Vec<PulseTuple>>, Error>

    // Flatten the successfully collected Vec<Vec<PulseTuple>>
    let flattened_results: Vec<PulseTuple> = results.into_iter().flatten().collect();

    // Estimate capacity based on actual results for efficiency
    let capacity = flattened_results.len();
    let mut pixel_cols: Vec<u32> = Vec::with_capacity(capacity);
    let mut pixel_rows: Vec<u32> = Vec::with_capacity(capacity);
    let mut trigger_indices: Vec<u16> = Vec::with_capacity(capacity);
    let mut pulse_times: Vec<u16> = Vec::with_capacity(capacity);
    let mut pulse_widths: Vec<u8> = Vec::with_capacity(capacity);
    let mut pulse_intensities: Vec<u16> = Vec::with_capacity(capacity);

    // Unzip the flattened results into column vectors
    // Correct destructuring order: col, row, trigger_idx, time, width, intensity
    for (col, row, trigger_idx, time, width, intensity) in flattened_results {
        pixel_cols.push(col);
        pixel_rows.push(row);
        trigger_indices.push(trigger_idx);
        // Correct variable usage for push
        pulse_times.push(time); // time is u16
        pulse_widths.push(width); // width is u8
        pulse_intensities.push(intensity); // intensity is u16
    }

    // Create Polars DataFrame from collected vectors
    let df = df!(
        "pixel_x" => pixel_cols,
        "pixel_y" => pixel_rows,
        "trigger_index" => trigger_indices,
        "pulse_time" => pulse_times,
        "pulse_width" => pulse_widths,
        "pulse_intensity" => pulse_intensities,
    )
    .map_err(PixelParseRegionError::DataFrameCreationFailed)?; // Map PolarsError

    Ok(df)
}

// Helper function to process a single pixel within the region
fn _process_pixel(
    coord: Coordinate,
    data: &[u8],
    offset_table: &OffsetLookupTable,
    header: &Header,
    img_rows: usize,
    img_cols: usize,
) -> Result<Vec<(u32, u32, u16, u16, u8, u16)>, PixelParseRegionError> {
    let row = coord.y;
    let col = coord.x;

    // Define the type for the tuple representing a single pulse event with coordinates
    // Correct order: col, row, trigger_idx, time (u16), width (u8), intensity (u16)
    type PulseTuple = (u32, u32, u16, u16, u8, u16);

    // --- Calculate offsets --- //
    let start_offset = offset_table.offsets[[row, col]];
    let end_offset = if col + 1 < img_cols {
        offset_table.offsets[[row, col + 1]]
    } else if row + 1 < img_rows {
        offset_table.offsets[[row + 1, 0]]
    } else {
        data.len() as u64
    };

    // --- Convert offsets to usize --- //
    let start_usize = start_offset as usize;
    let end_usize = end_offset as usize;

    // --- Slice data and parse --- //
    let chunk = &data[start_usize..end_usize];
    match parse_pixel_data(chunk, header) {
        Ok(pixel_data) => {
            // Collect the pulse tuples for this pixel into a Vec
            let pulses_for_pixel: Vec<PulseTuple> = pixel_data
                .trigger_events
                .into_iter()
                .enumerate()
                .flat_map(move |(trigger_idx, trigger_event)| {
                    trigger_event.pulses.into_iter().map(move |pulse| {
                        // Correct tuple creation order: col, row, trigger_idx, time, width, intensity
                        (
                            col as u32,
                            row as u32,
                            trigger_idx as u16,
                            pulse.time,      // u16
                            pulse.width,     // u8
                            pulse.intensity, // u16
                        )
                    })
                })
                .collect();
            Ok(pulses_for_pixel)
        }
        Err(e) => {
            // Propagate the error
            Err(PixelParseRegionError::PixelParsingFailed {
                pixel: coord,
                source: e,
            })
        }
    }
}
