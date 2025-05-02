use crate::{
    parser::pixel_parser::{PixelParseRegionError, batch_parse_region_to_dataframe},
    parser::{parse_descriptor, parse_header, parse_offset_table},
    types::offset_table::OffsetLookupTable,
    types::{MibiDescriptor, header::Header},
    utils::misc::Coordinate,
};

use crate::utils::{
    channel_manager::extract_channels, channel_manager::find_by_mass,
    channel_manager::find_by_target, file_utils::read_binary_file_mmap,
};
use polars::prelude::*;
use rayon::prelude::*;
use std::error::Error;

/// A structure representing a parsed MIBI file with efficient access methods
pub struct MibiFile {
    pub header: Header,
    pub descriptor: MibiDescriptor,
    pub offset_table: OffsetLookupTable,
    pub mmap_data: memmap2::Mmap,
    pub panel: DataFrame,
}

impl MibiFile {
    /// Open and parse a MIBI file, extracting header, descriptor and offset table
    pub fn new(path: &str) -> Result<Self, Box<dyn Error>> {
        let mibi_file = Self::open(path)?;
        Ok(mibi_file)
    }

    pub fn open(path: &str) -> Result<Self, Box<dyn Error>> {
        // Memory map the file
        let mmap_data = read_binary_file_mmap(path)?;

        // Parse the header
        let mut input = &mmap_data[..];
        let header =
            parse_header(&mut input).map_err(|e| format!("Failed to parse header: {:?}", e))?;

        // Parse the descriptor
        let descriptor = parse_descriptor(&mut input, &header)
            .map_err(|e| format!("Failed to parse descriptor: {:?}", e))?;

        // Parse the offset table
        let offset_table = parse_offset_table(&mut input, &header)
            .map_err(|e| format!("Failed to parse offset table: {:?}", e))?;

        // Initialize channel manager
        let panel = match extract_channels(&descriptor) {
            Ok(df) => df,
            Err(e) => {
                eprintln!("Warning: Failed to extract channels: {}", e);
                DataFrame::default()
            }
        };

        Ok(Self {
            header,
            descriptor,
            offset_table,
            panel,
            mmap_data,
        })
    }
    /// Get a summary of the file contents
    pub fn get_summary(&self) -> String {
        let mut result = String::new();

        // Header info
        result.push_str(&format!("Header:\n"));
        result.push_str(&format!(
            "  Frame size: {}x{}\n",
            self.header.size_x_pixels, self.header.size_y_pixels
        ));
        result.push_str(&format!("  Frames: {}\n", self.header.frame_count));
        result.push_str(&format!(
            "  Triggers per pixel: {}\n",
            self.header.triggers_per_pixel
        ));

        // Descriptor info
        result.push_str("\nMIBI File Information:\n");
        if let Some(run_name) = self.descriptor.run_name() {
            result.push_str(&format!("  Run: {}\n", run_name));
        }
        if let Some(id) = &self.descriptor.id {
            result.push_str(&format!("  ID: {}\n", id));
        }
        if let Some(instrument_id) = self.descriptor.instrument_id() {
            result.push_str(&format!("  Instrument: {}\n", instrument_id));
        }
        result
    }

    /// Find channels by target name (case-insensitive)
    pub fn find_channels_by_target(&self, target: &str) -> Result<DataFrame, PolarsError> {
        let lazy_frame = find_by_target(&self.panel, target)?;
        lazy_frame.collect()
    }

    /// Find a channel by mass with a given tolerance
    pub fn find_channels_by_mass(
        &self,
        mass: f64,
        tolerance: Option<f64>,
    ) -> Result<DataFrame, PolarsError> {
        let lazy_frame = find_by_mass(&self.panel, mass, tolerance)?;
        lazy_frame.collect()
    }

    /// Get the total number of channels
    pub fn channel_count(&self) -> usize {
        self.panel.height()
    }

    /// Get channel mass range information
    pub fn get_channel_mass_ranges(&self) -> Result<(Option<f64>, Option<f64>), PolarsError> {
        let df = &self.panel;

        let min_mass = df.column("mass")?.f64()?.min();

        let max_mass = df.column("mass")?.f64()?.max();

        Ok((min_mass, max_mass))
    }

    /// Parses pixel data within a specified rectangular region into a Polars DataFrame.
    ///
    /// Arguments:
    /// * `start_coord` - The top-left Coordinate { row, col } of the region (inclusive).
    /// * `end_coord` - The bottom-right Coordinate { row, col } of the region (inclusive).
    ///
    /// Returns a Polars DataFrame containing pulses from the region, or a PixelParseRegionError.
    pub fn parse_region_to_dataframe(
        &self,
        start_coord: Coordinate,
        end_coord: Coordinate,
    ) -> Result<DataFrame, PixelParseRegionError> {
        // Call the underlying parser function with the file's data
        batch_parse_region_to_dataframe(
            &self.mmap_data,
            &self.offset_table,
            &self.header,
            start_coord,
            end_coord,
        )
    }

    /// Parses the full image in parallel across multiple chunks.
    ///
    /// Arguments:
    /// * `num_chunks` - The desired number of chunks to divide the image into for parallel processing.
    ///
    /// Returns a single Polars DataFrame containing pulses from the entire image, or an error.
    pub fn parse_full_image_parallel(
        &self,
        num_chunks: usize,
    ) -> Result<DataFrame, PixelParseRegionError> {
        // Calculate chunk ranges for parallel processing
        let chunk_ranges = self.offset_table.calculate_row_chunks(num_chunks, None); // None for full-width chunks

        if chunk_ranges.is_empty() {
            println!("Warning: No chunks generated for parallel processing.");
            return Ok(DataFrame::default());
        }

        // Process chunks in parallel and collect resulting DataFrames
        let chunk_dfs: Vec<DataFrame> = chunk_ranges
            .par_iter()
            .map(|(start, end)| self.parse_region_to_dataframe(*start, *end))
            .collect::<Result<Vec<_>, _>>()?;

        chunk_dfs
            .into_iter()
            .fold(Ok(DataFrame::default()), |acc_result, next_df| {
                // Propagate previous error or perform vstack
                acc_result.and_then(|acc_df| {
                    if acc_df.height() == 0 {
                        // If accumulator is empty (initial state), just take the next df
                        Ok(next_df)
                    } else {
                        // Otherwise, vstack the next df onto the accumulator
                        acc_df
                            .vstack(&next_df)
                            .map_err(PixelParseRegionError::DataFrameCreationFailed)
                    }
                })
            })
    }
}
