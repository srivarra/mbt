use crate::parser::{parse_descriptor, parse_header, parse_offset_table};
use crate::types::{
    MibiDescriptor, header::Header, offset_table::OffsetLookupTable,
};

use crate::utils::{
    channel_manager::extract_channels, file_utils::read_binary_file_mmap,
    channel_manager::find_by_target, channel_manager::find_by_mass,
};
use crate::processing;
use memmap2;
use std::error::Error;
use ndarray::Array4;
use polars::prelude::*;

/// A structure representing a parsed MIBI file with efficient access methods
pub struct MibiFile {
    pub header: Header,
    pub descriptor: MibiDescriptor,
    pub offset_table: OffsetLookupTable,
    pub mmap_data: memmap2::Mmap,
    pub panel: DataFrame,
}

/// Statistics collected during streaming processing
pub struct ProcessingStats {
    pub total_pixels: usize,
    pub processed_pixels: usize,
    pub total_pulses: u64,
    pub max_pulse_time: u16,
    pub max_pulse_intensity: u16,
}

impl From<processing::ProcessingStats> for ProcessingStats {
    fn from(stats: processing::ProcessingStats) -> Self {
        Self {
            total_pixels: stats.total_pixels,
            processed_pixels: stats.processed_pixels,
            total_pulses: stats.total_pulses,
            max_pulse_time: stats.max_pulse_time,
            max_pulse_intensity: stats.max_pulse_intensity,
        }
    }
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
        let header = parse_header(&mut input)
            .map_err(|e| format!("Failed to parse header: {:?}", e))?;

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

        // Mass calibration
        if let Some(mass_cal) = self.descriptor.mass_calibration() {
            result.push_str("\nMass Calibration:\n");
            if let Some(masses) = &mass_cal.masses {
                if let Some(bins) = &mass_cal.bins {
                    for (i, (mass, bin)) in masses.iter().zip(bins.iter()).enumerate().take(5) {
                        result.push_str(&format!(
                            "  Point {}: Mass {} at bin {}\n",
                            i + 1,
                            mass,
                            bin
                        ));
                    }
                    if masses.len() > 5 {
                        result.push_str(&format!("  ... and {} more points\n", masses.len() - 5));
                    }
                }
            }
        }

        result
    }

    /// Extract bin file data into a 4D array (3 × x × y × channels)
    ///
    /// If low_range and high_range are provided, those values are prioritized.
    /// Otherwise, the channel ranges are automatically derived from the descriptor.
    ///
    /// # Arguments
    /// * `low_range` - Optional starting integration ranges for each channel
    /// * `high_range` - Optional stopping integration ranges for each channel
    /// * `calc_intensity` - Optional flags for calculating intensity and intensity*width
    ///
    /// # Returns
    /// A 4D array with dimensions (3, x, y, channels) where the first dimension
    /// represents [counts, intensity, intensity*width]
    pub fn extract_bin_data(
        &self,
        low_range: Option<&[u16]>,
        high_range: Option<&[u16]>,
        calc_intensity: Option<&[bool]>,
    ) -> Result<Array4<u32>, String> {
        // Convert the offset_table to the format needed by extract_bin
        let offset_table = self.offset_table.as_pixel_offsets(self.mmap_data.len() as u64);

        // Call the extract_bin function with our parameters
        processing::extract_bin(
            &self.mmap_data,
            &self.header,
            &offset_table,
            Some(&self.descriptor),
            low_range,
            high_range,
            calc_intensity,
        )
    }


    /// Find channels by target name (case-insensitive)
    pub fn find_channels_by_target(&self, target: &str) -> Result<DataFrame, PolarsError> {
        let lazy_frame = find_by_target(&self.panel, target)?;
        lazy_frame.collect()
    }

    /// Find a channel by mass with a given tolerance
    pub fn find_channels_by_mass(&self, mass: f64, tolerance: Option<f64>) -> Result<DataFrame, PolarsError> {
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

        let min_mass = df.column("mass")?
            .f64()?
            .min();

        let max_mass = df.column("mass")?
            .f64()?
            .max();

        Ok((min_mass, max_mass))
    }
}
