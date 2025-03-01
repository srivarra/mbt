use crate::parser::{
    parse_descriptor_tolerant, parse_descriptor_tolerant_with_fallback, parse_header,
    parse_offset_table,
};
use crate::types::{
    MibiDescriptor, header::Header, offset_table::OffsetLookupTable, pixel_data::PixelData,
};
use crate::utils::ChannelManager;
use crate::utils::file_utils::read_binary_file_mmap;
use crate::utils::processing;
use memmap2;
use std::error::Error;
use std::path::Path;
use std::fs;

/// A structure representing a parsed MIBI file with efficient access methods
pub struct MibiFile {
    pub header: Header,
    pub descriptor: MibiDescriptor,
    pub offset_table: OffsetLookupTable,
    mmap_data: memmap2::Mmap,
    channel_manager: ChannelManager,
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
    ///
    ///
    ///
    pub fn new(path: &str) -> Result<Self, Box<dyn Error>> {
        let mibi_file = Self::open(path)?;
        Ok(mibi_file)
    }

    pub fn open(path: &str) -> Result<Self, Box<dyn Error>> {
        // Memory map the file
        let mmap_data = read_binary_file_mmap(path)?;

        // Parse the header
        let (remaining, header) =
            parse_header(&mmap_data).map_err(|e| format!("Failed to parse header: {:?}", e))?;

        // Parse the descriptor
        let (remaining, descriptor) =
            parse_descriptor_tolerant_with_fallback(remaining, &header, false)
                .or_else(|_| parse_descriptor_tolerant(remaining, &header))
                .map_err(|e| format!("Failed to parse descriptor: {:?}", e))?;

        // Parse the offset table
        let (_, offset_table) = parse_offset_table(remaining, &header)
            .map_err(|e| format!("Failed to parse offset table: {:?}", e))?;

        // Initialize channel manager
        let channel_manager = ChannelManager::new(&descriptor);

        Ok(Self {
            header,
            descriptor,
            offset_table,
            mmap_data,
            channel_manager,
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

    /// Access individual pixel by coordinates
    pub fn get_pixel(&self, frame: usize, y: usize, x: usize) -> Result<PixelData, Box<dyn Error>> {
        processing::get_pixel_at_coordinates(
            &self.mmap_data,
            &self.header,
            &self.offset_table,
            frame,
            y,
            x,
        )
    }

    /// Process all pixels in a memory-efficient streaming manner
    pub fn process_streaming<F>(
        &self,
        frame: usize,
        process_fn: F,
    ) -> Result<ProcessingStats, Box<dyn Error>>
    where
        F: FnMut(usize, usize, &PixelData) -> (),
    {
        let stats = processing::process_frame_streaming(
            &self.mmap_data,
            &self.header,
            &self.offset_table,
            frame,
            process_fn,
        )?;
        Ok(stats.into())
    }

    /// Generate a heatmap of pulse counts
    pub fn generate_pulse_count_heatmap(
        &self,
        frame: usize,
    ) -> Result<Vec<Vec<u32>>, Box<dyn Error>> {
        processing::generate_pulse_count_heatmap(
            &self.mmap_data,
            &self.header,
            &self.offset_table,
            frame,
        )
    }

    /// Count pulses that fall within a specific mass range
    pub fn count_pulses_for_mass(
        &self,
        frame: usize,
        mass: f64,
        window: f64,
    ) -> Result<Vec<Vec<u32>>, Box<dyn Error>> {
        processing::create_mass_intensity_image(
            &self.mmap_data,
            &self.header,
            &self.offset_table,
            frame,
            mass,
            window,
        )
    }

    #[cfg(feature = "parallel")]
    /// Process pixels in parallel
    pub fn process_parallel<F, T>(
        &self,
        frame: usize,
        process_fn: F,
    ) -> Result<Vec<T>, Box<dyn Error>>
    where
        F: Fn(usize, usize, PixelData) -> T + Send + Sync,
        T: Send,
    {
        use rayon::prelude::*;

        // Get pixel coordinates
        let size_y = self.header.size_y_pixels as usize;
        let size_x = self.header.size_x_pixels as usize;
        let coords: Vec<(usize, usize)> = (0..size_y)
            .flat_map(|y| (0..size_x).map(move |x| (y, x)))
            .collect();

        // Process pixels in parallel
        let results: Vec<T> = coords
            .par_iter()
            .filter_map(|(y, x)| match self.get_pixel(frame, *y, *x) {
                Ok(pixel) => Some(process_fn(*y, *x, pixel)),
                Err(_) => None,
            })
            .collect();

        Ok(results)
    }

    /// Find a channel by name or mass
    pub fn find_channel(&self, identifier: &str) -> Option<&crate::types::Channel> {
        self.channel_manager.find_channel(identifier)
    }

    /// Get available channel names
    pub fn get_channel_names(&self) -> Vec<String> {
        self.channel_manager.get_channel_names()
    }

    /// Get a mass channel image
    pub fn get_mass_channel(
        &self,
        frame: usize,
        mass: f64,
        window: f64,
    ) -> Result<Vec<Vec<u32>>, Box<dyn Error>> {
        self.count_pulses_for_mass(frame, mass, window)
    }

    /// Get a specifically named channel
    pub fn get_channel_by_name(
        &self,
        frame: usize,
        channel_name: &str,
        window_factor: f64,
    ) -> Result<Vec<Vec<u32>>, Box<dyn Error>> {
        if let Some(channel) = self.find_channel(channel_name) {
            if let Some(mass) = channel.mass {
                // Default window is 0.5 Da, adjust by window_factor
                // Use a wider window by default (5.0 Da) to account for calibration issues
                let window = 5.0 * window_factor;
                return self.count_pulses_for_mass(frame, mass, window);
            }
            return Err(format!("Channel '{}' has no associated mass", channel_name).into());
        }
        Err(format!("Channel '{}' not found", channel_name).into())
    }

    /// Get an iterator of all valid (x, y) coordinates in the image
    pub fn iter_coordinates(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        let size_x = self.header.size_x_pixels as usize;
        let size_y = self.header.size_y_pixels as usize;
        (0..size_y).flat_map(move |y| (0..size_x).map(move |x| (x, y)))
    }

    /// Export the mass calibration data
    pub fn export_mass_calibration(&self) -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
        if let Some(mass_cal) = self.descriptor.mass_calibration() {
            if let (Some(masses), Some(bins)) = (&mass_cal.masses, &mass_cal.bins) {
                if !masses.is_empty() && masses.len() == bins.len() {
                    // Convert bins to f64 for easier usage
                    let bins_f64: Vec<f64> = bins.iter().map(|&bin| bin as f64).collect();
                    return Ok((masses.clone(), bins_f64));
                }
            }
        }
        Err("Mass calibration not available".into())
    }

    /// Convert TOF value to mass using calibration
    pub fn tof_to_mass(&self, tof: u16) -> Option<f64> {
        if let Some(mass_cal) = self.descriptor.mass_calibration() {
            if let (Some(masses), Some(bins)) = (&mass_cal.masses, &mass_cal.bins) {
                return processing::tof_to_mass(tof, masses, bins);
            }
        }

        // Fallback approximate conversion
        Some(tof as f64 / 100.0)
    }

    /// Save a channel image as a TIFF file
    ///
    /// # Arguments
    /// * `frame` - The frame number to process
    /// * `channel_name` - Name of the channel to save
    /// * `window_factor` - Factor to adjust the mass window (default is 1.0)
    /// * `output_path` - Path where the TIFF file should be saved
    ///
    /// # Returns
    /// Result indicating success or failure
    pub fn save_channel_as_tiff(
        &self,
        frame: usize,
        channel_name: &str,
        window_factor: f64,
        output_path: &str,
    ) -> Result<(), Box<dyn Error>> {
        use std::fs::File;
        use std::io::BufWriter;
        use tiff::encoder::TiffEncoder;
        use tiff::encoder::colortype;

        // Generate the channel image
        let image = self.get_channel_by_name(frame, channel_name, window_factor)?;

        // Get dimensions
        let height = image.len();
        if height == 0 {
            return Err("Empty image data".into());
        }
        let width = image[0].len();

        // Find maximum value to determine bit depth
        let max_value = image.iter()
            .flat_map(|row| row.iter())
            .max()
            .cloned()
            .unwrap_or(0);

        // Create a file for writing
        let file = File::create(output_path)?;
        let mut writer = BufWriter::new(file);

        // Handle different bit depths
        if max_value <= 255 {
            // 8-bit image
            let mut encoder = TiffEncoder::new(&mut writer)?;
            let mut data = Vec::with_capacity(width * height);

            // Convert to 8-bit values
            for row in &image {
                for &val in row {
                    if max_value > 0 {
                        // Scale to 0-255 range
                        data.push((val as f32 * 255.0 / max_value as f32) as u8);
                    } else {
                        data.push(0);
                    }
                }
            }

            // Create tiff
            encoder.write_image::<colortype::Gray8>(
                width as u32,
                height as u32,
                &data,
            )?;
        } else if max_value <= 65535 {
            // 16-bit image
            let mut encoder = TiffEncoder::new(&mut writer)?;
            let mut data = Vec::with_capacity(width * height);

            // Convert to 16-bit values
            for row in &image {
                for &val in row {
                    if max_value > 0 {
                        // Scale to 0-65535 range
                        data.push((val as f32 * 65535.0 / max_value as f32) as u16);
                    } else {
                        data.push(0);
                    }
                }
            }

            // Create tiff
            encoder.write_image::<colortype::Gray16>(
                width as u32,
                height as u32,
                &data,
            )?;
        } else {
            // 32-bit image - we'll use 16-bit and scale
            let mut encoder = TiffEncoder::new(&mut writer)?;
            let mut data = Vec::with_capacity(width * height);

            // Convert to 16-bit values with scaling
            for row in &image {
                for &val in row {
                    if max_value > 0 {
                        // Scale to 0-65535 range
                        data.push((val as f32 * 65535.0 / max_value as f32) as u16);
                    } else {
                        data.push(0);
                    }
                }
            }

            // Create tiff
            encoder.write_image::<colortype::Gray16>(
                width as u32,
                height as u32,
                &data,
            )?;
        }

        Ok(())
    }

    #[cfg(feature = "parallel")]
    /// Generate intensity images for multiple channels in parallel
    ///
    /// This function processes multiple mass channels in parallel, creating
    /// intensity images for each specified channel.
    ///
    /// # Arguments
    /// * `frame` - The frame number to process
    /// * `channel_indices` - Vector of channel identifiers (names or masses)
    /// * `window_factor` - Factor to adjust the mass window (default window is 0.5 Da)
    ///
    /// # Returns
    /// A vector of (channel_name, intensity_image) pairs
    pub fn generate_channel_images_parallel(
        &self,
        frame: usize,
        channel_indices: &[String],
        window_factor: f64,
    ) -> Result<Vec<(String, Vec<Vec<u32>>)>, Box<dyn Error>> {
        use rayon::prelude::*;

        // Default window is 0.5 Da, adjust by window_factor
        let window = 0.5 * window_factor;

        // Find all channels
        let channels: Vec<_> = channel_indices
            .iter()
            .filter_map(|name| {
                self.find_channel(name)
                    .map(|channel| (name.clone(), channel.clone()))
            })
            .collect();

        if channels.is_empty() {
            return Err("No valid channels found".into());
        }

        // Generate images in parallel
        let results: Vec<_> = channels
            .par_iter()
            .filter_map(|(name, channel)| {
                channel.mass.map(|mass| {
                    let image = self
                        .count_pulses_for_mass(frame, mass, window)
                        .unwrap_or_else(|_| {
                            // Create empty image on error
                            vec![
                                vec![0; self.header.size_x_pixels as usize];
                                self.header.size_y_pixels as usize
                            ]
                        });
                    (name.clone(), image)
                })
            })
            .collect();

        Ok(results)
    }

    /// Save multiple channel images as TIFF files
    ///
    /// # Arguments
    /// * `frame` - The frame number to process
    /// * `channel_names` - List of channel names to save
    /// * `window_factor` - Factor to adjust the mass window (default is 1.0)
    /// * `output_dir` - Directory where the TIFF files should be saved
    /// * `prefix` - Optional prefix for the filenames
    ///
    /// # Returns
    /// A vector of results indicating success or failure for each channel
    pub fn save_channels_as_tiff(
        &self,
        frame: usize,
        channel_names: &[String],
        window_factor: f64,
        output_dir: &str,
        prefix: Option<&str>,
    ) -> Vec<Result<String, String>> {
        use std::path::Path;
        use std::fs;

        // Create the output directory if it doesn't exist
        if let Err(e) = fs::create_dir_all(output_dir) {
            return vec![Err(format!("Failed to create output directory: {}", e))];
        }

        // Process each channel, potentially in parallel if the feature is enabled
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            channel_names.par_iter().map(|channel_name| {
                let sanitized_name = channel_name.replace(|c: char| !c.is_alphanumeric(), "_");
                let filename = match prefix {
                    Some(p) => format!("{}_{}.tiff", p, sanitized_name),
                    None => format!("{}.tiff", sanitized_name),
                };
                let output_path = Path::new(output_dir).join(filename);

                match self.save_channel_as_tiff(frame, channel_name, window_factor, output_path.to_str().unwrap()) {
                    Ok(_) => Ok(format!("Successfully saved {}", output_path.display())),
                    Err(e) => Err(format!("Failed to save {}: {}", output_path.display(), e)),
                }
            }).collect()
        }

        // Non-parallel version
        #[cfg(not(feature = "parallel"))]
        {
            channel_names.iter().map(|channel_name| {
                let sanitized_name = channel_name.replace(|c: char| !c.is_alphanumeric(), "_");
                let filename = match prefix {
                    Some(p) => format!("{}_{}.tiff", p, sanitized_name),
                    None => format!("{}.tiff", sanitized_name),
                };
                let output_path = Path::new(output_dir).join(filename);

                match self.save_channel_as_tiff(frame, channel_name, window_factor, output_path.to_str().unwrap()) {
                    Ok(_) => Ok(format!("Successfully saved {}", output_path.display())),
                    Err(e) => Err(format!("Failed to save {}: {}", output_path.display(), e)),
                }
            }).collect()
        }
    }
}
