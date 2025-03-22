use crate::types::header::Header;
use crate::parser::pixel_parser::parse_trigger_event;
use ndarray::{Array3, Array4};
use winnow::Parser;
use crate::types::MibiDescriptor;

/// Counts the total number of pulses across all pixels in the MIBI file
pub fn extract_total_counts(
    mmap_data: &[u8],
    header: &Header,
    offset_table: &[[u64; 2]], // Array of [start, end] offsets for each pixel
) -> Result<u64, String> {
    let mut total_pulses = 0u64;

    // Iterate through each pixel's data
    for &[start_offset, end_offset] in offset_table {
        // Get the pixel data slice
        let pixel_data_slice = &mmap_data[start_offset as usize..end_offset as usize];

        // Process events directly from the binary slice to avoid redundant parsing
        let mut input = pixel_data_slice;

        // Process each trigger event directly
        for _ in 0..header.triggers_per_pixel {
            // Parse a single trigger event
            let trigger_event = match parse_trigger_event.parse_next(&mut input) {
                Ok(event) => event,
                Err(e) => return Err(format!("Error parsing trigger event: {:?}", e)),
            };

            // Count pulses
            total_pulses += trigger_event.num_pulses as u64;
        }
    }

    Ok(total_pulses)
}

/// Extracts bin file data into a 4D array (3 × x × y × channels)
///
/// # Arguments
/// * `mmap_data` - The memory-mapped file data
/// * `header` - The MIBI file header
/// * `offset_table` - Array of [start, end] offsets for each pixel
/// * `descriptor` - Optional MIBI descriptor for channel information (used if low_range/high_range are None)
/// * `low_range` - Optional starting integration ranges for each channel (prioritized if provided)
/// * `high_range` - Optional stopping integration ranges for each channel (prioritized if provided)
/// * `calc_intensity` - Optional flags for whether to calculate intensity and intensity*width images
pub fn extract_bin(
    mmap_data: &[u8],
    header: &Header,
    offset_table: &[[u64; 2]],
    descriptor: Option<&MibiDescriptor>,
    low_range: Option<&[u16]>,
    high_range: Option<&[u16]>,
    calc_intensity: Option<&[bool]>,
) -> Result<Array4<u32>, String> {
    // Determine channel ranges based on input or descriptor
    let (low_bins, high_bins, intensity_flags) = match (low_range, high_range, descriptor) {
        // Case 1: User provided explicit ranges - these take priority
        (Some(low), Some(high), _) => {
            if low.len() != high.len() {
                return Err("low_range and high_range must have the same length".to_string());
            }

            let intensity = match calc_intensity {
                Some(flags) => {
                    if flags.len() != low.len() {
                        return Err("calc_intensity must have the same length as ranges".to_string());
                    }
                    flags.to_vec()
                },
                None => vec![true; low.len()]  // Default to calculating intensity for all channels
            };

            (low.to_vec(), high.to_vec(), intensity)
        },

        // Case 2: Use descriptor to get channel ranges
        (None, None, Some(desc)) => {
            // Get channel information from the descriptor
            let panel = match &desc.panel {
                Some(panel) if !panel.is_empty() => panel,
                _ => return Err("No channel panel found in descriptor".to_string()),
            };

            // Get mass calibration for converting mass to bin
            let mass_calibration = match desc.mass_calibration() {
                Some(cal) => cal,
                None => return Err("No mass calibration found in descriptor".to_string()),
            };

            // Extract mass gain and offset for conversion
            let mass_gain = mass_calibration.mass_gain.unwrap_or(1.0);
            let mass_offset = mass_calibration.mass_offset.unwrap_or(0.0);

            let mut low_bins = Vec::with_capacity(panel.len());
            let mut high_bins = Vec::with_capacity(panel.len());

            // Convert masses to bins for each channel
            for channel in panel {
                let start_mass = channel.mass_start.unwrap_or_else(|| {
                    // If mass_start not available, derive from mass with a small offset
                    channel.mass.unwrap_or(0.0) - 0.5
                });

                let stop_mass = channel.mass_stop.unwrap_or_else(|| {
                    // If mass_stop not available, derive from mass with a small offset
                    channel.mass.unwrap_or(0.0) + 0.5
                });

                // Convert mass to bin using the calibration
                let start_bin = ((start_mass - mass_offset) / mass_gain) as u16;
                let stop_bin = ((stop_mass - mass_offset) / mass_gain) as u16;

                low_bins.push(start_bin);
                high_bins.push(stop_bin);
            }

            // Default to calculating intensity for all channels
            let intensity = match calc_intensity {
                Some(flags) => {
                    if flags.len() != panel.len() {
                        return Err("calc_intensity must have the same length as channel count".to_string());
                    }
                    flags.to_vec()
                },
                None => vec![true; panel.len()]
            };

            (low_bins, high_bins, intensity)
        },

        // Case 3: No valid input
        _ => return Err("Either low_range/high_range or descriptor must be provided".to_string()),
    };

    let num_channels = low_bins.len();
    let size_x = header.size_x_pixels as usize;
    let size_y = header.size_y_pixels as usize;

    // Create 3D array: (3, x, y) where 3 represents [counts, intensity, intensity*width]
    let mut img_data = Array3::zeros((3, size_x * size_y, num_channels));

    // Helper function to find the channel index for a given time
    let find_channel = |time: u16| -> Option<usize> {
        for (i, (&low, &high)) in low_bins.iter().zip(high_bins.iter()).enumerate() {
            if time >= low && time <= high {
                return Some(i);
            }
        }
        None
    };

    // Iterate through each pixel's data
    for (pix_idx, &[start_offset, end_offset]) in offset_table.iter().enumerate() {
        // Get the pixel data slice
        let pixel_data_slice = &mmap_data[start_offset as usize..end_offset as usize];

        // Process events directly from the binary slice to avoid redundant parsing
        let mut input = pixel_data_slice;

        // Process each trigger event directly
        for _ in 0..header.triggers_per_pixel {
            // Parse a single trigger event
            let trigger_event = match parse_trigger_event.parse_next(&mut input) {
                Ok(event) => event,
                Err(e) => return Err(format!("Error parsing trigger event: {:?}", e)),
            };

            // Process each pulse in the trigger event
            for pulse in &trigger_event.pulses {
                if let Some(channel_idx) = find_channel(pulse.time) {
                    // Increment count for this channel
                    img_data[[0, pix_idx, channel_idx]] += 1;

                    // Calculate intensity and intensity*width if requested
                    if intensity_flags[channel_idx] {
                        img_data[[1, pix_idx, channel_idx]] += pulse.intensity as u32;
                        img_data[[2, pix_idx, channel_idx]] += (pulse.intensity as u32) * (pulse.width as u32);
                    }
                }
            }
        }
    }

    // Reshape the array to (3, x, y, channels)
    Ok(img_data.into_shape_with_order((3, size_x, size_y, num_channels)).unwrap())
}
