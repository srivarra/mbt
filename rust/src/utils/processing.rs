use crate::parser::pixel_parser::{parse_single_pixel, process_pixels_streaming};
use crate::types::header::Header;
use crate::types::offset_table::OffsetLookupTable;
use crate::types::pixel_data::PixelData;
use std::error::Error;

/// Statistics collected during streaming processing
#[derive(Default)]
pub struct ProcessingStats {
    pub total_pixels: usize,
    pub processed_pixels: usize,
    pub total_pulses: u64,
    pub max_pulse_time: u16,
    pub max_pulse_intensity: u16,
}

/// Convert TOF value to mass using calibration
///
/// # Arguments
/// * `tof` - Time of flight value to convert
/// * `masses` - Mass calibration points
/// * `bins` - Corresponding bin values for the calibration points
///
/// # Returns
/// * `Option<f64>` - The calculated mass value or None if calibration failed
pub fn tof_to_mass(tof: u16, masses: &[f64], bins: &[i32]) -> Option<f64> {
    if !masses.is_empty() && masses.len() == bins.len() {
        // Find the bin range that contains our TOF value
        let tof_i32 = tof as i32; // Convert tof to i32 for comparison
        for i in 0..bins.len() - 1 {
            let lower_bin = bins[i];
            let upper_bin = bins[i + 1];

            if tof_i32 >= lower_bin && tof_i32 <= upper_bin {
                // Perform linear interpolation
                let bin_range = upper_bin - lower_bin;
                let mass_range = masses[i + 1] - masses[i];
                let bin_pos = (tof_i32 - lower_bin) as f64 / bin_range as f64;
                let mass = masses[i] + (bin_pos * mass_range);
                return Some(mass);
            }
        }
    }

    // Fallback approximate conversion (placeholder)
    Some(tof as f64 / 100.0)
}

/// Count pulses in a specific mass range
pub fn count_pulses_at_mass(pixel: &PixelData, mass: f64, window: f64) -> u32 {
    // Simple TOF range calculation (replace with actual conversion logic if needed)
    let low_tof = ((mass - window) * 100.0) as u16; // Simple approximation
    let high_tof = ((mass + window) * 100.0) as u16; // Simple approximation

    let mut count = 0;
    for trigger in &pixel.trigger_events {
        for pulse in &trigger.pulses {
            if pulse.time >= low_tof && pulse.time <= high_tof {
                count += 1;
            }
        }
    }

    count
}

/// Process all pixels in a frame in a memory-efficient streaming manner
pub fn process_frame_streaming<F>(
    mmap_data: &[u8],
    header: &Header,
    offset_table: &OffsetLookupTable,
    frame: usize,
    mut process_fn: F,
) -> Result<ProcessingStats, Box<dyn Error>>
where
    F: FnMut(usize, usize, &PixelData) -> (),
{
    // Initialize statistics
    let mut stats = ProcessingStats {
        total_pixels: header.size_x_pixels as usize * header.size_y_pixels as usize,
        processed_pixels: 0,
        total_pulses: 0,
        max_pulse_time: 0,
        max_pulse_intensity: 0,
    };

    // Process all pixels without storing them in memory
    let mut pixel_processor = |_y: usize, _x: usize, pixel: &PixelData| {
        stats.processed_pixels += 1;

        for trigger in &pixel.trigger_events {
            stats.total_pulses += trigger.num_pulses as u64;

            for pulse in &trigger.pulses {
                stats.max_pulse_time = stats.max_pulse_time.max(pulse.time);
                stats.max_pulse_intensity = stats.max_pulse_intensity.max(pulse.intensity);
            }
        }
    };

    // Process pixels using the streaming parser
    process_pixels_streaming(mmap_data, header, offset_table, frame, |y, x, pixel| {
        // Process the pixel with the user-provided function first
        process_fn(y, x, pixel);
        // Then update our statistics
        pixel_processor(y, x, pixel);
    })?;

    Ok(stats)
}

/// Generate a heatmap of pulse counts for a frame
pub fn generate_pulse_count_heatmap(
    mmap_data: &[u8],
    header: &Header,
    offset_table: &OffsetLookupTable,
    frame: usize,
) -> Result<Vec<Vec<u32>>, Box<dyn Error>> {
    let size_y = header.size_y_pixels as usize;
    let size_x = header.size_x_pixels as usize;
    let mut heatmap = vec![vec![0u32; size_x]; size_y];

    process_pixels_streaming(mmap_data, header, offset_table, frame, |y, x, pixel| {
        let pulse_count: u32 = pixel
            .trigger_events
            .iter()
            .map(|trigger| trigger.num_pulses as u32)
            .sum();

        heatmap[y][x] = pulse_count;
    })?;

    Ok(heatmap)
}

/// Create an intensity image for counts of pulses in a specific mass range
pub fn create_mass_intensity_image(
    mmap_data: &[u8],
    header: &Header,
    offset_table: &OffsetLookupTable,
    frame: usize,
    mass: f64,
    window: f64,
) -> Result<Vec<Vec<u32>>, Box<dyn Error>> {
    let size_y = header.size_y_pixels as usize;
    let size_x = header.size_x_pixels as usize;
    let mut counts = vec![vec![0u32; size_x]; size_y];

    process_pixels_streaming(mmap_data, header, offset_table, frame, |y, x, pixel| {
        counts[y][x] = count_pulses_at_mass(pixel, mass, window);
    })?;

    Ok(counts)
}

/// Access an individual pixel by coordinates
pub fn get_pixel_at_coordinates(
    mmap_data: &[u8],
    header: &Header,
    offset_table: &OffsetLookupTable,
    frame: usize,
    y: usize,
    x: usize,
) -> Result<PixelData, Box<dyn Error>> {
    if let Some(offset) = offset_table.get_offset(frame, y, x) {
        let pixel_data = &mmap_data[offset as usize..];
        parse_single_pixel(pixel_data, header, offset_table, frame, y, x).map_err(|e| {
            format!(
                "Failed to parse pixel at ({},{}) in frame {}: {:?}",
                x, y, frame, e
            )
            .into()
        })
    } else {
        Err(format!(
            "Invalid pixel coordinates: ({},{}) in frame {}",
            x, y, frame
        )
        .into())
    }
}
