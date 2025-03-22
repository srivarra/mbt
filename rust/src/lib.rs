pub mod mibi_file;
pub mod parser;
pub mod python;
pub mod types;
pub mod utils;
pub mod processing;

// Re-export core types for easier access
pub use mibi_file::{MibiFile, ProcessingStats};
pub use types::{
    MibiDescriptor, header::Header, offset_table::OffsetLookupTable, pixel_data::PixelData,
};

// Only keep imports actually used in the file
use pyo3::prelude::*;

#[pyfunction]
fn hello_world() -> PyResult<String> {
    Ok("Hello, world!".to_string())
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_world, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Helper function to construct the file path for tissue data
    #[allow(dead_code)]
    fn get_tissue_file_path(filename: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("data")
            .join("tissue")
            .join(filename)
    }

    #[allow(dead_code)]
    fn get_real_world_bin_file_path(filename: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("data")
            .join("other")
            .join(filename)
    }

    // Define test cases that we might use in the future
    #[allow(dead_code)]
    const TEST_FILES: &[&str] = &[
        "fov-1-scan-1.bin",
        "fov-2-scan-1.bin",
        "fov-3-scan-1.bin",
        "fov-4-scan-1.bin",
        "fov-15-scan-1.bin",
    ];

    #[test]
    fn test_mibi_file_capabilities() -> Result<(), Box<dyn std::error::Error>> {
        let file_path = get_real_world_bin_file_path("fov-1-scan-1.bin");
        println!(
            "\nTesting MibiFile capabilities with: {}",
            file_path.display()
        );

        // Open the file
        let mibi_file = MibiFile::open(file_path.to_str().unwrap())?;

        // 1. Test basic file information
        println!(
            "File dimensions: {}x{}",
            mibi_file.header.size_x_pixels, mibi_file.header.size_y_pixels
        );
        println!(
            "Triggers per pixel: {}",
            mibi_file.header.triggers_per_pixel
        );
        println!("Frame count: {}", mibi_file.header.frame_count);

        // 2. Test descriptor access
        if let Some(id) = &mibi_file.descriptor.id {
            println!("File ID: {}", id);
        }
        if let Some(run_name) = mibi_file.descriptor.run_name() {
            println!("Run name: {}", run_name);
        }

        // Print offset table
        // println!("Offset table: {:?}", mibi_file.offset_table);

        // Test pixel data extraction
        let (x, y) = (10, 10);
        let offset = mibi_file.offset_table.offsets[[y, x]] as usize;
        let offset2 = mibi_file.offset_table.offsets[[y, x + 1]] as usize;

        // Create a copy of the data instead of a reference
        let pixel_data_vec = mibi_file.mmap_data[offset..offset2].to_vec();

        println!("Offsets: {}, {}", offset, offset2);
        use crate::parser::pixel_parser::parse_pixel_data;
        let pixel_data = parse_pixel_data(&pixel_data_vec, &mibi_file.header)
            .map_err(|e| Box::<dyn std::error::Error>::from(e))?;

        println!("Number of trigger events: {}", pixel_data.trigger_events.len());

        // Print first 3 trigger events in detail
        for (i, event) in pixel_data.trigger_events.iter().take(3).enumerate() {
            println!("Trigger Event {}: {} pulses", i, event.num_pulses);
            // Print first 2 pulses of each trigger event
            for (j, pulse) in event.pulses.iter().take(2).enumerate() {
                println!("  Pulse {}: time={}, width={}, intensity={}",
                    j, pulse.time, pulse.width, pulse.intensity);
            }
        }

        Ok(())
    }

    #[test]
    fn test_mibi_file_extract_bin_data() -> Result<(), Box<dyn std::error::Error>> {
        let file_path = get_real_world_bin_file_path("fov-1-scan-1.bin");
        println!("\nTesting MibiFile extract_bin_data with: {}", file_path.display());

        // Open the file
        let mibi_file = MibiFile::open(file_path.to_str().unwrap())?;


        let _mass_start = mibi_file.descriptor.fov_mass_start();
        let _mass_stop = mibi_file.descriptor.fov_mass_stop();

        let df = &mibi_file.panel;
        println!("DataFrame: {:?}", df);

        let df = mibi_file.find_channels_by_target("Calprotectin")?;
        println!("DataFrame: {:?}", df);


        // // Test 1: Using explicit ranges based on the real channels from the JSON
        // println!("Test 1: Using explicit ranges based on real channel data");

        // // Values from the JSON - Calprotectin (Ga69), Chymase (Ga71), SMA (Y89)
        // let low_range = vec![68, 70, 88];  // Mass - 1 as a simple bin range
        // let high_range = vec![70, 72, 90];  // Mass + 1 as a simple bin range
        // let calc_intensity = vec![true, true, true];  // Calculate intensity for all channels

        // let img_data_explicit = mibi_file.extract_bin_data(
        //     Some(&low_range),
        //     Some(&high_range),
        //     Some(&calc_intensity),
        // )?;

        // println!("Explicit ranges - Image data shape: {:?}", img_data_explicit.shape());
        // println!("Explicit ranges - Number of channels: {}", img_data_explicit.shape()[3]);

        // // Report on each channel separately
        // for i in 0..img_data_explicit.shape()[3] {
        //     let channel_name = match i {
        //         0 => "Calprotectin (Ga69)",
        //         1 => "Chymase (Ga71)",
        //         2 => "SMA (Y89)",
        //         _ => "Unknown",
        //     };

        //     let counts_layer = img_data_explicit.slice(ndarray::s![0, .., .., i]);

        //     // Calculate stats for this channel
        //     let mut min_val = u32::MAX;
        //     let mut max_val = 0u32;
        //     let mut sum = 0u32;
        //     let mut count = 0;

        //     counts_layer.iter().for_each(|&x| {
        //         if x < min_val { min_val = x; }
        //         if x > max_val { max_val = x; }
        //         sum += x;
        //         count += 1;
        //     });

        //     let mean = if count > 0 { sum as f64 / count as f64 } else { 0.0 };

        //     println!("Channel {} ({}) - min: {}, max: {}, mean: {:.2}",
        //         i, channel_name, min_val, max_val, mean
        //     );
        // }

        // // Test 2: Using descriptor-derived ranges
        // println!("\nTest 2: Using descriptor-derived ranges");
        // // Print available channels from descriptor
        // let channel_details = mibi_file.get_channel_details();
        // println!("Available channels in descriptor:");
        // for (idx, (name, mass, mass_start, mass_stop)) in channel_details.iter().enumerate() {
        //     println!(
        //         "  {}: {} (mass={:?}, range={:?}-{:?})",
        //         idx, name, mass, mass_start, mass_stop
        //     );
        // }

        // // Extract using descriptor ranges
        // let img_data_auto = mibi_file.extract_bin_data(None, None, None)?;

        // println!("Auto ranges - Image data shape: {:?}", img_data_auto.shape());
        // println!("Auto ranges - Number of channels: {}", img_data_auto.shape()[3]);

        // // Print some sample counts for the first channel from the auto-derived data
        // if img_data_auto.shape()[3] > 0 {
        //     let counts_layer = img_data_auto.slice(ndarray::s![0, .., .., 0]);

        //     // Use iteration to find min/max and sum for average
        //     let mut min_val = u32::MAX;
        //     let mut max_val = 0u32;
        //     let mut sum = 0u32;
        //     let mut count = 0;

        //     counts_layer.iter().for_each(|&x| {
        //         if x < min_val { min_val = x; }
        //         if x > max_val { max_val = x; }
        //         sum += x;
        //         count += 1;
        //     });

        //     let mean = if count > 0 { sum as f64 / count as f64 } else { 0.0 };

        //     println!("Auto first channel counts - min: {}, max: {}, mean: {:.2}",
        //         min_val, max_val, mean
        //     );
        // } else {
        //     println!("No channels found in auto-derived data");
        // }

        Ok(())
    }
}
