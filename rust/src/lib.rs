pub mod mibi_file;
pub mod parser;
pub mod python;
pub mod types;
pub mod utils;

// Re-export core types for easier access
pub use mibi_file::{MibiFile, ProcessingStats};
pub use types::{
    MibiDescriptor, header::Header, offset_table::OffsetLookupTable, pixel_data::PixelData,
};
pub use utils::ChannelManager;

// Only keep imports actually used in the file
use pyo3::prelude::*;
use std::collections::HashMap;
use std::io;

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

/// Parse a MIBI binary file using memory mapping and random access
/// for better performance with large files
pub fn parse_file_mmap(path: &str) -> Result<String, io::Error> {
    use crate::parser::data_start_offset;

    // Use the new MibiFile struct for parsing
    let mibi_file =
        MibiFile::open(path).map_err(|e| io::Error::new(io::ErrorKind::Other, format!("{}", e)))?;

    let mut result = String::new();

    // Add the file summary
    result.push_str(&mibi_file.get_summary());

    // Offset table information
    result.push_str(&format!(
        "\nOffset table entries: {}\n",
        mibi_file.offset_table.offsets.len()
    ));

    // Data start offset
    let data_start = data_start_offset(&mibi_file.header);
    result.push_str(&format!("Data starts at byte offset: {}\n", data_start));

    // Example of random access: get pixel at (10,10) in frame 0
    const SAMPLE_X: usize = 10;
    const SAMPLE_Y: usize = 10;
    const SAMPLE_FRAME: usize = 0;

    result.push_str(&format!("\nDirect pixel access example:\n"));
    match mibi_file.get_pixel(SAMPLE_FRAME, SAMPLE_Y, SAMPLE_X) {
        Ok(pixel) => {
            // Summary of the pixel data
            let total_pulses: usize = pixel
                .trigger_events
                .iter()
                .map(|trigger| trigger.pulses.len())
                .sum();

            result.push_str(&format!(
                "Pixel at ({}, {}) in frame {}:\n",
                SAMPLE_X, SAMPLE_Y, SAMPLE_FRAME
            ));
            result.push_str(&format!(
                "  Total trigger events: {}\n",
                pixel.trigger_events.len()
            ));
            result.push_str(&format!("  Total pulses: {}\n", total_pulses));

            // Show details of first trigger event if available
            if !pixel.trigger_events.is_empty() && !pixel.trigger_events[0].pulses.is_empty() {
                let first_pulse = &pixel.trigger_events[0].pulses[0];
                result.push_str(&format!(
                    "  First pulse - Time: {}, Width: {}, Intensity: {}\n",
                    first_pulse.time, first_pulse.width, first_pulse.intensity
                ));
            }
        }
        Err(e) => {
            result.push_str(&format!(
                "Failed to get pixel ({}, {}) in frame {}: {}\n",
                SAMPLE_X, SAMPLE_Y, SAMPLE_FRAME, e
            ));
        }
    }

    // Display available channels
    result.push_str("\nAvailable channels:\n");
    for channel_name in mibi_file.get_channel_names() {
        result.push_str(&format!("  {}\n", channel_name));
    }

    // Get mass calibration information
    result.push_str("\nMass calibration:\n");
    match mibi_file.export_mass_calibration() {
        Ok((masses, bins)) => {
            result.push_str(&format!("  Found {} calibration points\n", masses.len()));
            // Show a few example calibration points
            for i in 0..std::cmp::min(3, masses.len()) {
                result.push_str(&format!("  Mass {}: TOF {}\n", masses[i], bins[i]));
            }
        }
        Err(e) => {
            result.push_str(&format!("  No mass calibration available: {}\n", e));
        }
    }

    // Generate a pulse count heatmap
    result.push_str("\nPulse count heatmap:\n");
    match mibi_file.generate_pulse_count_heatmap(SAMPLE_FRAME) {
        Ok(heatmap) => {
            let sum: u32 = heatmap.iter().flat_map(|row| row.iter()).sum();
            let max: u32 = heatmap
                .iter()
                .flat_map(|row| row.iter())
                .max()
                .cloned()
                .unwrap_or(0);
            result.push_str(&format!(
                "  Heatmap size: {}x{}\n",
                heatmap.len(),
                heatmap[0].len()
            ));
            result.push_str(&format!("  Total pulses: {}\n", sum));
            result.push_str(&format!("  Max pulses in any pixel: {}\n", max));
        }
        Err(e) => {
            result.push_str(&format!("  Error generating heatmap: {}\n", e));
        }
    }

    Ok(result)
}

/// Analyze a MIBI file channels and extract specific channel data
pub fn analyze_mibi_channels(
    path: &str,
    channel_name: Option<&str>,
    mass_value: Option<f64>,
) -> Result<String, io::Error> {
    // Open the MIBI file
    let mibi_file =
        MibiFile::open(path).map_err(|e| io::Error::new(io::ErrorKind::Other, format!("{}", e)))?;

    let mut result = String::new();

    // Basic file information
    result.push_str(&format!("Analyzing MIBI file: {}\n", path));
    result.push_str(&format!(
        "Dimensions: {}x{}\n",
        mibi_file.header.size_x_pixels, mibi_file.header.size_y_pixels
    ));

    // List all available channels
    result.push_str("\nAvailable channels:\n");
    for name in mibi_file.get_channel_names() {
        if let Some(channel) = mibi_file.find_channel(&name) {
            if let Some(mass) = channel.mass {
                result.push_str(&format!("  {}: mass {}\n", name, mass));
            } else {
                result.push_str(&format!("  {}: no mass data\n", name));
            }
        }
    }

    // Extract specific channel data if requested
    if let Some(name) = channel_name {
        result.push_str(&format!("\nExtracting channel: {}\n", name));
        match mibi_file.get_channel_by_name(0, name, 1.0) {
            Ok(channel_data) => {
                let sum: u32 = channel_data.iter().flat_map(|row| row.iter()).sum();
                let max: u32 = channel_data
                    .iter()
                    .flat_map(|row| row.iter())
                    .max()
                    .cloned()
                    .unwrap_or(0);
                result.push_str(&format!(
                    "  Channel data size: {}x{}\n",
                    channel_data.len(),
                    channel_data[0].len()
                ));
                result.push_str(&format!("  Total counts: {}\n", sum));
                result.push_str(&format!("  Max counts in any pixel: {}\n", max));

                // Calculate some basic statistics
                let nonzero_pixels = channel_data
                    .iter()
                    .flat_map(|row| row.iter())
                    .filter(|&&count| count > 0)
                    .count();
                let total_pixels = channel_data.len() * channel_data[0].len();
                let percent_nonzero = (nonzero_pixels as f64 / total_pixels as f64) * 100.0;

                result.push_str(&format!(
                    "  Pixels with signal: {} ({:.2}%)\n",
                    nonzero_pixels, percent_nonzero
                ));
            }
            Err(e) => {
                result.push_str(&format!("  Error extracting channel: {}\n", e));
            }
        }
    }

    // Extract data for a specific mass value if requested
    if let Some(mass) = mass_value {
        result.push_str(&format!("\nExtracting mass: {}\n", mass));
        // Use a reasonable window of 0.5 Da around the target mass
        match mibi_file.get_mass_channel(0, mass, 0.5) {
            Ok(mass_data) => {
                let sum: u32 = mass_data.iter().flat_map(|row| row.iter()).sum();
                let max: u32 = mass_data
                    .iter()
                    .flat_map(|row| row.iter())
                    .max()
                    .cloned()
                    .unwrap_or(0);
                result.push_str(&format!(
                    "  Mass data size: {}x{}\n",
                    mass_data.len(),
                    mass_data[0].len()
                ));
                result.push_str(&format!("  Total counts: {}\n", sum));
                result.push_str(&format!("  Max counts in any pixel: {}\n", max));
            }
            Err(e) => {
                result.push_str(&format!("  Error extracting mass: {}\n", e));
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Channel;
    use std::path::PathBuf;

    /// Helper function to construct the file path for tissue data
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
    fn test_channel_manager() -> Result<(), Box<dyn std::error::Error>> {
        // Create test channels with required fields
        let channel1 = Channel {
            name: "Iron".to_string(),
            mass: Some(56.0),
            target: Some("Fe".to_string()),
            element: Some("Fe".to_string()),
            clone: None,
            mass_start: None,
            mass_stop: None,
            id: None,
            external_id: None,
            concentration: None,
            lot: None,
            manufacture_date: None,
            conjugate_id: None,
            extra: HashMap::new(),
        };

        let channel2 = Channel {
            name: "Gold".to_string(),
            mass: Some(197.0),
            target: Some("Au".to_string()),
            element: Some("Au".to_string()),
            clone: None,
            mass_start: None,
            mass_stop: None,
            id: None,
            external_id: None,
            concentration: None,
            lot: None,
            manufacture_date: None,
            conjugate_id: None,
            extra: HashMap::new(),
        };

        let channel3 = Channel {
            name: "Carbon".to_string(),
            mass: Some(12.0),
            target: Some("C".to_string()),
            element: Some("C".to_string()),
            clone: None,
            mass_start: None,
            mass_stop: None,
            id: None,
            external_id: None,
            concentration: None,
            lot: None,
            manufacture_date: None,
            conjugate_id: None,
            extra: HashMap::new(),
        };

        // Create a MibiDescriptor with just the panel field
        // (since that's all we need for ChannelManager testing)
        let descriptor = MibiDescriptor {
            id: None,
            backup_status: None,
            acquisition_status: None,
            fov_order: None,
            acquisition_start: None,
            acquisition_end: None,
            date: None,
            run_uuid: None,
            run_name: None,
            run_order: None,
            scan_order: None,
            fov_uuid: None,
            fov_id: None,
            fov_size_microns: None,
            instrument_identifier: None,
            instrument_control_version: None,
            tof_app_version: None,
            rsu_mode: None,
            frame_size: None,
            dwell_time_millis: None,
            sample_current: None,
            sample_bias: None,
            standard_target: None,
            panel: Some(vec![channel1.clone(), channel2.clone(), channel3.clone()]),
            fov: None,
            files: None,
            timing: None,
            gun: None,
            imaging_preset: None,
            coordinates: None,
            hv_adc: None,
            hv_dac: None,
            data_holder: None,
            extra: HashMap::new(),
        };

        // Initialize the channel manager
        let channel_manager = ChannelManager::new(&descriptor);

        // Test 1: Find by exact name
        let found_channel = channel_manager.find_by_name("Iron").unwrap();
        assert_eq!(found_channel.name, "Iron");
        assert_eq!(found_channel.mass, Some(56.0));

        // Test 2: Find by case-insensitive name
        let found_channel = channel_manager.find_by_name("iron").unwrap();
        assert_eq!(found_channel.name, "Iron");

        let found_channel = channel_manager.find_by_name("GOLD").unwrap();
        assert_eq!(found_channel.name, "Gold");

        // Test 3: Find by exact mass
        let found_channel = channel_manager.find_by_mass(197.0, None).unwrap();
        assert_eq!(found_channel.name, "Gold");

        // Test 4: Find by approximate mass with tolerance
        let found_channel = channel_manager.find_by_mass(56.05, Some(0.1)).unwrap();
        assert_eq!(found_channel.name, "Iron");

        // Test 5: Mass that should be too far (outside tolerance)
        let not_found = channel_manager.find_by_mass(56.2, Some(0.1));
        assert!(not_found.is_none());

        // Test 6: Find by identifier (name)
        let found_channel = channel_manager.find_channel("Carbon").unwrap();
        assert_eq!(found_channel.name, "Carbon");

        // Test 7: Find by identifier (mass as string)
        let found_channel = channel_manager.find_channel("12.0").unwrap();
        assert_eq!(found_channel.name, "Carbon");

        // Test 8: Get channel names
        let names = channel_manager.get_channel_names();
        assert_eq!(names.len(), 3);
        assert!(names.contains(&"Iron".to_string()));
        assert!(names.contains(&"Gold".to_string()));
        assert!(names.contains(&"Carbon".to_string()));

        // Test 9: Get channel masses
        let masses = channel_manager.get_channel_masses();
        assert_eq!(masses.len(), 3);
        assert!(masses.contains(&56.0));
        assert!(masses.contains(&197.0));
        assert!(masses.contains(&12.0));

        // Test 10: Non-existent channel
        let not_found = channel_manager.find_by_name("Silicon");
        assert!(not_found.is_none());

        Ok(())
    }

    // You can also add specific tests for individual files
    #[test]
    fn test_tissue_file() -> Result<(), Box<dyn std::error::Error>> {
        let file_path = get_tissue_file_path("fov-1-scan-1.bin");
        println!("\nTesting file: {}", file_path.display());

        // Open the MIBI file
        let mibi_file = MibiFile::new(file_path.to_str().unwrap())?;
        println!(
            "Image size: {}x{}",
            mibi_file.header.size_x_pixels, mibi_file.header.size_y_pixels
        );

        // Sample random pixels using direct coordinate generation
        println!("\nSampling random pixels:");
        use rand::Rng;
        let mut rng = rand::rng();

        const SAMPLE_COUNT: usize = 5;
        for i in 0..SAMPLE_COUNT {
            // Generate random coordinates within image bounds
            let x = rng.random_range(0..mibi_file.header.size_x_pixels as usize);
            let y = rng.random_range(0..mibi_file.header.size_y_pixels as usize);

            // Get and analyze pixel data
            match mibi_file.get_pixel(0, y, x) {
                Ok(pixel_data) => {
                    let pulse_count: usize = pixel_data
                        .trigger_events
                        .iter()
                        .map(|trigger| trigger.pulses.len())
                        .sum();

                    println!("Pixel {}: ({}, {}) has {} pulses", i + 1, x, y, pulse_count);

                    // If there are pulses, show a bit more detail about the first one
                    if pulse_count > 0 {
                        let first_trigger = &pixel_data.trigger_events[0];
                        if !first_trigger.pulses.is_empty() {
                            let first_pulse = &first_trigger.pulses[0];
                            println!(
                                "  First pulse: Time={}, Width={}, Intensity={}",
                                first_pulse.time, first_pulse.width, first_pulse.intensity
                            );
                        }
                    }
                }
                Err(e) => println!("Error getting pixel at ({}, {}): {}", x, y, e),
            }
        }

        Ok(())
    }

    #[test]
    fn test_mibi_file_capabilities() -> Result<(), Box<dyn std::error::Error>> {
        let file_path = get_tissue_file_path("fov-1-scan-1.bin");
        println!(
            "\nTesting MibiFile capabilities with: {}",
            file_path.display()
        );

        // Open the file
        let mibi_file = MibiFile::new(file_path.to_str().unwrap())?;

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

        // 3. Test streaming pixel processing
        println!("\nStreaming processing test:");
        let mut total_pulses = 0;
        let stats = mibi_file.process_streaming(0, |_, _, pixel| {
            for trigger in &pixel.trigger_events {
                total_pulses += trigger.pulses.len();
            }
        })?;

        println!(
            "  Processed {} of {} pixels",
            stats.processed_pixels, stats.total_pixels
        );
        println!("  Total pulses counted: {}", total_pulses);
        println!("  Stats report total pulses: {}", stats.total_pulses);

        // 4. Test pulse count heatmap
        println!("\nPulse count heatmap test:");
        let heatmap = mibi_file.generate_pulse_count_heatmap(0)?;
        let sum: u32 = heatmap.iter().flat_map(|row| row.iter()).sum();
        let max: u32 = heatmap
            .iter()
            .flat_map(|row| row.iter())
            .max()
            .cloned()
            .unwrap_or(0);
        println!("  Generated {}x{} heatmap", heatmap.len(), heatmap[0].len());
        println!("  Total pulses in heatmap: {}", sum);
        println!("  Max pulses in any pixel: {}", max);

        // 5. Test mass-specific channel image
        println!("\nMass channel test:");
        if let Some(mass_cal) = mibi_file.descriptor.mass_calibration() {
            if let Some(masses) = &mass_cal.masses {
                if !masses.is_empty() {
                    let test_mass = masses[0];
                    println!("  Testing with mass: {}", test_mass);

                    // Use a 0.5 mass unit window
                    let mass_image = mibi_file.count_pulses_for_mass(0, test_mass, 0.5)?;
                    let sum: u32 = mass_image.iter().flat_map(|row| row.iter()).sum();
                    println!(
                        "  Generated {}x{} mass channel image",
                        mass_image.len(),
                        mass_image[0].len()
                    );
                    println!("  Total pulses in mass range: {}", sum);
                }
            }
        }

        // Print information about the file structure
        println!("File header: size_x={}, size_y={}, frame_count={}",
            mibi_file.header.size_x_pixels,
            mibi_file.header.size_y_pixels,
            mibi_file.header.frame_count);

        // Check mass calibration
        println!("Checking mass calibration...");
        match mibi_file.export_mass_calibration() {
            Ok((masses, bins)) => {
                println!("Mass calibration available with {} points", masses.len());
                for i in 0..std::cmp::min(5, masses.len()) {
                    println!("  Calibration point {}: Mass {} at bin {}", i, masses[i], bins[i]);
                }

                // Try to convert some TOF values to masses
                for tof in [6900, 7000, 7100, 8000, 9000, 10000, 11000, 12000, 13000] {
                    if let Some(mass) = mibi_file.tof_to_mass(tof) {
                        println!("  TOF {} -> Mass {:.2}", tof, mass);
                    } else {
                        println!("  Could not convert TOF {} to mass", tof);
                    }
                }

                // Try to find channels at different masses
                for mass in [69.0, 70.0, 71.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0] {
                    match mibi_file.count_pulses_for_mass(0, mass, 0.5) {
                        Ok(image) => {
                            let sum: u32 = image.iter().flat_map(|row| row.iter()).sum();
                            let max: u32 = image.iter()
                                .flat_map(|row| row.iter())
                                .max()
                                .cloned()
                                .unwrap_or(0);
                            println!("  Mass {:.1} image: sum={}, max={}", mass, sum, max);
                        },
                        Err(e) => {
                            println!("  Failed to generate image for mass {:.1}: {}", mass, e);
                        }
                    }
                }
            },
            Err(e) => {
                println!("Mass calibration not available: {}", e);

                // Try with fallback conversion
                println!("Using fallback TOF to mass conversion:");
                for tof in [6900, 7000, 7100, 8000, 9000, 10000, 11000, 12000, 13000] {
                    let mass = tof as f64 / 100.0;
                    println!("  TOF {} -> Mass {:.2} (fallback)", tof, mass);
                }
            }
        }

        // Try to get some statistics about the file
        match mibi_file.process_streaming(0, |_, _, _| {}) {
            Ok(stats) => {
                println!("File statistics: total_pixels={}, processed_pixels={}, total_pulses={}, max_pulse_time={}, max_pulse_intensity={}",
                    stats.total_pixels,
                    stats.processed_pixels,
                    stats.total_pulses,
                    stats.max_pulse_time,
                    stats.max_pulse_intensity);
            },
            Err(e) => {
                println!("Failed to process file: {}", e);
            }
        }

        // Try to generate a pulse count heatmap
        match mibi_file.generate_pulse_count_heatmap(0) {
            Ok(heatmap) => {
                let sum: u32 = heatmap.iter().flat_map(|row| row.iter()).sum();
                let max: u32 = heatmap.iter()
                    .flat_map(|row| row.iter())
                    .max()
                    .cloned()
                    .unwrap_or(0);
                println!("Pulse count heatmap: {}x{}, sum={}, max={}",
                    heatmap.len(), heatmap[0].len(), sum, max);
            },
            Err(e) => {
                println!("Failed to generate pulse count heatmap: {}", e);
            }
        }

        Ok(())
    }

    #[test]
    fn test_channel_analysis() -> Result<(), Box<dyn std::error::Error>> {
        let file_path = get_tissue_file_path("fov-1-scan-1.bin");
        println!("\nTesting channel analysis with: {}", file_path.display());

        // First try with no specific channel or mass
        let result = analyze_mibi_channels(file_path.to_str().unwrap(), None, None)?;
        println!("{}", result);

        // Now open the file and find a channel name to test with
        let mibi_file = MibiFile::open(file_path.to_str().unwrap())?;

        let channel_names = mibi_file.get_channel_names();
        if !channel_names.is_empty() {
            println!("\nAnalyzing specific channel: {}", channel_names[0]);
            let result =
                analyze_mibi_channels(file_path.to_str().unwrap(), Some(&channel_names[0]), None)?;
            println!("{}", result);
        }

        // Try with a specific mass value
        if let Some(mass_cal) = mibi_file.descriptor.mass_calibration() {
            if let Some(masses) = &mass_cal.masses {
                if !masses.is_empty() {
                    println!("\nAnalyzing specific mass: {}", masses[0]);
                    let result =
                        analyze_mibi_channels(file_path.to_str().unwrap(), None, Some(masses[0]))?;
                    println!("{}", result);
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_direct_mibi_descriptor_parsing() -> Result<(), Box<dyn std::error::Error>> {
        // Use the re-exported function instead of accessing the private module
        use crate::parser::parse_descriptor_tolerant_with_fallback;
        use crate::parser::parse_header;
        use crate::utils::file_utils::read_binary_file_mmap;

        let file_path = get_tissue_file_path("fov-1-scan-1.bin");
        println!(
            "\nTesting direct MibiDescriptor parsing in file: {}",
            file_path.display()
        );

        // Map the file
        let mapped_file = read_binary_file_mmap(file_path.to_str().unwrap())?;

        // Parse header
        let (remaining, header) =
            parse_header(&mapped_file).map_err(|_: nom::Err<nom::error::Error<&[u8]>>| {
                io::Error::new(io::ErrorKind::InvalidData, "Failed to parse header")
            })?;

        // Try parsing without fallback to SimpleMibiDescriptor
        let result = parse_descriptor_tolerant_with_fallback(remaining, &header, false);

        match result {
            Ok((_, descriptor)) => {
                println!("Successfully parsed as MibiDescriptor without fallback!");
                println!("ID: {:?}", descriptor.id);
                println!("Run Name: {:?}", descriptor.run_name());
                println!("Instrument ID: {:?}", descriptor.instrument_id());
                println!(
                    "Has mass calibration: {}",
                    descriptor.mass_calibration().is_some()
                );
            }
            Err(e) => {
                println!(
                    "Failed to parse as MibiDescriptor without fallback: {:?}",
                    e
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_generate_channel_images_parallel() -> Result<(), Box<dyn std::error::Error>> {
        let file_path = get_real_world_bin_file_path("fov-1-scan-1.bin");
        println!("Testing file path: {}", file_path.display());

        // Check if file exists
        println!("File exists: {}", file_path.exists());

        let mibi_file = MibiFile::new(file_path.to_str().unwrap())?;

        // Print more detailed descriptor information
        println!("Descriptor ID: {:?}", mibi_file.descriptor.id);
        println!("Run name: {:?}", mibi_file.descriptor.run_name());
        println!("Has panel: {:?}", mibi_file.descriptor.panel.is_some());

        // Print structure of fov field for debugging
        if let Some(fov) = &mibi_file.descriptor.fov {
            println!("FOV exists in descriptor");

            // Check if panel exists in fov.extra
            if let Some(panel) = fov.extra.get("panel") {
                println!("Panel exists in fov.extra");

                // Check for conjugates
                if let Some(conjugates) = panel.get("conjugates") {
                    if let Some(conjugates_array) = conjugates.as_array() {
                        println!("Found {} conjugates in fov.panel.conjugates", conjugates_array.len());

                        // Print structure of first conjugate for debugging
                        if !conjugates_array.is_empty() {
                            println!("First conjugate structure: {:?}", conjugates_array[0]);

                            // Check specific fields we're interested in
                            if let Some(obj) = conjugates_array[0].as_object() {
                                for (key, _) in obj {
                                    println!("Field in conjugate: {}", key);
                                }
                            }
                        }
                    }
                } else {
                    println!("No 'conjugates' field found in panel");
                }
            } else {
                println!("No 'panel' field found in fov.extra");
            }
        } else {
            println!("No FOV field in descriptor");
        }

        if let Some(panel) = &mibi_file.descriptor.panel {
            println!("Panel size: {}", panel.len());
            for (i, channel) in panel.iter().enumerate().take(5) {
                println!("Channel {}: name={}, mass={:?}", i, channel.name, channel.mass);
            }
        } else {
            println!("Panel is None");
        }

        let channel_names = mibi_file.get_channel_names();
        println!("Channel names: {:?}", channel_names);

        // Try to find any channel by name or mass
        if !channel_names.is_empty() {
            let first_channel = mibi_file.find_channel(&channel_names[0]);
            println!("Found first channel: {:?}", first_channel.is_some());
            if let Some(channel) = first_channel {
                println!("First channel: name={}, mass={:?}", channel.name, channel.mass);
            }

            // Try with a limited subset of channels
            let test_channels = channel_names.iter()
                .take(3)
                .map(|s| s.to_string())
                .collect::<Vec<String>>();

            println!("Testing channel image generation with: {:?}", test_channels);

            // Print information about the file structure
            println!("File header: size_x={}, size_y={}, frame_count={}",
                mibi_file.header.size_x_pixels,
                mibi_file.header.size_y_pixels,
                mibi_file.header.frame_count);

            // Try to get some statistics about the file
            match mibi_file.process_streaming(0, |_, _, _| {}) {
                Ok(stats) => {
                    println!("File statistics: total_pixels={}, processed_pixels={}, total_pulses={}, max_pulse_time={}, max_pulse_intensity={}",
                        stats.total_pixels,
                        stats.processed_pixels,
                        stats.total_pulses,
                        stats.max_pulse_time,
                        stats.max_pulse_intensity);
                },
                Err(e) => {
                    println!("Failed to process file: {}", e);
                }
            }

            // Try to generate a pulse count heatmap
            match mibi_file.generate_pulse_count_heatmap(0) {
                Ok(heatmap) => {
                    let sum: u32 = heatmap.iter().flat_map(|row| row.iter()).sum();
                    let max: u32 = heatmap.iter()
                        .flat_map(|row| row.iter())
                        .max()
                        .cloned()
                        .unwrap_or(0);
                    println!("Pulse count heatmap: {}x{}, sum={}, max={}",
                        heatmap.len(), heatmap[0].len(), sum, max);
                },
                Err(e) => {
                    println!("Failed to generate pulse count heatmap: {}", e);
                }
            }

            // Generate one image for each channel
            let mut channel_images = Vec::new();
            for channel_name in &test_channels {
                // Try with a wider mass window (5.0 instead of 0.5)
                match mibi_file.get_channel_by_name(0, channel_name, 10.0) {
                    Ok(image) => {
                        let sum: u32 = image.iter().flat_map(|row| row.iter()).sum();
                        let max: u32 = image.iter()
                            .flat_map(|row| row.iter())
                            .max()
                            .cloned()
                            .unwrap_or(0);
                        println!("Successfully generated image for channel '{}' with wide window: sum={}, max={}",
                            channel_name, sum, max);

                        if sum > 0 {
                            channel_images.push((channel_name.clone(), image));
                        } else {
                            // Try with standard window
                            match mibi_file.get_channel_by_name(0, channel_name, 0.5) {
                                Ok(image) => {
                                    println!("Successfully generated image for channel '{}'", channel_name);
                                    channel_images.push((channel_name.clone(), image));
                                }
                                Err(e) => {
                                    println!("Failed to generate image for channel '{}': {}", channel_name, e);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        println!("Failed to generate image for channel '{}' with wide window: {}", channel_name, e);
                    }
                }
            }

            println!("Successfully generated {} channel images", channel_images.len());

            // Print info about the first image
            if !channel_images.is_empty() {
                let (name, image) = &channel_images[0];
                let sum: u32 = image.iter().flat_map(|row| row.iter()).sum();
                let max: u32 = image.iter()
                    .flat_map(|row| row.iter())
                    .max()
                    .cloned()
                    .unwrap_or(0);

                println!("Channel '{}' image: {}x{}, sum={}, max={}",
                    name, image.len(), image[0].len(), sum, max);
            }

            // Save the channel images as TIFF files
            println!("Saving channel images as TIFF files...");
            let output_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/output_tiff");
            println!("Output directory: {}", output_dir);
            let results = mibi_file.save_channels_as_tiff(
                0,
                &test_channels,
                10.0, // Use wide window factor
                output_dir,
                Some("tissue")
            );

            // Print the results of saving files
            for result in results {
                match result {
                    Ok(message) => println!("✅ {}", message),
                    Err(error) => println!("❌ Error: {}", error),
                }
            }
        } else {
            // Try with some common masses to see if we can find any channel
            for mass in &[12.0, 13.0, 14.0, 16.0, 31.0, 56.0, 197.0] {
                if let Some(channel) = mibi_file.find_channel(&mass.to_string()) {
                    println!("Found channel by mass {}: {}", mass, channel.name);
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_mass_calibration() -> Result<(), Box<dyn std::error::Error>> {
        let file_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../data/tissue/fov-1-scan-1.bin");
        println!("Testing file path: {}", file_path);

        let mibi_file = MibiFile::new(file_path)?;

        // Check mass calibration
        println!("Checking mass calibration...");
        match mibi_file.export_mass_calibration() {
            Ok((masses, bins)) => {
                println!("Mass calibration available with {} points", masses.len());
                for i in 0..std::cmp::min(5, masses.len()) {
                    println!("  Calibration point {}: Mass {} at bin {}", i, masses[i], bins[i]);
                }

                // Try to convert some TOF values to masses
                for tof in [6900, 7000, 7100, 8000, 9000, 10000, 11000, 12000, 13000] {
                    if let Some(mass) = mibi_file.tof_to_mass(tof) {
                        println!("  TOF {} -> Mass {:.2}", tof, mass);
                    } else {
                        println!("  Could not convert TOF {} to mass", tof);
                    }
                }

                // Try to find channels at different masses
                for mass in [69.0, 70.0, 71.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0] {
                    match mibi_file.count_pulses_for_mass(0, mass, 0.5) {
                        Ok(image) => {
                            let sum: u32 = image.iter().flat_map(|row| row.iter()).sum();
                            let max: u32 = image.iter()
                                .flat_map(|row| row.iter())
                                .max()
                                .cloned()
                                .unwrap_or(0);
                            println!("  Mass {:.1} image: sum={}, max={}", mass, sum, max);
                        },
                        Err(e) => {
                            println!("  Failed to generate image for mass {:.1}: {}", mass, e);
                        }
                    }
                }
            },
            Err(e) => {
                println!("Mass calibration not available: {}", e);

                // Try with fallback conversion
                println!("Using fallback TOF to mass conversion:");
                for tof in [6900, 7000, 7100, 8000, 9000, 10000, 11000, 12000, 13000] {
                    let mass = tof as f64 / 100.0;
                    println!("  TOF {} -> Mass {:.2} (fallback)", tof, mass);
                }
            }
        }

        Ok(())
    }
}
