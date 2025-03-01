//! Parsers for the MIBI descriptor JSON

use crate::types::descriptor::{MibiDescriptor, SimpleMibiDescriptor};
use crate::types::header::Header;
use nom::{IResult, bytes::complete::take, error::ErrorKind};
use serde_json;

/// Parses the descriptor block as JSON metadata
pub fn parse_descriptor<'a>(input: &'a [u8], header: &Header) -> IResult<&'a [u8], MibiDescriptor> {
    // First get the string as before
    let (input, desc_bytes) = take(header.metadata_length as usize)(input)?;
    let descriptor_str = String::from_utf8_lossy(desc_bytes).into_owned();

    // For debugging, print a sample of the JSON string
    println!(
        "Descriptor JSON sample (first 100 chars): {:?}",
        &descriptor_str.chars().take(100).collect::<String>()
    );

    // Now parse as JSON with better error handling
    match serde_json::from_str::<MibiDescriptor>(&descriptor_str) {
        Ok(descriptor) => Ok((input, descriptor)),
        Err(e) => {
            // Print the actual JSON error for debugging
            println!("JSON parsing error: {}", e);

            // Try to identify specific issues
            if descriptor_str.is_empty() {
                println!("Descriptor string is empty!");
            } else if !descriptor_str.starts_with('{') {
                println!(
                    "Descriptor doesn't start with '{{', first few bytes: {:?}",
                    &descriptor_str
                        .as_bytes()
                        .iter()
                        .take(10)
                        .collect::<Vec<_>>()
                );
            }

            // Continue to return a nom error for the parser chain
            Err(nom::Err::Error(nom::error::Error::new(
                input,
                ErrorKind::MapRes,
            )))
        }
    }
}

/// Alternative parser that uses a more tolerant approach
pub fn parse_descriptor_tolerant<'a>(
    input: &'a [u8],
    header: &Header,
) -> IResult<&'a [u8], MibiDescriptor> {
    parse_descriptor_tolerant_with_fallback(input, header, true)
}

/// Detailed parser that can optionally fall back to simplified parsing
pub fn parse_descriptor_tolerant_with_fallback<'a>(
    input: &'a [u8],
    header: &Header,
    use_fallback: bool,
) -> IResult<&'a [u8], MibiDescriptor> {
    // First get the string
    let (input, desc_bytes) = take(header.metadata_length as usize)(input)?;

    // Clean null bytes from descriptor bytes
    let cleaned_bytes: Vec<u8> = desc_bytes.iter().filter(|&&b| b != 0).cloned().collect();
    let descriptor_str = String::from_utf8_lossy(&cleaned_bytes).into_owned();

    // Try parsing the cleaned descriptor first
    match serde_json::from_str::<MibiDescriptor>(&descriptor_str) {
        Ok(descriptor) => {
            println!("Successfully parsed clean descriptor as MibiDescriptor");
            return Ok((input, descriptor));
        }
        Err(e) => {
            println!("Failed to parse clean descriptor as MibiDescriptor: {}", e);

            // Try truncating to the last closing brace
            if let Some(pos) = descriptor_str.rfind('}') {
                println!("Trying with truncated JSON ending at position {}", pos);
                let truncated = &descriptor_str[0..=pos];

                match serde_json::from_str::<MibiDescriptor>(truncated) {
                    Ok(descriptor) => {
                        println!("Successfully parsed truncated descriptor");
                        return Ok((input, descriptor));
                    }
                    Err(e) => println!("Failed to parse truncated descriptor: {}", e),
                }
            }

            // If fallback is disabled or all above attempts failed, either return error or try simplified parsing
            if !use_fallback {
                println!("Fallback disabled, returning error");
                return Err(nom::Err::Error(nom::error::Error::new(
                    input,
                    ErrorKind::MapRes,
                )));
            }

            println!(
                "Falling back to simplified descriptor parsing. Error was: {}",
                e
            );

            // Try with a simpler, more flexible structure
            match serde_json::from_str::<SimpleMibiDescriptor>(&descriptor_str) {
                Ok(simple) => {
                    // Convert to our regular struct with minimal information
                    let basic = MibiDescriptor {
                        id: simple
                            .data
                            .get("id")
                            .and_then(|v| v.as_str())
                            .map(String::from),
                        backup_status: None,
                        acquisition_status: None,
                        fov_order: None,
                        acquisition_start: None,
                        acquisition_end: None,
                        date: simple
                            .data
                            .get("date")
                            .and_then(|v| v.as_str())
                            .map(String::from),
                        run_name: simple
                            .data
                            .get("runName")
                            .and_then(|v| v.as_str())
                            .map(String::from),
                        run_uuid: None,
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
                        panel: None,
                        fov: None,
                        files: None,
                        timing: None,
                        gun: None,
                        imaging_preset: None,
                        coordinates: None,
                        hv_adc: None,
                        hv_dac: None,
                        data_holder: None,
                        extra: simple.data,
                    };

                    Ok((input, basic))
                }
                Err(e2) => {
                    println!("Even simplified parsing failed: {}", e2);
                    println!(
                        "First 100 chars of descriptor: {:?}",
                        &descriptor_str.chars().take(100).collect::<String>()
                    );

                    // Return error for the parser chain
                    Err(nom::Err::Error(nom::error::Error::new(
                        input,
                        ErrorKind::MapRes,
                    )))
                }
            }
        }
    }
}
