//! Parsers for the MIBI descriptor JSON

use crate::types::descriptor::MibiDescriptor;
use crate::types::header::Header;
use serde_json;
use winnow::{Parser, error::ContextError, token::take};

/// Parses the descriptor block as JSON metadata
pub fn parse_descriptor<'a>(
    input: &mut &'a [u8],
    header: &Header,
) -> Result<MibiDescriptor, ContextError<&'a [u8]>> {
    // Get the descriptor bytes
    let desc_bytes = take(header.metadata_length as usize).parse_next(input)?;

    // Clean null bytes from descriptor bytes
    let cleaned_bytes = clean_descriptor_bytes(desc_bytes);
    let descriptor_str = String::from_utf8_lossy(&cleaned_bytes).into_owned();

    // Now parse as JSON with better error handling
    match serde_json::from_str::<MibiDescriptor>(&descriptor_str) {
        Ok(descriptor) => Ok(descriptor),
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

            // Return a winnow error
            Err(ContextError::new())
        }
    }
}

fn clean_descriptor_bytes(desc_bytes: &[u8]) -> Vec<u8> {
    desc_bytes.iter().filter(|&&b| b != 0).cloned().collect()
}
