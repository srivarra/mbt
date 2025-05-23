//! MIBI file parsing functionality

mod descriptor_parser;
mod header_parser;
mod offset_table_parser;
pub mod pixel_parser;

// Re-export the parsing functions
pub use descriptor_parser::parse_descriptor;
pub use header_parser::parse_header;
pub use offset_table_parser::parse_offset_table;
pub use pixel_parser::{parse_pixel_data, parse_pulse_event, parse_trigger_event};
