//! Type definitions for the MIBI file format

pub mod descriptor;
pub mod header;
pub mod offset_table;
pub mod pixel_data;

// Re-export the main types for convenience
pub use descriptor::{MibiDescriptor, SimpleMibiDescriptor, Channel};
pub use header::Header;
pub use offset_table::OffsetLookupTable;
pub use pixel_data::{PixelData, TriggerEvent, PulseEvent};
