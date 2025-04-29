use crate::types::header::Header;
use bon::Builder;

/// Represents a 2D coordinate using row (y) and column (x).
#[derive(Debug, Copy, Clone, PartialEq, Eq, Builder)]
pub struct Coordinate {
    pub y: usize,
    pub x: usize,
}

/// Computes the data_start offset which is the sum of:
///  - A fixed header length (0x12 bytes)
///  - The descriptor (metadata_length) bytes
///  - The offset block (size_x_pixels * size_y_pixels * frame_count * 8 bytes)
pub fn data_start_offset(header: &Header) -> usize {
    let header_fixed_length = 0x12; // 18 bytes from the fixed header portion
    let descriptor_length = header.metadata_length;
    let offset_block_size = header.size_x_pixels * header.size_y_pixels * header.frame_count * 8;
    (header_fixed_length + descriptor_length + offset_block_size) as usize
}
