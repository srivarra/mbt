//! Parser functions for MIBI offset lookup table

use nom::{
    IResult, Parser, bytes::complete::take, multi::count,
    number::complete::le_u64,
};
use crate::types::header::Header;
use crate::types::offset_table::OffsetLookupTable;

/// Parses the offset block into a vector of u64 offsets using nom parsers
pub fn parse_offset_table<'a>(
    input: &'a [u8],
    header: &Header,
) -> IResult<&'a [u8], OffsetLookupTable> {
    let total_pixels =
        header.size_x_pixels as usize * header.size_y_pixels as usize * header.frame_count as usize;

    // Use nom's count combinator with le_u64 parser to parse exactly the right number of u64 values
    let (remaining, offsets) = count(le_u64, total_pixels).parse(input)?;

    let table = OffsetLookupTable::builder()
        .offsets(offsets)
        .size_x_pixels(header.size_x_pixels as usize)
        .size_y_pixels(header.size_y_pixels as usize)
        .frames(header.frame_count as usize)
        .build();

    Ok((remaining, table))
}

/// Skips over the offset block whose size is computed as:
///   size_x_pixels * size_y_pixels * frame_count * 8 bytes
/// The extracted value is discarded.
pub fn parse_offset_block<'a>(input: &'a [u8], header: &Header) -> IResult<&'a [u8], &'a [u8]> {
    let offset_block_size = (header.size_x_pixels as usize)
        * (header.size_y_pixels as usize)
        * (header.frame_count as usize)
        * 8;
    let (input, offset_block) = take(offset_block_size)(input)?;
    Ok((input, offset_block))
}

/// Computes the data_start offset which is the sum of:
///  - A fixed header length (0x12 bytes)
///  - The descriptor (metadata_length) bytes
///  - The offset block (size_x_pixels * size_y_pixels * frame_count * 8 bytes)
pub fn data_start_offset(header: &Header) -> usize {
    let header_fixed_length = 0x12; // 18 bytes from the fixed header portion
    let descriptor_length = header.metadata_length as usize;
    let offset_block_size = (header.size_x_pixels as usize)
        * (header.size_y_pixels as usize)
        * (header.frame_count as usize)
        * 8;
    header_fixed_length + descriptor_length + offset_block_size
}
