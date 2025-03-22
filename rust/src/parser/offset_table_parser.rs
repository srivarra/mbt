//! Parser functions for MIBI offset lookup table

use crate::types::header::Header;
use crate::types::offset_table::OffsetLookupTable;
use winnow::{
    Parser,
    binary::le_u64,
    combinator::repeat,
    error::ContextError,
};

/// Parses the offset block into a vector of u64 offsets using winnow parsers
pub fn parse_offset_table<'a>(
    input: &mut &'a [u8],
    header: &Header,
) -> Result<OffsetLookupTable, ContextError<&'a [u8]>> {
    // Cast each value to usize before multiplication to avoid overflow
    let size_x = header.size_x_pixels as usize;
    let size_y = header.size_y_pixels as usize;
    let frame_count = header.frame_count as usize;
    let total_pixels = size_x * size_y * frame_count;


    // Parse exactly total_pixels number of u64 values using repeat
    let offsets: Vec<u64> = repeat(total_pixels, le_u64).parse_next(input)?;

    let table = OffsetLookupTable::builder()
        .offsets(offsets)
        .size_x_pixels(size_x)
        .size_y_pixels(size_y)
        .frames(frame_count)
        .build();

    Ok(table)
}
