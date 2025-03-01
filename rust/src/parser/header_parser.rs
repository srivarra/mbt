use nom::{
    IResult,
    bytes::complete::take,
    number::complete::le_u16,
};
use crate::types::header::Header;


/// Parses the header section of the binary file.
///
/// The header layout is as follows:
/// - 6 bytes to skip (magic/reserved)
/// - 4 little‑endian u16 values: num_x, num_y, triggers_per_pixel, num_frames
/// - 2 bytes to skip (reserved)
/// - 1 little‑endian u16: desc_len
pub fn parse_header(input: &[u8]) -> IResult<&[u8], Header> {
    let (input, file_signature) = take(6usize)(input)?;
    let (input, size_x_pixels) = le_u16(input)?;
    let (input, size_y_pixels) = le_u16(input)?;
    let (input, triggers_per_pixel) = le_u16(input)?;
    let (input, frame_count) = le_u16(input)?;
    let (input, _) = take(2usize)(input)?;
    let (rest, metadata_length) = le_u16(input)?;

    let header = Header::builder()
        .size_x_pixels(size_x_pixels)
        .size_y_pixels(size_y_pixels)
        .triggers_per_pixel(triggers_per_pixel)
        .frame_count(frame_count)
        .metadata_length(metadata_length)
        .file_signature(String::from_utf8_lossy(file_signature).into_owned())
        .build();
    Ok((rest, header))
}
