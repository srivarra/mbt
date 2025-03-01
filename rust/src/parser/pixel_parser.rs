use nom::{
    IResult, Parser, bytes::complete::take, multi::count,
    number::complete::le_u8, number::complete::le_u16,
};
use std::convert::TryInto;
use crate::types::header::Header;
use crate::types::offset_table::OffsetLookupTable;
use crate::types::pixel_data::{PulseEvent, TriggerEvent, PixelData};

/// Parses a single pulse event record (5 bytes):
/// - 2 bytes: pulse time (u16, little‑endian)
/// - 1 byte: pulse width (u8)
/// - 2 bytes: pulse intensity (u16, little‑endian)
pub fn parse_pulse_event(input: &[u8]) -> IResult<&[u8], PulseEvent> {
    let (input, time) = le_u16(input)?;
    let (input, width) = le_u8(input)?;
    let (input, intensity) = le_u16(input)?;
    Ok((
        input,
        PulseEvent::builder()
            .time(time)
            .width(width)
            .intensity(intensity)
            .build(),
    ))
}

/// Parses a trigger event for a pixel.
/// First, read an 8‑byte trigger record and extract the number of pulse events from bytes 6..8.
/// Then, parse exactly that many pulse events.
pub fn parse_trigger_event(input: &[u8]) -> IResult<&[u8], TriggerEvent> {
    // Read the 8-byte trigger event record.
    let (input, trigger_record) = take(8usize)(input)?;
    // Extract num_pulses from bytes 6..8 of the record.
    let num_pulses = {
        let num_bytes: [u8; 2] = trigger_record[6..8]
            .try_into()
            .expect("Trigger record should be 8 bytes");
        u16::from_le_bytes(num_bytes)
    };

    let (remaining, pulses) = count(parse_pulse_event, num_pulses as usize).parse(input)?;
    Ok((
        remaining,
        TriggerEvent::builder()
            .num_pulses(num_pulses)
            .pulses(pulses)
            .build(),
    ))
}

/// Parses the pixel data without pre-computing boundaries.
/// The parser uses nom's `count` combinator to parse exactly the expected number of trigger events per pixel.
pub fn parse_pixel_data<'a>(
    input: &'a [u8],
    header: &Header,
    _offset_table: &OffsetLookupTable,  // Prefix with _ to indicate intentionally unused
) -> IResult<&'a [u8], Vec<PixelData>> {
    let num_pixels = (header.size_x_pixels as usize) * (header.size_y_pixels as usize);
    let mut pixels = Vec::with_capacity(num_pixels);
    let mut remaining = input;

    for _ in 0..num_pixels {
        let (new_remaining, trigger_events) =
            count(parse_trigger_event, header.triggers_per_pixel as usize).parse(remaining)?;
        pixels.push(PixelData::builder().trigger_events(trigger_events).build());
        remaining = new_remaining;
    }

    Ok((remaining, pixels))
}

/// Parses a single pixel's data at a specific location
/// This function is optimized for random access using the offset table
pub fn parse_single_pixel<'a>(
    input: &'a [u8],
    header: &Header,
    _offset_table: &OffsetLookupTable,
    _frame: usize,
    y: usize,
    x: usize
) -> Result<PixelData, &'static str> {
    // Parse trigger events
    let mut trigger_events = Vec::with_capacity(header.triggers_per_pixel as usize);
    let mut remaining = input;

    for _ in 0..header.triggers_per_pixel {
        match parse_trigger_event(remaining) {
            Ok((new_remaining, trigger_event)) => {
                trigger_events.push(trigger_event);
                remaining = new_remaining;
            },
            Err(_) => return Err("Failed to parse trigger event")
        }
    }

    Ok(PixelData::builder()
        .trigger_events(trigger_events)
        .build())
}

/// Process pixels in a memory-efficient streaming manner without storing all in memory
/// Takes a closure that's called for each pixel with its coordinates and data
pub fn process_pixels_streaming<F>(
    input: &[u8],
    header: &Header,
    offset_table: &OffsetLookupTable,
    frame: usize,
    mut process_fn: F
) -> Result<(), &'static str>
where
    F: FnMut(usize, usize, &PixelData) -> ()
{
    let size_y = header.size_y_pixels as usize;
    let size_x = header.size_x_pixels as usize;

    for y in 0..size_y {
        for x in 0..size_x {
            if let Some(offset) = offset_table.get_offset(frame, y, x) {
                let pixel_data = &input[offset as usize..];
                match parse_single_pixel(pixel_data, header, offset_table, frame, y, x) {
                    Ok(pixel) => {
                        process_fn(y, x, &pixel);
                    },
                    Err(_) => {
                        // Skip errors in individual pixels during streaming processing
                        continue;
                    }
                }
            }
        }
    }

    Ok(())
}

/// Helper function to convert mass to time-of-flight value
/// This is a placeholder - actual implementation would use calibration data
pub fn mass_to_tof(mass: f64, descriptor: &crate::types::MibiDescriptor) -> Option<u16> {
    // Get calibration from descriptor if available
    if let Some(mass_cal) = descriptor.mass_calibration() {
        if let (Some(masses), Some(bins)) = (&mass_cal.masses, &mass_cal.bins) {
            if !masses.is_empty() && masses.len() == bins.len() {
                // Simple linear interpolation between calibration points
                // In a real application, you might want to use a more sophisticated approach

                // Find closest calibration points
                let mut lower_idx = 0;
                let mut upper_idx = 0;

                for i in 1..masses.len() {
                    if masses[i] > mass {
                        upper_idx = i;
                        lower_idx = i - 1;
                        break;
                    }
                }

                // Perform linear interpolation
                let mass_range = masses[upper_idx] - masses[lower_idx];
                let tof_range = (bins[upper_idx] - bins[lower_idx]) as f64;
                let mass_pos = (mass - masses[lower_idx]) / mass_range;
                let tof = bins[lower_idx] as f64 + (mass_pos * tof_range);

                return Some(tof as u16);
            }
        }
    }

    // Fallback approximate conversion (placeholder)
    Some((mass * 100.0) as u16)
}
