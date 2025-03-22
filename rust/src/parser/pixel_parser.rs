use crate::types::header::Header;
use crate::types::pixel_data::{PixelData, PulseEvent, TriggerEvent};
use winnow::{
    Parser, token::take, combinator::repeat, binary::le_u8, binary::le_u16,
    error::ContextError,
};

/// Parses a single pulse event record (5 bytes):
/// - 2 bytes: pulse time (u16, little‑endian)
/// - 1 byte: pulse width (u8, little‑endian)
/// - 2 bytes: pulse intensity (u16, little‑endian)
pub fn parse_pulse_event<'a>(input: &mut &'a [u8]) -> Result<PulseEvent, ContextError> {
    let time = le_u16.parse_next(input)?;
    let width = le_u8.parse_next(input)?;
    let intensity = le_u16.parse_next(input)?;

    Ok(PulseEvent::builder()
        .time(time)
        .width(width)
        .intensity(intensity)
        .build())
}

/// Parses a trigger event for a pixel.
/// First, read an 8‑byte trigger record and extract the number of pulse events from bytes 6..8.
/// Then, parse exactly that many pulse events.
pub fn parse_trigger_event<'a>(input: &mut &'a [u8]) -> Result<TriggerEvent, ContextError> {
    // Read the 8-byte trigger event record.
    let trigger_record = take(8usize).parse_next(input)?;
    // Extract num_pulses from bytes 6..8 of the record.
    let num_pulses = {
        let num_bytes: [u8; 2] = trigger_record[6..8]
            .try_into()
            .expect("Trigger record should be 8 bytes");
        u16::from_le_bytes(num_bytes)
    };

    let pulses = repeat(num_pulses as usize, parse_pulse_event).parse_next(input)?;

    Ok(TriggerEvent::builder()
        .num_pulses(num_pulses)
        .pulses(pulses)
        .build())
}

/// Parses a single pixel's data at a specific location
/// This function is optimized for random access using the offset table
pub fn parse_pixel_data(
    input: &[u8],
    header: &Header,
) -> Result<PixelData, String> {
    let mut input = input;
    let trigger_events = match repeat(header.triggers_per_pixel as usize, parse_trigger_event).parse_next(&mut input) {
        Ok(events) => events,
        Err(e) => return Err(format!("Error parsing pixel data: {:?}", e)),
    };

    Ok(PixelData::builder().trigger_events(trigger_events).build())
}
