//! Types for representing pixel data in the MIBI file format

use bon::Builder;

/// A single pulse event (5 bytes):
/// - 2 bytes: pulse time
/// - 1 byte: pulse width
/// - 2 bytes: pulse intensity
#[derive(Debug, Clone, PartialEq, Builder)]
pub struct PulseEvent {
    pub time: u16,
    pub width: u8,
    pub intensity: u16,
}

/// A trigger event for one pixel. First an 8‑byte record is read (only the last 2 bytes are used
/// to extract the number of pulse events). Then, for each pulse event, a 5‑byte record follows.
#[derive(Debug, PartialEq, Builder)]
pub struct TriggerEvent {
    pub num_pulses: u16,
    pub pulses: Vec<PulseEvent>,
}

/// The full pixel data holds one or more trigger events. (The number of trigger events is specified
/// by header.triggers_per_pixel.)
#[derive(Debug, PartialEq, Builder)]
pub struct PixelData {
    pub trigger_events: Vec<TriggerEvent>,
}
