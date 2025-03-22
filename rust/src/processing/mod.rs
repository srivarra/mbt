pub mod processing;

// Re-export for easier access
pub use processing::*;

// Export extract_bin for easier access
pub use processing::extract_bin;

/// Statistics collected during streaming processing
pub struct ProcessingStats {
    pub total_pixels: usize,
    pub processed_pixels: usize,
    pub total_pulses: u64,
    pub max_pulse_time: u16,
    pub max_pulse_intensity: u16,
}
