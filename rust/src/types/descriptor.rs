//! Types for representing the MIBI descriptor JSON metadata

use bon::Builder;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
/// Represents the JSON descriptor metadata in the MIBI file
#[derive(Debug, Deserialize, Serialize)]
pub struct MibiDescriptor {
    // Common fields - add/modify based on your actual JSON structure
    #[serde(rename = "fovName")]
    pub fov_name: Option<String>,
    pub id: Option<String>,
    #[serde(rename = "backupStatus")]
    pub backup_status: Option<String>,
    #[serde(rename = "acquisitionStatus")]
    pub acquisition_status: Option<String>,
    #[serde(rename = "fovOrder")]
    pub fov_order: Option<i32>,

    // Acquisition timestamps
    #[serde(rename = "acquisitionStart")]
    pub acquisition_start: Option<i64>,
    #[serde(rename = "acquisitionEnd")]
    pub acquisition_end: Option<i64>,
    pub date: Option<String>,

    // Run information
    #[serde(rename = "runUUID")]
    pub run_uuid: Option<String>,
    #[serde(rename = "runName")]
    pub run_name: Option<String>,
    #[serde(rename = "runOrder")]
    pub run_order: Option<i32>,
    #[serde(rename = "scanOrder")]
    pub scan_order: Option<i32>,

    // FOV information
    #[serde(rename = "fovUUID")]
    pub fov_uuid: Option<String>,
    #[serde(rename = "fovId")]
    pub fov_id: Option<String>,
    #[serde(rename = "fovSizeMicrons")]
    pub fov_size_microns: Option<f64>,

    // Instrument details
    #[serde(rename = "instrumentIdentifier")]
    pub instrument_identifier: Option<String>,
    #[serde(rename = "instrumentControlVersion")]
    pub instrument_control_version: Option<String>,
    #[serde(rename = "tofAppVersion")]
    pub tof_app_version: Option<String>,
    #[serde(rename = "rsuMode")]
    pub rsu_mode: Option<String>,

    // Frame and timing information
    #[serde(rename = "frameSize")]
    pub frame_size: Option<i32>,
    #[serde(rename = "dwellTimeMillis")]
    pub dwell_time_millis: Option<f64>,

    // Sample settings
    #[serde(rename = "sampleCurrent")]
    pub sample_current: Option<f64>,
    #[serde(rename = "sampleBias")]
    pub sample_bias: Option<f64>,
    #[serde(rename = "standardTarget")]
    pub standard_target: Option<String>,

    // Panel of channels
    pub panel: Option<Vec<Channel>>,

    // Nested structures
    pub fov: Option<FovInfo>,
    pub files: Option<FilesInfo>,
    pub timing: Option<TimingInfo>,
    pub gun: Option<GunInfo>,
    #[serde(rename = "imagingPreset")]
    pub imaging_preset: Option<ImagingPreset>,
    pub coordinates: Option<Coordinates>,
    #[serde(rename = "hvAdc")]
    pub hv_adc: Option<Vec<HvAdcChannel>>,
    #[serde(rename = "hvDac")]
    pub hv_dac: Option<Vec<HvDacChannel>>,
    // Data holder info (if present)
    #[serde(rename = "dataHolder")]
    pub data_holder: Option<DataHolder>,

    // Catch-all for additional fields
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

// Nested structures for MibiDescriptor

#[derive(Debug, Deserialize, Serialize)]
pub struct FovInfo {
    pub name: Option<String>,
    pub run: Option<RunInfo>,
    #[serde(rename = "scanCount")]
    pub scan_count: Option<i32>,
    #[serde(rename = "runOrder")]
    pub run_order: Option<i32>,
    #[serde(rename = "timingChoice")]
    pub timing_choice: Option<i32>,
    #[serde(rename = "timingDescription")]
    pub timing_description: Option<String>,
    #[serde(rename = "fovSizeMicrons")]
    pub fov_size_microns: Option<f64>,
    #[serde(rename = "standardTarget")]
    pub standard_target: Option<String>,
    #[serde(rename = "imagingPreset")]
    pub imaging_preset: Option<ImagingPreset>,
    #[serde(rename = "fullTiming")]
    pub full_timing: Option<TimingInfo>,
    pub scans: Option<Vec<ScanInfo>>,
    #[serde(rename = "massStart")]
    pub mass_start: Option<f64>,
    #[serde(rename = "massStop")]
    pub mass_stop: Option<f64>,
    pub id: Option<String>,
    #[serde(rename = "centerPointMicrons")]
    pub center_point_microns: Option<Coordinates>,
    #[serde(rename = "centerPointPixels")]
    pub center_point_pixels: Option<Coordinates>,
    #[serde(rename = "frameSizePixels")]
    pub frame_size_pixels: Option<FrameSize>,
    #[serde(rename = "dwellTimeMillis")]
    pub dwell_time_millis: Option<f64>,
    pub status: Option<String>,
    #[serde(rename = "triggerCount")]
    pub trigger_count: Option<i32>,
    pub panel: Option<Panel>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct RunInfo {
    pub id: Option<String>,
    pub name: Option<String>,
    pub status: Option<String>,
    #[serde(rename = "opticalImage")]
    pub optical_image: Option<OpticalImageInfo>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct OpticalImageInfo {
    pub id: Option<i32>,
    #[serde(rename = "originalFilename")]
    pub original_filename: Option<String>,
    pub path: Option<String>,
    pub format: Option<String>,
    #[serde(rename = "imageCaptureTimestampMillis")]
    pub image_capture_timestamp_millis: Option<i64>,
    #[serde(rename = "slideId")]
    pub slide_id: Option<i32>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ImagingPreset {
    pub preset: Option<String>,
    pub aperture: Option<String>,
    #[serde(rename = "displayName")]
    pub display_name: Option<String>,
    pub defaults: Option<PresetDefaults>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct PresetDefaults {
    #[serde(rename = "timingChoice")]
    pub timing_choice: Option<i32>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimingInfo {
    pub id: Option<i32>,
    pub description: Option<String>,
    #[serde(rename = "tofPeriodNanoS")]
    pub tof_period_nano_s: Option<i32>,
    #[serde(rename = "pushPull")]
    pub push_pull: Option<TimingSignal>,
    pub flapper: Option<TimingSignal>,
    pub pulse: Option<TimingSignal>,
    pub sp: Option<TimingSignal>,
    #[serde(rename = "tofApp")]
    pub tof_app: Option<TofAppSettings>,
    #[serde(rename = "isActive")]
    pub is_active: Option<bool>,
    #[serde(rename = "massCalibration")]
    pub mass_calibration: Option<MassCalibration>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TimingSignal {
    pub active: Option<bool>,
    #[serde(rename = "delayNanoS")]
    pub delay_nano_s: Option<i32>,
    #[serde(rename = "widthNanoS")]
    pub width_nano_s: Option<i32>,
    pub polarity: Option<i32>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TofAppSettings {
    #[serde(rename = "triggerPerPixel")]
    pub trigger_per_pixel: Option<i32>,
    #[serde(rename = "recordLength")]
    pub record_length: Option<i32>,
    #[serde(rename = "triggerDelay")]
    pub trigger_delay: Option<i32>,
    #[serde(rename = "triggerThresholdVoltage")]
    pub trigger_threshold_voltage: Option<i32>,
    #[serde(rename = "triggerResetVoltage")]
    pub trigger_reset_voltage: Option<i32>,
    #[serde(rename = "filterBand")]
    pub filter_band: Option<Vec<i32>>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct MassCalibration {
    pub masses: Option<Vec<f64>>,
    pub bins: Option<Vec<i32>>,
    #[serde(rename = "massGain")]
    pub mass_gain: Option<f64>,
    #[serde(rename = "massOffset")]
    pub mass_offset: Option<f64>,
    #[serde(rename = "timeResolution")]
    pub time_resolution: Option<f64>,
    #[serde(rename = "timeResolutionNanoS")]
    pub time_resolution_nano_s: Option<f64>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ScanInfo {
    pub id: Option<String>,
    #[serde(rename = "backupStatus")]
    pub backup_status: Option<String>,
    #[serde(rename = "acquisitionStatus")]
    pub acquisition_status: Option<String>,
    #[serde(rename = "fovOrder")]
    pub fov_order: Option<i32>,
    #[serde(rename = "acquisitionStart")]
    pub acquisition_start: Option<i64>,
    #[serde(rename = "acquisitionEnd")]
    pub acquisition_end: Option<i64>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Coordinates {
    pub x: Option<f64>,
    pub y: Option<f64>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct FrameSize {
    pub width: Option<i32>,
    pub height: Option<i32>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct FilesInfo {
    pub directory: Option<String>,
    #[serde(rename = "binFilePath")]
    pub bin_file_path: Option<String>,
    #[serde(rename = "ionImageFilePath")]
    pub ion_image_file_path: Option<String>,
    #[serde(rename = "spectraFilePath")]
    pub spectra_file_path: Option<String>,
    #[serde(rename = "descriptionFilePath")]
    pub description_file_path: Option<String>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct GunInfo {
    #[serde(rename = "lens1V")]
    pub lens1_v: Option<f64>,
    #[serde(rename = "lens2V")]
    pub lens2_v: Option<f64>,
    #[serde(rename = "stigmatorAngle")]
    pub stigmator_angle: Option<f64>,
    #[serde(rename = "stigmatorMagnitude")]
    pub stigmator_magnitude: Option<f64>,
    #[serde(rename = "deflectorX1")]
    pub deflector_x1: Option<f64>,
    #[serde(rename = "deflectorX2")]
    pub deflector_x2: Option<f64>,
    #[serde(rename = "deflectorX3")]
    pub deflector_x3: Option<f64>,
    #[serde(rename = "deflectorY1")]
    pub deflector_y1: Option<f64>,
    #[serde(rename = "deflectorY2")]
    pub deflector_y2: Option<f64>,
    #[serde(rename = "deflectorY3")]
    pub deflector_y3: Option<f64>,
    #[serde(rename = "blankingV")]
    pub blanking_v: Option<f64>,
    #[serde(rename = "anodeV")]
    pub anode_v: Option<f64>,
    #[serde(rename = "apertureActual")]
    pub aperture_actual: Option<String>,
    pub aperture: Option<String>,
    #[serde(rename = "apertureObj")]
    pub aperture_obj: Option<ApertureObj>,
    #[serde(rename = "gunDescription")]
    pub gun_description: Option<Box<GunInfo>>,
    #[serde(rename = "sampleCurrent")]
    pub sample_current: Option<f64>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ApertureObj {
    #[serde(rename = "motorPosition")]
    pub motor_position: Option<f64>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct HvAdcChannel {
    pub name: Option<String>,
    pub channel: Option<i32>,
    pub value: Option<f64>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct HvDacChannel {
    pub name: Option<String>,
    #[serde(rename = "currentSetPoint")]
    pub current_set_point: Option<f64>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct DataHolder {
    #[serde(rename = "frameSize")]
    pub frame_size: Option<i32>,
    #[serde(rename = "recordLength")]
    pub record_length: Option<i32>,
    pub id: Option<String>,
    #[serde(rename = "spectraBuffer")]
    pub spectra_buffer: Option<HashMap<String, serde_json::Value>>,
    #[serde(rename = "ionImageBuffer")]
    pub ion_image_buffer: Option<HashMap<String, serde_json::Value>>,
    #[serde(rename = "ionImage")]
    pub ion_image: Option<HashMap<String, serde_json::Value>>,
    pub spectra: Option<HashMap<String, serde_json::Value>>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

// Channel information
#[derive(Debug, Deserialize, Serialize, Clone, Builder)]
pub struct Channel {
    pub mass: Option<f64>,
    pub target: Option<String>,
    pub element: Option<String>,
    pub clone: Option<String>,
    #[serde(rename = "massStart")]
    pub mass_start: Option<f64>,
    #[serde(rename = "massStop")]
    pub mass_stop: Option<f64>,
    pub id: Option<i32>,
    #[serde(rename = "external_id")]
    pub external_id: Option<String>,
    pub concentration: Option<f64>,
    pub lot: Option<String>,
    pub manufacture_date: Option<String>,
    pub conjugate_id: Option<i32>,
}

// Panel structure to match JSON format
#[derive(Debug, Deserialize, Serialize)]
pub struct Panel {
    pub id: Option<i32>,
    pub description: Option<String>,
    pub name: Option<String>,
    pub dilution: Option<f64>,
    #[serde(rename = "manufacture_date")]
    pub manufacture_date: Option<String>,
    pub conjugates: Option<Vec<Channel>>,
    pub batches: Option<Vec<PanelBatch>>,
    pub active: Option<bool>,
    #[serde(rename = "previous_id")]
    pub previous_id: Option<i32>,
    #[serde(rename = "next_id")]
    pub next_id: Option<i32>,
    #[serde(rename = "is_draft")]
    pub is_draft: Option<bool>,
    #[serde(rename = "owned_by_user")]
    pub owned_by_user: Option<bool>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

// Implement AsRef<[Channel]> for Panel
impl AsRef<[Channel]> for Panel {
    fn as_ref(&self) -> &[Channel] {
        match &self.conjugates {
            Some(conjugates) => conjugates.as_slice(),
            None => &[],
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct PanelBatch {
    #[serde(rename = "antibody_volume")]
    pub antibody_volume: Option<f64>,
    #[serde(rename = "buffer_volume")]
    pub buffer_volume: Option<f64>,
    pub id: Option<i32>,
    #[serde(rename = "batch_id")]
    pub batch_id: Option<i32>,
    pub volume: Option<f64>,
    #[serde(rename = "is_draft")]
    pub is_draft: Option<bool>,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

// Implementation methods for MibiDescriptor
impl MibiDescriptor {
    /// Returns the number of frames in the file
    pub fn frame_count(&self) -> Option<i32> {
        // Try to get from timing, then from fov
        self.frame_size.or_else(|| {
            self.timing
                .as_ref()?
                .tof_app
                .as_ref()?
                .trigger_per_pixel
                .or_else(|| {
                    self.fov
                        .as_ref()?
                        .full_timing
                        .as_ref()?
                        .tof_app
                        .as_ref()?
                        .trigger_per_pixel
                })
        })
    }

    /// Returns the mass calibration information if available
    pub fn mass_calibration(&self) -> Option<&MassCalibration> {
        self.timing.as_ref()?.mass_calibration.as_ref().or_else(|| {
            self.fov
                .as_ref()?
                .full_timing
                .as_ref()?
                .mass_calibration
                .as_ref()
        })
    }

    /// Get the file acquisition date as a string
    pub fn acquisition_date(&self) -> Option<String> {
        self.date.clone()
    }

    /// Get the instrument ID
    pub fn instrument_id(&self) -> Option<String> {
        self.instrument_identifier.clone()
    }

    /// Get the run name
    pub fn run_name(&self) -> Option<String> {
        self.run_name
            .clone()
            .or_else(|| self.fov.as_ref()?.run.as_ref()?.name.clone())
    }

    /// Safely get fov mass_start value
    pub fn fov_mass_start(&self) -> Option<f64> {
        self.fov.as_ref()?.mass_start
    }

    /// Safely get fov mass_stop value
    pub fn fov_mass_stop(&self) -> Option<f64> {
        self.fov.as_ref()?.mass_stop
    }
}
