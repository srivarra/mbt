pub mod mibi_file;
pub mod parser;
pub mod processing;
pub mod types;
pub mod utils;

// Add specific pyo3 imports
pub use mibi_file::MibiFile;
use pyo3::prelude::*; // Import common PyO3 items
use pyo3::exceptions::{PyFileNotFoundError, PyValueError};
pub use types::{
    MibiDescriptor, header::Header, offset_table::OffsetLookupTable, pixel_data::PixelData,
};

use crate::parser::pixel_parser::PixelParseRegionError;
use polars::prelude::DataFrame;
use pyo3_polars::PyDataFrame;
use serde_json;
use std::path::PathBuf;

// --- Error Conversion for PyO3 ---
// Implement conversion from our custom error to PyErr
impl From<PixelParseRegionError> for PyErr {
    fn from(err: PixelParseRegionError) -> PyErr {
        PyValueError::new_err(err.to_string())
    }
}

#[pyclass(module = "mbt._core")]
struct MibiReader {
    mibi_file: MibiFile,
    num_chunks: usize,
    dataframe: Option<DataFrame>,
}

#[pymethods]
impl MibiReader {
    #[new]
    #[pyo3(signature = (file_path, num_chunks = 16))]
    fn new(file_path: PathBuf, num_chunks: usize) -> PyResult<Self> {
        let path_str = file_path.to_str().ok_or_else(|| {
            PyValueError::new_err(format!(
                "Invalid UTF-8 sequence in path: {}",
                file_path.display()
            ))
        })?;

        let mibi_file = MibiFile::open(path_str).map_err(|e| {
            PyFileNotFoundError::new_err(format!(
                "Failed to open MIBI file '{}': {}",
                path_str, e
            ))
        })?;

        Ok(MibiReader {
            mibi_file,
            num_chunks,
            dataframe: None,
        })
    }

    #[getter]
    fn panel(&self) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame(self.mibi_file.panel.clone()))
    }

    #[getter]
    fn mass_calibration_json(&self) -> PyResult<String> {
        let mass_cal = self.mibi_file.descriptor.mass_calibration().ok_or_else(|| {
            PyValueError::new_err("Mass calibration data not found in file descriptor")
        })?;

        serde_json::to_string(&mass_cal).map_err(|e| {
            PyValueError::new_err(format!(
                "Failed to serialize mass calibration to JSON: {}",
                e
            ))
        })
    }

    #[getter]
    fn dimensions(&self) -> PyResult<(u16, u16)> {
        Ok((
            self.mibi_file.header.n_x_pixels() as u16,
            self.mibi_file.header.n_y_pixels() as u16,
        ))
    }

    #[getter]
    fn fov_size_microns(&self) -> PyResult<f64> {
        self.mibi_file.descriptor.fov_size_microns.ok_or_else(|| {
            PyValueError::new_err("FOV size (microns) not found in descriptor")
        })
    }

    #[getter]
    fn fov_id(&self) -> PyResult<String> {
        self.mibi_file.descriptor.fov_id.clone().ok_or_else(|| {
            PyValueError::new_err("FOV ID not found in descriptor")
        })
    }

    #[getter]
    fn fov_name(&self) -> PyResult<String> {
        self.mibi_file.descriptor.fov_name.clone().ok_or_else(|| {
            PyValueError::new_err("FOV name not found in descriptor")
        })
    }

    #[getter]
    fn run_name(&self) -> PyResult<String> {
        self.mibi_file.descriptor.run_name.clone().ok_or_else(|| {
            PyValueError::new_err("Run name not found in descriptor")
        })
    }

    #[getter]
    fn run_uuid(&self) -> PyResult<String> {
        self.mibi_file.descriptor.run_uuid.clone().ok_or_else(|| {
            PyValueError::new_err("Run UUID not found in descriptor")
        })
    }

    #[getter]
    fn instrument_identifier(&self) -> PyResult<String> {
        self.mibi_file
            .descriptor
            .instrument_identifier
            .clone()
            .ok_or_else(|| PyValueError::new_err("Instrument identifier not found in descriptor"))
    }

    #[getter]
    fn instrument_control_version(&self) -> PyResult<String> {
        self.mibi_file
            .descriptor
            .instrument_control_version
            .clone()
            .ok_or_else(|| {
                PyValueError::new_err("Instrument control version not found in descriptor")
            })
    }

    #[getter]
    fn tof_app_version(&self) -> PyResult<String> {
        self.mibi_file
            .descriptor
            .tof_app_version
            .clone()
            .ok_or_else(|| PyValueError::new_err("TOF app version not found in descriptor"))
    }

    #[getter]
    fn dwell_time_millis(&self) -> PyResult<f64> {
        // Prefer fov.dwell_time_millis if available, otherwise top-level
        self.mibi_file.descriptor.fov.as_ref().and_then(|fov| fov.dwell_time_millis)
            .or(self.mibi_file.descriptor.dwell_time_millis)
            .ok_or_else(|| PyValueError::new_err("Dwell time (ms) not found in descriptor"))
    }

    #[getter]
    fn acquisition_status(&self) -> PyResult<String> {
        self.mibi_file
            .descriptor
            .acquisition_status
            .clone()
            .ok_or_else(|| PyValueError::new_err("Acquisition status not found in descriptor"))
    }

    #[getter]
    fn scan_count(&self) -> PyResult<i32> {
        self.mibi_file
            .descriptor
            .fov.as_ref()
            .and_then(|fov| fov.scan_count)
            .ok_or_else(|| PyValueError::new_err("Scan count not found in descriptor (fov.scan_count)"))
    }

    #[getter]
    fn imaging_preset_name(&self) -> PyResult<String> {
        self.mibi_file
            .descriptor
            .fov.as_ref()
            .and_then(|fov| fov.imaging_preset.as_ref())
            .and_then(|preset| preset.preset.clone())
            .ok_or_else(|| PyValueError::new_err("Imaging preset name not found in descriptor (fov.imaging_preset.preset)"))
    }

    #[getter]
    fn imaging_aperture(&self) -> PyResult<String> {
        self.mibi_file
            .descriptor
            .fov.as_ref()
            .and_then(|fov| fov.imaging_preset.as_ref())
            .and_then(|preset| preset.aperture.clone())
             // Fallback to gun.aperture if fov preset aperture is missing
            .or_else(|| self.mibi_file.descriptor.gun.as_ref().and_then(|gun| gun.aperture.clone()))
            .ok_or_else(|| PyValueError::new_err("Imaging aperture not found in descriptor (fov.imaging_preset.aperture or gun.aperture)"))
    }

    #[getter]
    fn acquisition_start_timestamp(&self) -> PyResult<i64> {
        self.mibi_file
            .descriptor
            .acquisition_start
            .ok_or_else(|| PyValueError::new_err("Acquisition start timestamp not found in descriptor"))
    }

    #[getter]
    fn acquisition_end_timestamp(&self) -> PyResult<i64> {
         self.mibi_file
            .descriptor
            .acquisition_end
            .ok_or_else(|| PyValueError::new_err("Acquisition end timestamp not found in descriptor"))
    }

    fn get_dataframe(&mut self) -> PyResult<PyDataFrame> {
        if let Some(ref df) = self.dataframe {
            Ok(PyDataFrame(df.clone()))
        } else {
            let df = self
                .mibi_file
                .parse_full_image_parallel(self.num_chunks)
                .map_err(|e: PixelParseRegionError| PyValueError::new_err(format!("Pixel parsing error: {}", e)))?;

            self.dataframe = Some(df.clone());
            Ok(PyDataFrame(df))
        }
    }
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MibiReader>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Helper function to construct the file path for tissue data
    #[allow(dead_code)]
    fn get_tissue_file_path(filename: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("data")
            .join("tissue")
            .join(filename)
    }

    #[allow(dead_code)]
    fn get_real_world_bin_file_path(filename: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("data")
            .join("other")
            .join(filename)
    }

    // Define test cases that we might use in the future
    #[allow(dead_code)]
    const TEST_FILES: &[&str] = &[
        "fov-1-scan-1.bin",
        "fov-2-scan-1.bin",
        "fov-3-scan-1.bin",
        "fov-4-scan-1.bin",
        "fov-15-scan-1.bin",
    ];

    #[test]
    fn test_mibi_file_capabilities() -> Result<(), Box<dyn std::error::Error>> {
        let file_path = get_real_world_bin_file_path("fov-1-scan-1.bin");
        println!(
            "\nTesting MibiFile capabilities with: {}",
            file_path.display()
        );

        // Open the file
        let mibi_file = MibiFile::open(file_path.to_str().unwrap())?;

        // 1. Test basic file information
        println!(
            "File dimensions: {}x{}",
            mibi_file.header.size_x_pixels, mibi_file.header.size_y_pixels
        );
        println!(
            "Triggers per pixel: {}",
            mibi_file.header.triggers_per_pixel
        );
        println!("Frame count: {}", mibi_file.header.frame_count);

        // 2. Test descriptor access
        if let Some(id) = &mibi_file.descriptor.id {
            println!("File ID: {}", id);
        }
        if let Some(run_name) = mibi_file.descriptor.run_name() {
            println!("Run name: {}", run_name);
        }

        // ===== 4. Test batch pixel parsing for a specific REGION =====
        println!("\n===== Testing parallel full image parsing ====");

        // Define number of chunks for parallel processing
        let num_chunks = 32;
        println!(
            "\nParsing full image in parallel using {} chunks...",
            num_chunks
        );

        // Call the new method on MibiFile
        let mut combined_df = mibi_file.parse_full_image_parallel(num_chunks)?;

        println!(
            "Full image parsing successful. Combined DataFrame shape: {:?}",
            combined_df.shape()
        );
        use polars_io::prelude::ParquetWriter;
        let mut data_file = std::fs::File::create("../combined_df.parquet").unwrap();
        ParquetWriter::new(&mut data_file).finish(&mut combined_df)?;

        let mut metadata_file = std::fs::File::create("../metadata.parquet").unwrap();
        let mut mibi_panel = mibi_file.panel;
        ParquetWriter::new(&mut metadata_file).finish(&mut mibi_panel)?;

        Ok(())
    }

    #[test]
    fn test_mibi_file_extract_bin_data() -> Result<(), Box<dyn std::error::Error>> {
        let file_path = get_real_world_bin_file_path("fov-1-scan-1.bin");
        println!(
            "\nTesting MibiFile extract_bin_data with: {}",
            file_path.display()
        );

        // Open the file
        let mibi_file = MibiFile::open(file_path.to_str().unwrap())?;

        let df = &mibi_file.panel;
        println!("DataFrame: {:?}", df);

        let mass_start = df.column("mass_start")?;
        let mass_stop = df.column("mass_stop")?;
        println!("mass_start column: {:?}", mass_start);
        println!("mass_stop column: {:?}", mass_stop);
        println!("{}", mibi_file.get_summary());

        // let pixel_data = mibi_file

        Ok(())
    }
}
