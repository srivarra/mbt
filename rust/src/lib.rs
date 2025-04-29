pub mod mibi_file;
pub mod parser;
pub mod processing;
pub mod python;
pub mod types;
pub mod utils;

// Add specific pyo3 imports
pub use mibi_file::MibiFile;
use pyo3::{
    Bound,
    PyErr,
    PyResult,
    pyfunction,
    pymodule,
    types::PyModule,
    types::PyModuleMethods,
    wrap_pyfunction,
};
pub use types::{
    MibiDescriptor, header::Header, offset_table::OffsetLookupTable, pixel_data::PixelData,
};

use crate::parser::pixel_parser::PixelParseRegionError;
use polars::prelude::DataFrame;
use pyo3_polars::PyDataFrame;
use std::path::PathBuf;

// Removed unused import for the old standalone function
// use crate::utils::misc::calculate_full_width_row_chunks;

// --- Error Conversion for PyO3 ---
// Implement conversion from our custom error to PyErr
impl From<PixelParseRegionError> for PyErr {
    fn from(err: PixelParseRegionError) -> PyErr {
        pyo3::exceptions::PyValueError::new_err(err.to_string())
    }
}

#[pyfunction]
fn hello_world() -> PyResult<String> {
    Ok("Hello, world!".to_string())
}

/// Parses a MIBI file and returns its pixel data as a Polars DataFrame.
///
/// Args:
///     file_path (str): Path to the .bin MIBI file.
///     num_chunks (int): Number of chunks for parallel processing.
///
/// Returns:
///     polars.DataFrame: A Polars DataFrame containing the pixel data.
///
/// Raises:
///     FileNotFoundError: If the specified file cannot be opened.
///     ValueError: If parsing fails due to invalid data, offsets, or file issues.
///     RuntimeError: For other unexpected errors during file processing.
#[pyfunction]
#[pyo3(signature = (file_path, num_chunks = 16))]
fn parse_mibi_file_to_py_df(file_path: PathBuf, num_chunks: u64) -> PyResult<(PyDataFrame, PyDataFrame)> {
    let path_str = file_path.to_str().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid UTF-8 sequence in path: {}",
            file_path.display()
        ))
    })?;

    // 1. Open the MIBI file using the &str path
    // Handle error directly instead of relying on From trait
    let mibi_file = MibiFile::open(path_str).map_err(|e| -> PyErr {
        pyo3::exceptions::PyFileNotFoundError::new_err(format!(
            "Failed to open file '{}': {}",
            path_str, // Use path_str here for error message
            e
        ))
    })?;

    // 2. Parse the full image in parallel
    // The From<PixelParseRegionError> for PyErr implementation is still valid
    let df: DataFrame = mibi_file.parse_full_image_parallel(num_chunks as usize)?;
    let panel = mibi_file.panel;

    // 3. Wrap the Rust Polars DataFrame in PyDataFrame for returning to Python
    Ok((PyDataFrame(df), PyDataFrame(panel)))
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello_world, m)?)?;
    m.add_function(wrap_pyfunction!(parse_mibi_file_to_py_df, m)?)?;
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
