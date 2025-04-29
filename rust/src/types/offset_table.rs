use crate::utils::misc::Coordinate;
use bon::bon;
use itertools::Itertools;
use ndarray::{Array, Array2, Ix1};
#[derive(Debug, PartialEq)]
pub struct OffsetLookupTable {
    pub offsets: Array2<u64>,
    pub size_y_pixels: usize,
    pub size_x_pixels: usize,
    pub frames: usize,
}
#[bon]
impl OffsetLookupTable {
    #[builder]
    pub fn new(
        offsets: Vec<u64>,
        size_x_pixels: usize,
        size_y_pixels: usize,
        frames: usize,
    ) -> Self {
        let offset_table = Array2::from_shape_vec((size_y_pixels, size_x_pixels), offsets)
            .expect("Failed to create offset table");

        Self {
            offsets: offset_table,
            size_y_pixels: size_y_pixels,
            size_x_pixels: size_x_pixels,
            frames: frames,
        }
    }
    /// Get offset for specific coordinate
    pub fn get_offset(&self, frame: usize, y_idx: usize, x_idx: usize) -> Option<u64> {
        if frame >= self.frames || y_idx >= self.size_y_pixels || x_idx >= self.size_x_pixels {
            return None;
        }
        Some(self.offsets[[y_idx, x_idx]])
    }
    /// Convert the offset lookup table to a vector of [start, end] pixel offsets
    ///
    /// # Arguments
    /// * `data_len` - The total length of the data buffer for calculating the last pixel's end
    pub fn as_pixel_offsets(&self, data_len: u64) -> Vec<[u64; 2]> {
        let size_y = self.size_y_pixels;
        let size_x = self.size_x_pixels;
        let mut result = Vec::with_capacity(size_x * size_y);

        for y in 0..size_y {
            for x in 0..size_x {
                let start = self.offsets[[y, x]];
                let end = if x < size_x - 1 {
                    self.offsets[[y, x + 1]]
                } else if y < size_y - 1 {
                    self.offsets[[y + 1, 0]]
                } else {
                    data_len
                };
                result.push([start, end]);
            }
        }

        result
    }

    pub fn flatten(&self) -> Array<u64, Ix1> {
        self.offsets.flatten().to_owned()
    }

    /// Calculates coordinate ranges for processing the image (or a region) in row chunks.
    ///
    /// Divides the relevant image height as evenly as possible among the desired number of chunks.
    /// Each chunk's coordinate range spans the relevant width (full image or region).
    ///
    /// # Arguments
    /// * `num_chunks` - The desired number of chunks (threads).
    /// * `region` - An optional tuple `(start_coord, end_coord)` defining the region.
    ///              If `None`, chunks cover the full image dimensions.
    ///
    /// # Returns
    /// A vector of tuples, where each tuple represents a chunk:
    /// `(start_coordinate, end_coordinate)`
    /// Returns an empty vector if image dimensions or `num_chunks` is 0,
    /// or if provided region coordinates are invalid/out of bounds.
    pub fn calculate_row_chunks(
        &self,
        num_chunks: usize,
        region: Option<(Coordinate, Coordinate)>,
    ) -> Vec<(Coordinate, Coordinate)> {
        let img_rows = self.size_y_pixels;
        let img_cols = self.size_x_pixels;

        // Validate basic conditions
        if img_rows == 0 || img_cols == 0 || num_chunks == 0 {
            return Vec::new();
        }

        // Determine the processing boundaries and validate if region is provided
        // Coordinates are struct { y: usize, x: usize }
        let (process_start_coord, process_end_coord) = match region {
            Some((start_coord, end_coord)) => {
                // Validate region coordinates using struct fields
                if start_coord.y > end_coord.y || start_coord.x > end_coord.x {
                    return Vec::new(); // Invalid range
                }
                // Check bounds (indices must be < dimension)
                if end_coord.y >= img_rows || end_coord.x >= img_cols {
                    return Vec::new(); // Out of bounds
                }
                (start_coord, end_coord)
            }
            None => {
                // Use full image dimensions
                let start_coord = Coordinate { y: 0, x: 0 };
                let end_coord = Coordinate {
                    y: img_rows.saturating_sub(1),
                    x: img_cols.saturating_sub(1),
                };
                (start_coord, end_coord)
            }
        };

        let process_rows = process_end_coord.y - process_start_coord.y + 1;

        // Ensure we don't have more chunks than rows in the processing area
        let actual_num_chunks = std::cmp::min(num_chunks, process_rows);
        if actual_num_chunks == 0 {
            return Vec::new(); // Avoid division by zero if process_rows is 0
        }

        // Calculate chunk size for Y-axis (rows) based on processing rows
        let chunk_size_y = (process_rows + actual_num_chunks - 1) / actual_num_chunks;

        let chunk_ranges: Vec<(Coordinate, Coordinate)> = (process_start_coord.y
            ..=process_end_coord.y)
            .chunks(chunk_size_y)
            .into_iter()
            .map(|mut chunk| {
                // Get the first and last row coordinate from the chunk
                let chunk_start_y = chunk.next().unwrap(); // Chunks are non-empty
                let chunk_end_row = chunk.last().unwrap_or(chunk_start_y);
                // Create Coordinate structs for the chunk boundaries
                let chunk_start_coord = Coordinate {
                    y: chunk_start_y,
                    x: process_start_coord.x, // Use region's start column
                };
                let chunk_end_coord = Coordinate {
                    y: chunk_end_row,
                    x: process_end_coord.x, // Use region's end column
                };
                (chunk_start_coord, chunk_end_coord)
            })
            .collect();

        chunk_ranges
    }
}
