use bon::bon;
use ndarray::Array2;

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
}
