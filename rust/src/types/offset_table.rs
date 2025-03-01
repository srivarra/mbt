use bon::Builder;



#[derive(Debug, PartialEq, Builder)]
pub struct OffsetLookupTable {
    pub offsets: Vec<u64>,
    pub size_x_pixels: usize,
    pub size_y_pixels: usize,
    pub frames: usize,
}


impl OffsetLookupTable {
    /// Get offset for specific coordinate
    pub fn get_offset(&self, frame: usize, y_idx: usize, x_idx: usize) -> Option<u64> {
        if frame >= self.frames || y_idx >= self.size_y_pixels || x_idx >= self.size_x_pixels {
            return None;
        }

        let index =
            frame * self.size_y_pixels * self.size_x_pixels + y_idx * self.size_x_pixels + x_idx;
        self.offsets.get(index).copied()
    }
}
