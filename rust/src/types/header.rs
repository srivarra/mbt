use bon::Builder;

#[derive(Debug, PartialEq, Builder)]
pub struct Header {
    pub file_signature: String,
    pub size_x_pixels: u16,
    pub size_y_pixels: u16,
    pub triggers_per_pixel: u16,
    pub frame_count: u16,
    pub metadata_length: u16,
}

impl Header {
    pub fn n_x_pixels(&self) -> usize {
        self.size_x_pixels as usize
    }

    pub fn n_y_pixels(&self) -> usize {
        self.size_y_pixels as usize
    }
}
