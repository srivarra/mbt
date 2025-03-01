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
