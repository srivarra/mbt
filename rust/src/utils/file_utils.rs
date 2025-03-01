use memmap2::Mmap;
use std::fs::File;
use std::io;
use std::path::Path;

/// Read a binary file using memory mapping for improved performance
/// This is more efficient for large files as it doesn't load the entire file into RAM
pub fn read_binary_file_mmap(path: impl AsRef<Path>) -> io::Result<Mmap> {
    let file = File::open(path)?;
    // Safety: The file is not modified while the mmap is active
    unsafe { Mmap::map(&file) }.map_err(|e| io::Error::new(io::ErrorKind::Other, e))
}
