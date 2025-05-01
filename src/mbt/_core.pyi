from os import PathLike

import polars as pl

class MibiReader:
    """A class to read and access data from MIBI binary files.

    This class is implemented in Rust and provides an interface to interact
    with MIBI file data, including metadata and pixel data.
    """

    def __init__(self, file_path: PathLike, num_chunks: int = 16):
        """Initializes the MibiReader by opening the specified MIBI file.

        Args:
            file_path: The path to the MIBI binary file (.bin).
            num_chunks: The number of chunks to use for parallel processing
                        when parsing the full image data.

        Raises
        ------
            FileNotFoundError: If the file cannot be found or opened.
            ValueError: If the file path is invalid or for other file access issues.
        """
        ...

    @property
    def panel(self) -> pl.DataFrame:
        """Returns the panel information DataFrame.

        This DataFrame is typically loaded when the file is opened.
        """
        ...

    @property
    def mass_calibration_json(self) -> str:
        """Returns the mass calibration data as a JSON string.

        Raises
        ------
            ValueError: If mass calibration data is not found or cannot be serialized.
        """
        ...

    @property
    def dimensions(self) -> tuple[int, int]:
        """Returns the image dimensions (width, height) in pixels.

        Returns
        -------
            A tuple (width, height).
        """
        ...

    @property
    def fov_size_microns(self) -> float:
        """Returns the field of view size in microns."""
        ...

    @property
    def fov_id(self) -> str:
        """Returns the field of view ID."""
        ...

    @property
    def fov_name(self) -> str:
        """Returns the field of view name."""
        ...

    @property
    def run_name(self) -> str:
        """Returns the run name."""
        ...

    @property
    def run_uuid(self) -> str:
        """Returns the run UUID."""
        ...

    @property
    def instrument_identifier(self) -> str:
        """Returns the instrument identifier string."""
        ...

    @property
    def instrument_control_version(self) -> str:
        """Returns the instrument control software version string."""
        ...

    @property
    def tof_app_version(self) -> str:
        """Returns the TOF application version string."""
        ...

    @property
    def dwell_time_millis(self) -> float:
        """Returns the dwell time per pixel in milliseconds."""
        ...

    @property
    def acquisition_status(self) -> str:
        """Returns the acquisition status string."""
        ...

    @property
    def scan_count(self) -> int:
        """Returns the number of scans performed for the FOV."""
        ...

    @property
    def imaging_preset_name(self) -> str:
        """Returns the name of the imaging preset used."""
        ...

    @property
    def imaging_aperture(self) -> str:
        """Returns the imaging aperture setting used."""
        ...

    @property
    def acquisition_start_timestamp(self) -> int:
        """Returns the acquisition start timestamp (likely Unix epoch milliseconds)."""
        ...

    @property
    def acquisition_end_timestamp(self) -> int:
        """Returns the acquisition end timestamp (likely Unix epoch milliseconds)."""
        ...

    def get_dataframe(self) -> pl.DataFrame:
        """Parses and returns the full pixel data as a Polars DataFrame.

        The DataFrame is cached after the first call. Subsequent calls return
        the cached DataFrame.

        Returns
        -------
            A Polars DataFrame containing the pixel data.

        Raises
        ------
            ValueError: If pixel data parsing fails.
        """
        ...
