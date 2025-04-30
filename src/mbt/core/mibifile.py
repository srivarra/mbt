import time
from collections.abc import Mapping
from os import PathLike
from pathlib import Path

import numpy as np
import patito as pt
import polars as pl
import xarray as xr
from numpydantic import NDArray

from mbt._core import parse_mibi_file_to_py_df
from mbt.core.models import MassCalibrationModel, MibiDataModel, MibiFilePanelModel, UserPanelModel
from mbt.core.utils import _set_tof_ranges
from mbt.im import to_xarray


class MibiFile:
    """
    Context manager and data loader for MIBI binary files.

    Loads pixel data and associated metadata (panel, mass calibration) lazily
    using a Rust backend. Validates data structures using Patito/Pydantic models.
    A user-supplied panel can be provided for validation against UserPanelModel;
    otherwise, the panel embedded in the MIBI file is loaded and validated
    against MibiFilePanelModel.
    """

    def __init__(
        self,
        file_path: PathLike,
        panel: pl.DataFrame | None = None,
        time_resolution: float | None = 500e-6,
        num_chunks: int = 16,
    ):
        self.file_path = Path(file_path)
        if not self.file_path.is_file():
            raise FileNotFoundError(f"MIBI file not found: {self.file_path}")

        self.num_chunks = num_chunks
        self._data: pl.LazyFrame | None = None
        self._panel: pt.DataFrame[MibiFilePanelModel] | pt.DataFrame[UserPanelModel] | None = None
        self._mass_calibration: MassCalibrationModel | None = None
        self._loaded_data: bool = False
        self._time_resolution = time_resolution

        if panel is not None:
            # Validate the user-provided panel immediately
            try:
                self._panel = UserPanelModel.validate(panel)
                print("User-supplied panel validated successfully.")
            except (pt.DataFrameValidationError, Exception) as e:
                print(f"User-supplied panel validation failed for file: {self.file_path}")
                raise e  # Re-raise validation or other error
        # Else: self._panel remains None, will be loaded from file later

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Returning False (or omitting return) propagates exceptions
        pass

    def _load_data_if_needed(self) -> None:
        """Internal method to load and validate data using the Rust backend."""
        if self._loaded_data:
            return

        try:
            print(f"Parsing MIBI file {self.file_path}...", end=" ", flush=True)
            # 1. Call Rust function: returns data, embedded panel, mass cal JSON
            raw_data_df, file_panel_df, mass_calibration_json, (size_x_pixels, size_y_pixels) = (
                parse_mibi_file_to_py_df(
                    file_path=self.file_path,
                    num_chunks=self.num_chunks,
                )
            )
            self._size_x_pixels = size_x_pixels
            self._size_y_pixels = size_y_pixels

            # 2. Validate Mass Calibration
            self._mass_calibration = MassCalibrationModel.model_validate_json(mass_calibration_json)
            print("(Mass Cal OK)", end=" ", flush=True)

            # 3. Validate MIBI Data (Optional)
            validated_data_df = MibiDataModel.validate(raw_data_df)
            self._data = validated_data_df.lazy()
            print("(Data OK)", end=" ", flush=True)

            # 4. Validate Panel (File panel only if no user panel exists)
            if self._panel is None:
                self._panel = MibiFilePanelModel.validate(file_panel_df)
                print("(File Panel OK)")
            else:
                print("(Using User Panel)")

            self._loaded_data = True

        except (pt.DataFrameValidationError, Exception) as e:
            self._data = None
            self._mass_calibration = None
            if "file_panel_df" in locals() and self._panel is file_panel_df:
                self._panel = None
            self._loaded_data = False
            print(f"\nError loading/validating file {self.file_path}: {e}")
            raise e

    @property
    def data(self) -> pl.LazyFrame:
        """Lazily loads and returns the pixel data LazyFrame."""
        self._load_data_if_needed()
        assert self._data is not None
        return self._data

    @property
    def panel(self) -> pt.DataFrame[MibiFilePanelModel] | pt.DataFrame[UserPanelModel]:
        """Returns the panel DataFrame (user-supplied or from file)."""
        self._load_data_if_needed()
        assert self._panel is not None
        return self._panel

    @property
    def mass_calibration(self) -> MassCalibrationModel:
        """Returns the validated mass calibration data."""
        self._load_data_if_needed()
        assert self._mass_calibration is not None
        return self._mass_calibration

    def add_tof_ranges_to_panel(self) -> None:
        """
        Adds lower and upper time-of-flight (TOF) range columns to the panel.

        Uses the mass calibration parameters associated with this MIBI file
        to calculate TOF ranges based on the 'mass_start' and 'mass_stop'
        columns in the panel.

        Returns
        -------
            A new Polars DataFrame derived from the instance's panel, with
            'lower_tof_range' (UInt16) and 'upper_tof_range' (UInt16) added.

        Raises
        ------
            ValueError: If the panel DataFrame lacks 'mass_start' or 'mass_stop'.
            AttributeError: If data hasn't been loaded successfully.
        """
        self._load_data_if_needed()
        if self._panel is None or self._mass_calibration is None:
            raise AttributeError("Panel or mass calibration data not loaded.")

        self._panel = MibiFilePanelModel.validate(
            _set_tof_ranges(
                panel=self.panel,
                mass_offset=self.mass_calibration.mass_offset,
                mass_gain=self.mass_calibration.mass_gain,
                time_resolution=self._time_resolution or self.mass_calibration.time_resolution,
            ),
            allow_superfluous_columns=True,
        )

    def get_all_channel_images(self, sparsity: bool = False) -> xr.DataArray:
        """Generates a dictionary of 2D NumPy arrays, one for each channel in the panel.

        Returns
        -------
            A dictionary where keys are channel names (from 'target' or 'channel_name')
            and values are 2D NumPy arrays (dtype=np.uint32) representing pulse counts.

        Raises
        ------
            AttributeError: If data hasn't been loaded successfully or panel is missing.
        """
        start_time = time.time()
        self._load_data_if_needed()

        if self._panel is None:
            raise AttributeError("Panel data not loaded or is missing.")

        if "lower_tof_range" not in self.panel.columns:
            print("TOF ranges not found in panel, calculating now...")
            self.add_tof_ranges_to_panel()  # Modifies self.panel in place

        channel_col = "target" if "target" in self.panel.columns else "channel_name"
        if channel_col not in self.panel.columns:
            raise AttributeError("Panel missing channel identifier column ('target' or 'channel_name').")

        # Prepare panel for iteration (it's small and eager)
        panel_to_iterate = self.panel.select(channel_col, "lower_tof_range", "upper_tof_range")

        lazy_frames_to_concat = []
        channel_names = panel_to_iterate.select(channel_col).to_series().to_list()

        # Create lazy frames for each channel

        for row in panel_to_iterate.sort(channel_col).iter_rows(named=True):
            channel_name = row[channel_col]
            lower_tof = row["lower_tof_range"]
            upper_tof = row["upper_tof_range"]

            lf = (
                self.data.filter(pl.col("pulse_time").is_between(lower_tof, upper_tof, closed="both"))
                .with_columns(pl.lit(channel_name).alias(channel_col))
                # Select only needed columns for aggregation
                .select("pixel_x", "pixel_y", channel_col)
            )
            lazy_frames_to_concat.append(lf)

        # Concatenate and aggregate
        combined_lf: pl.LazyFrame = pl.concat(lazy_frames_to_concat, how="vertical")

        pixel_counts_all_channels = (
            combined_lf.group_by([channel_col, "pixel_y", "pixel_x"])
            .agg(pl.count().alias("counts").cast(pl.UInt32))
            .collect()
        )

        # Prepare output structure
        output_images: dict[str, np.ndarray] = {
            name: np.zeros((self._size_y_pixels + 1, self._size_x_pixels + 1), dtype=np.uint32)
            for name in channel_names
        }
        stop_time = time.time()
        print(f"Time taken to gather data and prepare output arrays: {stop_time - start_time:.4f} seconds")

        start_time = time.time()

        print(f"Populating {len(channel_names)} images using optimized method...")
        for ch_name in channel_names:
            channel_specific_counts = pixel_counts_all_channels.filter(pl.col(channel_col) == ch_name)

            y_coords = channel_specific_counts.get_column("pixel_y").to_numpy()
            x_coords = channel_specific_counts.get_column("pixel_x").to_numpy()
            counts = channel_specific_counts.get_column("counts").to_numpy()

            output_images[ch_name][y_coords, x_coords] = counts

        stop_time = time.time()
        print(f"Time taken to populate images: {stop_time - start_time:.4f} seconds")
        img_xr = to_xarray(output_images, sparsity=sparsity)  # type: ignore

        return img_xr

    def get_channel_info(self, channel_name: str) -> pl.DataFrame | None:
        """Example: Get panel info for a specific channel using the 'target' column."""
        try:
            return self.panel.filter(pl.col("target") == channel_name)
        except (pt.DataFrameValidationError, Exception) as e:
            print(f"Error filtering panel for channel '{channel_name}': {e}")
            return None

    def get_channel_pulse_widths(self, channel_name: str) -> pl.DataFrame:
        """
        Calculates the histogram of pulse widths for a specific channel.

        Args:
            channel_name: The name of the channel (from 'target' or 'channel_name' column).

        Returns
        -------
            A Polars DataFrame with columns 'pulse_width' (UInt8) and 'count' (UInt64),
            representing the frequency of each pulse width within the channel's TOF range.

        Raises
        ------
            AttributeError: If data or panel hasn't been loaded or is missing required columns.
            ValueError: If the specified channel_name is not found in the panel.
        """
        self._load_data_if_needed()
        if self._panel is None or self._data is None:
            raise AttributeError("Data or panel not loaded.")
        if "lower_tof_range" not in self.panel.columns:
            print("TOF ranges not found in panel, calculating now...")
            self.add_tof_ranges_to_panel()

        channel_info = self.get_channel_info(channel_name)
        if channel_info is None or channel_info.is_empty():
            raise ValueError(f"Channel '{channel_name}' not found in the panel.")

        lower_tof = channel_info.select("lower_tof_range").item()
        upper_tof = channel_info.select("upper_tof_range").item()

        width_histogram = (
            self.data.filter(pl.col("pulse_time").is_between(lower_tof, upper_tof, closed="both"))
            .group_by("pulse_width")
            .agg(pl.count())  # Default output column name is 'count'
            .sort("pulse_width")
            .collect()
        )
        return width_histogram

    def get_channel_pulse_intensities(self, channel_name: str) -> pl.DataFrame:
        """
        Calculates the histogram of pulse intensities for a specific channel.

        Args:
            channel_name: The name of the channel (from 'target' or 'channel_name' column).

        Returns
        -------
            A Polars DataFrame with columns 'pulse_intensity' (UInt16) and 'count' (UInt64),
            representing the frequency of each pulse intensity within the channel's TOF range.

        Raises
        ------
            AttributeError: If data or panel hasn't been loaded or is missing required columns.
            ValueError: If the specified channel_name is not found in the panel.
        """
        self._load_data_if_needed()
        if self._panel is None or self._data is None:
            raise AttributeError("Data or panel not loaded.")
        if "lower_tof_range" not in self.panel.columns:
            print("TOF ranges not found in panel, calculating now...")
            self.add_tof_ranges_to_panel()

        channel_info = self.get_channel_info(channel_name)
        if channel_info is None or channel_info.is_empty():
            raise ValueError(f"Channel '{channel_name}' not found in the panel.")

        lower_tof = channel_info.select("lower_tof_range").item()
        upper_tof = channel_info.select("upper_tof_range").item()

        intensity_histogram = (
            self.data.filter(pl.col("pulse_time").is_between(lower_tof, upper_tof, closed="both"))
            .group_by("pulse_intensity")
            .agg(pl.count())  # Default output column name is 'count'
            .sort("pulse_intensity")
            .collect()
        )
        return intensity_histogram

    def get_all_channel_intensity_images(self, sparsity: bool = False) -> xr.DataArray:
        """Create an Xarray DataArray of the multichannel intensity image.

        Returns
        -------
            An xarray DataArray where coordinates are channel names and pixel coordinates,
            and values are 2D NumPy arrays (dtype=np.uint64) representing summed pulse intensities.

        Raises
        ------
            AttributeError: If data hasn't been loaded successfully or panel is missing.
        """
        start_time = time.time()
        self._load_data_if_needed()

        if self._panel is None or self._data is None:
            raise AttributeError("Panel or data not loaded or is missing.")

        if "lower_tof_range" not in self.panel.columns:
            print("TOF ranges not found in panel, calculating now...")
            self.add_tof_ranges_to_panel()

        channel_col = "target" if "target" in self.panel.columns else "channel_name"
        if channel_col not in self.panel.columns:
            raise AttributeError("Panel missing channel identifier column ('target' or 'channel_name').")

        # Prepare panel
        panel_to_iterate = self.panel.select(channel_col, "lower_tof_range", "upper_tof_range")
        channel_names = panel_to_iterate.select(channel_col).to_series().to_list()

        lazy_frames_to_concat = []
        print(f"Preparing lazy frames for {len(channel_names)} intensity images...")
        for row in panel_to_iterate.sort(channel_col).iter_rows(named=True):
            channel_name = row[channel_col]
            lower_tof = row["lower_tof_range"]
            upper_tof = row["upper_tof_range"]

            lf = (
                self.data.filter(pl.col("pulse_time").is_between(lower_tof, upper_tof, closed="both"))
                .with_columns(pl.lit(channel_name).alias(channel_col))
                # Select needed columns for aggregation
                .select("pixel_x", "pixel_y", channel_col, "pulse_intensity")
            )
            lazy_frames_to_concat.append(lf)

        # Concatenate and aggregate
        combined_lf: pl.LazyFrame = pl.concat(lazy_frames_to_concat, how="vertical")

        pixel_intensities_all_channels = (
            combined_lf.group_by([channel_col, "pixel_y", "pixel_x"])
            .agg(pl.sum("pulse_intensity").alias("sum_intensity").cast(pl.UInt64))
            .collect()
        )

        # Prepare output structure (using uint64 for potentially large sums)
        output_images: dict[str, NDArray] = {
            name: np.zeros((self._size_y_pixels + 1, self._size_x_pixels + 1), dtype=np.uint64)
            for name in channel_names
        }
        stop_time = time.time()
        print(f"Time taken to gather data and prepare output arrays: {stop_time - start_time:.4f} seconds")

        start_time = time.time()
        print(f"Populating {len(channel_names)} intensity images...")
        for ch_name in channel_names:
            channel_specific_intensities = pixel_intensities_all_channels.filter(pl.col(channel_col) == ch_name)

            y_coords = channel_specific_intensities.get_column("pixel_y").to_numpy()
            x_coords = channel_specific_intensities.get_column("pixel_x").to_numpy()
            intensities = channel_specific_intensities.get_column("sum_intensity").to_numpy()

            output_images[ch_name][y_coords, x_coords] = intensities

        stop_time = time.time()
        print(f"Time taken to populate images: {stop_time - start_time:.4f} seconds")
        img_xr = to_xarray(output_images, sparsity=sparsity)

        return img_xr

    def get_all_channel_intensity_width_images(self, sparsity: bool = False) -> xr.DataArray:
        """Create an Xarray DataArray of the multichannel intensity*width image.

        Returns
        -------
            An xarray DataArray where coordinates are channel names and pixel coordinates,
            and values are 2D NumPy arrays (dtype=np.uint64) representing summed (intensity * width).

        Raises
        ------
            AttributeError: If data hasn't been loaded successfully or panel is missing.
        """
        start_time = time.time()
        self._load_data_if_needed()

        if self._panel is None or self._data is None:
            raise AttributeError("Panel or data not loaded or is missing.")

        if "lower_tof_range" not in self.panel.columns:
            print("TOF ranges not found in panel, calculating now...")
            self.add_tof_ranges_to_panel()

        channel_col = "target" if "target" in self.panel.columns else "channel_name"
        if channel_col not in self.panel.columns:
            raise AttributeError("Panel missing channel identifier column ('target' or 'channel_name').")

        # Prepare panel
        panel_to_iterate = self.panel.select(channel_col, "lower_tof_range", "upper_tof_range")
        channel_names = panel_to_iterate.select(channel_col).to_series().to_list()

        lazy_frames_to_concat = []
        print(f"Preparing lazy frames for {len(channel_names)} intensity*width images...")
        for row in panel_to_iterate.sort(channel_col).iter_rows(named=True):
            channel_name = row[channel_col]
            lower_tof = row["lower_tof_range"]
            upper_tof = row["upper_tof_range"]

            lf = (
                self.data.filter(pl.col("pulse_time").is_between(lower_tof, upper_tof, closed="both"))
                .with_columns(pl.lit(channel_name).alias(channel_col))
                # Select needed columns for aggregation
                .select("pixel_x", "pixel_y", channel_col, "pulse_intensity", "pulse_width")
            )
            lazy_frames_to_concat.append(lf)

        # Concatenate and aggregate
        combined_lf: pl.LazyFrame = pl.concat(lazy_frames_to_concat, how="vertical")

        pixel_intensity_widths_all_channels = (
            combined_lf.group_by([channel_col, "pixel_y", "pixel_x"])
            .agg((pl.col("pulse_intensity") * pl.col("pulse_width")).sum().alias("sum_intensity_width").cast(pl.UInt64))
            .collect()
        )

        # Prepare output structure (using uint64 for potentially large sums)
        output_images: Mapping[str, np.ndarray] = {
            name: np.zeros((self._size_y_pixels + 1, self._size_x_pixels + 1), dtype=np.uint64)
            for name in channel_names
        }
        stop_time = time.time()
        print(f"Time taken to gather data and prepare output arrays: {stop_time - start_time:.4f} seconds")

        start_time = time.time()
        print(f"Populating {len(channel_names)} intensity*width images...")
        for ch_name in channel_names:
            channel_specific_data = pixel_intensity_widths_all_channels.filter(pl.col(channel_col) == ch_name)

            y_coords = channel_specific_data.get_column("pixel_y").to_numpy()
            x_coords = channel_specific_data.get_column("pixel_x").to_numpy()
            intensity_widths = channel_specific_data.get_column("sum_intensity_width").to_numpy()

            output_images[ch_name][y_coords, x_coords] = intensity_widths

        stop_time = time.time()
        print(f"Time taken to populate images: {stop_time - start_time:.4f} seconds")
        img_xr = to_xarray(output_images, sparsity=sparsity)  # type: ignore

        return img_xr
