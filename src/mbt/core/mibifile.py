import time
from collections.abc import Sequence
from functools import cached_property
from os import PathLike
from typing import Any, Literal, TypedDict, Unpack

import numpy as np
import numpydantic.dtype as ndt
import patito as pt
import polars as pl
import xarray as xr
from numpydantic import NDArray
from polars._typing import ParquetCompression
from polars.io.cloud.credential_provider._providers import CredentialProviderFunction
from tqdm.auto import tqdm
from upath import UPath
from zarr.storage import StoreLike

from mbt._core import MibiReader
from mbt.core.models import C, MassCalibrationModel, MibiDataModel, MibiFilePanelModel, UserPanelModel, X, Y
from mbt.core.utils import _set_tof_ranges, format_image_name
from mbt.im import to_xarray


class WriteParquetKwargs(TypedDict):
    """Parameters for writing Parquet files."""

    compression: ParquetCompression
    compression_level: int | None
    statistics: bool | str | dict[str, bool]
    row_group_size: int | None
    data_page_size: int | None
    use_pyarrow: bool
    pyarrow_options: dict[str, Any]
    partition_by: str | Sequence[str]
    partition_chunk_size_bytes: int
    storage_options: dict[str, Any]
    credential_provider: CredentialProviderFunction | Literal["auto"] | None
    retries: int


class MibiFile:
    """
    Context manager and data loader for MIBI binary files.

    Loads pixel data and associated metadata (panel, mass calibration) lazily
    using a Rust backend. Validates data structures using Patito/Pydantic models.
    A user-supplied panel can be provided for validation against UserPanelModel;
    otherwise, the panel embedded in the MIBI file is loaded and validated
    against MibiFilePanelModel.
    """

    # --- Initialization & Context Management --- #

    def __init__(
        self,
        file_path: PathLike,
        panel: pl.DataFrame | None = None,
        time_resolution: float | None = 500e-6,
        num_chunks: int = 16,
        sparsity: bool = False,
    ):
        if (fp := UPath(file_path)).is_file():
            self.file_path = fp
        else:
            raise FileNotFoundError(f"MIBI file not found: {fp}")

        self.reader = MibiReader(self.file_path, num_chunks)
        self.num_chunks = num_chunks
        self._data: pt.LazyFrame[MibiDataModel] | None = None
        self._panel: pt.DataFrame[MibiFilePanelModel] | pt.DataFrame[UserPanelModel] | None = None
        self._mass_calibration: MassCalibrationModel | None = None
        self._loaded_data = False
        self._time_resolution = time_resolution
        self.sparsity = sparsity

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

    # --- Internal Data Loading --- #

    def _load_data_if_needed(self) -> None:
        """Internal method to load and validate data using the Rust backend."""
        if self._loaded_data:
            return

        try:
            # 1. Get Metadata & Dimensions from Reader
            (n_x_pixels, n_y_pixels) = self.reader.dimensions
            mass_calibration_json = self.reader.mass_calibration_json
            # Only get panel from reader if no user panel was provided
            file_panel_df = self.reader.panel if self._panel is None else None

            # Name of the FOV, run
            self.fov_id = self.reader.fov_id
            self.fov_name = self.reader.fov_name
            self.run_name = self.reader.run_name
            self.run_uuid = self.reader.run_uuid
            self.fov_size_microns = self.reader.fov_size_microns
            self.formatted_fov_name = format_image_name(self.fov_id, self.fov_name)

            # 2. Validate Metadata
            self._mass_calibration = MassCalibrationModel.model_validate_json(mass_calibration_json)

            # Validate file panel if needed, otherwise confirm user panel
            if self._panel is None and file_panel_df is not None:
                self._panel = MibiFilePanelModel.validate(file_panel_df)
            elif self._panel is not None:
                # Assuming user panel was validated in __init__
                print("(User Panel OK)", end=" ", flush=True)
            else:
                # Handle case where panel is None but shouldn't be?
                print("Panel validation skipped/failed?", end=" ", flush=True)

            # 3. Get & Validate Main Data
            start_time = time.time()
            raw_data_df: pl.DataFrame = self.reader.get_dataframe()
            end_time = time.time()
            print(f"({end_time - start_time:.5f}s)", end=" ", flush=True)
            validated_data_df = MibiDataModel.validate(raw_data_df)
            self._data = validated_data_df.lazy()

            # 4. Set internal state
            self._n_x_pixels = n_x_pixels
            self._n_y_pixels = n_y_pixels
            self._loaded_data = True

        except (pt.DataFrameValidationError, Exception) as e:
            # Reset state on error
            self._data = None
            self._mass_calibration = None
            self._loaded_data = False
            print(f"\nError loading/validating file {self.file_path}: {e}")
            raise e

    # --- Core Data Properties --- #

    @property
    def data(self) -> pt.LazyFrame[MibiDataModel]:
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

    # --- Panel Operations --- #

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

    def get_channel_info(self, channel_name: str) -> pl.DataFrame | None:
        """Example: Get panel info for a specific channel using the 'target' column."""
        try:
            return self.panel.filter(pl.col("target") == channel_name)
        except (pt.DataFrameValidationError, Exception) as e:
            print(f"Error filtering panel for channel '{channel_name}': {e}")
            return None

    # --- Image Generation --- #

    def _generate_channel_images(
        self,
        aggregation_expr: pl.Expr,
        output_dtype: type,
        required_data_cols: list[str],
        result_col_alias: str,
        log_suffix: str,
    ) -> xr.DataArray:
        """Internal helper to generate multichannel images based on an aggregation."""
        self._load_data_if_needed()

        if self._panel is None or self._data is None:
            raise AttributeError("Panel or data not loaded or is missing.")

        if "lower_tof_range" not in self.panel.columns or "upper_tof_range" not in self.panel.columns:
            print("TOF ranges not found in panel, calculating now...")
            self.add_tof_ranges_to_panel()

        channel_col = "target" if "target" in self.panel.columns else "channel_name"
        if channel_col not in self.panel.columns:
            raise AttributeError("Panel missing channel identifier column ('target' or 'channel_name').")

        # Prepare panel
        panel_to_iterate = self.panel.select(channel_col, "lower_tof_range", "upper_tof_range")
        channel_names = panel_to_iterate.select(channel_col).to_series().to_list()

        # Prepare output structure - Initialized before the loop
        output_images: dict[str, NDArray] = {
            name: np.zeros((self._n_y_pixels, self._n_x_pixels), dtype=output_dtype) for name in channel_names
        }

        for row in tqdm(panel_to_iterate.sort(channel_col).iter_rows(named=True), total=len(panel_to_iterate)):
            channel_name = row[channel_col]
            lower_tof = row["lower_tof_range"]
            upper_tof = row["upper_tof_range"]

            # Filter data for the current channel, select necessary columns and aggregate *only* for this channel
            pixel_aggregated_channel = (
                self.data.filter(pl.col("pulse_time").is_between(lower_tof, upper_tof, closed="both"))
                .select(["pixel_y", "pixel_x"] + required_data_cols)
                .group_by(["pixel_y", "pixel_x"])
                .agg(aggregation_expr)
                .collect()
            )

            # Populate the NumPy array for this channel
            if not pixel_aggregated_channel.is_empty():
                y_coords: NDArray = pixel_aggregated_channel.get_column("pixel_y").to_numpy(allow_copy=False)
                x_coords: NDArray = pixel_aggregated_channel.get_column("pixel_x").to_numpy(allow_copy=False)
                values: NDArray = pixel_aggregated_channel.get_column(result_col_alias).to_numpy(allow_copy=False)
                output_images[channel_name][y_coords, x_coords] = values

        img_xr = to_xarray(
            output_images,
            name=format_image_name(self.fov_id, self.fov_name),
            sparsity=self.sparsity,
            scale={
                C: 1,
                X: self.fov_size_microns / self._n_x_pixels,
                Y: self.fov_size_microns / self._n_y_pixels,
            },
        )
        return img_xr

    @cached_property
    def counts_image(self) -> xr.DataArray:
        """Returns the counts image (computed on first access)."""
        return self._generate_channel_images(
            aggregation_expr=pl.count().alias("counts"),
            output_dtype=ndt.UInt32,
            required_data_cols=[],
            result_col_alias="counts",
            log_suffix="count images",
        )

    @cached_property
    def intensity_image(self) -> xr.DataArray:
        """Returns the intensity image (computed on first access)."""
        return self._generate_channel_images(
            aggregation_expr=pl.sum("pulse_intensity").alias("sum_intensity"),
            output_dtype=ndt.Int64,
            required_data_cols=["pulse_intensity"],
            result_col_alias="sum_intensity",
            log_suffix="intensity images",
        )

    @cached_property
    def intensity_width_image(self) -> xr.DataArray:
        """Returns the intensity*width image (computed on first access)."""
        return self._generate_channel_images(
            aggregation_expr=(pl.col("pulse_intensity") * pl.col("pulse_width")).sum().alias("sum_intensity_width"),
            output_dtype=ndt.Int64,
            required_data_cols=["pulse_intensity", "pulse_width"],
            result_col_alias="sum_intensity_width",
            log_suffix="intensity*width images",
        )

    # --- Data Export --- #

    def write_data(self, file_path: PathLike, **kwargs: Unpack[WriteParquetKwargs]) -> None:
        """Writes the raw pulse data (LazyFrame) to a Parquet file."""
        self._load_data_if_needed()
        output_path = UPath(file_path)
        self.data.collect().write_parquet(output_path, **kwargs)

    def _configure_zarr(self):
        """Sets the global zarr configuration for optimized writing."""
        import zarr
        import zarrs  # noqa: F401

        zarr.config.set(
            {
                "threading.max_workers": None,
                "array.write_empty_chunks": False,
                "codec_pipeline": {
                    "path": "zarrs.ZarrsCodecPipeline",
                    "validate_checksums": True,
                    "store_empty_chunks": False,
                    "chunk_concurrent_maximum": None,
                    "chunk_concurrent_minimum": 4,
                    "batch_size": 1,
                },
            }
        )

    def write_dataset_to_zarr(self, store: StoreLike):
        """Convert the generated image DataArrays to a Dataset and write to a Zarr store."""
        self._configure_zarr()
        import xarray as xr

        xr.Dataset(
            data_vars={
                "counts": self.counts_image,
                "intensity": self.intensity_image,
                "intensity_width": self.intensity_width_image,
            },
            coords={
                C: self.counts_image.coords[C].to_list(),
                X: self.counts_image.coords[X].to_list(),
                Y: self.counts_image.coords[Y].to_list(),
            },
        ).to_zarr(store, consolidated=True)  # type: ignore

    def write_ome_zarr(
        self, output_directory: PathLike, image_type: list[Literal["counts", "intensity", "intensity_width"]]
    ):
        """
        Write specified image types to individual OME-Zarr stores in a directory.

        Args:
            output_directory: The directory where the OME-Zarr stores will be created.
            image_type: A list of image types to write (e.g., ["counts", "intensity"]).
                      Valid types are "counts", "intensity", "intensity_width".
        """
        self._configure_zarr()  # Call helper method
        self._load_data_if_needed()  # Ensure data/dims are loaded

        from ngff_zarr import to_multiscales, to_ngff_image, to_ngff_zarr

        # Create the output directory if it doesn't exist
        out_dir = UPath(output_directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Map image type names to their corresponding DataArray properties
        image_map = {
            "counts": self.counts_image,
            "intensity": self.intensity_image,
            "intensity_width": self.intensity_width_image,
        }

        for img_type in image_type:
            img_data = image_map[img_type]
            run_dir = out_dir / self.run_name
            run_dir.mkdir(parents=True, exist_ok=True)

            output_path = run_dir / f"{self.reader.fov_name}-{img_type}.ome.zarr"

            ngff_image = to_ngff_image(
                data=img_data,
                dims=[C, X, Y],
                scale={
                    C: 1,
                    Y: self.fov_size_microns / self._n_y_pixels,
                    X: self.fov_size_microns / self._n_x_pixels,
                },
                translation=None,
                name=f"{self.reader.fov_name}_{img_type}",  # Use specific name
                axes_units={X: "micrometer", Y: "micrometer"},  # Assuming micrometer
            )
            multiscales = to_multiscales(
                ngff_image,
                scale_factors=[{Y: 1, X: 1}],
                chunks={C: 1, X: self._n_x_pixels, Y: self._n_y_pixels},
            )

            # Write the specific image to its own zarr store
            to_ngff_zarr(
                str(output_path),
                multiscales=multiscales,
                consolidated=True,
            )
