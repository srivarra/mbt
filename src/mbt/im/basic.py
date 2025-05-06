from collections.abc import Hashable, Mapping

import dask.array as da
import natsort as ns
import numpy as np
import sparse
import xarray as xr
from numpydantic import NDArray
from spatial_image import to_spatial_image

from mbt.core.models import ImageType


def to_xarray(
    img_data: dict[str, NDArray],
    run_name: str,
    fov_name: str,
    image_type: ImageType,
    sparsity: bool = False,
    scale: Mapping[Hashable, float] | None = None,
) -> xr.DataArray:
    """Convert a dictionary of channel images to an xarray DataArray.

    Parameters
    ----------
    img_data
        A dictionary of channel images.
    name, optional
        The name of the xarray DataArray.
    image_type, optional
        The type of image (e.g. "counts", "intensity", "intensity_width").
    sparsity, optional
        Whether to use sparse storage, by default False.
    scale, optional
        A dictionary of scale factors for the xarray DataArray, by default None.

    Returns
    -------
        An xarray DataArray.
    """
    channel_names = ns.natsorted(img_data.keys())
    reordered_data = np.stack([img_data[ch] for ch in channel_names])
    _, chunk_size_y, chunk_size_x = reordered_data.shape
    data = da.from_array(
        x=sparse.as_coo(reordered_data) if sparsity else reordered_data,
        chunks=(1, chunk_size_y, chunk_size_x),  # type: ignore
    )

    xr_img = to_spatial_image(
        array_like=data,
        dims=("c", "y", "x"),
        scale=scale,
        axis_names={"c": "channel", "y": "y-axis", "x": "x-axis"},
        axis_units={"c": "channel", "y": "micrometers", "x": "micrometers"},
        c_coords=channel_names,
        name=f"{fov_name}_{image_type}",
    )
    xr_img.attrs["image_type"] = image_type
    xr_img.attrs["fov_name"] = fov_name
    xr_img.attrs["run_name"] = run_name
    return xr_img
