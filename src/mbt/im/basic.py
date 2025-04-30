import dask.array as da
import natsort as ns
import numpy as np
import sparse
import xarray as xr
from numpydantic import NDArray
from spatial_image import default_name, to_spatial_image


def to_xarray(img_data: dict[str, NDArray], name: str | None = None, sparsity: bool = False) -> xr.DataArray:
    """Convert a dictionary of channel images to an xarray DataArray.

    Parameters
    ----------
    img_data
        A dictionary of channel images.
    name, optional
        The name of the xarray DataArray, by default None.
    sparsity, optional
        Whether to use sparse storage, by default False.

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

    return to_spatial_image(
        array_like=data,
        dims=("c", "y", "x"),
        c_coords=channel_names,
        name=name or default_name,
    )
