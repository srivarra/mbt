from os import PathLike

import polars as pl

def hello_world() -> str: ...
def parse_mibi_file_to_py_df(file_path: PathLike, num_chunks: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Parse the MIBI binary file into two Polars DataFrames: one for pixel data and one for panel information.

    Parameters
    ----------
    file_path
        The path to the MIBI binary file.
    num_chunks
        The number of chunks to use for parallel processing.

    Returns
    -------
        A tuple containing two Polars DataFrames:
        - The first DataFrame contains the pixel data.
        - The second DataFrame contains the panel information.
    """
    ...
