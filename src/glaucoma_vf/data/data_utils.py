import numpy as np
import polars as pl
import polars.selectors as cs

from glaucoma_vf.enums import GlaucomaSeverity


def df_to_hvf_grids_uwhvf(
    df: pl.DataFrame, columns_prefix: str = "Sens_", fill_value: float = 100.0
) -> np.ndarray:
    """
    Converts a Polars DataFrame of 54 columns into a 3D NumPy array (N, 8, 9).
    """
    num_rows = len(df)

    # 1. Convert the entire DF to a flat 2D numpy array (N, 54)
    # Ensure we only take the Sens columns
    data_54 = df.select(cs.starts_with(columns_prefix)).to_numpy().astype(np.float32)

    # 2. Create the empty destination stack (N, 8, 9) filled with 100.0
    stack = np.full((num_rows, 8, 9), fill_value, dtype=np.float32)

    # 3. Define the HVF 24-2 map indices (The "Holes" in the 8x9 grid)
    # These are the (row, col) coordinates for the 54 points
    # fmt: off
    hvf_map = [
        (0,3),(0,4),(0,5),(0,6),
        (1,2),(1,3),(1,4),(1,5),(1,6),(1,7),
        (2,1),(2,2),(2,3),(2,4),(2,5),(2,6),(2,7),(2,8),
        (3,0),(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7),(3,8),
        (4,0),(4,1),(4,2),(4,3),(4,4),(4,5),(4,6),(4,7),(4,8),
        (5,1),(5,2),(5,3),(5,4),(5,5),(5,6),(5,7),(5,8),
        (6,2),(6,3),(6,4),(6,5),(6,6),(6,7),
        (7,3),(7,4),(7,5),(7,6)
    ]
    # fmt: on

    # 4. "Scatter" the data into the stack
    # We unzip the map into row indices and column indices
    rows, cols = zip(*hvf_map)

    # Advanced Indexing: Fill all N rows at these specific (r, c) spots
    stack[:, rows, cols] = data_54

    return stack


# TODO
def df_to_hvf_grids_grape(df: pl.DataFrame, fill_value: float = 100.0) -> np.ndarray:
    """
    Converts a Polars DataFrame of 61 columns into a 3D NumPy array (N, 10, 10).
    """
    # Load VFs
    cols_to_load = ["VF"] + [str(i) for i in range(5, 65)]
    data_61 = df.select(cols_to_load)

    # Initialize with -1 (the dataset's value for 'unseen' or 'blind')
    grid = np.full((10, 10), -1.0)

    # Mapping Dictionary: index_data_61-> (row, col)
    # This approximates the G1 spiral coordinates into 6-degree bins
    # fmt: off
    #
    # TODO
    #
    # fmt: on

    # Logic to handle multiple points falling into the same cell
    counts = np.zeros((10, 10))
    for i, val in enumerate(data_61):
        if i in mapping:
            r, c = mapping[i]
            if grid[r, c] == -1:
                grid[r, c] = val
            else:
                # Average the values for dense central points
                grid[r, c] = (grid[r, c] * counts[r, c] + val) / (counts[r, c] + 1)
            counts[r, c] += 1

    return grid


def map_mtd_to_enum(mtd_array: np.ndarray) -> np.ndarray:
    """Maps Mean Total Deviation values (MTD) to severity label

    Args:
        mtd_array (np.ndarray): array of floats

    Returns:
        np.ndarray: array of ints of categories 0, 1, or 2
    """

    # Define HPA thresholds:
    # Mild: > -6dB
    # Moderate: -6 to -12dB
    # Severe: < -12dB
    # We use -6 and -12 as our "bins"

    # fmt: off
    conditions = [
        mtd_array > -6.0,                                # MILD
        (mtd_array <= -6.0) & (mtd_array > -12.0),       # MODERATE
        mtd_array <= -12.0                               # SEVERE
    ]
    choices = [
        GlaucomaSeverity.MILD, 
        GlaucomaSeverity.MODERATE, 
        GlaucomaSeverity.SEVERE
    ]
    # fmt: on

    return np.select(conditions, choices, default=GlaucomaSeverity.MILD)
