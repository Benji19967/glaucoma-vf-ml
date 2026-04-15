import numpy as np
import polars as pl
import polars.selectors as cs

from glaucoma_vf.enums import GlaucomaSeverity
from glaucoma_vf.utils import get_git_root

REPO_ROOT = get_git_root(__file__)
GRAPE_DIR = REPO_ROOT / "data" / "GRAPE"
ASSETS_DIR = REPO_ROOT / "assets"
MASTER_LOOKUP_FILENAME = ASSETS_DIR / "grape_master_lookup_61.npy"
TRAINING_MASK_FILENAME = ASSETS_DIR / "grape_training_mask_61.npy"


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


def df_to_vf_grids_grape(df: pl.DataFrame) -> np.ndarray:
    """
    Converts a Polars DataFrame of (N, 61) into a 3D NumPy array (N, 61, 61).
    """
    # Load VFs
    cols_to_load = ["VF", ""] + [str(i) for i in range(59)]
    data_61 = df.select(cols_to_load).to_numpy().astype(np.float32)

    # 1. Ensure master_lookup is a 61x61 numpy array of integers
    # master_lookup should contain values from 0 to 60 (which of
    # the 61 cells does each pixel--of the 61x61 image--belong to).
    master_lookup = np.load(MASTER_LOOKUP_FILENAME).astype(int)
    training_mask = np.load(TRAINING_MASK_FILENAME).astype(int)

    # 2. Perform the mapping
    # NumPy indexing magic: this creates (N, 61, 61)
    data_grid = data_61[:, master_lookup]

    # 3. Apply the Eye Mask (Zero out the corners)
    # training_mask should be a (61, 61) array of 1s and 0s
    data_grid = data_grid * training_mask

    return data_grid


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
