import numpy as np
from scipy.spatial import cKDTree

from glaucoma_vf.utils import get_git_root

REPO_ROOT = get_git_root(__file__)
ASSETS_DIR = REPO_ROOT / "assets"
RAW_POINTS_FILENAME = ASSETS_DIR / "grape_vf_report_coords.txt"
COORDS_DEG_FILENAME = ASSETS_DIR / "grape_vf_report_coords_degrees.txt"


def generate():
    """
    Generate master lookup (61, 61): allows to go from (61,) raw VF data to
    (61, 61) Voronoi image. The Voronoi image contains 61 cells, each
    comprised of some pixels. All pixels in one cell share the same dB
    value.

    Generate training mask (61, 61): mask to ignore points outside the 30° radius of
    the VF (0: outside, 1: inside) during training.
    """
    coords_deg = np.loadtxt(COORDS_DEG_FILENAME, delimiter=",")

    # 1. Create the 61x61 Grid
    res = 61
    lin = np.linspace(-30, 30, res)
    gx, gy = np.meshgrid(lin, lin)
    grid_points = np.stack([gx.ravel(), gy.ravel()], axis=1)

    # 2. Generate Lookup (Nearest Neighbor)
    tree = cKDTree(coords_deg)
    _, indices = tree.query(grid_points)
    master_lookup = indices.reshape(res, res)

    # 3. Generate Training Mask (The circle)
    dist = np.sqrt(gx**2 + gy**2)
    training_mask = (dist <= 30.5).astype(np.float32)  # Slight buffer for edge points

    # 4. Save assets
    np.save("assets/grape_master_lookup_61.npy", master_lookup)
    np.save("assets/grape_training_mask_61.npy", training_mask)
    print("Success: Assets saved to assets/")


if __name__ == "__main__":
    generate()
