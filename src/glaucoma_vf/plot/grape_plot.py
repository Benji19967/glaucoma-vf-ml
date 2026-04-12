import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree  # type: ignore

from glaucoma_vf.utils import get_git_root

REPO_ROOT = get_git_root(__file__)
ASSETS_DIR = REPO_ROOT / "assets"
PROCESSED_POINTS_FILENAME = ASSETS_DIR / "grape_vf_report_coords_degrees.txt"
MASTER_LOOKUP_FILENAME = ASSETS_DIR / "grape_master_lookup_61.npy"


def plot_grape_predictions(x_grids, y_grids, preds_grids, n_samples=5):
    preds_grids = preds_grids.cpu().numpy()

    # (61, 61)
    preds_grid = preds_grids[0]

    # Ungrid: (61, 61) -> (61,)
    master_lookup = np.load(MASTER_LOOKUP_FILENAME).astype(int)
    _, ungrid_indices = np.unique(master_lookup, return_index=True)
    preds = preds_grid.flatten()[ungrid_indices]

    # Generate a 500x500 high-res mask for a professional look
    res = 500
    coords_deg = np.loadtxt(PROCESSED_POINTS_FILENAME, delimiter=",")
    smooth_mask = create_smooth_circular_mask(res, res, radius=245, smoothness=1.5)
    master_lookup_highres = create_highres_lookup(coords_deg)

    plot(preds, smooth_mask, master_lookup_highres)


def create_smooth_circular_mask(h, w, center=None, radius=None, smoothness=0.5):
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance from center to image boundary
        radius = min(center[0], center[1], w - center[0], h - center[1])

    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)

    # Use a sigmoid or a simple linear ramp for the edge
    # This creates a very thin 1-pixel 'blur' that looks smooth to the eye
    mask = 1.0 - np.clip(
        (dist_from_center - (radius - smoothness)) / (2 * smoothness), 0, 1
    )
    return mask


def create_highres_lookup(coords_deg, resolution=500):
    # 1. Create a high-definition grid (-30 to 30 degrees)
    lin = np.linspace(-30, 30, resolution)
    grid_x, grid_y = np.meshgrid(lin, lin)
    highres_pixels = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    # 2. Use the same G1 coordinates from your JSON
    tree = cKDTree(coords_deg)
    # 3. Find the nearest G1 point for every one of the 250,000 pixels
    _, idx = tree.query(highres_pixels)
    # 4. Reshape into a 500x500 map
    return idx.reshape(resolution, resolution)


def plot(preds, smooth_mask, master_lookup_highres, idx=5):
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_under("white")
    p_min = preds.min()
    p_max = preds.max()

    img = get_smooth_vf_plot(smooth_mask, preds, master_lookup_highres)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        img,
        cmap=cmap,
        vmin=p_min,
        vmax=p_max,
        origin="upper",
        # Use 'interpolation=lanczos' or 'bicubic' for the smoothest visual output
        interpolation="lanczos",
    )
    # Add the Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.05)
    cbar.set_label("Sensitivity (dB)", rotation=270, labelpad=15, fontweight="bold")

    # Optional: Set specific tick marks
    cbar.set_ticks(np.linspace(0, int(p_max), 5))  # type: ignore

    plt.title("Visual Field Prediction", pad=20)
    plt.axis("off")  # Hide the pixel coordinates for a cleaner look
    plt.show()


def get_smooth_vf_plot(smooth_mask, preds, master_lookup_highres):
    WHITE = -999
    # 1. Map values to high-res grid (e.g., 500x500)
    # This ensures the tiles don't look 'blocky' at the edges
    high_res_img = preds[master_lookup_highres]

    # 2. Apply negative values to the background
    # Logic: If mask is 1, keep img. If mask is 0, set to neg_value.
    final_viz = np.where(smooth_mask > 0.5, high_res_img, WHITE)

    # Add edges to voronoi cells
    edges = np.abs(np.gradient(master_lookup_highres)).sum(axis=0) > 0
    final_viz[edges] = WHITE

    return final_viz
