import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree  # type: ignore

from glaucoma_vf.utils import get_git_root

REPO_ROOT = get_git_root(__file__)
ASSETS_DIR = REPO_ROOT / "assets"
PROCESSED_POINTS_FILENAME = ASSETS_DIR / "grape_vf_report_coords_degrees.txt"
MASTER_LOOKUP_FILENAME = ASSETS_DIR / "grape_master_lookup_61.npy"


def plot_grape_predictions(
    x_annotated_images, y_grids, image_names, preds_grids, n_samples=5
):
    preds_grids = preds_grids.cpu().numpy()

    idx = 0

    # (61, 61)
    pred_grid = preds_grids[idx]
    y_grid = y_grids[idx]

    # Ungrid: (61, 61) -> (61,)
    master_lookup = np.load(MASTER_LOOKUP_FILENAME).astype(int)
    _, ungrid_indices = np.unique(master_lookup, return_index=True)
    pred_vf = pred_grid.flatten()[ungrid_indices]
    actual_vf = y_grid.flatten()[ungrid_indices]

    # Generate a 500x500 high-res mask for a professional look
    res = 500
    coords_deg = np.loadtxt(PROCESSED_POINTS_FILENAME, delimiter=",")
    smooth_mask = create_smooth_circular_mask(res, res, radius=245, smoothness=1.5)
    master_lookup_highres = create_highres_lookup(coords_deg)

    plot(actual_vf, pred_vf, smooth_mask, master_lookup_highres, image_names[idx])


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


def plot(actual_vf, pred_vf, smooth_mask, master_lookup_highres, image_name, idx=5):
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_under("white")
    a_min = actual_vf.min()
    a_max = actual_vf.max()

    p_min = pred_vf.min()
    p_max = pred_vf.max()

    img_actual = get_smooth_vf_plot(smooth_mask, actual_vf, master_lookup_highres)
    img_pred = get_smooth_vf_plot(smooth_mask, pred_vf, master_lookup_highres)
    diff_vf = actual_vf - pred_vf
    img_diff = get_smooth_vf_plot(smooth_mask, diff_vf, master_lookup_highres)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # --- Panel 1: Actual ---
    im1 = axes[0].imshow(img_actual, cmap=cmap, vmin=a_min, vmax=a_max, origin="upper")
    axes[0].set_title(f"Actual VF\n{image_name}", fontweight="bold")
    fig.colorbar(im1, ax=axes[0], shrink=0.6)

    # --- Panel 2: Prediction ---
    im2 = axes[1].imshow(img_pred, cmap=cmap, vmin=p_min, vmax=p_max, origin="upper")
    axes[1].set_title("Model Prediction", fontweight="bold")
    fig.colorbar(im2, ax=axes[1], shrink=0.6)

    # --- Panel 3: Difference (Error) ---
    # Use a diverging colormap: Red = Model underestimated, Blue = Model overestimated
    im3 = axes[2].imshow(img_diff, cmap="bwr", vmin=-15, vmax=15, origin="upper")
    axes[2].set_title("Difference (Actual - Pred)", fontweight="bold")
    cbar_diff = fig.colorbar(im3, ax=axes[2], shrink=0.6)
    cbar_diff.set_label("dB Error")

    # Clean up
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def get_smooth_vf_plot(smooth_mask, pred_vf, master_lookup_highres):
    WHITE = -999
    # 1. Map values to high-res grid (e.g., 500x500)
    # This ensures the tiles don't look 'blocky' at the edges
    high_res_img = pred_vf[master_lookup_highres]

    # 2. Apply negative values to the background
    # Logic: If mask is 1, keep img. If mask is 0, set to neg_value.
    final_viz = np.where(smooth_mask > 0.5, high_res_img, WHITE)

    # Add edges to voronoi cells
    edges = np.abs(np.gradient(master_lookup_highres)).sum(axis=0) > 0
    final_viz[edges] = WHITE

    return final_viz
