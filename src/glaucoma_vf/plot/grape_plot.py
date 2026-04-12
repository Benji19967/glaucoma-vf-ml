import json

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import scipy.ndimage as ndimage
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree  # type: ignore

from glaucoma_vf.utils import get_git_root

REPO_ROOT = get_git_root(__file__)
GRAPE_DIR = REPO_ROOT / "data" / "GRAPE"
VF_DATA_FILENAME = GRAPE_DIR / "VFs_and_clinical_info.xlsx"
ASSETS_DIR = REPO_ROOT / "assets"
PROCESSED_POINTS_FILENAME = ASSETS_DIR / "grape_vf_report_coords_degrees.txt"

def plot_hvf_predictions(grids, true_labels, pred_labels, n_samples=5):
    pass

coords_deg = np.loadtxt(PROCESSED_POINTS_FILENAME, delimiter=",")

res = 61
grid_nodes = np.linspace(-30, 30, res)
gx, gy = np.meshgrid(grid_nodes, grid_nodes)
pixel_coords = np.stack([gx.ravel(), gy.ravel()], axis=1)
tree = cKDTree(coords_deg)
_, lookup_indices = tree.query(pixel_coords)
master_lookup = lookup_indices.reshape(res, res)


df = pl.read_excel(
    VF_DATA_FILENAME,
    sheet_name="Baseline",
    read_options={"skip_rows_after_header": 1},
    drop_empty_rows=True,
    engine="xlsx2csv",
)

cols_to_load = ["VF"] + [str(i) for i in range(5, 65)]
data_61 = df.select(cols_to_load)

patient_img = data_61[0].to_numpy()[0][master_lookup]
   
def create_smooth_circular_mask(h, w, center=None, radius=None, smoothness=0.5):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance from center to image boundary
        radius = min(center[0], center[1], w-center[0], h-center[1])

    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)

    # Use a sigmoid or a simple linear ramp for the edge
    # This creates a very thin 1-pixel 'blur' that looks smooth to the eye
    mask = 1.0 - np.clip((dist_from_center - (radius - smoothness)) / (2 * smoothness), 0, 1)
    return mask

# Generate a 500x500 high-res mask for a professional look
res = 500
smooth_mask = create_smooth_circular_mask(res, res, radius=245, smoothness=1.5)


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
# Usage:
master_lookup_highres = create_highres_lookup(coords_deg)

# 1. SETUP: Grid and Coordinates
grid_size = 61
vf_range = 30 # Degrees (radius)
# Generate grid coordinates (-30 to 30)
lin = np.linspace(-vf_range, vf_range, grid_size)
grid_x, grid_y = np.meshgrid(lin, lin)
grid_points = np.stack((grid_x.ravel(), grid_y.ravel()), axis=1)
# 2. CREATE THE VORONOI LOOKUP (As detailed in previous turns)
# (Assumes raw_points.txt exists with your extracted pixels)
# ... Load raw_points, normalize to coords_deg, and create the KDTree ...

tree = cKDTree(coords_deg)
_, lookup_indices = tree.query(grid_points)
master_lookup = lookup_indices.reshape(grid_size, grid_size)
# 3. CREATE THE PHYSIOLOGICAL MASK (The Circle)
# Calculate the distance of every grid pixel from the center (0,0)
grid_dist = np.sqrt(grid_x**2 + grid_y**2)
# Create a mask where True = inside the eye (radius <= 30), False = outside
# Note: Use '< 30' or slightly smaller for a clean edge
mask_active_area = grid_dist <= vf_range
# Create an inverse mask for the 'void' areas
mask_void = ~mask_active_area
# 4. PROCESS PATIENT DATA (For Visualization/GenViT)
# Generate a test patient row (61 random dB values)
patient_row_from_excel = np.random.uniform(15, 30, 61)
# Add a severe scotoma to Point 21 (near the blind spot)
patient_row_from_excel[20] = 5
# Create the Voronoi Image
patient_image = patient_img
# Apply the Physological Mask: set outside areas to a neutral value
# We will set 'outside' to -1, which our colormap will handle.
masked_patient_image = np.where(mask_active_area, patient_image, -1)

    
 def get_smooth_vf_plot(patient_values, master_lookup_highres):
     # 1. Map values to high-res grid (e.g., 500x500)
     # This ensures the tiles don't look 'blocky' at the edges
     high_res_img = patient_values[master_lookup_highres]

     # 2. Apply the smooth mask
     # We set the background to White (1.0) and multiply the VF data
     # (Assuming data is normalized 0-1 for this visualization)
     final_viz = high_res_img * smooth_mask + (1 - smooth_mask) # Blends to white

     # Add edges to voronoi cells
     edges = np.abs(np.gradient(master_lookup_highres)).sum(axis=0) > 0
     final_viz[edges] = 0

     return final_viz

 # --- How to display it ---
 # Use 'interpolation=lanczos' or 'bicubic' for the smoothest visual output
 cmap = plt.get_cmap('RdYlGn').copy()
 cmap.set_under('white')
 patient_values = patient_img[5]
 p_min = patient_values.min()
 p_max = patient_values.max()
 plt.imshow(get_smooth_vf_plot(patient_values, master_lookup_highres), cmap='nipy_spectral', vmin=p_min, vmax=p_max, origin='upper', interpolation='lanczos')
 plt.axis('off') # Hide the pixel coordinates for a cleaner look
 plt.show()

