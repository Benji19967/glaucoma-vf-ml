import numpy as np

from glaucoma_vf.utils import get_git_root

REPO_ROOT = get_git_root(__file__)
ASSETS_DIR = REPO_ROOT / "assets"
RAW_POINTS_FILENAME = ASSETS_DIR / "grape_vf_report_coords.txt"
PROCESSED_POINTS_FILENAME = ASSETS_DIR / "grape_vf_report_coords_degrees.txt"


def load_raw_points():
    return np.loadtxt(RAW_POINTS_FILENAME, delimiter=",")


def finalize_mapping(pts):
    center = pts[0]  # Using first point as fixation (0,0)

    # 1. Center the coordinates
    centered = pts - center

    # 2. Calculate scale (assuming 30 degrees max radius)
    max_px_dist = np.max(np.linalg.norm(centered, axis=1))
    scale = 30.0 / max_px_dist

    # 3. Scale and Flip Y for Cartesian coordinates
    final_coords = centered * scale
    final_coords[:, 1] *= -1  # Flip Y so 'up' is positive

    return final_coords


def main():
    raw_points = load_raw_points()

    # Process and Save
    coords_deg = finalize_mapping(raw_points)
    # mapping_dict = {i: list(coord) for i, coord in enumerate(coords_deg)}

    with open(PROCESSED_POINTS_FILENAME, "w") as f:
        np.savetxt(PROCESSED_POINTS_FILENAME, coords_deg, delimiter=",", fmt="%.2f")
        # json.dump(mapping_dict, f, indent=4)

    print("Mapping saved! Data now ranges from -30 to +30 degrees.")


if __name__ == "__main__":
    main()
