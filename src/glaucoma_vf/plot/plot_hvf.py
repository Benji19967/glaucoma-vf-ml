import time

import matplotlib.pyplot as plt
import numpy as np


def plot_hvf_predictions(grids, true_labels, pred_labels, n_samples=5):
    """
    Visualizes a batch of Humphrey Visual Field (HVF) 24-2 grids with
    their corresponding True and Predicted classifications.

    Args:
        grids (np.ndarray): A stack of HVF grids of shape (N, 8, 9).
            Values should be sensitivity in dB (typically 0-35).
            Non-testable points can contain any value as they are masked.
        true_labels (np.ndarray or list): Ground truth class indices
            of shape (N,). Expected values: 0 (Mild), 1 (Moderate), 2 (Severe).
        pred_labels (np.ndarray or list): Predicted class indices
            of shape (N,). Expected values: 0 (Mild), 1 (Moderate), 2 (Severe).
        n_samples (int): Number of random samples from the batch to
            visualize. Defaults to 5.
    """
    # Mapping for your Enum/Integer labels back to strings
    class_map = {0: "Mild", 1: "Moderate", 2: "Severe"}

    # 24-2 HVF Mask: 1 for testable points, 0 for "holes"
    hvf_mask = np.zeros((8, 9))

    # fmt: off
    hvf_coords = [
        (0,3),(0,4),(0,5),(0,6), (1,2),(1,3),(1,4),(1,5),(1,6),(1,7),
        (2,1),(2,2),(2,3),(2,4),(2,5),(2,6),(2,7),(2,8), (3,0),(3,1),
        (3,2),(3,3),(3,4),(3,5),(3,6),(3,7),(3,8), (4,0),(4,1),(4,2),
        (4,3),(4,4),(4,5),(4,6),(4,7),(4,8), (5,1),(5,2),(5,3),(5,4),
        (5,5),(5,6),(5,7),(5,8), (6,2),(6,3),(6,4),(6,5),(6,6),(6,7),
        (7,3),(7,4),(7,5),(7,6)
    ]
    # fmt: on

    for r, c in hvf_coords:
        hvf_mask[r, c] = 1

    np.random.seed(int(time.time()))
    batch_size = len(grids)
    # We use min() to ensure we don't try to pick 5 samples from a batch of 2
    actual_n = min(n_samples, batch_size)
    indices = np.random.choice(batch_size, actual_n, replace=False)

    fig, axes = plt.subplots(1, actual_n, figsize=(18, 4))

    for i, idx in enumerate(indices):
        grid = grids[idx].squeeze()
        true_txt = class_map[true_labels[idx]]
        pred_txt = class_map[pred_labels[idx]]

        # Mask the non-testable points for a clean look
        masked_grid = np.where(hvf_mask == 1, grid, np.nan)

        im = axes[i].imshow(masked_grid, cmap="viridis", vmin=0, vmax=35)
        axes[i].set_title(
            f"True: {true_txt}\nPred: {pred_txt}",
            color="green" if true_txt == pred_txt else "red",
        )

        # Add the decibel numbers on top of the grid
        for r in range(8):
            for c in range(9):
                if hvf_mask[r, c]:
                    val = grid[r, c]
                    axes[i].text(
                        c,
                        r,
                        f"{int(val)}",
                        ha="center",
                        va="center",
                        color="white" if val < 15 else "black",
                        fontsize=8,
                    )

        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def print_hvf_ascii(grid, true_label, pred_label, sample_idx):
    class_map = {0: "Mild", 1: "Mod ", 2: "Seve"}  # Uniform length for alignment

    # 24-2 Mask (Row, Col)
    # fmt: off
    hvf_coords = set([
        (0,3),(0,4),(0,5),(0,6), (1,2),(1,3),(1,4),(1,5),(1,6),(1,7),
        (2,1),(2,2),(2,3),(2,4),(2,5),(2,6),(2,7),(2,8), (3,0),(3,1),
        (3,2),(3,3),(3,4),(3,5),(3,6),(3,7),(3,8), (4,0),(4,1),(4,2),
        (4,3),(4,4),(4,5),(4,6),(4,7),(4,8), (5,1),(5,2),(5,3),(5,4),
        (5,5),(5,6),(5,7),(5,8), (6,2),(6,3),(6,4),(6,5),(6,6),(6,7),
        (7,3),(7,4),(7,5),(7,6)
    ])
    # fmt: on

    print(
        f"\n--- Sample #{sample_idx} | True: {class_map[true_label]} | Pred: {class_map[pred_label]} ---"
    )

    for r in range(8):
        row_str = ""
        for c in range(9):
            if (r, c) in hvf_coords:
                val = int(grid[r, c])
                # Format to 2 spaces for alignment
                row_str += f"{val:2} "
            else:
                row_str += "   "  # Empty space for non-testable points
        print(row_str)
