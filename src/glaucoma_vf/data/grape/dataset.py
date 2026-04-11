from typing import TypedDict

import torch
from torch.utils.data import Dataset

from glaucoma_vf.utils import get_git_root

REPO_ROOT = get_git_root(__file__)
GRAPE_DIR = REPO_ROOT / "data" / "GRAPE"
ANNOTATED_IMAGES_DIR = GRAPE_DIR / "Annotated Images"
COORDINATES_DIR = GRAPE_DIR / "json"

# Not sure yet if we'll need these, but keeping here for now
# CFPS_DIR = GRAPE_DIR / "CFPs"
# ROI_IMAGES_DIR = GRAPE_DIR / "ROI images"


class FeatureSet(TypedDict):
    grids: torch.Tensor


class LabelSet(TypedDict):
    grids: torch.Tensor


class DatasetItem(TypedDict):
    X: FeatureSet
    y: LabelSet


class GRAPEDataset(Dataset):
    def __init__(
        self,
        x_grids,
        y_grids,
    ):
        """
        Args:
            x_grids (`np.array`):
                Humphrey Visual Field grids.
                Shape: (N, 8, 9) where 1 is the single-channel
                sensitivity map.
        """
        self.x_grids = x_grids
        self.y_grids = y_grids

    def __len__(self) -> int:
        return len(self.x_grids)

    def __getitem__(self, idx) -> DatasetItem:
        """
        Returns:
            BatchItem: A dictionary containing:
                - FeatureSet:
                    - x_grid: The current 8x9 HVF sensitivity grid
                    Shape: (1, 8, 9) (C, H, W)
                - LabelSet:
                    - y_grid: The next 8x9 HVF sensitivity grid
                    Shape: (1, 8, 9) (C, H, W)
        """
        x_grid = torch.as_tensor(self.x_grids[idx], dtype=torch.float32).unsqueeze(0)
        y_grid = torch.as_tensor(self.y_grids[idx], dtype=torch.float32).unsqueeze(0)

        return DatasetItem(
            X=FeatureSet(
                grids=x_grid,
            ),
            y=LabelSet(
                grids=y_grid,
            ),
        )
