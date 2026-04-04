import torch
from torch.utils.data import Dataset

from glaucoma_vf.utils import get_git_root

REPO_ROOT = get_git_root(__file__)
UWHVF_DIR = REPO_ROOT / "data" / "UWHVF"
VF_DATA_FILENAME = UWHVF_DIR / "CSV" / "VF_Data.csv"


class UWHVFDataset(Dataset):
    def __init__(
        self,
        x_grids,
        x_age,
        x_years_since_first,
        x_years_since_last,
        y_class,
        y_mtd,
        y_grids,
    ):
        """
        Args:
            x_grids (`np.array`):
                Humphrey Visual Field grids.
                Shape: (N, 8, 9) where 1 is the single-channel
                sensitivity map.
            x_age (`np.array`):
                A list or numpy array of the age of the patient at measurement
                Shape: (N,)
            x_years_since_first (`np.array`):
                A list or numpy array of the years since first measurement
                Shape: (N,)
            x_years_since_last (`np.array`):
                A list or numpy array of the years since the last measurement
                Shape: (N,)
            y_class (`np.array`):
                A list or numpy array of the Enum/Integer labels
                Shape: (N,)
            y_mtd (`np.array`):
                A list or numpy array of the mean total deviation labels
                Shape: (N,)
            y_grids (`np.array`):
                Humphrey Visual Field grids.
                Shape: (N, 8, 9) where 1 is the single-channel
                sensitivity map.
        """
        self.x_grids = x_grids
        self.x_age = x_age
        self.x_years_since_first = x_years_since_first
        self.x_years_since_last = x_years_since_last
        self.y_class = y_class
        self.y_mtd = y_mtd
        self.y_grids = y_grids

    def __len__(self) -> int:
        return len(self.x_grids)

    def __getitem__(self, idx) -> tuple[torch.Tensor, ...]:
        """
        Returns:
            tuple[torch.Tensor, ...]: A tuple containing:
                - x_grid: The current 8x9 HVF sensitivity grid
                  Shape: (1, 8, 9) (C, H, W)
                - x_age: The age of the patient at measurement
                  Shape: () (Scalar LongTensor)
                - x_years_since_first: Years since first measurement
                  Shape: () (Scalar LongTensor)
                - x_years_since_last: Years since last measurement
                  Shape: () (Scalar LongTensor)
                - y_class: The classification category
                  Shape: () (Scalar LongTensor)
                - y_mtd: The mean total deviation
                  Shape: () (Scalar LongTensor)
                - y_grid: The next 8x9 HVF sensitivity grid
                  Shape: (1, 8, 9) (C, H, W)
        """
        x_grid = torch.as_tensor(self.x_grids[idx], dtype=torch.float32).unsqueeze(0)
        x_age = torch.as_tensor(self.x_age[idx].squeeze(), dtype=torch.float32)
        x_years_since_first = torch.as_tensor(
            self.x_years_since_first[idx].squeeze(), dtype=torch.float32
        )
        x_years_since_last = torch.as_tensor(
            self.x_years_since_last[idx].squeeze(), dtype=torch.float32
        )
        x_age = torch.as_tensor(self.x_age[idx].squeeze(), dtype=torch.float32)
        y_grid = torch.as_tensor(self.y_grids[idx], dtype=torch.float32).unsqueeze(0)
        y_class = torch.as_tensor(self.y_class[idx].squeeze(), dtype=torch.int16)
        y_mtd = torch.as_tensor(self.y_mtd[idx].squeeze(), dtype=torch.float32)

        return (
            x_grid,
            x_age,
            x_years_since_first,
            x_years_since_last,
            y_class,
            y_mtd,
            y_grid,
        )
