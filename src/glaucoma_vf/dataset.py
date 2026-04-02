import lightning as L
import numpy as np
import polars as pl
import polars.selectors as cs
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch._C import dtype
from torch.utils.data import DataLoader, Dataset

from glaucoma_vf.data_utils import df_to_hvf_grids, map_mtd_to_enum
from glaucoma_vf.utils import get_git_root

REPO_ROOT = get_git_root(__file__)
UWHVF_DIR = REPO_ROOT / "data" / "UWHVF"
VF_DATA_FILENAME = UWHVF_DIR / "CSV" / "VF_Data.csv"


class UWHVFDataset(Dataset):
    def __init__(
        self,
        x_grids,
        x_age,
        x_years_from_baseline,
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
            x_years_from_baseline (`np.array`):
                A list or numpy array of the years since first measurement
                Shape: (N,)
            x_years_since_last_measurement (`np.array`):
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
        self.x_years_from_baseline = x_years_from_baseline
        self.x_years_since_last = x_years_since_last
        self.y_class = y_class
        self.y_mtd = y_mtd
        self.y_grids = y_grids

    def __len__(self) -> int:
        return len(self.x_grids)

    def __getitem__(
        self, idx
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - feature: The 8x9 HVF sensitivity grid
                  Shape: (1, 8, 9) (C, H, W)
                - y_class: The classification category
                  Shape: () (Scalar LongTensor)
                - y_mtd: The mean total deviation
                  Shape: () (Scalar LongTensor)
        """
        x_grid = torch.as_tensor(self.x_grids[idx], dtype=torch.float32).unsqueeze(0)
        y_grid = torch.as_tensor(self.y_grids[idx], dtype=torch.float32).unsqueeze(0)
        y_class = torch.as_tensor(self.y_class[idx].squeeze(), dtype=torch.int16)
        y_mtd = torch.as_tensor(self.y_mtd[idx].squeeze(), dtype=torch.float32)

        return (
            x_grid,
            y_class,
            y_mtd,
            y_grid,
        )


class UWHVFDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.csv_path = VF_DATA_FILENAME
        self.batch_size = batch_size

    def setup(self, stage: str):
        df = pl.read_csv(self.csv_path, schema_overrides={"Sens_35": pl.Float32})

        # Shift for forecasting -- if a patient has 10 measurements, we keep 9 (intervals)
        # SensPrevious_ will be the x grid (feature)
        # Sens_ will be the y grid (label)
        for i in range(1, 55):
            df = df.with_columns(pl.col(f"Sens_{i}").alias(f"SensPrevious_{i}").shift())
        df = df.remove(pl.col("Time_from_Baseline") == 0.0)

        # Load grids from CSV and normalize
        x_grids = df_to_hvf_grids(df, columns_prefix="SensPrevious_") / 40
        y_grids = df_to_hvf_grids(df, columns_prefix="Sens_") / 40

        # Load age from CSV and normalize
        x_age = df.select(cs.by_name("Age")).to_numpy().squeeze() / 100

        # Load years from baseline from CSV and normalize
        x_years_from_baseline = (
            df.select(cs.by_name("Time_from_Baseline")).to_numpy().squeeze() / 10
        )

        # Compute years since last visit and normalize
        years_from_baseline = df.select(cs.by_name("Time_from_Baseline"))
        x_years_since_last_measurement = (
            years_from_baseline.with_columns(
                Time_Delta=pl.col("Time_from_Baseline").diff().clip(lower_bound=0)
            )
            .fill_null(0)
            .to_numpy()
            .squeeze()
            / 10
        )

        # Load mtd from CSV and normalize
        y_mtd = (df.select(cs.by_name("MTD")).to_numpy().squeeze() + 35) / 35

        # Create class labels from mtd
        y_class = map_mtd_to_enum(y_mtd)

        # Create the full dataset object
        full_dataset = UWHVFDataset(
            x_grids,
            x_age,
            x_years_from_baseline,
            x_years_since_last_measurement,
            y_class,
            y_mtd,
            y_grids,
        )

        # --- Patient-Level Split Logic ---
        # We need the PatientID column to define our groups
        patient_ids = df.select(pl.col("PatID")).to_numpy().squeeze()

        # Split 1: Separate Test (10%) from the rest (90%)
        gss_test = GroupShuffleSplit(n_splits=1, train_size=0.9, random_state=42)
        train_val_idx, test_idx = next(
            gss_test.split(x_grids, y_class, groups=patient_ids)
        )

        # Split 2: Separate Train (80% total) and Val (10% total)
        # Since 0.1 is 1/9th of 0.9, we use train_size=0.888 (approx 8/9)
        gss_val = GroupShuffleSplit(n_splits=1, train_size=0.888, random_state=42)

        # Filter the IDs to only include the non-test patients for the second split
        train_idx_sub, val_idx_sub = next(
            gss_val.split(
                x_grids[train_val_idx],
                y_class[train_val_idx],
                groups=patient_ids[train_val_idx],
            )
        )

        # Map back to original indices
        train_idx = train_val_idx[train_idx_sub]
        val_idx = train_val_idx[val_idx_sub]

        # 4. Create Subsets using the indices
        train_set = torch.utils.data.Subset(full_dataset, train_idx.tolist())
        val_set = torch.utils.data.Subset(full_dataset, val_idx.tolist())
        test_set = torch.utils.data.Subset(full_dataset, test_idx.tolist())

        print(
            f"Split complete: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}"
        )

        if stage == "fit":
            self.train_ds = train_set
            self.val_ds = val_set

        if stage == "validate":
            self.val_ds = val_set

        if stage == "test":
            self.test_ds = test_set

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)
