import lightning as L
import numpy as np
import polars as pl
import polars.selectors as cs
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset

from glaucoma_vf.data_utils import df_to_hvf_grids, map_mtd_to_enum
from glaucoma_vf.utils import get_git_root

REPO_ROOT = get_git_root(__file__)
UWHVF_DIR = REPO_ROOT / "data" / "UWHVF"
VF_DATA_FILENAME = UWHVF_DIR / "CSV" / "VF_Data.csv"


class UWHVFDataset(Dataset):
    def __init__(self, grids, labels):
        """
        Args:
            grids (`np.array`):
                Humphrey Visual Field grids.
                Shape: (N, 8, 9) where 1 is the single-channel
                sensitivity map.
            labels (`np.array`):
                A list or numpy array of the Enum/Integer labels
                Shape: (N,)
        """
        self.grids = grids
        self.labels = labels

    def __len__(self) -> int:
        return len(self.grids)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - feature: The 8x9 HVF sensitivity grid
                  Shape: (1, 8, 9) (C, H, W)
                - label: The classification category
                  Shape: () (Scalar LongTensor)
        """
        grid = torch.as_tensor(self.grids[idx], dtype=torch.float32).unsqueeze(0)
        return grid, self.labels[idx].squeeze()


class UWHVFDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.csv_path = VF_DATA_FILENAME
        self.batch_size = batch_size

    def setup(self, stage: str):
        df = pl.read_csv(self.csv_path, schema_overrides={"Sens_35": pl.Float32})

        # Load grids from CSV
        grids = df_to_hvf_grids(df)

        # Load labels from CSV
        mtd = df.select(cs.by_name("MTD")).to_numpy().squeeze()
        labels = map_mtd_to_enum(mtd)

        # Create the full dataset object
        full_dataset = UWHVFDataset(grids, labels)

        # --- Patient-Level Split Logic ---
        # We need the PatientID column to define our groups
        patient_ids = df.select(pl.col("PatID")).to_numpy().squeeze()

        # Split 1: Separate Test (10%) from the rest (90%)
        gss_test = GroupShuffleSplit(n_splits=1, train_size=0.9, random_state=42)
        train_val_idx, test_idx = next(
            gss_test.split(grids, labels, groups=patient_ids)
        )

        # Split 2: Separate Train (80% total) and Val (10% total)
        # Since 0.1 is 1/9th of 0.9, we use train_size=0.888 (approx 8/9)
        gss_val = GroupShuffleSplit(n_splits=1, train_size=0.888, random_state=42)

        # Filter the IDs to only include the non-test patients for the second split
        train_idx_sub, val_idx_sub = next(
            gss_val.split(
                grids[train_val_idx],
                labels[train_val_idx],
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

        # # Split into train, val, and test sets
        # train_set, val_set, test_set = torch.utils.data.random_split(
        #     UWHVFDataset(grids, labels), [0.8, 0.1, 0.1]
        # )

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
