import lightning as L
import polars as pl
import polars.selectors as cs
import torch
from torch.utils.data import DataLoader, Dataset

from glaucoma_vf.data_utils import map_mtd_to_enum, polars_to_hvf_grids


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
        return self.grids[idx], self.labels[idx].squeeze()


class GlaucomaDataModule(L.LightningDataModule):
    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path

    def setup(self, stage=None):
        df = pl.read_csv(self.csv_path, schema_overrides={"Sens_35": pl.Float32})

        # Load grids from CSV
        grids = polars_to_hvf_grids(df)

        # Load labels from CSV
        mtd = df.select(cs.by_name("MTD")).to_numpy().squeeze()
        labels = map_mtd_to_enum(mtd)

        self.train_ds = UWHVFDataset(grids, labels)
        self.val_ds = UWHVFDataset(grids, labels)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=32)
