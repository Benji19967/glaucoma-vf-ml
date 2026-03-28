import lightning as L
import polars as pl
import polars.selectors as cs
import torch
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
        return self.grids[idx], self.labels[idx].squeeze()


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

        # Split into train, val, and test sets
        train_set, val_set, test_set = torch.utils.data.random_split(
            UWHVFDataset(grids, labels), [0.8, 0.1, 0.1]
        )

        if stage == "fit":
            self.train_ds = train_set
            self.val_ds = val_set

        if stage == "test":
            self.test_ds = test_set

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)
