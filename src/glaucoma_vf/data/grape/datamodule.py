import lightning as L
import polars as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from glaucoma_vf.data.data_utils import df_to_vf_grids_grape
from glaucoma_vf.data.grape.dataset import GRAPEDataset
from glaucoma_vf.utils import get_git_root

REPO_ROOT = get_git_root(__file__)
GRAPE_DIR = REPO_ROOT / "data" / "GRAPE"
VF_DATA_FILENAME = GRAPE_DIR / "VFs_and_clinical_info.xlsx"

VF_DATA_BASELINE_SHEET = "Baseline"


class GRAPEDataModule(L.LightningDataModule):
    """
    Prepares the train/val/test Dataloaders
    """

    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.xlsx_path = VF_DATA_FILENAME
        self.batch_size = batch_size

    def setup(self, stage: str):
        """
        Create the train/val/test datasets from the CSV file.
        """
        df_baseline = self._load_df(VF_DATA_BASELINE_SHEET)

        # (N, 61, 61)
        x_grids = self._get_normalized_grids(df_baseline)

        # Dummy for now
        y_grids = x_grids

        train_set, val_set, test_set = self._split_dataset(x_grids, y_grids)

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

    def _split_dataset(self, x_grids, y_grids):
        # 1. First split: Separate the Test set (10%)
        x_train_val, x_test, y_train_val, y_test = train_test_split(
            x_grids, y_grids, test_size=0.10, random_state=42, shuffle=True
        )

        # 2. Second split: Separate Train and Val from the remaining 90%
        # To get 10% of the original total for Val, we take ~11% of the 90%
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_val,
            y_train_val,
            test_size=0.111,  # 0.1 / 0.9 ≈ 0.111
            random_state=42,
            shuffle=True,
        )

        train_set = GRAPEDataset(x_train, y_train)
        val_set = GRAPEDataset(x_val, y_val)
        test_set = GRAPEDataset(x_test, y_test)

        return train_set, val_set, test_set

    def _get_normalized_grids(self, df):
        x_grids = df_to_vf_grids_grape(df) / 40
        return x_grids

    def _load_df(self, sheet_name: str):
        pl.read_excel(
            self.xlsx_path,
            sheet_name=sheet_name,
            read_options={"skip_rows_after_header": 1},
            drop_empty_rows=True,
            engine="xlsx2csv",
        )
