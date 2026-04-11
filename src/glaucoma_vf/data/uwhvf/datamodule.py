import lightning as L
import polars as pl
import polars.selectors as cs
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

from glaucoma_vf.data.data_utils import df_to_hvf_grids_uwhvf, map_mtd_to_enum
from glaucoma_vf.data.uwhvf.dataset import UWHVFDataset
from glaucoma_vf.utils import get_git_root

REPO_ROOT = get_git_root(__file__)
UWHVF_DIR = REPO_ROOT / "data" / "UWHVF"
VF_DATA_FILENAME = UWHVF_DIR / "CSV" / "VF_Data.csv"


class UWHVFDataModule(L.LightningDataModule):
    """
    Prepares the train/val/test Dataloaders
    """

    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.csv_path = VF_DATA_FILENAME
        self.batch_size = batch_size

    def setup(self, stage: str):
        """
        Create the train/val/test datasets from the CSV file.

        We do a patient-level split to avoid data leakage.
        """
        df = self._load_df()
        df = self._add_shifted_grids(df)

        x_grids, y_grids = self._get_normalized_grids(df)
        x_age = self._get_normalized_age(df)
        x_years_since_first = self._get_normalized_years_since_first(df)
        x_years_since_last = self._get_normalized_years_since_last(df)
        y_mtd = self._get_normalized_mtd(df)

        # Create class labels from mtd
        y_class = map_mtd_to_enum(y_mtd)

        full_dataset = UWHVFDataset(
            x_grids,
            x_age,
            x_years_since_first,
            x_years_since_last,
            y_class,
            y_mtd,
            y_grids,
        )

        train_set, val_set, test_set = self._split_dataset_by_patient(
            df, x_grids, y_class, full_dataset
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

    def _split_dataset_by_patient(self, df, x_grids, y_class, full_dataset):
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
            f"Patient-level split complete: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}"
        )

        return train_set, val_set, test_set

    def _get_normalized_mtd(self, df):
        y_mtd = (df.select(cs.by_name("MTD")).to_numpy().squeeze() + 35) / 35
        return y_mtd

    def _get_normalized_years_since_last(self, df):
        years_since_first = df.select(cs.by_name("Time_from_Baseline"))
        x_years_since_last = (
            years_since_first.with_columns(
                Time_Delta=pl.col("Time_from_Baseline").diff().clip(lower_bound=0)
            )
            .select(pl.col("Time_Delta"))
            .fill_null(0)
            .to_numpy()
            .squeeze()
            / 10
        )

        return x_years_since_last

    def _get_normalized_years_since_first(self, df):
        x_years_since_first = (
            df.select(cs.by_name("Time_from_Baseline")).to_numpy().squeeze() / 10
        )

        return x_years_since_first

    def _get_normalized_age(self, df):
        x_age = df.select(cs.by_name("Age")).to_numpy().squeeze() / 100
        return x_age

    def _get_normalized_grids(self, df):
        x_grids = df_to_hvf_grids_uwhvf(df, columns_prefix="SensPrevious_") / 40
        y_grids = df_to_hvf_grids_uwhvf(df, columns_prefix="Sens_") / 40
        return x_grids, y_grids

    def _add_shifted_grids(self, df):
        """
        Shift for forecasting -- if a patient has 10 measurements, we keep 9 (intervals)
        SensPrevious_ will be the x grid (feature)
        Sens_ will be the y grid (label)
        """
        for i in range(1, 55):
            df = df.with_columns(pl.col(f"Sens_{i}").alias(f"SensPrevious_{i}").shift())
        df = df.remove(pl.col("Time_from_Baseline") == 0.0)
        return df

    def _load_df(self):
        return pl.read_csv(self.csv_path, schema_overrides={"Sens_35": pl.Float32})
