from pathlib import Path

import lightning as L
import numpy as np
import polars as pl
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from glaucoma_vf.data.data_utils import df_to_vf_grids_grape
from glaucoma_vf.data.grape.dataset import GRAPEDataset
from glaucoma_vf.utils import get_git_root

REPO_ROOT = get_git_root(__file__)
GRAPE_DIR = REPO_ROOT / "data" / "GRAPE"
# ANNOTATED_IMAGES_DIR = GRAPE_DIR / "Annotated Images"
CFP_IMAGES_DIR = GRAPE_DIR / "CFPs"
VF_DATA_FILENAME = GRAPE_DIR / "VFs_and_clinical_info.xlsx"

VF_DATA_FOLLOWUP_SHEET = "Follow-up"

TARGET_FILE_SIZE = (1556, 1556)


# TODO: Do we need colored images, or are grayscale ones enough?
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
        Create the train/val/test datasets from the CSV file and
        annotated images.
        """
        df_follow_up = self._load_df(sheet_name=VF_DATA_FOLLOWUP_SHEET)

        # 631 image names
        image_names_sorted = self._get_image_names(df_follow_up)

        # 631 images
        x_images = self._load_images(
            dirname=CFP_IMAGES_DIR,
            image_names=image_names_sorted,
            target_size=TARGET_FILE_SIZE,
        )

        # 631 rows
        df_follow_up = df_follow_up.filter(pl.col("Corresponding CFP") != "/")

        # (631, 61, 61)
        y_grids = self._get_normalized_grids(df_follow_up)

        train_set, val_set, test_set = self._split_dataset(
            x_images, y_grids, image_names_sorted
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

    def _get_image_names(self, df_follow_up):
        image_names_sorted = list(
            df_follow_up.filter(pl.col("Corresponding CFP") != "/")
            .select(pl.col("Corresponding CFP"))
            .to_series()
        )
        return image_names_sorted

    def _load_images(
        self, dirname: Path, image_names: list[str], target_size: tuple[int, int]
    ) -> list[Image.Image]:
        images = []
        for name in image_names:
            path = dirname / name
            img = Image.open(path).convert("RGB").resize(target_size)
            images.append(img)

        return images

    def _split_dataset(self, x_images, y_grids, image_names):
        assert len(x_images) == len(y_grids)

        # Only load the master data once
        full_dataset = GRAPEDataset(x_images, y_grids, image_names)
        n_total = len(full_dataset)

        # Split logic
        indices = np.arange(n_total)
        np.random.shuffle(indices)

        # 3. Calculate split points
        train_end = int(0.8 * n_total)
        val_end = int(0.9 * n_total)  # 0.8 + 0.1

        # 4. Slice indices
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        # 5. Create Subsets
        train_set = Subset(full_dataset, train_idx.tolist())
        val_set = Subset(full_dataset, val_idx.tolist())
        test_set = Subset(full_dataset, test_idx.tolist())

        return train_set, val_set, test_set

    def _get_normalized_grids(self, df):
        x_grids = df_to_vf_grids_grape(df) / 40
        return x_grids

    def _load_df(self, sheet_name: str):
        return pl.read_excel(
            self.xlsx_path,
            sheet_name=sheet_name,
            read_options={"skip_rows_after_header": 1},
            drop_empty_rows=True,
            engine="xlsx2csv",
        )
