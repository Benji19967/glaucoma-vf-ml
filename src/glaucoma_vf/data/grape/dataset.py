from typing import TypedDict

import torch
import torchvision.transforms as transforms
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
    annotated_image: torch.Tensor


class LabelSet(TypedDict):
    grid: torch.Tensor
    image_name: str


class DatasetItem(TypedDict):
    X: FeatureSet
    y: LabelSet


# TODO: Shape / channels of images
class GRAPEDataset(Dataset):
    def __init__(
        self,
        x_annotated_images,
        y_grids,
        image_names,
    ):
        """
        Args:
            x_annotated_images (list[PIL Image]):
                Annotated images of optic nerves
                Shape: N x (TARGET_FILE_SIZE, TARGET_FILE_SIZE, 3)
            y_grids (`np.array`):
                Visual Field grids.
                Shape: (N, 61, 61)
            image_names (list[str]):
                list of image names, used as global ID to
                identify each item.
                Shape: N
        """
        self.x_annotated_images = x_annotated_images
        self.y_grids = y_grids
        self.image_names = image_names

    def __len__(self) -> int:
        return len(self.x_annotated_images)

    def __getitem__(self, idx) -> DatasetItem:
        """
        Returns:
            BatchItem: A dictionary containing:
                - FeatureSet:
                    - x_annotated_images: The images of optic nerves
                    Shape: (3, TARGET_FILE_SIZE, TARGET_FILE_SIZE)
                - LabelSet:
                    - y_grid: The next 61x61 VF sensitivity grid
                    Shape: (1, 61, 61) (C, H, W)
                    - image_name: name of the optic nerve image
                    str
        """
        transform = transforms.Compose([transforms.ToTensor()])
        x_annotated_image = transform(self.x_annotated_images[idx])

        y_grid = torch.as_tensor(self.y_grids[idx], dtype=torch.float32).unsqueeze(0)

        return DatasetItem(
            X=FeatureSet(
                annotated_image=x_annotated_image,
            ),
            y=LabelSet(
                grid=y_grid,
                image_name=self.image_names[idx],
            ),
        )
