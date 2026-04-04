import lightning as L

from glaucoma_vf.dataset import UWHVFDataModule
from glaucoma_vf.models.hvf_cnn import HVFSystem
from glaucoma_vf.models.hvf_cnn_backbone import HVFCNNBackbone


def train():
    backbone = HVFCNNBackbone()
    model = HVFSystem(backbone=backbone)
    datamodule = UWHVFDataModule()

    trainer = L.Trainer(max_epochs=10)
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    train()
