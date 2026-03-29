import lightning as L

from glaucoma_vf.dataset import UWHVFDataModule
from glaucoma_vf.models.hvf_cnn_classifier import HVFClassifier


def train():
    model = HVFClassifier()
    datamodule = UWHVFDataModule()

    trainer = L.Trainer(max_epochs=10)
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    train()
