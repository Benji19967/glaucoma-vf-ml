import datetime

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI

from glaucoma_vf.dataset import UWHVFDataModule
from glaucoma_vf.models.hvf_cnn_classifier import HVFClassifier


def cli_main():
    # Generate timestamp: YYYYMMDD_HHMMSS
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = "logs"
    name = "hvf_classifier"
    version = f"v_{timestamp}"

    cli = LightningCLI(
        model_class=HVFClassifier,
        datamodule_class=UWHVFDataModule,
        save_config_kwargs={"overwrite": True},
        trainer_defaults={
            "callbacks": [
                ModelCheckpoint(
                    monitor="val_balanced_acc",
                    mode="max",
                    save_top_k=1,
                    save_last=True,
                    filename="best",
                )
            ],
            "logger": {
                "class_path": "lightning.pytorch.loggers.TensorBoardLogger",
                "init_args": {
                    "save_dir": log_path,
                    "name": name,
                    "version": version,
                },
            },
        },
    )


if __name__ == "__main__":
    cli_main()
