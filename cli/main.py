import datetime

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI

from glaucoma_vf.data.uwhvf.datamodule import UWHVFDataModule
from glaucoma_vf.models.hvf_cnn import HVFSystem

default_backbone = {
    "class_path": "glaucoma_vf.models.hvf_cnn_backbone.HVFCNNBackbone",
}


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.set_defaults({"model.backbone": default_backbone})


def cli_main():
    # Generate timestamp: YYYYMMDD_HHMMSS
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = "logs"
    name = "hvf_system"
    version = f"v_{timestamp}"

    cli = MyLightningCLI(
        model_class=HVFSystem,
        datamodule_class=UWHVFDataModule,
        save_config_kwargs={"overwrite": True},
        trainer_defaults={
            "callbacks": [
                ModelCheckpoint(
                    monitor="val/curr_f1",
                    mode="max",
                    save_top_k=1,
                    save_last=True,
                    filename="best",
                )
            ],#removed logger for test
        },
    )


if __name__ == "__main__":
    cli_main()
