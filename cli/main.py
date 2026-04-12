import datetime

import lightning as L
from lightning.pytorch.cli import LightningCLI


def cli_main():
    # Generate timestamp: YYYYMMDD_HHMMSS
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = "logs"
    name = "hvf_system"
    version = f"v_{timestamp}"

    cli = LightningCLI(
        model_class=L.LightningModule,
        datamodule_class=L.LightningDataModule,
        subclass_mode_model=True,  # Now we allow swapping the whole system
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        trainer_defaults={
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
