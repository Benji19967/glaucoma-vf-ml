from lightning.pytorch.callbacks import Callback

from glaucoma_vf.models.dummy_grape_vf_model import (
    Batch,
    FeatureSet,
    LabelSet,
    ModelOutput,
)
from glaucoma_vf.plot.grape_plot import plot_grape_predictions

# See: https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html


class GRAPELogger(Callback):
    """
    Logs Visual Fields from the GRAPE dataset.
    """

    def __init__(self):
        super().__init__()
        self.test_outputs = {}  # Buffer to hold samples

    def on_test_batch_end(  # type: ignore
        self,
        trainer,
        pl_module,
        outputs: ModelOutput,
        batch,
        batch_idx,
        dataloader_idx=0,
    ):
        features = FeatureSet(**batch["X"])
        labels = LabelSet(**batch["y"])
        batch = Batch(X=features, y=labels)

        x_grids = batch.X.grids.cpu().numpy().squeeze(1)

        y_grids = batch.y.grids.cpu().numpy().squeeze(1)
        preds_grids = outputs.pred_grids.squeeze(1)

        # Un-normalize
        x_grids *= 40
        y_grids *= 40
        preds_grids *= 40

        # --- PRINT MATPLOTLIB ---
        # Only save the first batch to avoid filling up RAM
        if batch_idx == 0:
            # 'outputs' usually contains the logits/preds if you return them in test_step
            # If your test_step returns {'loss': loss, 'preds': preds}, access it here:
            self.test_outputs = {
                "x_grids": x_grids,
                "y_grids": y_grids,
                "preds_grids": preds_grids,
            }

    def on_test_epoch_end(self, trainer, pl_module):
        if self.test_outputs:
            plot_grape_predictions(
                self.test_outputs["x_grids"],
                self.test_outputs["y_grids"],
                self.test_outputs["preds_grids"],
                n_samples=5,
            )
            # Clear the buffer for the next run
            self.test_outputs = {}
