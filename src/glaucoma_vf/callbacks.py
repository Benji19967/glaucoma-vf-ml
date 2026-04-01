from lightning.pytorch.callbacks import Callback

from glaucoma_vf.plot.plot_hvf import plot_hvf_predictions, print_hvf_ascii

# See: https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html


class HVFPrinter(Callback):
    def __init__(self, message: str = "Epoch Finished"):
        super().__init__()
        self.message = message
        self.test_outputs = {}  # Buffer to hold samples

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        # Extract data from the outputs dictionary
        # Squeeze removes the channel dim: (Batch, 1, 8, 9) -> (Batch, 8, 9)
        x, y_true = batch
        grids = x.cpu().numpy().squeeze(1)
        y_trues = y_true.cpu().numpy()
        y_preds = outputs["preds"].cpu().numpy()  # type: ignore

        batch_size = len(y_trues)

        # Calculate starting index for this batch
        start_idx = batch_idx * batch_size

        for i in range(batch_size):
            print_hvf_ascii(
                grid=grids[i],
                true_label=y_trues[i],
                pred_label=y_preds[i],
                sample_idx=start_idx + i,
            )

        # --- PRINT MATPLOTLIB ---
        # # Only save the first batch to avoid filling up RAM
        # # if batch_idx == 0:
        # x, y_true = batch
        # # 'outputs' usually contains the logits/preds if you return them in test_step
        # # If your test_step returns {'loss': loss, 'preds': preds}, access it here:
        # self.test_outputs = {
        #     "grids": x.cpu().numpy(),
        #     "y_true": y_true.cpu().numpy(),
        #     "y_pred": outputs["preds"].cpu().numpy(),  # type: ignore
        # }

    # def on_test_epoch_end(self, trainer, pl_module):
    #     if self.test_outputs:
    #         plot_hvf_predictions(
    #             self.test_outputs["grids"],
    #             self.test_outputs["y_true"],
    #             self.test_outputs["y_pred"],
    #             n_samples=5,
    #         )
    #         # Clear the buffer for the next run
    #         self.test_outputs = {}
