import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from pytorch_lightning import Callback
import json


def list_of_timeseries_to_tensor(lst):
    """
    Converts a list of time series instances into a PyTorch tensor that can be used to calculate loss.

    Args:
    - lst (list): List containing time series instances.

    Returns:
    - torch.Tensor: PyTorch tensor representing the time series data.
    """
    tensor_builder = []
    for t in lst:
        tensor_builder.append(np.array(t.pd_dataframe()))
    return torch.Tensor(np.array(tensor_builder))


def normalize_list(lst, features_len):
    """
    Normalize a list of values between zero and one. This function is meant to be used on a timeseries
    that includes features and targets. However the targets are not used in finding the maximum and
    minimum values for normalization and thus might be slightly lower than zero or higher than one.

    Args:
    - lst (list): List of values to be normalized.
    - features_len (int): Number of features to consider for normalization.

    Returns:
    - list: Normalized list of values.
    - tuple: Tuple containing min and max values used for normalization.
    """
    lst = np.array(lst)
    min_value = min(lst[:features_len])
    max_value = max(lst[:features_len])
    if min_value == max_value:
        return lst, (min_value, max_value)
    else:
        return list(((lst - min_value) / (max_value - min_value))), (
            min_value,
            max_value,
        )


def reverse_normalization(array, min_value, max_value):
    """
    Reverse normalization of an array based on provided min and max values.

    Args:
    - array (np.ndarray): Array to be reverse normalized.
    - min_value (float): Minimum value used in normalization.
    - max_value (float): Maximum value used in normalization.

    Returns:
    - np.ndarray: Reverse normalized array.
    """
    assert not isinstance(array, list)
    return (array * (max_value - min_value)) + min_value


def geomean(iterable):
    """
    Calculate the geometric mean of an iterable.

    Args:
    - iterable (iterable): Iterable containing numerical values.

    Returns:
    - float: Geometric mean of the input iterable.
    """
    a = np.array(iterable)
    return a.prod() ** (1.0 / len(a))


class PlotLossCallback(Callback):
    """
    Callback for plotting training and validation losses during training.
    """

    def __init__(self, save_path):
        """
        Initializes the PlotLossCallback.

        Args:
        - save_path (str): Path to save the plotted loss figures.
        """
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.save_path = save_path
        self.fig, self.ax = plt.subplots()

    def on_train_epoch_end(self, trainer, pl_module):
        """Callback function called at the end of each training epoch.

        Args:
            trainer: The PyTorch Lightning trainer object.
            pl_module: The PyTorch Lightning module being trained.

        Returns:
            None
        """
        self.epochs.append(trainer.current_epoch)
        self.train_losses.append(trainer.callback_metrics["train_loss"].item())
        self.val_losses.append(trainer.callback_metrics["val_loss"].item())
        for log in [True, False]:
            self.ax.clear()
            self.ax.plot(self.epochs, self.train_losses, label="Training Loss")
            self.ax.plot(self.epochs, self.val_losses, label="Validation Loss")
            self.ax.set_xlabel("Epochs")
            self.ax.set_ylabel("Loss")
            self.ax.set_title("Training and Validation Loss")
            self.ax.legend()

            if log:
                self.ax.set_yscale("log")
                self.fig.savefig(self.save_path + "_last_active_log.png")
            else:
                self.fig.savefig(self.save_path + "_last_active.png")
                self.fig.savefig(self.save_path + f"_epoch_{len(self.epochs) - 1}.png")
                with open("loss.json", "w") as file:
                    json.dump(
                        {
                            "validation": self.val_losses,
                            "train": self.train_losses,
                            "epochs": self.epochs,
                        },
                        file,
                    )
            plt.close(self.fig)

    def on_train_end(self, trainer, pl_module):
        """Saves the final training plot and closes the figure.

        Args:
            trainer: The trainer object used for training.
            pl_module: The LightningModule object used for training.

        Returns:
            None

        Raises:
            None
        """
        self.fig.savefig(
            self.save_path
            + f"_final_time_{time.time()}_val_loss_{self.val_losses[-1]}.png"
        )
        plt.close(self.fig)
