"""this module contains our implementation of the N-Beats module and is used in almost all other modules^   """

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from darts import TimeSeries
from darts.models import NBEATSModel
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from utils.utils import list_of_timeseries_to_tensor, PlotLossCallback
from utils.OurDataset import OurDataset
from utils.metric import *

# this speeds up calculations during matrix multiplication
torch.set_float32_matmul_precision("high")


class OurNBeatsModel:
    """Class representing our NBeats model.

    Args:
        params (dict): Dictionary containing hyperparameters for the model.
        train_dataset (list): List of training datasets.
        val_dataset (list): List of validation datasets.
        verbose (bool, optional): Verbosity mode. Defaults to True.
    """

    def __init__(
        self, params: dict, train_dataset: list, val_dataset: list, verbose: bool = True
    ):
        """Initialize the NBeats model."""
        self.verbose = verbose

        # IMPORT HYPERPARAMETERS
        self.TIMESERIES_LENGTH = params.get("TIMESERIES_LENGTH", None)
        self.WINDOW_SIZE = params.get("WINDOW_SIZE", None)
        self.FORWARD_WINDOW = self.TIMESERIES_LENGTH - self.WINDOW_SIZE
        self.EPOCHS = params.get("EPOCHS", None)
        self.BATCH_SIZE = params.get("BATCH_SIZE", None)
        self.NUM_STACKS = params.get("NUM_STACKS", None)
        self.NUM_BLOCKS = params.get("NUM_BLOCKS", None)
        self.NUM_LAYERS = params.get("NUM_LAYERS", None)
        self.EXPANSION_COEFFICIENT_DIMENSION = params.get(
            "EXPANSION_COEFFICIENT_DIMENSION", None
        )
        self.LAYER_WIDTHS = params.get("LAYER_WIDTHS", None)
        self.LOSS_FN = params.get("LOSS_FN", None)
        self.LEARNING_RATE = params.get("LEARNING_RATE", None)

        optimizer = params.get("OPTIMIZER", None)
        if optimizer is None:
            self.OPTIMIZER = None
        elif optimizer.lower() == "adam":
            self.OPTIMIZER = torch.optim.Adam
        elif optimizer.lower() == "rmsprop":
            self.OPTIMIZER = torch.optim.RMSprop
        elif optimizer.lower() == "rprop":
            self.OPTIMIZER = torch.optim.Rprop
        elif optimizer.lower() == "radam":
            self.OPTIMIZER = torch.optim.RAdam
        else:
            raise ValueError(f"unknown optimizer: {optimizer}")

        self.DROPOUT = params.get("DROPOUT", None)
        self.EARLYSTOP_PATIENCE = params.get("EARLYSTOP_PATIENCE", None)
        self.EARLYSTOP_MIN_DELTA = params.get("EARLYSTOP_MIN_DELTA", None)
        self.CPUS = params.get("CPUS", os.cpu_count())
        self.MODEL_NAME = params.get("MODEL_NAME", "unknown_model")

        plot_loss_callback = PlotLossCallback(f"imgs/{self.MODEL_NAME}_loss")
        self.CALLBACKS = [plot_loss_callback] + params.get("CALLBACKS", [])

        if self.EARLYSTOP_MIN_DELTA is None or self.EARLYSTOP_PATIENCE is None:
            pass
        else:
            val_loss_stopper = EarlyStopping(
                monitor="val_loss",
                patience=self.EARLYSTOP_PATIENCE,
                min_delta=self.EARLYSTOP_MIN_DELTA,
                mode="min",
            )
            self.CALLBACKS.append(val_loss_stopper)

        if self.verbose:
            print(f"HYPERPARAMETERS: \n{params}")

        self.model = None
        self.trained = False

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        assert self.train_dataset
        if isinstance(self.train_dataset, list):
            for x in self.train_dataset:
                assert len(x) == self.TIMESERIES_LENGTH
        if isinstance(self.val_dataset, list):
            for x in self.val_dataset:
                assert len(x) == self.TIMESERIES_LENGTH

    def setup_new_model(self):
        """Set up a new NBeats model."""
        self.model = NBEATSModel(
            input_chunk_length=self.WINDOW_SIZE,
            output_chunk_length=self.FORWARD_WINDOW,
            num_stacks=self.NUM_STACKS,
            num_blocks=self.NUM_BLOCKS,
            num_layers=self.NUM_LAYERS,
            expansion_coefficient_dim=self.EXPANSION_COEFFICIENT_DIMENSION,
            layer_widths=self.LAYER_WIDTHS,
            dropout=self.DROPOUT,
            generic_architecture=True,  # use our own architecture, not the one from the paper
            n_epochs=self.EPOCHS,
            nr_epochs_val_period=1,  # default, start recording val loss after 1 epoch
            batch_size=self.BATCH_SIZE,
            model_name=self.MODEL_NAME,
            save_checkpoints=True,
            loss_fn=self.LOSS_FN,
            log_tensorboard=True,
            pl_trainer_kwargs={"accelerator": "auto", "callbacks": self.CALLBACKS},
            optimizer_cls=self.OPTIMIZER,
            optimizer_kwargs={"lr": self.LEARNING_RATE},
            force_reset=True,
            show_warnings=True,
        )
        if self.verbose:
            print("Model initialized")

    def load_model(self):
        """Load the best checkpoint for the model."""
        if self.verbose:
            print(f"loading best model from checkpoint: {self.MODEL_NAME}")
        self.model = NBEATSModel.load_from_checkpoint(self.MODEL_NAME, best=True)
        self.trained = True

    def _check_model(self):
        if self.model is None:
            raise RuntimeError(
                "error: please setup a model first using self.setup_model"
            )

    def get_model(self):
        self._check_model()
        return self.model

    def train(self):
        """Train the NBeats model.

        Fits the NBeats model to the provided training dataset.
        """
        self._check_model()
        self.trained = True  # this is done first so interupted training is still considered trained
        self.model.fit(
            series=self.train_dataset,
            val_series=self.val_dataset,
            verbose=self.verbose,
            max_samples_per_ts=1,
            num_loader_workers=self.CPUS,
        )
        self.load_model()  # loading the best model, just in case

    def predict(self, n, features):
        """Make predictions using the trained model.

        Args:
            n (int): Number of steps to predict.
            features (TimeSeries): Features used for prediction.

        Returns:
            TimeSeries: Predicted values.
        """
        self._check_model()
        if not self.trained:
            raise RuntimeError("error: please train the model first")
        return self.model.predict(
            n=n, series=features, num_loader_workers=self.CPUS, verbose=False
        )

    def save_prediction(self, n_plots=10):
        """Save predictions as images.

        Saves the model predictions as images for visualization.

        Args:
            n_plots (int, optional): Number of plots to save. Defaults to 10.
        """
        if not self.trained:
            raise RuntimeError("you must train the model first")

        path = f"imgs/prediction_{self.MODEL_NAME}"

        for i in range(n_plots):
            data = list(np.array(self.val_dataset[i].pd_dataframe()).flatten())
            assert len(data) == self.WINDOW_SIZE + self.FORWARD_WINDOW
            features, targets = data[: self.WINDOW_SIZE], data[self.WINDOW_SIZE :]

            features_timeseries = TimeSeries.from_values(np.array(features))
            pred = self.predict(self.FORWARD_WINDOW, features_timeseries)

            display_features_and_targets = (features + targets)[-300:]

            plt.plot(display_features_and_targets, label="ground truth")
            plt.plot(
                [None for _ in range(len(display_features_and_targets) - len(targets))]
                + list(np.array(pred.pd_dataframe()).flatten()),
                label="pred",
            )
            plt.legend()
            plt.title(f"actual features_length = {len(features)}")
            plt.savefig(f"{path}_{i}_{time.time()}.png")
            plt.clf()
            plt.close()

    def get_test_loss(
        self,
        loss_functions: list,
        path: str = "/data/processed/test/",
        BREAK_BETWEEN_WINDOW_STARTS=114,
        sample_size=1,
    ):
        """Calculate test loss for the model.

        Args:
            loss_functions (list): List of loss functions to calculate the test loss.
            path (str, optional): Path to test data. Defaults to "/data/processed/test/".
            BREAK_BETWEEN_WINDOW_STARTS (int, optional): Break between window starts. Defaults to 114.
            sample_size (int, optional): Sample size. Defaults to 1 (100%).

        Returns:
            list: List of calculated losses for each loss function.
        """
        params = {
            "TIMESERIES_LENGTH": self.TIMESERIES_LENGTH,
            "WINDOW_SIZE": self.WINDOW_SIZE,
            "BREAK_BETWEEN_WINDOW_STARTS": BREAK_BETWEEN_WINDOW_STARTS,
            "ENRICH_RATIO_WEATHER": 0,
            "ENRICH_RATIO_DARTS": 0,
            "VAL_SHARE": 0,
            "SAMPLE_SIZE": sample_size,
        }

        dataset = OurDataset(params)

        cache_name = f"cache/PRIVATE_validation_break_{BREAK_BETWEEN_WINDOW_STARTS}_sample_{sample_size}_length_{params['TIMESERIES_LENGTH']}.pkl"
        if os.path.isfile(cache_name):
            dataset.load_cached(cache_name)
        else:
            print("load test data from files")
            dataset.load_from_files(
                data_path_stocks=path, data_path_weather=None, data_path_darts=None
            )
            print("save test data to cache:", cache_name)
            dataset.save_datasets(cache_name)

        test, _ = dataset.get_datasets()
        normalization_factors_val, _ = dataset.get_normalization_factors()

        del dataset
        dataset_size = sys.getsizeof(test)
        print(f"Size of test dataset: {dataset_size/(1e6)} MB")
        print("Calculating test loss.")

        features_all = []
        targets_all = []

        for i in range(len(test)):
            features, targets = (
                test[i][: -self.FORWARD_WINDOW],
                test[i][-self.FORWARD_WINDOW :],
            )
            features_all.append(features)
            targets_all.append(targets)

        pred = self.predict(self.FORWARD_WINDOW, features_all)
        pred_ = list_of_timeseries_to_tensor(pred)
        targets_ = list_of_timeseries_to_tensor(targets_all)

        losses = []
        for i, loss_fn in enumerate(loss_functions):
            if hasattr(loss_fn, "pass_features_and_normal_factors"):
                loss = loss_fn(pred_, targets_, features_all, normalization_factors_val)
            else:
                loss = loss_fn(pred_, targets_)

            losses.append(float(loss))
        return losses
