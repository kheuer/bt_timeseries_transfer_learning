import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from darts import TimeSeries

from utils.utils import normalize_list


class OurDataset:
    def __init__(self, params: dict):
        """
        Initialize the OurDataset class.

        Args:
        - params (dict): Dictionary containing parameters for dataset creation.
        """
        # GET DATA
        self.TIMESERIES_LENGTH = params["TIMESERIES_LENGTH"]
        self.WINDOW_SIZE = params["WINDOW_SIZE"]
        self.FORWARD_WINDOW = self.TIMESERIES_LENGTH - self.WINDOW_SIZE

        self.BREAK_BETWEEN_WINDOW_STARTS = params["BREAK_BETWEEN_WINDOW_STARTS"]
        self.ENRICH_RATIO_WEATHER = params["ENRICH_RATIO_WEATHER"]
        self.ENRICH_RATIO_DARTS = params["ENRICH_RATIO_DARTS"]
        self.VAL_SHARE = params["VAL_SHARE"]

        self.train_dataset = []
        self.val_dataset = []
        self.files_stock = []
        self.files_weather = []

        self.data_path_stocks = None
        self.data_path_weather = None
        self.data_path_darts = None

        self.SAMPLE_SIZE = params["SAMPLE_SIZE"]

    def load_from_files(
        self,
        data_path_stocks: str = "/data/processed/stocks/",
        data_path_weather: str = "/data/processed/weather/",
        data_path_darts: str = "/data/processed/darts",
    ):
        """
        Load data from specified file paths.

        Args:
        - data_path_stocks (str): Path to the directory containing stock data.
        - data_path_weather (str): Path to the directory containing weather data.
        - data_path_darts (str): Path to the directory containing darts data.
        """
        # ADJUST PATHS
        if data_path_stocks[0] != "/":
            data_path_stocks = "/" + data_path_stocks
        if data_path_stocks[-1] != "/":
            data_path_stocks += "/"

        # if data_path_weather is None, weather data will not be used
        if isinstance(data_path_weather, str):
            self.use_weather = True
            if data_path_weather[0] != "/":
                data_path_weather = "/" + data_path_weather
            if data_path_weather[-1] != "/":
                data_path_weather += "/"
            self.data_path_weather = os.getcwd() + data_path_weather
        else:
            self.use_weather = False

        # if data_path_darts is None, darts data will not be used
        if isinstance(data_path_darts, str):
            self.use_darts = True
            if data_path_darts[0] != "/":
                data_path_darts = "/" + data_path_darts
            if data_path_darts[-1] != "/":
                data_path_darts += "/"
            self.data_path_darts = os.getcwd() + data_path_darts
        else:
            self.use_darts = False

        self.data_path_stocks = os.getcwd() + data_path_stocks

        self._collect_files()
        self._get_data()

    def save_datasets(self, filepath):
        """
        Save datasets to a pickle file.

        Args:
        - filepath (str): Filepath to save the datasets.
        """
        assert filepath[-4:] == ".pkl"
        self._check_data_loaded()
        with open(filepath, "wb") as f:
            pickle.dump(
                (
                    self.train_dataset,
                    self.val_dataset,
                    self.normal_factors_train,
                    self.normal_factors_val,
                ),
                f,
            )

    def load_cached(self, filepath):
        """
        Load datasets from a cached pickle file.

        Args:
        - filepath (str): Filepath to the cached pickle file.
        """
        assert filepath[-4:] == ".pkl"
        print(f"loading data from file {filepath}")
        with open(filepath, "rb") as f:
            (
                self.train_dataset,
                self.val_dataset,
                self.normal_factors_train,
                self.normal_factors_val,
            ) = pickle.load(f)

    def _collect_files(self):
        """
        Collect file paths from specified directories.
        """
        self.files_stock = [
            self.data_path_stocks + file for file in os.listdir(self.data_path_stocks)
        ]
        np.random.shuffle(self.files_stock)

        if self.use_weather:
            self.files_weather = [
                self.data_path_weather + file
                for file in os.listdir(self.data_path_weather)
            ]
            np.random.shuffle(self.files_weather)
        else:
            self.files_weather = []

        if self.use_darts:
            self.files_darts = [
                self.data_path_darts + file for file in os.listdir(self.data_path_darts)
            ]
            np.random.shuffle(self.files_darts)
        else:
            self.files_darts = []

    def _add_timeseries_from_files(self, files, target_train_len):
        """
        Add time series data from files to the dataset.

        Args:
        - files (list): List of file paths containing time series data.
        - target_train_len (int): Target length for the training dataset.
        """
        possible_selections = []

        for file_index in tqdm(range(len(files))):
            file = self.files_weather[file_index]
            df = pd.read_parquet(file)
            data = np.array(df).flatten()
            if len(data) < self.TIMESERIES_LENGTH:
                self.too_short += 1
                continue

            for i in range(0, len(data), self.BREAK_BETWEEN_WINDOW_STARTS):
                start_index = i
                end_index = start_index + self.TIMESERIES_LENGTH

                # Ensure the slice is not shorter than WINDOW_SIZE+FORWARD_WINDOW
                if end_index <= len(data) and np.random.random() < self.SAMPLE_SIZE:
                    selection = data[start_index:end_index]
                    possible_selections.append(selection)

            self.lengths.append(
                len(data)
            )  # technically, this sample might not be choosen but this is still valuable to know to us

        while possible_selections:
            i = np.random.randint(len(possible_selections))
            selection = possible_selections.pop(i)

            selection, factors = normalize_list(selection, self.WINDOW_SIZE)
            selection = TimeSeries.from_dataframe(pd.DataFrame(selection))
            self.train_dataset.append(selection)
            self.normal_factors_train.append(factors)

            if len(self.train_dataset) == target_train_len:
                print("stopped enrichment, reached target length of", target_train_len)
                break

        if len(self.train_dataset) < target_train_len:
            print(
                f"WARNING: there is not enough data to enrich the train_dataset to {target_train_len} timeseries, could only reach {len(self.train_dataset)} timeseries, enrich ratio: {self.ENRICH_RATIO_DARTS}"
            )

    def _get_data(self):
        """
        Load and preprocess data for training and validation.
        """
        self.too_short = 0
        self.lengths = []
        self.normal_factors_train = []
        self.normal_factors_val = []

        # load STOCK data into train and validation set
        for file_index in tqdm(range(len(self.files_stock))):
            file = self.files_stock[file_index]
            df = pd.read_parquet(file)
            data = np.array(df).flatten()
            if len(data) < self.TIMESERIES_LENGTH:
                self.too_short += 1
                continue

            saved = False
            for i in range(0, len(data), self.BREAK_BETWEEN_WINDOW_STARTS):
                start_index = i
                end_index = start_index + self.TIMESERIES_LENGTH

                # Ensure the slice is not shorter than WINDOW_SIZE+FORWARD_WINDOW
                if end_index <= len(data) and np.random.random() < self.SAMPLE_SIZE:
                    selection = data[start_index:end_index]
                    selection, factors = normalize_list(selection, self.WINDOW_SIZE)
                    if max(selection) > 2 or min(selection) < -1:
                        continue
                    selection = TimeSeries.from_dataframe(pd.DataFrame(selection))

                    saved = True
                    if np.random.random() < self.VAL_SHARE:
                        self.val_dataset.append(selection)
                        self.normal_factors_val.append(factors)
                    else:
                        self.train_dataset.append(selection)
                        self.normal_factors_train.append(factors)
            if saved:
                self.lengths.append(len(data))

        # Explain stock data to console
        n_timeseries_stocks = len(self.train_dataset)
        percent_in_val = len(self.val_dataset) / (
            len(self.val_dataset) + len(self.train_dataset)
        )
        print(f"{round(percent_in_val*100, 2)}% of stock data in validation set")
        tollerance = 0.05
        if percent_in_val < self.VAL_SHARE - tollerance:
            raise RuntimeWarning(
                f"only {round(percent_in_val*100, 2)}% of stock data is in the validation set, this is less than required {self.VAL_SHARE*100}% +- {tollerance*100}% tollerance"
            )

        print(
            f"Used {len(self.lengths)}/{len(self.files_stock)} stock files ({round(100*(len(self.lengths)/len(self.files_stock)), 2)}%) {f'{self.too_short} timeseries were excluded because of insufficient length,' if self.too_short else ''}"
        )
        print(f"Train dataset has {n_timeseries_stocks} timeseries from stocks")
        # load weather data into train set
        if self.use_weather:
            print("LOADING IN WEATHER DATA")
            n_timeseries_before = len(self.train_dataset)
            files_used_before = len(self.lengths)
            target_train_len = int(
                n_timeseries_stocks * (1 + self.ENRICH_RATIO_WEATHER)
            )
            self._add_timeseries_from_files(self.files_weather, target_train_len)
            n_timeseries_weather = len(self.train_dataset) - n_timeseries_before
            print(
                f"Loaded in {n_timeseries_weather} weather timeseries from {len(self.lengths) - files_used_before} files"
            )

        if self.use_darts:
            print("LOADING IN DARTS DATA")
            n_timeseries_before = len(self.train_dataset)
            files_used_before = len(self.lengths)
            target_train_len = int(
                n_timeseries_stocks * (1 + self.ENRICH_RATIO_DARTS)
                + n_timeseries_weather
            )
            self._add_timeseries_from_files(self.files_darts, target_train_len)
            print(
                f"Loaded in {len(self.train_dataset) - n_timeseries_before} darts timeseries from {len(self.lengths) - files_used_before} files"
            )

        print(
            f"\nloaded in data for {len(self.lengths)} timeseries\nMinimum length: {min(self.lengths)}\nMean length: {int(np.mean(self.lengths))}"
        )
        print(
            f"Train Dataset: {len(self.train_dataset)} timeseries\nValidation Dataset: {len(self.val_dataset)} timeseries"
        )
        print(
            f"Mean sub-series per timeseries: {round((len(self.train_dataset)+len(self.val_dataset))/len(self.lengths), 1)}"
        )
        print(
            f"Train dataset: {n_timeseries_stocks}/{len(self.train_dataset)} timeseries are from stocks ({round(100*n_timeseries_stocks/len(self.train_dataset), 2)}%)"
        )

        # ensure the data loading worked fine
        for dp in self.train_dataset:
            assert len(dp) == self.TIMESERIES_LENGTH
        for dp in self.val_dataset:
            assert len(dp) == self.TIMESERIES_LENGTH

        assert self.train_dataset

        train_perm = np.random.permutation(len(self.train_dataset))
        self.train_dataset[:] = [self.train_dataset[i] for i in train_perm]
        self.normal_factors_train[:] = [
            self.normal_factors_train[i] for i in train_perm
        ]

        val_perm = np.random.permutation(len(self.val_dataset))
        self.val_dataset[:] = [self.val_dataset[i] for i in val_perm]
        self.normal_factors_val[:] = [self.normal_factors_val[i] for i in val_perm]

    def _check_data_loaded(self):
        """
        Check if data has been loaded.
        Raises a RuntimeError if data is not loaded.
        """

        if not self.train_dataset:
            raise RuntimeError("you must load data first")

    def get_datasets(self):
        """
        Get the training and validation datasets.

        Returns:
        - Tuple containing (train_dataset, val_dataset)
        """
        self._check_data_loaded()
        return self.train_dataset, self.val_dataset

    def get_normalization_factors(self):
        """
        Get normalization factors used for dataset normalization.

        Returns:
        - Tuple containing (normal_factors_train, normal_factors_val)
        """
        return self.normal_factors_train, self.normal_factors_val
