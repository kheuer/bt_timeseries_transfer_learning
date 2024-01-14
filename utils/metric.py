"""this module creates all of our custom metrics, to be used in other files"""
import numpy as np
import torch
import torch.nn as nn
import darts
from utils.utils import reverse_normalization, geomean

device = "cuda" if torch.cuda.is_available() else "cpu"


class MedianAbsoluteError(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Computes the median of mean absolute errors between predictions and targets.

        Args:
            predictions (torch.Tensor): Predicted values.
            targets (torch.Tensor): Target or true values.

        Returns:
            float: Median absolute error.
        """
        assert predictions.shape == targets.shape
        # shape (n_timeseries, length_timeseries, 1)
        diff = np.abs(predictions - targets)
        mean_errors = diff.mean(axis=1)
        media_error = mean_errors.median()

        return float(media_error)


class MedianSquaredError(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Computes the median of mean squared errors between predictions and targets.

        Args:
            predictions (torch.Tensor): Predicted values.
            targets (torch.Tensor): Target or true values.

        Returns:
            float: Median squared error.
        """
        assert predictions.shape == targets.shape
        # shape (n_timeseries, length_timeseries, 1)
        diff = (predictions - targets) ** 2
        mean_errors = diff.mean(axis=1)
        media_error = mean_errors.median()

        return float(media_error)


class MeanLastValueError(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Computes the mean last value error between predictions and targets.

        Args:
            predictions (torch.Tensor): Predicted values.
            targets (torch.Tensor): Target or true values.

        Returns:
            float: Mean last value error.
        """
        assert predictions.shape == targets.shape
        # shape (n_timeseries, length_timeseries, 1)
        last_pred = predictions[:, -1]
        last_target = targets[:, -1]
        mae = torch.nn.L1Loss()
        return float(mae(last_pred, last_target))


class MedianLastValueError(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Computes the median of mean last value errors between predictions and targets.

        Args:
            predictions (torch.Tensor): Predicted values.
            targets (torch.Tensor): Target or true values.

        Returns:
            float: Median last value error.
        """
        assert predictions.shape == targets.shape
        # shape (n_timeseries, length_timeseries, 1)
        last_pred = predictions[:, -1]
        last_target = targets[:, -1]

        diff = (last_pred - last_target).abs()
        median_error = diff.median()
        return float(median_error)


class MeanTotalReturnError(nn.Module):
    def __init__(self):
        super().__init__()
        self.pass_features_and_normal_factors = (
            True  # set to detect these classes with hasattr
        )

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        features: list,
        normalization_factors: list,
    ):
        """
        Calculates the forward pass of the model and returns the mean total return error.

        Args:
            predictions (torch.Tensor): The predicted values.
            targets (torch.Tensor): The target values.
            features (list): A list of time series features.
            normalization_factors (list): A list of tuples containing the minimum and maximum values for each time series.

        Returns:
            float: The mean total return error.

        Raises:
            AssertionError: If the shape of predictions and targets do not match.
            AssertionError: If features is not a list.
            AssertionError: If the first element of features is not an instance of darts.TimeSeries.
        """
        assert predictions.shape == targets.shape
        assert isinstance(features, list)
        assert isinstance(features[0], darts.TimeSeries)
        last_feature_list = [ts[-1].values()[0][0] for ts in features]
        predictions = np.array(predictions)
        targets = np.array(targets)

        abs_errors = []
        for ts_i in range(len(predictions)):
            min_val, max_val = normalization_factors[ts_i]

            _preds = reverse_normalization(predictions[ts_i], min_val, max_val)
            _targets = reverse_normalization(targets[ts_i], min_val, max_val)
            last_feature = (last_feature_list[ts_i] * (max_val - min_val)) + min_val
            total_returns_pred = (last_feature - _preds) / last_feature
            total_returns_targets = (last_feature - _targets) / last_feature
            errors = np.abs(total_returns_pred - total_returns_targets)
            abs_errors.append(np.mean(errors))
        return float(np.mean(abs_errors))


class MedianTotalReturnError(nn.Module):
    def __init__(self):
        super().__init__()
        self.pass_features_and_normal_factors = (
            True  # set to detect these classes with hasattr
        )

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        features: list,
        normalization_factors: list,
    ):
        """
        Calculates the forward pass of the model and returns the median of mean total return error.

        Args:
            predictions (torch.Tensor): The predicted values.
            targets (torch.Tensor): The target values.
            features (list): A list of time series features.
            normalization_factors (list): A list of tuples containing the minimum and maximum values for each time series.

        Returns:
            float: The median of mean total return error

        Raises:
            AssertionError: If the shape of predictions and targets do not match.
            AssertionError: If features is not a list.
            AssertionError: If the first element of features is not an instance of darts.TimeSeries.
        """
        assert predictions.shape == targets.shape
        assert isinstance(features, list)
        assert isinstance(features[0], darts.TimeSeries)
        last_feature_list = [ts[-1].values()[0][0] for ts in features]
        predictions = np.array(predictions)
        targets = np.array(targets)

        abs_errors = []
        for ts_i in range(len(predictions)):
            min_val, max_val = normalization_factors[ts_i]

            _preds = reverse_normalization(predictions[ts_i], min_val, max_val)
            _targets = reverse_normalization(targets[ts_i], min_val, max_val)
            last_feature = (last_feature_list[ts_i] * (max_val - min_val)) + min_val
            total_returns_pred = (last_feature - _preds) / last_feature
            total_returns_targets = (last_feature - _targets) / last_feature
            errors = np.abs(total_returns_pred - total_returns_targets)
            abs_errors.append(np.mean(errors))
        return float(np.median(abs_errors))


class GeometricMeanDailyReturnError(nn.Module):
    def __init__(self):
        super().__init__()
        self.pass_features_and_normal_factors = (
            True  # set to detect these classes with hasatt
        )

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        features: list,
        normalization_factors: list,
    ):
        """
        Calculates the forward pass of the model and returns the geometric mean daily return error.

        Args:
            predictions (torch.Tensor): The predicted values.
            targets (torch.Tensor): The target values.
            features (list): A list of time series features.
            normalization_factors (list): A list of tuples containing the minimum and maximum values for each time series.

        Returns:
            float: The geometric mean daily return error.

        Raises:
            AssertionError: If the shape of predictions and targets do not match.
            AssertionError: If features is not a list.
            AssertionError: If the first element of features is not an instance of darts.TimeSeries.
        """
        assert predictions.shape == targets.shape
        assert isinstance(features, list)
        assert isinstance(features[0], darts.TimeSeries)
        last_feature_list = [ts[-1].values()[0][0] for ts in features]
        predictions = np.array(predictions)
        targets = np.array(targets)

        abs_errors = []
        for ts_i in range(len(predictions)):
            min_val, max_val = normalization_factors[ts_i]
            last_feature = (last_feature_list[ts_i] * (max_val - min_val)) + min_val

            _preds = reverse_normalization(
                predictions[ts_i], min_val, max_val
            ).flatten()
            _preds_with_last = np.concatenate((np.array([last_feature]), _preds))
            daily_returns_pred = np.diff(_preds_with_last) / _preds_with_last[:-1]

            _targets = reverse_normalization(targets[ts_i], min_val, max_val).flatten()
            _targets_with_last = np.concatenate((np.array([last_feature]), _targets))
            daily_returns_targets = (
                np.diff(_targets_with_last) / _targets_with_last[:-1]
            )

            errors = np.abs(daily_returns_pred - daily_returns_targets)
            abs_errors.append(geomean(errors))
        return float(np.mean(abs_errors))


class MeanFinalReturnError(nn.Module):
    def __init__(self):
        super().__init__()
        self.pass_features_and_normal_factors = (
            True  # set to detect these classes with hasatt
        )

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        features: list,
        normalization_factors: list,
    ):
        """
        Calculates the forward pass of the model and returns the mean final return error.

        Args:
            predictions (torch.Tensor): The predicted values.
            targets (torch.Tensor): The target values.
            features (list): A list of time series features.
            normalization_factors (list): A list of tuples containing the minimum and maximum values for each time series.

        Returns:
            float: The mean final return error.

        Raises:
            AssertionError: If the shape of predictions and targets do not match.
            AssertionError: If features is not a list.
            AssertionError: If the first element of features is not an instance of darts.TimeSeries.
        """
        assert predictions.shape == targets.shape
        assert isinstance(features, list)
        assert isinstance(features[0], darts.TimeSeries)
        last_feature_list = [ts[-1].values()[0][0] for ts in features]
        predictions = np.array(predictions)
        targets = np.array(targets)

        abs_errors = []
        for ts_i in range(len(predictions)):
            min_val, max_val = normalization_factors[ts_i]

            last_feature = (last_feature_list[ts_i] * (max_val - min_val)) + min_val
            last_target = (float(targets[ts_i][-1]) * (max_val - min_val)) + min_val
            last_pred = (float(predictions[ts_i][-1]) * (max_val - min_val)) + min_val

            predicted_return = (last_pred - last_feature) / last_feature
            actual_return = (last_target - last_feature) / last_feature

            error = np.abs(predicted_return - actual_return)
            abs_errors.append(error)
        return float(np.mean(abs_errors))


class MedianFinalReturnError(nn.Module):
    def __init__(self):
        super().__init__()
        self.pass_features_and_normal_factors = (
            True  # set to detect these classes with hasatt
        )

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        features: list,
        normalization_factors: list,
    ):
        """
        Calculates the forward pass of the model and returns the median total return error.

        Args:
            predictions (torch.Tensor): The predicted values.
            targets (torch.Tensor): The target values.
            features (list): A list of time series features.
            normalization_factors (list): A list of tuples containing the minimum and maximum values for each time series.

        Returns:
            float: The median final return error.

        Raises:
            AssertionError: If the shape of predictions and targets do not match.
            AssertionError: If features is not a list.
            AssertionError: If the first element of features is not an instance of darts.TimeSeries.
        """
        assert predictions.shape == targets.shape
        assert isinstance(features, list)
        assert isinstance(features[0], darts.TimeSeries)
        last_feature_list = [ts[-1].values()[0][0] for ts in features]
        predictions = np.array(predictions)
        targets = np.array(targets)

        abs_errors = []
        for ts_i in range(len(predictions)):
            min_val, max_val = normalization_factors[ts_i]

            last_feature = (last_feature_list[ts_i] * (max_val - min_val)) + min_val
            last_target = (float(targets[ts_i][-1]) * (max_val - min_val)) + min_val
            last_pred = (float(predictions[ts_i][-1]) * (max_val - min_val)) + min_val

            predicted_return = (last_pred - last_feature) / last_feature
            actual_return = (last_target - last_feature) / last_feature

            error = np.abs(predicted_return - actual_return)
            abs_errors.append(error)
        return float(np.median(abs_errors))


class BackTestingProfitError(nn.Module):
    def __init__(self, choose_top_percent=0.05, print=True):
        super().__init__()
        self.pass_features_and_normal_factors = (
            True  # set to detect these classes with hasattr
        )
        self.choose_top_percent = choose_top_percent
        self.print = print

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        features: list,
        normalization_factors: list,
    ):
        """
        Calculates the forward pass of the model and returns the backtesting profit error.

        Args:
            predictions (torch.Tensor): The predicted values.
            targets (torch.Tensor): The target values.
            features (list): A list of time series features.
            normalization_factors (list): A list of tuples containing the minimum and maximum values for each time series.

        Returns:
            float: The backtesting profit error.

        Raises:
            AssertionError: If the shape of predictions and targets do not match.
            AssertionError: If features is not a list.
            AssertionError: If the first element of features is not an instance of darts.TimeSeries.
        """
        assert predictions.shape == targets.shape
        assert isinstance(features, list)
        assert isinstance(features[0], darts.TimeSeries)
        last_feature_list = [ts[-1].values()[0][0] for ts in features]
        predictions = np.array(predictions)
        targets = np.array(targets)

        actual_returns = []
        predicted_returns = []
        for ts_i in range(len(predictions)):
            min_val, max_val = normalization_factors[ts_i]

            last_feature = (last_feature_list[ts_i] * (max_val - min_val)) + min_val
            last_target = (float(targets[ts_i][-1]) * (max_val - min_val)) + min_val
            last_pred = (float(predictions[ts_i][-1]) * (max_val - min_val)) + min_val

            predicted_return = (last_pred - last_feature) / last_feature
            actual_return = (last_target - last_feature) / last_feature

            predicted_returns.append(predicted_return)
            actual_returns.append(actual_return)

        top_n_percent = int(self.choose_top_percent * len(predicted_returns))

        top_indices = list(np.argsort(predicted_returns)[-top_n_percent:])
        ai_portfolio_returns_individual = np.array(actual_returns)[top_indices]
        ai_portfolio_return = np.mean(ai_portfolio_returns_individual)
        benchmark_return = np.mean(actual_returns)
        outperformance = ai_portfolio_return - benchmark_return

        if self.print:
            print(
                f"Market Outperformance Analysis (top {round(100*self.choose_top_percent, 2)}%):\nMarket Benchmark: {benchmark_return}\nPortfolio Return: {ai_portfolio_return}\nOutperformance: {outperformance}"
            )
        return outperformance


all_metrices = [
    torch.nn.L1Loss(),
    MedianAbsoluteError(),
    torch.nn.MSELoss(),
    MedianSquaredError(),
    torch.nn.HuberLoss(delta=0.3),
    MeanLastValueError(),
    MedianLastValueError(),
    MeanTotalReturnError(),
    MedianTotalReturnError(),
    GeometricMeanDailyReturnError(),
    MeanFinalReturnError(),
    MedianFinalReturnError(),
    BackTestingProfitError(),
]
