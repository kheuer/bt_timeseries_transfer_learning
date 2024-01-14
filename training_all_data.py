"""
This module trains an N-Beats model with the best hyperparameters for a module with the enrichment ratio 10%
"""
import os
import sys
import torch
from utils.OurNBeatsModel import OurNBeatsModel
from utils.OurDataset import OurDataset
from torchmetrics.regression import (
    MeanAbsolutePercentageError,
    WeightedMeanAbsolutePercentageError,
)
from utils.metric import *

parameters = {
    # Data Structure
    "TIMESERIES_LENGTH": 252 + 14,
    "BREAK_BETWEEN_WINDOW_STARTS": 300,
    "VAL_SHARE": 0.3,  # percent of timeseries to be choosen for validation set
    "MAX_SAMPLES_PER_TS": 1,  # fix for this custom data selection approach
    "ENRICH_RATIO_WEATHER": 0.1,
    "ENRICH_RATIO_DARTS": 0.1,
    "SAMPLE_SIZE": 1,
    # Models
    "WINDOW_SIZE": 252,
    "EPOCHS": 2000,
    "LEARNING_RATE": 0.0007585775750291836,
    "BATCH_SIZE": 2000,
    "NUM_STACKS": 4,
    "NUM_BLOCKS": 4,
    "NUM_LAYERS": 5,
    "EXPANSION_COEFFICIENT_DIMENSION": 4,
    "LAYER_WIDTHS": 136,
    "LOSS_FN": torch.nn.HuberLoss(delta=0.3),
    "OPTIMIZER": "Adam",
    "DROPOUT": 0.156693,
    "EARLYSTOP_PATIENCE": 50,
    "EARLYSTOP_MIN_DELTA": 0.00005,
    "ACTIVATION_FN": "Sigmoid",
    "CALLBACKS": [],
    # Other
    "CPUS": os.cpu_count(),
    "MODEL_NAME": "train_best_params_all",
}

if parameters["SAMPLE_SIZE"] != 1:
    print(f"WARNING: the sample size is only {parameters['SAMPLE_SIZE']}")


if __name__ == "__main__":
    dataset = OurDataset(parameters)
    cache_name = f"cache/all_data_sample_{parameters['SAMPLE_SIZE']}_break_{parameters['BREAK_BETWEEN_WINDOW_STARTS']}_enrich_{parameters['ENRICH_RATIO_WEATHER']}_{parameters['ENRICH_RATIO_DARTS']}.pkl"
    if os.path.isfile(cache_name):
        dataset.load_cached(cache_name)
    else:
        dataset.load_from_files()
        dataset.save_datasets(cache_name)

    train, val = dataset.get_datasets()

    del dataset
    dataset_size = sys.getsizeof(train) + sys.getsizeof(val)
    print(f"Full Dataset size: {dataset_size/(1e6)} MB")

    model = OurNBeatsModel(parameters, train, val, verbose=True)
    model.setup_new_model()
    print("finished setting up new model")
    model.train()
    print("finished traning")
    model.save_prediction(5)
    print("Getting test loss")

    model.TIMESERIES_LENGTH = 504 + 14
    model.WINDOW_SIZE = 504

    losses = model.get_test_loss(
        all_metrices,
        BREAK_BETWEEN_WINDOW_STARTS=parameters["BREAK_BETWEEN_WINDOW_STARTS"],
        sample_size=parameters["SAMPLE_SIZE"],
    )

    for i, loss in enumerate(losses):
        print(f"Loss for {all_metrices[i]}: {loss}")
