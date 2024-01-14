"""
This module trains an N-Beats model with the best hyperparameters for a module without enrichment
"""
import os
import sys
import torch
from utils.OurNBeatsModel import OurNBeatsModel
from utils.OurDataset import OurDataset
from utils.metric import *

parameters = {
    # Data Structure
    "TIMESERIES_LENGTH": 504 + 14,
    "BREAK_BETWEEN_WINDOW_STARTS": 300,
    "VAL_SHARE": 0.3,  # percent of timeseries to be choosen for validation set
    "MAX_SAMPLES_PER_TS": 1,  # fix for this custom data selection approach
    "ENRICH_RATIO_WEATHER": 0,
    "ENRICH_RATIO_DARTS": 0,
    "SAMPLE_SIZE": 1,
    # Models
    "WINDOW_SIZE": 504,
    "EPOCHS": 5000,
    "LEARNING_RATE": 1.2022644346174132e-06,
    "BATCH_SIZE": 2000,
    "NUM_STACKS": 5,
    "NUM_BLOCKS": 6,
    "NUM_LAYERS": 5,
    "EXPANSION_COEFFICIENT_DIMENSION": 6,
    "LAYER_WIDTHS": 129,
    "LOSS_FN": torch.nn.HuberLoss(delta=0.3),
    "OPTIMIZER": "Adam",
    "DROPOUT": 0.138856,
    "EARLYSTOP_PATIENCE": 50,
    "EARLYSTOP_MIN_DELTA": 0.00005,
    "ACTIVATION_FN": "Sigmoid",
    "CALLBACKS": [],
    # Other
    "CPUS": os.cpu_count(),
    "MODEL_NAME": "train_best_params_stocks",
}

if parameters["SAMPLE_SIZE"] != 1:
    print(f"WARNING: the sample size is only {parameters['SAMPLE_SIZE']}")


if __name__ == "__main__":
    dataset = OurDataset(parameters)
    cache_name = f"cache/stock_sample_{parameters['SAMPLE_SIZE']}_break_{parameters['BREAK_BETWEEN_WINDOW_STARTS']}.pkl"
    if os.path.isfile(cache_name):
        dataset.load_cached(cache_name)
    else:
        dataset.load_from_files(data_path_weather=None, data_path_darts=None)
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

    print("Model: Stock Data")
    for i, loss in enumerate(losses):
        print(f"Loss for {all_metrices[i]}: {loss}")
