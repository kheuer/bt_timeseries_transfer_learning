"""this module obtains losses of all models on all metrics, this is neccessary to produce plots"""
import os
import json
import sys
from tqdm import tqdm
import numpy as np
import torch
from utils.OurNBeatsModel import OurNBeatsModel
from utils.OurDataset import OurDataset
from utils.metric import *

BREAK_BETWEEN_WINDOW_STARTS = 300
WINDOW_SIZE = 504
sample_size = 1

parameters = {
    # Data Structure
    "TIMESERIES_LENGTH": 14 + WINDOW_SIZE,
    "WINDOW_SIZE": WINDOW_SIZE,
    "BREAK_BETWEEN_WINDOW_STARTS": BREAK_BETWEEN_WINDOW_STARTS,
    "VAL_SHARE": 0,
    "MAX_SAMPLES_PER_TS": 1,  # fix for this custom data selection approach
    "ENRICH_RATIO_WEATHER": 0,
    "ENRICH_RATIO_DARTS": 0,
    "SAMPLE_SIZE": sample_size,
}

dataset = OurDataset(parameters)

cache_name = f"cache/PRIVATE_validation_break_{BREAK_BETWEEN_WINDOW_STARTS}_sample_{sample_size}_length_{parameters['TIMESERIES_LENGTH']}.pkl"
if os.path.isfile(cache_name):
    dataset.load_cached(cache_name)
else:
    print("load test data from files")
    dataset.load_from_files(
        data_path_stocks="data/processed/test/",
        data_path_weather=None,
        data_path_darts=None,
    )
    print("save test data to cache:", cache_name)
    dataset.save_datasets(cache_name)

test, _ = dataset.get_datasets()
normalization_factors_val, _ = dataset.get_normalization_factors()

del dataset
dataset_size = sys.getsizeof(test)
print(f"Size of test dataset: {dataset_size/(1e6)} MB")
print("Calculating test loss.")

data = {}


for model_name in [
    "best_stock",
    "best_enrich_10",
    "best_enrich_40",
    "best_enrich_10_without_darts",
    "best_enrich_40_without_darts",
]:
    parameters["MODEL_NAME"] = model_name
    data[model_name] = {}

    model = OurNBeatsModel(
        parameters, train_dataset=True, val_dataset=True, verbose=True
    )
    model.load_model()

    features_all = []
    targets_all = []

    for i in range(len(test)):
        features, targets = (
            test[i][: -model.FORWARD_WINDOW],
            test[i][-model.FORWARD_WINDOW :],
        )
        features_all.append(features)
        targets_all.append(targets)

    pred = model.predict(model.FORWARD_WINDOW, features_all)

    losses = []

    for i in tqdm(range(len(pred))):
        losses.append([])
        for j, loss_fn in enumerate(all_metrices[:-1]):
            single_target = torch.Tensor(np.array([targets_all[i].values()]))
            single_pred = torch.Tensor(np.array([pred[i].values()]))

            if hasattr(loss_fn, "pass_features_and_normal_factors"):
                loss = loss_fn(
                    single_pred,
                    single_target,
                    [features_all[i]],
                    [normalization_factors_val[i]],
                )

            else:
                loss = loss_fn(single_pred, single_target)
            losses[j].append(float(loss))

    data[model_name] = losses

    with open("results/losses.json", "w") as file:
        json.dump(data, file)
