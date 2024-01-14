"""this module optimizes the hyperparameters"""
import optuna
import os
import time
from torch.nn import HuberLoss

from utils.OurDataset import OurDataset
from utils.OurNBeatsModel import OurNBeatsModel
from utils.metric import *

# set this to true or false, this changes the data to be enriched or not
enrich = True

if enrich:
    STUDY_NAME = f"ALL_DATA_STUDY_{time.time()}"
    model_name = "tuning_all_data"
else:
    STUDY_NAME = f"STOCK_DATA_STUDY_{time.time()}"
    model_name = "tuning_stock_data"


def objective(trial):
    """
    Objective function for Optuna to optimize hyperparameters.

    Args:
    - trial: optuna.trial.Trial object

    Returns:
    - Loss value to minimize
    """

    WINDOW_SIZE = trial.suggest_categorical(
        "window_size", [252, 504, 1000, 1500, 2000, 3000]
    )

    TIMESERIES_LENGTH = WINDOW_SIZE + 14

    parameters = {
        # MODEL
        "TIMESERIES_LENGTH": TIMESERIES_LENGTH,
        "WINDOW_SIZE": WINDOW_SIZE,
        "EPOCHS": 20,
        "LEARNING_RATE": trial.suggest_float("learning_rate", 1e-6, 1e-2),
        "BATCH_SIZE": 100,
        "NUM_STACKS": trial.suggest_int("num_stacks", 4, 20),
        "NUM_BLOCKS": trial.suggest_int("num_blocks", 1, 8),
        "NUM_LAYERS": trial.suggest_int("num_layers", 4, 6),
        "EXPANSION_COEFFICIENT_DIMENSION": trial.suggest_int(
            "expansion_coefficient_dimension", 3, 8
        ),
        "LAYER_WIDTHS": trial.suggest_int("layer_widths", 128, 512),
        "LOSS_FN": HuberLoss(delta=0.3),
        "OPTIMIZER": "Adam",
        "DROPOUT": trial.suggest_float("dropout", 0, 0.5),
        "EARLYSTOP_PATIENCE": 10,
        "EARLYSTOP_MIN_DELTA": 0.005,
        # can be [ReLU,RReLU, PReLU, Softplus, Tanh, SELU, LeakyReLU, Sigmoid]
        "ACTIVATION_FN": trial.suggest_categorical(
            "activation", ["ReLU", "SELU", "Sigmoid"]
        ),
        "CALLBACKS": [],  # callback,
        # Other
        "CPUS": os.cpu_count(),
        "MODEL_NAME": model_name,
    }

    if enrich:
        enrich_ratio = trial.suggest_int(
            "enrich_ratio", 1, 6
        )  # trial result is in percent

    else:
        enrich_ratio = 0

    parameters_data = {
        "BREAK_BETWEEN_WINDOW_STARTS": 300,
        "VAL_SHARE": 0.3,  # percent of timeseries to be choosen for validation set
        "ENRICH_RATIO_WEATHER": enrich_ratio / 10,
        "ENRICH_RATIO_DARTS": enrich_ratio / 10,
        "TIMESERIES_LENGTH": TIMESERIES_LENGTH,
        "WINDOW_SIZE": WINDOW_SIZE,
        "SAMPLE_SIZE": 0.05,
    }

    dataset = OurDataset(parameters_data)
    dataset.load_cached(
        f"cache/tuner_all_enrich_{enrich_ratio}_break_{parameters_data['BREAK_BETWEEN_WINDOW_STARTS']}_sample_{parameters_data['SAMPLE_SIZE']}_window_{WINDOW_SIZE}.pkl"
    )
    train, val = dataset.get_datasets()

    model = OurNBeatsModel(
        params=parameters, train_dataset=train, val_dataset=val, verbose=True
    )
    model.setup_new_model()

    model.train()

    model.save_prediction(n_plots=5)

    model.WINDOW_SIZE = 3000  # this should be consistant and cannot be shorter than the WINDOW_SIZE the model was trained on
    model.TIMESERIES_LENGTH = model.WINDOW_SIZE + model.FORWARD_WINDOW

    losses = model.get_test_loss(
        all_metrices,
        BREAK_BETWEEN_WINDOW_STARTS=300,
        sample_size=parameters_data["SAMPLE_SIZE"],
    )

    messages = ["#####################################################\n"]
    for i, loss in enumerate(losses):
        messages.append(f"Loss for {all_metrices[i]}: {loss}\n")

    for msg in messages:
        with open(f"optuna_logs/{STUDY_NAME}.txt", "a") as file:
            file.write(msg)
        print(msg)

    return losses[0]  # MAE Loss


def print_callback(study, trial):
    """Print trial information to the console"""
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(
        f"     Best value: {study.best_trial.value}, Best params: {study.best_trial.params}"
    )
    print("#####################################################")


def write_callback(study, trial):
    """Write trial information to a file"""
    with open(f"optuna_logs/{STUDY_NAME}.txt", "a") as file:
        file.write(f"Current value: {trial.value}, Current params: {trial.params}\n")
        file.write(
            f"     Best value: {study.best_trial.value}, Best params: {study.best_trial.params}\n\n"
        )
        file.write("#####################################################\n")


if __name__ == "__main__":
    # Optimize the study using objective function
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
        study_name=STUDY_NAME,
    )
    study.optimize(
        objective, gc_after_trial=True, callbacks=[print_callback, write_callback]
    )
