import os
from utils.OurNBeatsModel import OurNBeatsModel
from utils.OurDataset import OurDataset
from utils.metric import *


model_name = input(f"insert the model name {os.listdir('darts_logs')}: ")
sample_size = float(input("insert sample size: "))
BREAK_BETWEEN_WINDOW_STARTS = 100
window_size = int(input("insert WINDOW_SIZE: "))

parameters = {
    # Data Structure
    "TIMESERIES_LENGTH": 14 + window_size,
    "WINDOW_SIZE": window_size,
    "BREAK_BETWEEN_WINDOW_STARTS": BREAK_BETWEEN_WINDOW_STARTS,
    "VAL_SHARE": 0.5,
    "MAX_SAMPLES_PER_TS": 1,  # fix for this custom data selection approach
    "ENRICH_RATIO_WEATHER": 0,
    "ENRICH_RATIO_DARTS": 0,
    "SAMPLE_SIZE": 0.01,
    "CPUS": os.cpu_count(),
    "MODEL_NAME": model_name,
}

dataset = OurDataset(parameters)
dataset.load_from_files(data_path_weather=None, data_path_darts=None)
train, val = dataset.get_datasets()


model = OurNBeatsModel(parameters, train_dataset=train, val_dataset=val, verbose=True)
model.load_model()

model.save_prediction()

print("getting losses.")

losses = model.get_test_loss(
    loss_functions=all_metrices,
    BREAK_BETWEEN_WINDOW_STARTS=BREAK_BETWEEN_WINDOW_STARTS,
    sample_size=sample_size,
)
print("Model:", model_name)

for i, loss in enumerate(losses):
    print(f"Loss for {all_metrices[i]}: {loss}")
