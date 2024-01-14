"""This modulte created datasets on the file system that the HyperparameterTuner can use to spped up data loading"""
from utils.OurDataset import OurDataset


FORWARD_WINDOW = 14
window_sizes = [252, 504, 1000, 1500, 2000, 3000]

for enrich_ratio in [0, 1, 2, 3, 4, 5, 6]:
    print("        ENRICH RATIO:", enrich_ratio)
    for WINDOW_SIZE in window_sizes:
        print("    WINDOW_SIZE", WINDOW_SIZE)
        TIMESERIES_LENGTH = WINDOW_SIZE + FORWARD_WINDOW
        parameters_data = {
            "BREAK_BETWEEN_WINDOW_STARTS": 300,
            "VAL_SHARE": 0.3,  # percent of timeseries to be choosen for validation set
            "ENRICH_RATIO_WEATHER": enrich_ratio / 10,
            "ENRICH_RATIO_DARTS": enrich_ratio / 10,
            "TIMESERIES_LENGTH": TIMESERIES_LENGTH,
            "SAMPLE_SIZE": 0.05,
            "WINDOW_SIZE": WINDOW_SIZE,
        }

        dataset = OurDataset(parameters_data)
        if enrich_ratio:
            dataset.load_from_files()
        else:
            dataset.load_from_files(data_path_weather=None, data_path_darts=None)

        dataset.save_datasets(
            f"cache/tuner_all_enrich_{enrich_ratio}_break_{parameters_data['BREAK_BETWEEN_WINDOW_STARTS']}_sample_{parameters_data['SAMPLE_SIZE']}_window_{WINDOW_SIZE}.pkl"
        )
