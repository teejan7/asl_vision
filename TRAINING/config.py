import os

DATASET_DIR = os.path.join("asl_alphabet_train", "asl_alphabet_train")
MODEL_OUTPUT_DIR = "models"

MEDIAPIPE_CONFIG = {
    "static_image_mode": True,
    "max_num_hands": 1,
    "min_detection_confidence": 0.5,
}

TEST_SIZE = 0.2
RANDOM_STATE = 42

RANDOM_FOREST_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
    "verbose": 1,
}

ENABLE_TUNING = True
N_ITER_SEARCH = 30
CV_FOLDS = 5

TUNING_PARAM_DIST = {
    "n_estimators": [100, 200, 300, 400, 500],
    "max_depth": [None, 20, 30, 40],
    "min_samples_split": [2, 4, 6, 8],
    "min_samples_leaf": [1, 2, 3],
    "max_features": ["sqrt", "log2", 0.3],
    "class_weight": ["balanced", None],
    "bootstrap": [True, False],
}

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
