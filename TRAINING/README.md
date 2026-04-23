# Training

## Purpose

This folder contains the training pipeline for the ASL Vision model. It scans the dataset, extracts 91-dimensional hand features using MediaPipe, trains a Random Forest classifier, evaluates the result, and saves the model artifacts.

## Files

```text
TRAINING/
|-- train.py
|-- config.py
|-- data_loader.py
|-- feature_extractor.py
|-- model_trainer.py
|-- model_io.py
`-- requirements_train.txt
```

Expected local-only dataset folders:

```text
TRAINING/asl_alphabet_train/
TRAINING/asl_alphabet_test/
```

These are intentionally not committed to the repo.

## Training Workflow

```text
Dataset folders
  -> data_loader.py scans class folders and image paths
  -> feature_extractor.py runs MediaPipe on each image
  -> 91-D feature matrix is built
  -> labels are encoded
  -> train/test split is created
  -> Random Forest is trained or tuned
  -> evaluation report is generated
  -> rf_model_68.pkl and label_encoder.pkl are saved
```

## `config.py`

Purpose: Central configuration for training.

Key settings:

```python
DATASET_DIR = "asl_alphabet_train/asl_alphabet_train"
MODEL_OUTPUT_DIR = "models"
TEST_SIZE = 0.2
RANDOM_STATE = 42
ENABLE_TUNING = True
N_ITER_SEARCH = 30
CV_FOLDS = 5
```

Important note:

- Set `ENABLE_TUNING = False` when training on Google Colab or a limited CPU environment.

## `data_loader.py`

Purpose: Discover class folders and collect aligned image paths and labels.

Functions:

```python
def get_class_labels(dataset_dir)
def load_image_paths(dataset_dir, class_labels)
def load_dataset(dataset_dir)
```

## `feature_extractor.py`

Purpose: Convert a hand image into a 91-value feature vector.

Landmark groups:

```python
FINGERTIP_IDS = [4, 8, 12, 16, 20]
FINGER_BASE_IDS = [2, 5, 9, 13, 17]
KNUCKLE_IDS = [3, 6, 10, 14, 18]
FEATURE_SIZE = 91
```

91-D breakdown:

```text
63 = normalized (x, y, z) landmark coordinates
 5 = wrist-to-fingertip distances
 5 = fingertip-to-base distances
10 = pairwise fingertip distances
 5 = finger bend cosine values
 1 = thumb-index spread
 1 = middle-finger orientation angle
 1 = palm normal value
----
91 total
```

Key functions:

```python
def _build_coords(landmarks)
def extract_features_from_image(image_path, hands_detector)
def build_feature_matrix(image_paths, string_labels)
```

Implementation note:

- `build_feature_matrix()` preallocates the output array to reduce overhead on large runs.

## `model_trainer.py`

Purpose: Encode labels, split data, train the classifier, optionally tune it, and evaluate it.

Functions:

```python
def encode_labels(string_labels)
def split_data(X, y, test_size=0.2)
def build_model(hyperparams)
def train_model(model, X_train, y_train)
def tune_model(...)
def evaluate_model(model, X_test, y_test, encoder)
```

## `model_io.py`

Purpose: Save and load trained artifacts.

Artifact filenames:

```python
MODEL_FILENAME = "rf_model_68.pkl"
ENCODER_FILENAME = "label_encoder.pkl"
```

## `train.py`

Purpose: Entry point for the full training pipeline.

Pipeline order:

```python
# Step 1: load_dataset()
# Step 2: build_feature_matrix()
# Step 3: encode_labels() + split_data()
# Step 4: tune_model() or train_model()
# Step 5: evaluate_model()
# Step 6: save_artifacts()
```

## Run Training

```bash
pip install -r requirements_train.txt
python train.py
```

## Colab Note

Recommended before running:

```python
ENABLE_TUNING = False
```

Then save the produced model files immediately after training.
