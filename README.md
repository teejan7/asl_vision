# ASL Vision - Complete Project Knowledge Base

## What This Project Is

ASL Vision is a real-time American Sign Language recognition web application that runs on a standard laptop CPU. A user signs in front of a webcam, the system predicts the ASL class in real time, and the browser builds words and sentences from stable predictions.

Reference model metrics:

- Accuracy: 98.89%
- Classes: 29 (`A-Z`, `del`, `nothing`, `space`)

Run from the `DEPLOYMENT` folder:

```bash
py -3.10 app.py
```

Open:

```text
http://localhost:7860
```

## Complete File Structure

```text
asl_vision/
|-- DEPLOYMENT/
|   |-- app.py
|   |-- index.html
|   |-- rf_model_68.pkl
|   |-- label_encoder.pkl
|   |-- requirements.txt
|   `-- Dockerfile
|-- TRAINING/
|   |-- train.py
|   |-- config.py
|   |-- data_loader.py
|   |-- feature_extractor.py
|   |-- model_trainer.py
|   |-- model_io.py
|   `-- requirements_train.txt
`-- README.md
```

Dataset folders such as `TRAINING/asl_alphabet_train/` and `TRAINING/asl_alphabet_test/` are expected locally for retraining, but they are not committed to this repo because of size.

The deployment model files are tracked with Git LFS.

## How It Works

```text
User shows hand to webcam
    ->
index.html captures frames from the browser camera
    ->
camera request uses 640x480 input, then frames are resized to 320x240
    ->
JPEG quality 0.5 is used before sending to the backend
    ->
frontend throttles prediction requests to about every 66 ms
    ->
only one /predict request is kept in flight at a time
    ->
app.py decodes base64 to a numpy image
    ->
cv2.flip(frame, 1) mirrors the frame for alignment
    ->
cv2.cvtColor(BGR -> RGB) prepares input for MediaPipe
    ->
MediaPipe detects 21 hand landmarks
    ->
extract_features() builds a 91-dimensional feature vector
    ->
Random Forest predict_proba() produces 29 class probabilities
    ->
confidence filter rejects predictions below 0.40
    ->
10-frame majority voting stabilizes the output
    ->
backend returns {letter, voted_letter, hand, landmarks}
    ->
frontend draws the skeleton and updates the letter display
    ->
stable letters are committed into the current word
    ->
committed words are sent to /spellcheck and added to the sentence
```

## File 1 - `DEPLOYMENT/app.py`

Purpose: FastAPI backend server for prediction, spell correction, and health reporting.

Key constants:

```python
MODEL_PATH = "rf_model_68.pkl"
ENCODER_PATH = "label_encoder.pkl"
VOTING_WINDOW = 10
CONF_THRESHOLD = 0.40
```

Key functions:

```python
def load_artifacts():
    # Loads rf_model_68.pkl and label_encoder.pkl at startup
    # Converts encoder classes to strings

def extract_features(hand_landmarks):
    # Takes 21 MediaPipe landmarks
    # Returns numpy array of shape (91,)

def majority_vote(predictions):
    # Filters out "-" rejected frames
    # Returns the most common stable prediction

@app.post("/predict")
async def predict(req: FrameRequest):
    # 1. Decode base64 -> numpy image
    # 2. Mirror with cv2.flip(frame, 1)
    # 3. Run MediaPipe hand detection
    # 4. Build 91-D feature vector
    # 5. model.predict_proba()
    # 6. Reject if confidence < 0.40
    # 7. Append to a deque-based prediction buffer
    # 8. majority_vote() for stable output
    # 9. Return JSON with letter, voted_letter, and landmarks

@app.post("/spellcheck")
async def spellcheck(req: WordRequest):
    # Uses PySpellChecker to correct committed words

@app.get("/health")
    # Returns runtime health and detector mode
```

MediaPipe setup:

```python
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
```

Optimization notes:

- `index.html` is cached in memory at startup instead of being re-read on every `/` request.
- The voting buffer uses `deque(maxlen=10)` instead of a list with manual `pop(0)`.
- Landmark coordinates are rounded before JSON response to reduce payload size.
- Prediction work is guarded by a lock so the shared detector and buffer do not get corrupted by overlapping requests.

## File 2 - `DEPLOYMENT/index.html`

Purpose: Single-page frontend for camera capture, skeleton drawing, stable letter commits, word building, and sentence output.

Key state values:

```javascript
STABLE_THRESH: 10
COOLDOWN: 18
MIN_PREDICT_INTERVAL: 66
```

Key functions:

```javascript
async function startCamera()
// Requests webcam with ideal 640x480
// Starts the throttled render/predict loop

function scheduleLoop()
// Ensures only one requestAnimationFrame loop is active

function loop(now)
// Updates FPS clock
// Uses one reusable offscreen canvas
// Resizes capture to 320x240
// Encodes as JPEG at quality 0.5
// Sends /predict only if no request is already in flight

function processResult(data)
// If hand:false -> show overlay and reset stability
// If hand:true -> draw skeleton, update letter, update vote bar
// Commits a letter once stability threshold is reached

function drawSkeleton(landmarks)
// Draws cyan connections and pink fingertips
// Uses lm.x * width because the backend already flips the frame

async function commitWord()
// Sends currentWord to /spellcheck
// Adds corrected word to the sentence array

function clearAll()
// Clears word, sentence, counters, and stability UI
```

Optimization notes:

- The frontend no longer creates a new canvas every prediction cycle.
- Only one `/predict` request can be active at a time.
- Prediction calls are throttled, which reduces CPU load and browser-backend congestion.
- Frequently used DOM nodes are cached instead of queried repeatedly every frame.

Encoding and display fixes:

- The no-hand icon uses `&#9995;` instead of a raw emoji character.
- The stability counter is reset centrally with `resetStability()`.

## File 3 - `TRAINING/config.py`

Purpose: Central training configuration.

Current key settings:

```python
DATASET_DIR = "asl_alphabet_train/asl_alphabet_train"
MODEL_OUTPUT_DIR = "models"

MEDIAPIPE_CONFIG = {
    "static_image_mode": True,
    "max_num_hands": 1,
    "min_detection_confidence": 0.5
}

TEST_SIZE = 0.2
RANDOM_STATE = 42
ENABLE_TUNING = True
N_ITER_SEARCH = 30
CV_FOLDS = 5
```

Important note:

- The repository currently has `ENABLE_TUNING = True`.
- If retraining on Google Colab or another limited CPU environment, set `ENABLE_TUNING = False` first.

## File 4 - `TRAINING/data_loader.py`

Purpose: Scan the dataset folders and return image paths plus labels.

Functions:

```python
def get_class_labels(dataset_dir)
def load_image_paths(dataset_dir, class_labels)
def load_dataset(dataset_dir)
```

Expected class count:

- 29 class directories

## File 5 - `TRAINING/feature_extractor.py`

Purpose: Convert raw images into 91-value hand feature vectors for training.

Landmark groups:

```python
FINGERTIP_IDS = [4, 8, 12, 16, 20]
FINGER_BASE_IDS = [2, 5, 9, 13, 17]
KNUCKLE_IDS = [3, 6, 10, 14, 18]
FEATURE_SIZE = 91
```

91-D breakdown:

```text
63 = normalized (x, y, z) coordinates for 21 landmarks
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

Optimization note:

- `build_feature_matrix()` preallocates the output feature array and trims it at the end, which reduces list-growth overhead during full-dataset processing.

## File 6 - `TRAINING/model_trainer.py`

Purpose: Encode labels, split data, train the Random Forest, optionally tune it, and evaluate the result.

Functions:

```python
def encode_labels(string_labels)
def split_data(X, y, test_size=0.2)
def build_model(hyperparams)
def train_model(model, X_train, y_train)
def tune_model(...)
def evaluate_model(model, X_test, y_test, encoder)
```

Notes:

- `split_data()` uses a stratified train/test split.
- `evaluate_model()` returns accuracy plus a full classification report.

## File 7 - `TRAINING/model_io.py`

Purpose: Save and load the trained model artifacts.

```python
MODEL_FILENAME = "rf_model_68.pkl"
ENCODER_FILENAME = "label_encoder.pkl"
```

Saved files:

- `rf_model_68.pkl`
- `label_encoder.pkl`

Current deployment artifact sizes:

- `rf_model_68.pkl`: about 188 MB
- `label_encoder.pkl`: about 1 KB

## File 8 - `TRAINING/train.py`

Purpose: End-to-end training pipeline runner.

Pipeline:

```python
# Step 1: load_dataset()
# Step 2: build_feature_matrix()
# Step 3: encode_labels() + split_data()
# Step 4: tune_model() or train_model()
# Step 5: evaluate_model()
# Step 6: save_artifacts()
```

Run from `TRAINING`:

```bash
python train.py
```

## File 9 - `DEPLOYMENT/requirements.txt`

```text
scikit-learn==1.6.1
opencv-python-headless
mediapipe==0.10.13
fastapi
uvicorn
numpy
pyspellchecker
```

Compatibility note:

- `scikit-learn==1.6.1` should match the version used for the serialized model.
- `mediapipe==0.10.13` is the version pinned by the deployment app.

## API Endpoints

| Endpoint | Method | Input | Output |
|---|---|---|---|
| `/` | GET | none | `index.html` |
| `/predict` | POST | `{image: base64_string}` | `{letter, voted_letter, hand, landmarks}` |
| `/spellcheck` | POST | `{word: string}` | `{original, corrected, changed}` |
| `/health` | GET | none | `{status, classes, voting_window, conf_threshold, detector_mode, spellcheck}` |

## Known Issues and Fixes

| Item | Location | Status |
|---|---|---|
| Emoji mojibake in the no-hand overlay | `DEPLOYMENT/index.html` | Fixed with `&#9995;` |
| Skeleton offset from mirrored video | `DEPLOYMENT/index.html` | Fixed by using `lm.x * width` |
| Stability counter overflow display | `DEPLOYMENT/index.html` | Fixed by central stability reset logic |
| Slow browser/backend request pile-up | `DEPLOYMENT/index.html` | Improved with throttling and one in-flight request |
| Slower MediaPipe frame-by-frame detection | `DEPLOYMENT/app.py` | Improved by switching deployment to video mode |
| Long Colab training runs | `TRAINING/config.py` | Set `ENABLE_TUNING = False` before retraining |

## How To Run Locally

From `DEPLOYMENT`:

```bash
pip install -r requirements.txt
py -3.10 app.py
```

Open:

```text
http://localhost:7860
http://localhost:7860/health
```

## How To Retrain On Colab

Recommended steps:

```python
# 1. Set ENABLE_TUNING = False in config.py
# 2. Run:
!python train.py

# 3. Save artifacts immediately
import os
import shutil

os.makedirs("/content/drive/MyDrive/asl_models", exist_ok=True)
shutil.copy("models/rf_model_68.pkl", "/content/drive/MyDrive/asl_models/")
shutil.copy("models/label_encoder.pkl", "/content/drive/MyDrive/asl_models/")
```
