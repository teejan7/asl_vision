# Deployment

## Purpose

This folder contains the runnable ASL Vision web application. It serves the frontend, loads the trained model, receives webcam frames, predicts ASL signs in real time, and builds words and sentences in the browser.

## Files

```text
DEPLOYMENT/
|-- app.py
|-- index.html
|-- rf_model_68.pkl
|-- label_encoder.pkl
|-- requirements.txt
`-- Dockerfile
```

## Runtime Workflow

```text
Browser camera
  -> capture at 640x480 input
  -> resize to 320x240
  -> JPEG quality 0.5
  -> POST /predict
  -> base64 decode in app.py
  -> cv2.flip(frame, 1)
  -> MediaPipe hand detection
  -> 91-D feature extraction
  -> Random Forest predict_proba()
  -> confidence threshold at 0.40
  -> 10-frame majority vote
  -> JSON response with letter and landmarks
  -> frontend skeleton + text update
```

## `app.py`

Purpose: FastAPI backend for prediction, spell correction, and health reporting.

Important constants:

```python
MODEL_PATH = "rf_model_68.pkl"
ENCODER_PATH = "label_encoder.pkl"
VOTING_WINDOW = 10
CONF_THRESHOLD = 0.40
```

Important functions:

```python
def load_artifacts()
def extract_features(hand_landmarks)
def majority_vote(predictions)

@app.post("/predict")
@app.post("/spellcheck")
@app.get("/health")
```

MediaPipe configuration:

```python
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
```

Implementation notes:

- The backend caches `index.html` in memory.
- The prediction buffer uses `deque(maxlen=10)`.
- A lock protects shared prediction state.
- Landmark values are rounded before being returned.

## `index.html`

Purpose: Single-page frontend for camera capture, UI rendering, stability logic, and sentence building.

Important state values:

```javascript
STABLE_THRESH: 10
COOLDOWN: 18
MIN_PREDICT_INTERVAL: 66
```

Important functions:

```javascript
async function startCamera()
function scheduleLoop()
function loop(now)
function processResult(data)
function drawSkeleton(landmarks)
async function commitWord()
function clearAll()
```

Frontend optimization details:

- One reusable offscreen canvas is used for capture.
- Only one `/predict` request is allowed in flight at a time.
- Prediction requests are throttled to reduce CPU load.
- Frequently used DOM elements are cached.
- The no-hand icon uses `&#9995;` to avoid encoding issues.

Skeleton drawing detail:

- `drawSkeleton()` uses `lm.x * width`, not `(1 - lm.x) * width`, because the backend already mirrors the frame.

## API Endpoints

| Endpoint | Method | Output |
|---|---|---|
| `/` | GET | app UI |
| `/predict` | POST | `{letter, voted_letter, hand, landmarks}` |
| `/spellcheck` | POST | `{original, corrected, changed}` |
| `/health` | GET | `{status, classes, voting_window, conf_threshold, detector_mode, spellcheck}` |

## Run Locally

```bash
pip install -r requirements.txt
py -3.10 app.py
```

Open:

```text
http://localhost:7860
```

## Compatibility

```text
scikit-learn==1.6.1
mediapipe==0.10.13
```

The deployment model files are tracked with Git LFS.
