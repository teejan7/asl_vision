import base64
import logging
import pickle
from collections import Counter, deque
from pathlib import Path
from threading import Lock

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from spellchecker import SpellChecker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = "rf_model_68.pkl"
ENCODER_PATH = "label_encoder.pkl"
VOTING_WINDOW = 10
CONF_THRESHOLD = 0.40


def load_artifacts():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not Path(ENCODER_PATH).exists():
        raise FileNotFoundError(f"Encoder not found: {ENCODER_PATH}")

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
    with open(ENCODER_PATH, "rb") as file:
        encoder = pickle.load(file)

    encoder.classes_ = np.array([str(label) for label in encoder.classes_])
    logger.info(f"Model loaded | Classes: {list(encoder.classes_)}")
    return model, encoder


model, encoder = load_artifacts()
INDEX_HTML = Path("index.html").read_text(encoding="utf-8")
spell = SpellChecker()
logger.info("SpellChecker initialized")

prediction_buffer = deque(maxlen=VOTING_WINDOW)
predict_lock = Lock()

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

FINGERTIP_IDS = [4, 8, 12, 16, 20]
FINGER_BASE_IDS = [2, 5, 9, 13, 17]
KNUCKLE_IDS = [3, 6, 10, 14, 18]


def majority_vote(predictions: list) -> str:
    if not predictions:
        return "-"

    filtered = [prediction for prediction in predictions if prediction != "-"]
    if not filtered:
        return "-"

    return Counter(filtered).most_common(1)[0][0]


def extract_features(hand_landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
    coords -= coords[0]
    scale = np.max(np.linalg.norm(coords, axis=1)) + 1e-6
    coords /= scale

    normalized = coords.flatten()
    tip_dists = np.linalg.norm(coords[FINGERTIP_IDS], axis=1)
    extension = np.linalg.norm(coords[FINGERTIP_IDS] - coords[FINGER_BASE_IDS], axis=1)

    inter = []
    for i in range(5):
        for j in range(i + 1, 5):
            inter.append(np.linalg.norm(coords[FINGERTIP_IDS[i]] - coords[FINGERTIP_IDS[j]]))
    inter = np.array(inter, dtype=np.float32)

    angles = []
    for tip, knuckle, base in zip(FINGERTIP_IDS, KNUCKLE_IDS, FINGER_BASE_IDS):
        v1 = coords[tip] - coords[knuckle]
        v2 = coords[base] - coords[knuckle]
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angles.append(float(np.clip(cos_a, -1.0, 1.0)))

    angles.append(float(np.linalg.norm(coords[4] - coords[8])))

    v_mid = coords[9] - coords[0]
    angles.append(float(np.arctan2(v_mid[1], v_mid[0])))

    v1 = coords[5] - coords[0]
    v2 = coords[17] - coords[0]
    angles.append(float(v1[0] * v2[1] - v1[1] * v2[0]))
    angles = np.array(angles, dtype=np.float32)

    return np.concatenate([normalized, tip_dists, extension, inter, angles])


def correct_word(word: str) -> str:
    if len(word) <= 1:
        return word

    word_lower = word.lower()
    corrected = spell.correction(word_lower)
    if corrected and corrected != word_lower:
        logger.info(f"SpellCheck: '{word}' -> '{corrected}'")
        return corrected.upper()

    return word


app = FastAPI(title="ASL Vision API")
app.mount("/static", StaticFiles(directory="."), name="static")


class FrameRequest(BaseModel):
    image: str


class WordRequest(BaseModel):
    word: str


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return HTMLResponse(content=INDEX_HTML)


@app.post("/predict")
async def predict(req: FrameRequest):
    try:
        _header, encoded = req.image.split(",", 1) if "," in req.image else ("", req.image)
        img_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return JSONResponse({"letter": "-", "voted_letter": "-", "hand": False})

        with predict_lock:
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_detector.process(frame_rgb)

            if not results.multi_hand_landmarks:
                prediction_buffer.clear()
                return JSONResponse({"letter": "-", "voted_letter": "-", "hand": False})

            hand_lm = results.multi_hand_landmarks[0].landmark
            features = extract_features(hand_lm)

            probs = model.predict_proba(features.reshape(1, -1))[0]
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            raw_letter = str(encoder.classes_[pred_idx])

            if confidence < CONF_THRESHOLD:
                raw_letter = "-"

            prediction_buffer.append(raw_letter)
            voted_letter = majority_vote(prediction_buffer)

            landmarks = [
                {"x": round(lm.x, 4), "y": round(lm.y, 4)}
                for lm in hand_lm
            ]

        return JSONResponse(
            {
                "letter": raw_letter,
                "voted_letter": voted_letter,
                "hand": True,
                "landmarks": landmarks,
            }
        )
    except Exception as exc:
        logger.error(f"Prediction error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/spellcheck")
async def spellcheck(req: WordRequest):
    original = req.word.strip()
    corrected = correct_word(original)
    return JSONResponse(
        {
            "original": original,
            "corrected": corrected,
            "changed": corrected.upper() != original.upper(),
        }
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "classes": list(encoder.classes_),
        "voting_window": VOTING_WINDOW,
        "conf_threshold": CONF_THRESHOLD,
        "detector_mode": "video",
        "spellcheck": "enabled",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
