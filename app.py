"""
app.py
------
FastAPI backend for ASL Sign Language Recognition.
Features:
  - MediaPipe hand landmark extraction
  - 91-D geometric feature vector
  - Random Forest Classifier
  - Majority Voting (temporal stabilization)
  - Confidence Filtering
  - PySpellChecker (post-recognition correction)
"""

import os
import base64
import pickle
import logging
import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path
from collections import Counter
from spellchecker import SpellChecker
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Load Model ─────────────────────────────────────────────────────────────────

MODEL_PATH   = "rf_model_68.pkl"
ENCODER_PATH = "label_encoder.pkl"

def load_artifacts():
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not Path(ENCODER_PATH).exists():
        raise FileNotFoundError(f"Encoder not found: {ENCODER_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    encoder.classes_ = np.array([str(c) for c in encoder.classes_])
    logger.info(f"Model loaded | Classes: {list(encoder.classes_)}")
    return model, encoder

model, encoder = load_artifacts()

# ── Spell Checker ──────────────────────────────────────────────────────────────

spell = SpellChecker()
logger.info("SpellChecker initialized")

# ── Majority Voting Buffer ─────────────────────────────────────────────────────

VOTING_WINDOW    = 8     # frames in voting buffer
CONF_THRESHOLD   = 0.40   # minimum confidence to accept prediction
prediction_buffer = []

def majority_vote(predictions: list) -> str:
    if not predictions:
        return "–"
    filtered = [p for p in predictions if p != "–"]
    if not filtered:
        return "–"
    return Counter(filtered).most_common(1)[0][0]

# ── MediaPipe ──────────────────────────────────────────────────────────────────

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

FINGERTIP_IDS   = [4, 8, 12, 16, 20]
FINGER_BASE_IDS = [2, 5,  9, 13, 17]
KNUCKLE_IDS     = [3, 6, 10, 14, 18]

# ── 91-D Feature Extraction ────────────────────────────────────────────────────

def extract_features(hand_landmarks):
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32
    )
    coords -= coords[0]
    scale = np.max(np.linalg.norm(coords, axis=1)) + 1e-6
    coords /= scale

    normalized = coords.flatten()                                              # 63
    tip_dists  = np.linalg.norm(coords[FINGERTIP_IDS], axis=1)                # 5
    extension  = np.linalg.norm(
        coords[FINGERTIP_IDS] - coords[FINGER_BASE_IDS], axis=1)              # 5

    inter = []
    for i in range(5):
        for j in range(i + 1, 5):
            inter.append(np.linalg.norm(
                coords[FINGERTIP_IDS[i]] - coords[FINGERTIP_IDS[j]]))
    inter = np.array(inter, dtype=np.float32)                                 # 10

    angles = []
    for tip, knuckle, base in zip(FINGERTIP_IDS, KNUCKLE_IDS, FINGER_BASE_IDS):
        v1 = coords[tip] - coords[knuckle]
        v2 = coords[base] - coords[knuckle]
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angles.append(float(np.clip(cos_a, -1.0, 1.0)))
    angles.append(float(np.linalg.norm(coords[4] - coords[8])))
    v_mid = coords[9] - coords[0]
    angles.append(float(np.arctan2(v_mid[1], v_mid[0])))
    v1, v2 = coords[5] - coords[0], coords[17] - coords[0]
    angles.append(float(v1[0] * v2[1] - v1[1] * v2[0]))
    angles = np.array(angles, dtype=np.float32)                               # 8

    return np.concatenate([normalized, tip_dists, extension, inter, angles])  # 91

# ── Spell Correction ───────────────────────────────────────────────────────────

def correct_word(word: str) -> str:
    if len(word) <= 1:
        return word
    word_lower = word.lower()
    corrected  = spell.correction(word_lower)
    if corrected and corrected != word_lower:
        logger.info(f"SpellCheck: '{word}' → '{corrected}'")
        return corrected.upper()
    return word

# ── FastAPI App ────────────────────────────────────────────────────────────────

app = FastAPI(title="ASL Vision API")
app.mount("/static", StaticFiles(directory="."), name="static")


class FrameRequest(BaseModel):
    image: str

class WordRequest(BaseModel):
    word: str


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.post("/predict")
async def predict(req: FrameRequest):
    global prediction_buffer
    try:
        header, encoded = req.image.split(",", 1) if "," in req.image else ("", req.image)
        img_bytes = base64.b64decode(encoded)
        np_arr    = np.frombuffer(img_bytes, dtype=np.uint8)
        frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return JSONResponse({"letter": "–", "voted_letter": "–", "hand": False})

        frame     = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = hands_detector.process(frame_rgb)

        if not results.multi_hand_landmarks:
            prediction_buffer = []
            return JSONResponse({"letter": "–", "voted_letter": "–", "hand": False})

        hand_lm  = results.multi_hand_landmarks[0].landmark
        features = extract_features(hand_lm)

        # Raw prediction + confidence filter
        probs      = model.predict_proba([features])[0]
        pred_idx   = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        raw_letter = str(encoder.classes_[pred_idx])

        if confidence < CONF_THRESHOLD:
            raw_letter = "–"

        # Majority voting
        prediction_buffer.append(raw_letter)
        if len(prediction_buffer) > VOTING_WINDOW:
            prediction_buffer.pop(0)
        voted_letter = majority_vote(prediction_buffer)

        landmarks = [{"x": lm.x, "y": lm.y} for lm in hand_lm]

        return JSONResponse({
            "letter":       raw_letter,
            "voted_letter": voted_letter,
            "hand":         True,
            "landmarks":    landmarks
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/spellcheck")
async def spellcheck(req: WordRequest):
    """Called when user commits a word — returns spell-corrected version."""
    original  = req.word.strip()
    corrected = correct_word(original)
    return JSONResponse({
        "original":  original,
        "corrected": corrected,
        "changed":   corrected.upper() != original.upper()
    })


@app.get("/health")
async def health():
    return {
        "status":        "ok",
        "classes":       list(encoder.classes_),
        "voting_window": VOTING_WINDOW,
        "conf_threshold": CONF_THRESHOLD,
        "spellcheck":    "enabled"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
