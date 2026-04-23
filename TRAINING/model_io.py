import logging
import pickle
from pathlib import Path
from typing import Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

MODEL_FILENAME = "rf_model_68.pkl"
ENCODER_FILENAME = "label_encoder.pkl"


def save_artifacts(
    model: RandomForestClassifier,
    encoder: LabelEncoder,
    output_dir: str = ".",
) -> Tuple[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_path = out / MODEL_FILENAME
    encoder_path = out / ENCODER_FILENAME

    with open(model_path, "wb") as file:
        pickle.dump(model, file)
    logger.info(f"Model saved to {model_path}")

    with open(encoder_path, "wb") as file:
        pickle.dump(encoder, file)
    logger.info(f"Label encoder saved to {encoder_path}")

    return str(model_path), str(encoder_path)


def load_artifacts(model_dir: str = ".") -> Tuple[RandomForestClassifier, LabelEncoder]:
    model_path = Path(model_dir) / MODEL_FILENAME
    encoder_path = Path(model_dir) / ENCODER_FILENAME

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder file not found: {encoder_path}")

    with open(model_path, "rb") as file:
        model = pickle.load(file)
    logger.info(f"Model loaded from {model_path}")

    with open(encoder_path, "rb") as file:
        encoder = pickle.load(file)
    logger.info(f"Label encoder loaded from {encoder_path}")

    return model, encoder
