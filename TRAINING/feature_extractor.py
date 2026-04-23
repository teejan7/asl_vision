import logging
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

_mp_hands = mp.solutions.hands

WRIST_INDEX = 0
FEATURE_SIZE = 91
FINGERTIP_IDS = [4, 8, 12, 16, 20]
FINGER_BASE_IDS = [2, 5, 9, 13, 17]
KNUCKLE_IDS = [3, 6, 10, 14, 18]


def _build_coords(landmarks) -> np.ndarray:
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    coords -= coords[WRIST_INDEX]
    scale = np.max(np.linalg.norm(coords, axis=1)) + 1e-6
    coords /= scale
    return coords


def _normalized_coords(coords: np.ndarray) -> np.ndarray:
    return coords.flatten()


def _fingertip_distances(coords: np.ndarray) -> np.ndarray:
    return np.linalg.norm(coords[FINGERTIP_IDS], axis=1)


def _finger_extension(coords: np.ndarray) -> np.ndarray:
    return np.linalg.norm(coords[FINGERTIP_IDS] - coords[FINGER_BASE_IDS], axis=1)


def _inter_fingertip_distances(coords: np.ndarray) -> np.ndarray:
    dists = []
    for i in range(len(FINGERTIP_IDS)):
        for j in range(i + 1, len(FINGERTIP_IDS)):
            dists.append(np.linalg.norm(coords[FINGERTIP_IDS[i]] - coords[FINGERTIP_IDS[j]]))
    return np.array(dists, dtype=np.float32)


def _joint_angles(coords: np.ndarray) -> np.ndarray:
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

    return np.array(angles, dtype=np.float32)


def extract_features_from_image(image_path: str, hands_detector) -> Optional[np.ndarray]:
    img = cv2.imread(image_path)
    if img is None:
        logger.debug(f"Could not read image: {image_path}")
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(img_rgb)

    if not results.multi_hand_landmarks:
        img_flipped = cv2.flip(img, 1)
        results = hands_detector.process(cv2.cvtColor(img_flipped, cv2.COLOR_BGR2RGB))
        if not results.multi_hand_landmarks:
            return None

    hand_landmarks = results.multi_hand_landmarks[0].landmark
    coords = _build_coords(hand_landmarks)

    feature_vector = np.concatenate(
        [
            _normalized_coords(coords),
            _fingertip_distances(coords),
            _finger_extension(coords),
            _inter_fingertip_distances(coords),
            _joint_angles(coords),
        ]
    )

    return feature_vector.astype(np.float32)


def build_feature_matrix(
    image_paths: list,
    string_labels: list,
    static_image_mode: bool = True,
    max_num_hands: int = 1,
    min_detection_confidence: float = 0.5,
):
    x_data = np.empty((total := len(image_paths), FEATURE_SIZE), dtype=np.float32)
    y_data = []
    skipped = 0
    valid_count = 0

    with _mp_hands.Hands(
        static_image_mode=static_image_mode,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
    ) as hands:
        for idx, (path, label) in enumerate(zip(image_paths, string_labels)):
            features = extract_features_from_image(path, hands)
            if features is not None:
                x_data[valid_count] = features
                y_data.append(label)
                valid_count += 1
            else:
                skipped += 1

            if (idx + 1) % 1000 == 0:
                logger.info(f"Processed {idx + 1}/{total} images")

    logger.info(f"Feature extraction complete: {valid_count} valid, {skipped} skipped")
    return x_data[:valid_count], y_data, skipped
