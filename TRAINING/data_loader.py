import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


def get_class_labels(dataset_dir: str) -> List[str]:
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    labels = sorted([d.name for d in dataset_path.iterdir() if d.is_dir()])

    if not labels:
        raise ValueError(f"No subdirectories found in: {dataset_dir}")

    logger.info(f"Found {len(labels)} classes: {labels}")
    return labels


def load_image_paths(
    dataset_dir: str,
    class_labels: List[str],
    valid_extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
) -> Tuple[List[str], List[str]]:
    dataset_path = Path(dataset_dir)
    image_paths: List[str] = []
    labels: List[str] = []

    for label in class_labels:
        class_dir = dataset_path / label
        if not class_dir.exists():
            logger.warning(f"Class folder not found, skipping: {class_dir}")
            continue

        count = 0
        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() in valid_extensions:
                image_paths.append(str(img_file))
                labels.append(label)
                count += 1

        logger.info(f"[{label}] {count} images loaded")

    logger.info(f"Total images collected: {len(image_paths)}")
    return image_paths, labels


def load_dataset(dataset_dir: str) -> Tuple[List[str], List[str], List[str]]:
    class_labels = get_class_labels(dataset_dir)
    image_paths, labels = load_image_paths(dataset_dir, class_labels)
    return image_paths, labels, class_labels
