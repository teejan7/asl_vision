import logging
import time

import config
from data_loader import load_dataset
from feature_extractor import build_feature_matrix
from model_io import save_artifacts
from model_trainer import (
    build_model,
    encode_labels,
    evaluate_model,
    split_data,
    train_model,
    tune_model,
)

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATE_FORMAT,
)
logger = logging.getLogger(__name__)


def main():
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("ASL Sign Language - Training Pipeline")
    logger.info("=" * 60)

    logger.info("\n[STEP 1/6] Loading dataset")
    image_paths, string_labels, class_labels = load_dataset(config.DATASET_DIR)
    logger.info(f"Loaded {len(image_paths)} images across {len(class_labels)} classes")

    logger.info("\n[STEP 2/6] Extracting 91-D geometric features via MediaPipe")
    logger.info("This is the slowest step.")
    x_data, y_strings, skipped = build_feature_matrix(
        image_paths,
        string_labels,
        static_image_mode=config.MEDIAPIPE_CONFIG["static_image_mode"],
        max_num_hands=config.MEDIAPIPE_CONFIG["max_num_hands"],
        min_detection_confidence=config.MEDIAPIPE_CONFIG["min_detection_confidence"],
    )
    logger.info(f"Feature matrix shape: {x_data.shape} | Skipped: {skipped}")

    logger.info("\n[STEP 3/6] Encoding labels and splitting data")
    y_encoded, encoder = encode_labels(y_strings)
    x_train, x_test, y_train, y_test = split_data(
        x_data,
        y_encoded,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
    )

    if config.ENABLE_TUNING:
        logger.info("\n[STEP 4/6] Tuning Random Forest with RandomizedSearchCV")
        model = tune_model(
            x_train,
            y_train,
            param_dist=config.TUNING_PARAM_DIST,
            n_iter=config.N_ITER_SEARCH,
            cv_folds=config.CV_FOLDS,
            random_state=config.RANDOM_STATE,
        )
    else:
        logger.info("\n[STEP 4/6] Training Random Forest")
        model = build_model(config.RANDOM_FOREST_PARAMS)
        model = train_model(model, x_train, y_train)

    logger.info("\n[STEP 5/6] Evaluating on test set")
    results = evaluate_model(model, x_test, y_test, encoder)
    logger.info(f"Final Test Accuracy: {results['accuracy'] * 100:.2f}%")

    logger.info("\n[STEP 6/6] Saving model and encoder")
    model_path, encoder_path = save_artifacts(
        model,
        encoder,
        output_dir=config.MODEL_OUTPUT_DIR,
    )

    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("Training pipeline complete")
    logger.info(f"Total time: {elapsed / 60:.1f} minutes")
    logger.info(f"Accuracy: {results['accuracy'] * 100:.2f}%")
    logger.info(f"Model saved: {model_path}")
    logger.info(f"Encoder saved: {encoder_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
