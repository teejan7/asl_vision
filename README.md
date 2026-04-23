# ASL Vision

ASL Vision is a real-time American Sign Language recognition web application that runs on a laptop CPU. A user signs in front of a webcam, the system predicts the ASL class, and the browser builds text from stable predictions.

## Workflow

```text
Webcam frame
  -> browser capture and compression
  -> FastAPI /predict
  -> MediaPipe hand landmarks
  -> 91-D feature extraction
  -> Random Forest prediction
  -> confidence filtering
  -> majority voting
  -> live letter, word, and sentence output
```

## Output

- Real-time predicted ASL letters
- Stabilized word formation in the browser
- Spell-corrected sentence output
- Health endpoint at `/health`

## Project Structure

- `DEPLOYMENT/` contains the runnable web app and model files
- `TRAINING/` contains the training pipeline and model-building code

Detailed internal documentation:

- [DEPLOYMENT/README.md](DEPLOYMENT/README.md)
- [TRAINING/README.md](TRAINING/README.md)
