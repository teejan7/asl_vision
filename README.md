<<<<<<< HEAD
# 🤟 ASL Vision — Deployment

Real-time ASL sign language recognition app.
Part of the ASL Vision project.

🚀 **Live Demo:** [teejan7-asl-vision.hf.space](https://teejan7-asl-vision.hf.space)

---

## 📁 Folder Contents

```
asl_vision/
├── app.py              ← FastAPI backend
├── index.html          ← Neon UI frontend
├── Dockerfile          ← HuggingFace deployment
├── requirements.txt    ← Python dependencies
├── rf_model_68.pkl     ← Trained model (98.89% accuracy)
└── label_encoder.pkl   ← Class labels A-Z
```

---

## 🚀 Run Locally

**Step 1 — Install dependencies:**
```bash
py -3.10 -m pip install fastapi uvicorn opencv-python mediapipe==0.10.13 scikit-learn==1.6.1 numpy pyspellchecker
```

**Step 2 — Run:**
```bash
py -3.10 app.py
```

**Step 3 — Open browser:**
```
http://localhost:7860
```

---

## 🧠 How It Works

```
Webcam → MediaPipe → 91-D Features
       → Random Forest → Majority Voting
       → Confidence Filter → SpellCheck
       → Real-Time Text Output
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | **98.89%** |
| Classes | 29 (A–Z + del + space) |
| Features | 91-D geometric vector |
| Classifier | Random Forest (300 trees) |

---

## ✋ How To Use

1. Allow camera access
2. Show hand clearly to webcam
3. Hold each ASL sign for **~1 second**
4. Press **SPACE** to commit word
5. Press **DELETE** to remove last letter
6. Press **CLEAR ALL** to reset

---

## 👥 Team

| Name | Roll Number |
|------|-------------|
| Bebino Khesoh | LHGW23CS033 |
| Bhagyalakshmi K B | SGI23CS021 |
| Christina Raphel | LHGW23CS034 |
| Teejan Teepee | LHGW23CS035 |
=======
# asl_vision
>>>>>>> 75c23a8785276c9c2ad8fe3fe8d2eaca38d333f3
