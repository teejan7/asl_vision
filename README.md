# 🤟 ASL Vision — Neural Interface

A lightweight, real-time American Sign Language (ASL) recognition web application. This project captures live webcam feeds, extracts 3D hand landmarks, and translates gestures into text with high accuracy.

🚀 **Live Demo:** [Launch ASL Vision Here](https://asl-vision-app.onrender.com)
*(Note: Replace the link above with your actual Render URL if it is different)*

---

## ⚡ Key Features

- **Real-Time Processing:** Uses an efficient FastAPI backend with WebSocket-like streaming via base64 encoding.
- **High Accuracy:** Custom Random Forest Classifier trained on 91-Dimensional geometric feature vectors achieving **98.89%** accuracy.
- **Temporal Stabilization:** Implements a Majority Voting buffer to prevent flickering and ensure confident letter predictions.
- **Smart Correction:** Integrated PySpellChecker to automatically fix misspelled words after the user completes a sign sequence.
- **Cyberpunk UI:** Custom-built "Neural Interface" frontend featuring neon styling, a live skeleton overlay, and responsive controls.

---

## 🧠 Technical Architecture

**The Pipeline:**

`Webcam` → `MediaPipe Hands` → `91-D Geometric Feature Extraction` → `Random Forest Classifier` → `Majority Voting & Confidence Filter` → `PySpellChecker` → `Real-Time Text Output`

**Tech Stack:**

| Layer | Technology |
|---|---|
| Backend | Python 3.10, FastAPI, Uvicorn |
| Computer Vision | OpenCV (Headless), MediaPipe 0.10.13 |
| Machine Learning | Scikit-Learn 1.6.1, NumPy |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Hosting | Render (Web Service) |
| Model Storage | Git LFS (Large File Storage) |

---

## 💻 Run Locally

Because this project uses **Git LFS** for the trained model weights, make sure you have [Git LFS installed](https://git-lfs.com/) on your machine before cloning.

**1. Clone the repository:**
```bash
git clone https://github.com/teejan7/asl_vision.git
cd asl_vision
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Start the local server:**
```bash
python app.py
```

**4. Access the interface:**

Open your browser and navigate to `http://localhost:7860`

---

## ✋ How To Use

1. **Allow Camera Access** — Your browser will ask for permission when the page loads.
2. **Position Your Hand** — Hold your hand clearly in front of the webcam.
3. **Sign** — Hold each ASL sign for ~1 second until the "Majority Vote" bar fills up.
4. **Controls:**
   - Click **SPACE** (or sign `space`) to commit the current word and trigger the spellchecker.
   - Click **DELETE** (or sign `del`) to remove the last letter.
   - Click **CLEAR ALL** to reset the sentence.

---

## 👥 The Team — Team Artemis Crew

| Name | Roll Number |
|---|---|
| Bebino Khesoh | LHGW23CS033 |
| Bhagyalakshmi K B | SGI23CS021 |
| Christina Raphel | LHGW23CS034 |
| Teejan Teepee | LHGW23CS035 |
