import cv2
from ultralytics import YOLO
import time
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

# ==========================================================
# LOAD MODELS
# ==========================================================

# YOLO face detector
face_model = YOLO("models/Face_detect_v1.pt")

# Emotion classifier
emotion_model = load_model("models/emotion_model_v2.keras")

# Label
emotion_labels = ["Angry", "Happy", "Neutral", "Sad"]

IMG_SIZE = 96

# Temporal smoothing buffer
pred_buffer = deque(maxlen=7)

# ==========================================================
# EMOTION CLASSIFIER
# ==========================================================
def classify_emotion(face_img):
    if face_img is None or face_img.size == 0:
        return "Unknown", 0.0

    if face_img.shape[0] < 50 or face_img.shape[1] < 50:
        return "Unknown", 0.0

    # BGR → RGB
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
    face_img = face_img.astype("float32") / 255.0
    face_img = np.expand_dims(face_img, axis=0)

    pred = emotion_model.predict(face_img, verbose=0)[0]
    pred_buffer.append(pred)

    # Temporal smoothing
    smoothed_pred = np.mean(np.array(pred_buffer), axis=0)

    idx = np.argmax(smoothed_pred)
    confidence = float(smoothed_pred[idx])

    # Apply confidence threshold: below 0.4 → Neutral
    if confidence < 0.4:
        idx = emotion_labels.index("Neutral")
        confidence = float(smoothed_pred[idx])

    emotion = emotion_labels[idx]

    return emotion, confidence

# ==========================================================
# VIDEO MODE (WEBCAM)
# ==========================================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame. mencoba lagi...")
        break

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    results = face_model(frame)

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)

        face = frame[y1:y2, x1:x2]
        emotion, conf = classify_emotion(face)

        # Draw face box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw emotion label
        label = f"{emotion} ({conf*100:.1f}%)"
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # FPS
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YOLO Face + Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()