import cv2
from ultralytics import YOLO
import time
import os

model = YOLO("models/traffic_sign_v1.pt") 

source = "testing/6.png"  #source gambar/video

# Detect if source is image or video
is_image = os.path.splitext(source)[1].lower() in [".jpg", ".png", ".jpeg", ".bmp"]

if is_image:
    img = cv2.imread(source)
    results = model(img)
    annotated = results[0].plot()
    cv2.imshow("Result", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

# Otherwise: Video mode
cap = cv2.VideoCapture(source) # 0 untuk facecam

if not cap.isOpened():
    print("Gagal membuka video / RTSP")
    exit()

prev = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame. mencoba lagi...")
        break

    now = time.time()
    fps = 1 / (now - prev)
    prev = now

    results = model(frame, device=0)
    annotated = results[0].plot()

    cv2.putText(annotated, f"FPS: {int(fps)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("YOLO Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()