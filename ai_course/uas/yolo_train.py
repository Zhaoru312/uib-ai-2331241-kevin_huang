from ultralytics import YOLO
import shutil
import os
import re

if __name__ == "__main__":

    model = YOLO("models/yolov8n.pt")  # build a new model from pretrained

    results = model.train(
        data="config.yaml",
        epochs=50,
        device=0,
    )

   # source
    src = "runs/detect/train/weights/best.pt"
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    existing = [
        f for f in os.listdir(model_dir)
        if re.match(r"Face_detect_v\d+\.pt", f)
    ]

    if existing:
        versions = [int(re.findall(r"\d+", f)[0]) for f in existing]
        next_version = max(versions) + 1
    else:
        next_version = 1

    # model version
    dst = f"{model_dir}/Face_detect_v{next_version}.pt"
    shutil.copy(src, dst)

    print(f"Saved trained model as: {dst}")