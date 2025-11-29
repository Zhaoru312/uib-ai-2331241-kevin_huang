from ultralytics import YOLO

if __name__ == "__main__":
    
    # Load a model
    model = YOLO("yolov10n.yaml")  # build a new model from YAML

    # training
    results = model.train(data="config.yaml", epochs=1)
