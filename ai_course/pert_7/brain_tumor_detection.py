"""Brain tumor classifier training & evaluation script.

This script mirrors the workflow from the `brain tumor detections.ipynb` notebook so it can
be run end-to-end from the command line. It:
  * Builds train/validation/test splits from the dataset folder structure
  * Creates TensorFlow ImageDataGenerators
  * Trains an EfficientNetV2-B0-based classifier
  * Evaluates on the three splits and saves reports/figures
  * (Optionally) runs a single-image inference after training

Example usage (from repository root):
    python ai course/pert 7/brain_tumor_detection.py \
        --train_dir "dataset/training" \
        --test_dir "dataset/testing" \
        --epochs 5 \
        --model_path "brain_tumor.h5" \
        --history_plot "training_history.png" \
        --confusion_plot "confusion_matrix.png"
"""

from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate the brain tumor detector.")
    parser.add_argument("--train_dir", default="dataset/training", help="Path to the training dataset root.")
    parser.add_argument("--test_dir", default="dataset/testing", help="Path to the testing dataset root.")
    parser.add_argument("--image_height", type=int, default=224, help="Input image height.")
    parser.add_argument("--image_width", type=int, default=244, help="Input image width.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generators.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.5,
        help="Fraction of the testing dataframe reserved for validation (rest is held-out test).",
    )
    parser.add_argument(
        "--model_path",
        default="brain_tumor.h5",
        help="Where to save the trained Keras model (HDF5).",
    )
    parser.add_argument(
        "--history_plot",
        default="training_history.png",
        help="Path to save the training/validation loss & accuracy curves.",
    )
    parser.add_argument(
        "--confusion_plot",
        default="confusion_matrix.png",
        help="Path to save the confusion matrix heatmap.",
    )
    parser.add_argument(
        "--sample_image",
        default=None,
        help="Optional path to a single image used for inference after training.",
    )
    parser.add_argument(
        "--freeze_base",
        action="store_true",
        help="Freeze EfficientNet base weights instead of fine-tuning the whole network.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for dataframe splitting and TensorFlow.",
    )
    return parser.parse_args()


def build_dataframe(root_dir: Path) -> pd.DataFrame:
    """Create a dataframe with absolute filepaths and labels derived from directory names."""

    if not root_dir.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    filepaths: List[str] = []
    labels: List[str] = []

    for class_dir in sorted(root_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for image_path in sorted(class_dir.glob("**/*")):
            if image_path.is_file():
                filepaths.append(str(image_path.resolve()))
                labels.append(class_dir.name)

    if not filepaths:
        raise ValueError(f"No image files found under {root_dir}")

    return pd.DataFrame({"filepaths": filepaths, "labels": labels})


def create_generators(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    image_size: Tuple[int, int],
    batch_size: int,
):
    """Create ImageDataGenerators matching the original notebook configuration."""

    tr_gen = ImageDataGenerator()
    ts_gen = ImageDataGenerator()

    train_gen = tr_gen.flow_from_dataframe(
        train_df,
        x_col="filepaths",
        y_col="labels",
        target_size=image_size,
        class_mode="categorical",
        color_mode="rgb",
        shuffle=True,
        batch_size=batch_size,
    )

    valid_gen = ts_gen.flow_from_dataframe(
        valid_df,
        x_col="filepaths",
        y_col="labels",
        target_size=image_size,
        class_mode="categorical",
        color_mode="rgb",
        shuffle=True,
        batch_size=batch_size,
    )

    test_gen = ts_gen.flow_from_dataframe(
        test_df,
        x_col="filepaths",
        y_col="labels",
        target_size=image_size,
        class_mode="categorical",
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size,
    )

    return train_gen, valid_gen, test_gen


def build_model(img_shape: Tuple[int, int, int], num_classes: int, freeze_base: bool) -> tf.keras.Model:
    base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=img_shape,
        pooling="max",
    )
    base_model.trainable = not freeze_base

    model = Sequential(
        [
            base_model,
            BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
            Dense(
                256,
                kernel_regularizer=regularizers.l2(0.016),
                activity_regularizer=regularizers.l1(0.006),
                bias_regularizer=regularizers.l1(0.006),
                activation="relu",
            ),
            Dropout(rate=0.4, seed=75),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(Adamax(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def plot_history(history: tf.keras.callbacks.History, out_path: str) -> None:
    train_acc = history.history.get("accuracy", [])
    train_loss = history.history.get("loss", [])
    val_acc = history.history.get("val_accuracy", [])
    val_loss = history.history.get("val_loss", [])
    epochs = range(1, len(train_acc) + 1)

    plt.figure(figsize=(20, 8))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, "r", label="Training Loss")
    plt.plot(epochs, val_loss, "g", label="Validation Loss")
    if val_loss:
        min_epoch = int(np.argmin(val_loss)) + 1
        plt.scatter(min_epoch, val_loss[min_epoch - 1], s=150, c="blue", label=f"Best epoch = {min_epoch}")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, "r", label="Training Accuracy")
    plt.plot(epochs, val_acc, "g", label="Validation Accuracy")
    if val_acc:
        max_epoch = int(np.argmax(val_acc)) + 1
        plt.scatter(max_epoch, val_acc[max_epoch - 1], s=150, c="blue", label=f"Best epoch = {max_epoch}")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_confusion(cm: np.ndarray, class_names: Sequence[str], out_path: str) -> None:
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def run_inference(
    model: tf.keras.Model,
    image_path: str,
    image_size: Tuple[int, int],
    class_names: Sequence[str],
) -> None:
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=image_size)
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)
    predicted_index = int(np.argmax(preds, axis=-1)[0])
    predicted_class = class_names[predicted_index]

    print(f"Inference on {image_path} => class index {predicted_index} ({predicted_class})")


def main() -> None:
    args = parse_args()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    train_df = build_dataframe(Path(args.train_dir))
    test_df = build_dataframe(Path(args.test_dir))
    valid_df, test_df = train_test_split(
        test_df,
        train_size=args.validation_split,
        shuffle=True,
        random_state=args.seed,
        stratify=test_df["labels"],
    )

    image_size = (args.image_height, args.image_width)
    img_shape = (args.image_height, args.image_width, 3)

    train_gen, valid_gen, test_gen = create_generators(train_df, valid_df, test_df, image_size, args.batch_size)
    class_names = list(train_gen.class_indices.keys())

    print(f"Training on {len(train_gen.filenames)} images with classes: {class_names}")
    model = build_model(img_shape, len(class_names), args.freeze_base)
    model.summary()

    history = model.fit(
        train_gen,
        epochs=args.epochs,
        verbose=1,
        validation_data=valid_gen,
        shuffle=True,
    )

    plot_history(history, args.history_plot)
    print(f"Saved training curves to {args.history_plot}")

    train_score = model.evaluate(train_gen, verbose=1)
    valid_score = model.evaluate(valid_gen, verbose=1)
    test_score = model.evaluate(test_gen, verbose=1)

    print(f"Train Loss: {train_score[0]:.4f} | Train Accuracy: {train_score[1]:.4f}")
    print(f"Validation Loss: {valid_score[0]:.4f} | Validation Accuracy: {valid_score[1]:.4f}")
    print(f"Test Loss: {test_score[0]:.4f} | Test Accuracy: {test_score[1]:.4f}")

    preds = model.predict(test_gen)
    y_pred = np.argmax(preds, axis=1)
    cm = confusion_matrix(test_gen.classes, y_pred)
    plot_confusion(cm, class_names, args.confusion_plot)
    print(f"Saved confusion matrix to {args.confusion_plot}")

    report = classification_report(test_gen.classes, y_pred, target_names=class_names)
    print("\nClassification Report:\n", report)

    model.save(args.model_path)
    print(f"Model saved to {args.model_path}")

    if args.sample_image:
        run_inference(model, args.sample_image, image_size, class_names)


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()
