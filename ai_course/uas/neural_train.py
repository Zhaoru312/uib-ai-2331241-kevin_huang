# ENV SETUP
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    CSVLogger,
    ReduceLROnPlateau
)
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

# SEED
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# CONFIG
IMG_SIZE = 96         
BATCH_SIZE = 32
EPOCHS = 50

TRAIN_DIR = "Neutral_Dataset/Train"
TEST_DIR  = "Neutral_Dataset/Test"
SAVE_PATH = "models/emotion_mobilenet.keras"

os.makedirs("models", exist_ok=True)
os.makedirs("report", exist_ok=True)
os.makedirs("matrix", exist_ok=True)

# DATA GENERATORS
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    validation_split=0.2
)

val_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

test_gen = ImageDataGenerator(rescale=1./255)

# Load training set
train_ds = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=SEED
)

val_ds = val_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

test_ds = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

NUM_CLASSES = train_ds.num_classes
CLASS_NAMES = list(train_ds.class_indices.keys())
print("Classes:", CLASS_NAMES)

# COMPUTE CLASS WEIGHTS
# Compute class weights to help "angry" and "sad" detection
labels = train_ds.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weight_dict = dict(enumerate(class_weights))
print("Class weights:", class_weight_dict)

# LOSS
loss_fn = tf.keras.losses.CategoricalCrossentropy(
    label_smoothing=0.1
)

# MODEL
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze layers (transfer learning)
base_model.trainable = False

# Build model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),

    layers.Dense(256, activation="relu",
                 kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dropout(0.4),

    layers.Dense(128, activation="relu",
                 kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dropout(0.3),

    layers.Dense(NUM_CLASSES, activation="softmax")
])

# CALLBACKS
callbacks = [
    ModelCheckpoint(
        "models/best_model.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1
    ),
    CSVLogger("report/training_log.csv")
]

# LOOP 1 — FEATURE EXTRACTION
print("\n===== LOOP 1: FEATURE EXTRACTION =====")
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=loss_fn,
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weight_dict,
    callbacks=callbacks
)

# LOOP 2 — FINE TUNING
print("\n===== LOOP 2: FINE TUNING =====")

base_model.trainable = True
for layer in base_model.layers[:-60]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=loss_fn,
    metrics=["accuracy"]
)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS * 2,
    class_weight=class_weight_dict,
    callbacks=callbacks
)

# TEST EVALUATION
print("\n===== TEST SET EVALUATION =====")
test_ds.reset()
pred = model.predict(test_ds)
y_pred = np.argmax(pred, axis=1)
y_true = test_ds.classes

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="d",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    cmap="Blues"
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Test)")
plt.tight_layout()
plt.savefig("confusion_matrix_test.png")
plt.close()

report = classification_report(
    y_true, y_pred,
    target_names=CLASS_NAMES,
    digits=4
)

with open("classification_report_test.txt", "w") as f:
    f.write(report)

print(report)

# SAVE FINAL MODEL
model.save(SAVE_PATH)
print(f"\n✅ Model saved to {SAVE_PATH}")
