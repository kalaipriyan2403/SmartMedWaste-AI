import os
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

# ============================================================
# SMARTMEDWASTE - MOBILE NET V3 SMALL TRAINING
# ============================================================

# -----------------------------
# Paths
# -----------------------------
DATASET_DIR = r"C:\Users\DELL\Downloads\SmartMedWaste-Dataset"
PROJECT_DIR = r"C:\Users\DELL\Downloads\SmartMedWaste-AI-Project"

MODEL_DIR = os.path.join(PROJECT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "smartmedwaste_mobilenetv3.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")

# -----------------------------
# Configuration
# -----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 4

EPOCHS_HEAD = 15
EPOCHS_FINE = 10

SEED = 42

print("=" * 70)
print("SMARTMEDWASTE AI MODEL TRAINING")
print("=" * 70)

print("\nTensorFlow version:", tf.__version__)

# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLoading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "train"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED
)

print("\nLoading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "validation"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

print("\nLoading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "test"),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

# ============================================================
# CLASS NAMES
# ============================================================

class_names = train_ds.class_names

print("\nDetected classes:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

if len(class_names) != NUM_CLASSES:
    raise ValueError(
        f"Expected {NUM_CLASSES} classes, but found {len(class_names)}"
    )

# Save class names
with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names, f, indent=4)

print("\nClass mapping saved to:")
print(CLASS_NAMES_PATH)

# ============================================================
# PERFORMANCE OPTIMIZATION
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.10),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ],
    name="data_augmentation"
)

# ============================================================
# BUILD MOBILE NET V3 SMALL
# ============================================================

print("\nBuilding MobileNetV3Small model...")

base_model = MobileNetV3Small(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
    pooling="avg"
)

# Initially freeze pretrained model
base_model.trainable = False

inputs = keras.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

# MobileNetV3 includes its own preprocessing by default
x = base_model(x, training=False)

x = layers.Dropout(0.30)(x)

outputs = layers.Dense(
    NUM_CLASSES,
    activation="softmax",
    name="classification"
)(x)

model = keras.Model(inputs, outputs)

# ============================================================
# COMPILE - INITIAL TRAINING
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel created successfully.")

model.summary()

# ============================================================
# CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_accuracy",
    patience=4,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

# ============================================================
# PHASE 1 - TRAIN CLASSIFICATION HEAD
# ============================================================

print("\n")
print("=" * 70)
print("PHASE 1: TRAINING CLASSIFICATION HEAD")
print("=" * 70)

history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)

# ============================================================
# PHASE 2 - FINE TUNING
# ============================================================

print("\n")
print("=" * 70)
print("PHASE 2: FINE-TUNING MOBILENETV3")
print("=" * 70)

# Unfreeze the base model
base_model.trainable = True

# Freeze most layers and fine-tune only the upper layers
fine_tune_from = max(0, len(base_model.layers) - 30)

for layer in base_model.layers[:fine_tune_from]:
    layer.trainable = False

for layer in base_model.layers[fine_tune_from:]:
    layer.trainable = True

# Recompile with a very small learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.00001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print(
    f"\nFine-tuning last "
    f"{len(base_model.layers) - fine_tune_from} layers."
)

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FINE,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)

# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n")
print("=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)

best_model = keras.models.load_model(MODEL_PATH)

print("\nBest model loaded successfully.")

# ============================================================
# TEST EVALUATION
# ============================================================

print("\n")
print("=" * 70)
print("TEST DATASET EVALUATION")
print("=" * 70)

test_loss, test_accuracy = best_model.evaluate(
    test_ds,
    verbose=1
)

print("\nTest Loss     :", round(test_loss, 4))
print("Test Accuracy :", round(test_accuracy * 100, 2), "%")

# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n")
print("=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print("\nModel saved at:")
print(MODEL_PATH)

print("\nClass names saved at:")
print(CLASS_NAMES_PATH)

print("\nFinal Test Accuracy:")
print(f"{test_accuracy * 100:.2f}%")

print("\nClass mapping:")
print("ANATOMICAL              -> YELLOW")
print("CONTAMINATED_RECYCLABLE -> RED")
print("SHARPS                  -> WHITE")
print("PHARMA_GLASS            -> BLUE")

print("\nNext stage:")
print("Evaluate the model using a classification report and")
print("confusion matrix before connecting it to FastAPI.")

print("=" * 70)