import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# ==============================
# CONFIGURATION
# ==============================

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "smartmedwaste_mobilenetv3.keras"
)

CLASS_NAMES_PATH = os.path.join(
    PROJECT_DIR,
    "models",
    "class_names.json"
)

TEST_DIR = r"C:\Users\DELL\Downloads\SmartMedWaste-Dataset\test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


# ==============================
# LOAD CLASS NAMES
# ==============================

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print("\nClass names:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")


# ==============================
# LOAD TEST DATASET
# ==============================

print("\nLoading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("\nTest dataset loaded successfully.")
print("Number of test images:", len(test_ds.file_paths))


# ==============================
# LOAD MODEL
# ==============================

print("\nLoading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ==============================
# MODEL EVALUATION
# ==============================

print("\nEvaluating model...")

loss, accuracy = model.evaluate(test_ds, verbose=1)

print("\n==============================")
print("FINAL TEST RESULTS")
print("==============================")

print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy * 100:.2f}%")


# ==============================
# PREDICTIONS
# ==============================

print("\nGenerating predictions...")

y_true = np.concatenate([
    labels.numpy()
    for _, labels in test_ds
])

predictions = model.predict(test_ds, verbose=1)

y_pred = np.argmax(predictions, axis=1)


# ==============================
# CLASSIFICATION REPORT
# ==============================

print("\n==============================")
print("CLASSIFICATION REPORT")
print("==============================")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
)


# ==============================
# CONFUSION MATRIX
# ==============================

print("\n==============================")
print("CONFUSION MATRIX")
print("==============================")

cm = confusion_matrix(y_true, y_pred)

print(cm)


# ==============================
# PER-CLASS ACCURACY
# ==============================

print("\n==============================")
print("PER-CLASS ACCURACY")
print("==============================")

for i, class_name in enumerate(class_names):

    total = cm[i].sum()

    correct = cm[i, i]

    class_accuracy = (
        correct / total * 100
        if total > 0
        else 0
    )

    print(
        f"{class_name:30s}: "
        f"{class_accuracy:.2f}% "
        f"({correct}/{total})"
    )


print("\n==============================")
print("EVALUATION COMPLETED")
print("==============================")