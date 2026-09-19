import json
import tensorflow as tf

from api.config import MODEL_PATH, CLASS_NAMES_PATH


def load_model():
    """
    Load the trained SmartMedWaste MobileNetV3 model.
    """

    print("Loading trained model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded successfully.")

    return model


def load_class_names():
    """
    Load class names from class_names.json.
    """

    with open(
        CLASS_NAMES_PATH,
        "r"
    ) as file:

        class_names = json.load(file)

    print("Class names:")

    for index, class_name in enumerate(class_names):
        print(f"{index}: {class_name}")

    return class_names


def predict_class(
    model,
    image_array
):
    """
    Run prediction on a prepared image.
    """

    predictions = model.predict(
        image_array,
        verbose=0
    )

    predicted_index = int(
        tf.argmax(
            predictions[0]
        )
    )

    confidence = float(
        predictions[0][predicted_index]
    )

    return predicted_index, confidence