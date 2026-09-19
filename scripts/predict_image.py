import sys
import os

# Add project root to Python path
PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    PROJECT_DIR
)

from api.config import COLOUR_MAPPING
from api.utils.model_utils import (
    load_model,
    load_class_names,
    predict_class
)
from api.utils.image_utils import (
    validate_image_type,
    validate_image_size,
    prepare_image
)


def predict_image(image_path):

    print("\n========================================")
    print("SmartMedWaste AI Prediction")
    print("========================================")

    # Check file exists
    if not os.path.exists(image_path):
        print("ERROR: Image file not found.")
        return

    # Read image
    with open(
        image_path,
        "rb"
    ) as file:

        image_bytes = file.read()

    # Detect MIME type from extension
    extension = os.path.splitext(
        image_path
    )[1].lower()

    if extension in [".jpg", ".jpeg"]:
        content_type = "image/jpeg"

    elif extension == ".png":
        content_type = "image/png"

    else:
        print(
            "ERROR: Only JPG, JPEG and PNG images are supported."
        )
        return

    # Validate
    validate_image_type(
        content_type
    )

    validate_image_size(
        image_bytes
    )

    # Prepare image
    image_array = prepare_image(
        image_bytes
    )

    # Load model
    model = load_model()

    # Load classes
    class_names = load_class_names()

    # Prediction
    predicted_index, confidence = predict_class(
        model,
        image_array
    )

    category = class_names[
        predicted_index
    ]

    colour_code = COLOUR_MAPPING.get(
        category,
        "UNKNOWN"
    )

    # Display result
    print("\n------------- RESULT -------------")

    print(
        f"Category       : {category}"
    )

    print(
        f"Colour Code    : {colour_code}"
    )

    print(
        f"Confidence     : {confidence:.4f}"
    )

    print(
        f"Confidence (%) : {confidence * 100:.2f}%"
    )

    print("----------------------------------")


if __name__ == "__main__":

    if len(sys.argv) != 2:

        print(
            "\nUsage:"
        )

        print(
            "python scripts/predict_image.py <image_path>"
        )

        print(
            "\nExample:"
        )

        print(
            r"python scripts/predict_image.py C:\Users\DELL\Downloads\test.jpg"
        )

        sys.exit(1)

    image_path = sys.argv[1]

    predict_image(
        image_path
    )