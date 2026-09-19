import io

import numpy as np

from PIL import Image
from fastapi import HTTPException

from api.config import (
    IMAGE_SIZE,
    ALLOWED_IMAGE_TYPES,
    MAX_IMAGE_SIZE_BYTES
)


def validate_image_type(content_type):
    """
    Validate uploaded image MIME type.
    """

    if content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG images are supported."
        )


def validate_image_size(image_bytes):
    """
    Validate uploaded image file size.
    """

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail="Image size exceeds the maximum allowed limit."
        )


def prepare_image(image_bytes):
    """
    Convert uploaded image bytes into
    a NumPy array suitable for MobileNetV3.
    """

    try:

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

        image = image.resize(
            IMAGE_SIZE
        )

        # IMPORTANT:
        # Do NOT divide by 255.
        # MobileNetV3 preprocessing is handled
        # by the Keras model itself.

        image_array = np.array(
            image,
            dtype=np.float32
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        return image_array

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file: {str(e)}"
        )