# SmartMedWaste AI Service Configuration

import os


# --------------------------------------------------
# Project Paths
# --------------------------------------------------

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


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


# --------------------------------------------------
# Image Configuration
# --------------------------------------------------

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

IMAGE_SIZE = (
    IMAGE_WIDTH,
    IMAGE_HEIGHT
)


ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png"
}


MAX_IMAGE_SIZE_MB = 5

MAX_IMAGE_SIZE_BYTES = (
    MAX_IMAGE_SIZE_MB * 1024 * 1024
)


# --------------------------------------------------
# AI Configuration
# --------------------------------------------------

CONFIDENCE_THRESHOLD = 0.80


# --------------------------------------------------
# Biomedical Waste Categories
# --------------------------------------------------

COLOUR_MAPPING = {
    "ANATOMICAL": "YELLOW",
    "CONTAMINATED_RECYCLABLE": "RED",
    "PHARMA_GLASS": "BLUE",
    "SHARPS": "WHITE"
}