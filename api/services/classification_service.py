from api.config import (
    COLOUR_MAPPING,
    CONFIDENCE_THRESHOLD
)

from api.utils.model_utils import predict_class


class ClassificationService:

    def __init__(
        self,
        model,
        class_names
    ):
        self.model = model
        self.class_names = class_names

    def classify(
        self,
        image_array
    ):
        predicted_index, confidence = predict_class(
            self.model,
            image_array
        )

        category = self.class_names[
            predicted_index
        ]

        colour_code = COLOUR_MAPPING.get(
            category,
            "UNKNOWN"
        )

        requires_manual_review = (
            confidence < CONFIDENCE_THRESHOLD
        )

        return {
            "category": category,
            "colour_code": colour_code,
            "confidence": round(
                confidence,
                4
            ),
            "confidence_percentage": round(
                confidence * 100,
                2
            ),
            "requires_manual_review": (
                requires_manual_review
            )
        }