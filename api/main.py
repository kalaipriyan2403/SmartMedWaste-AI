from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException
)

from api.schemas import PredictionResponse

from api.utils.model_utils import (
    load_model,
    load_class_names
)

from api.utils.image_utils import (
    validate_image_type,
    validate_image_size,
    prepare_image
)

from api.services.classification_service import (
    ClassificationService
)


# ========================================
# FastAPI Application
# ========================================

app = FastAPI(
    title="SmartMedWaste AI Service",
    description=(
        "AI-assisted biomedical waste classification "
        "using MobileNetV3-Small"
    ),
    version="1.0.0"
)


# ========================================
# Load AI Model
# ========================================

print("========================================")
print("SmartMedWaste AI Service")
print("========================================")

print("Loading trained model...")

model = load_model()

class_names = load_class_names()

print("AI model initialization completed.")


# ========================================
# Classification Service
# ========================================

classification_service = ClassificationService(
    model,
    class_names
)


# ========================================
# Root Endpoint
# ========================================

@app.get("/")
def root():

    return {
        "service": "SmartMedWaste AI Service",
        "status": "running",
        "model": "MobileNetV3-Small",
        "version": "1.0.0"
    }


# ========================================
# Health Check
# ========================================

@app.get("/health")
def health_check():

    return {
        "service": "SmartMedWaste AI Service",
        "status": "healthy",
        "model_loaded": model is not None
    }


# ========================================
# Prediction Endpoint
# ========================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
async def predict(
    file: UploadFile = File(...)
):

    try:

        # --------------------------------
        # 1. Validate image type
        # --------------------------------

        validate_image_type(
            file.content_type
        )


        # --------------------------------
        # 2. Read uploaded image
        # --------------------------------

        image_bytes = await file.read()


        # --------------------------------
        # 3. Validate image size
        # --------------------------------

        validate_image_size(
            image_bytes
        )


        # --------------------------------
        # 4. Prepare image
        # --------------------------------

        image_array = prepare_image(
            image_bytes
        )


        # --------------------------------
        # 5. AI Classification
        # --------------------------------

        result = classification_service.classify(
            image_array
        )


        # --------------------------------
        # 6. Return result
        # --------------------------------

        return result


    except HTTPException:
        raise


    except Exception as e:

        print(
            f"Prediction error: {str(e)}"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "AI classification failed. "
                "Please try again."
            )
        )