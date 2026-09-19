\# SmartMedWaste AI Service



AI-assisted biomedical waste classification service built using \*\*Python, FastAPI, TensorFlow, and MobileNetV3-Small\*\*.



The service accepts a biomedical waste image and predicts its category, colour code, confidence score, and whether manual review is required.



\## Features



\* Biomedical waste image classification

\* MobileNetV3-Small CNN model

\* 4 waste categories

\* Confidence-based manual review

\* JPEG and PNG image support

\* Maximum image size: 5 MB

\* REST API using FastAPI

\* Swagger API documentation



\## Waste Categories



| Category                | Colour |

| ----------------------- | ------ |

| ANATOMICAL              | YELLOW |

| CONTAMINATED\_RECYCLABLE | RED    |

| PHARMA\_GLASS            | BLUE   |

| SHARPS                  | WHITE  |



\## Project Structure



```text

SmartMedWaste-AI-Project/

│

├── api/

│   ├── main.py

│   ├── config.py

│   ├── schemas.py

│   ├── services/

│   │   └── classification\_service.py

│   └── utils/

│       ├── image\_utils.py

│       └── model\_utils.py

│

├── models/

│   ├── class\_names.json

│   └── smartmedwaste\_mobilenetv3.keras

│

├── scripts/

├── requirements.txt

├── .gitignore

└── README.md

```



\## Requirements



\* Python 3.12

\* pip

\* Git

\* 8 GB RAM recommended



\## Installation



Clone the repository:



```bash

git clone https://github.com/YOUR\_USERNAME/YOUR\_REPOSITORY.git

cd YOUR\_REPOSITORY

```



Create a virtual environment:



```bash

python -m venv smartmedwaste-ai

```



Activate it on Windows:



```bash

smartmedwaste-ai\\Scripts\\activate

```



Install dependencies:



```bash

pip install -r requirements.txt

```



\## Run the AI Service



Start FastAPI:



```bash

python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

```



The service will run at:



```text

http://127.0.0.1:8000

```



\## API Documentation



Swagger UI:



```text

http://127.0.0.1:8000/docs

```



Health check:



```text

http://127.0.0.1:8000/health

```



Prediction endpoint:



```text

POST /predict

```



Upload an image using the `file` parameter.



\## Example Response



```json

{

&#x20; "category": "SHARPS",

&#x20; "colour\_code": "WHITE",

&#x20; "confidence": 0.943,

&#x20; "confidence\_percentage": 94.3,

&#x20; "requires\_manual\_review": false

}

```



\## Confidence Rule



The AI service uses an \*\*80% confidence threshold\*\*.



```text

Confidence >= 80%

→ AI suggestion accepted for confirmation



Confidence < 80%

→ Manual review required

```



The AI is designed as an \*\*assistive classification system\*\*. The final classification can be manually confirmed or changed by the user.



\## Model



The project uses \*\*MobileNetV3-Small\*\* with transfer learning.



Model input:



```text

224 × 224 × 3

```



Output:



```text

4 classes

```



