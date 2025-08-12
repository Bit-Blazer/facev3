from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO

app = FastAPI()

# Load YOLO face model
model = YOLO("yolov12n-face.pt")


@app.post("/tag-faces")
async def tag_faces(file: UploadFile = File(...)):
    try:
        # Read image
        img_bytes = await file.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        height, width = img.shape[:2]

        # Run YOLO detection
        results = model.predict(source=img, conf=0.25, verbose=False)

        tags = []
        for idx, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            tags.append(
                {
                    "x": x1 / width,
                    "y": y1 / height,
                    "width": (x2 - x1) / width,
                    "height": (y2 - y1) / height,
                    "label": f"Face {idx+1}",
                }
            )

        return JSONResponse(content=tags)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
