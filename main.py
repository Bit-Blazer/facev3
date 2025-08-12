from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from ultralytics import YOLO
import io

app = FastAPI()

# Load model once on startup
model = YOLO("yolov12n-face.pt")

@app.post("/tag-faces")
async def tag_faces(file: UploadFile):
    # Read uploaded image into OpenCV
    img_bytes = await file.read()
    img_np = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image file"}

    # Detect faces
    results = model.predict(source=img, conf=0.25, verbose=False)

    # Draw boxes + labels
    for idx, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"Face {idx+1}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Encode back to JPEG
    _, buffer = cv2.imencode(".jpg", img)

    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
