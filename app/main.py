from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import shutil
import os
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Load YOLO model
model = YOLO('app/best.pt')

# CORS settings
origins = [
    "http://localhost",
    "http://localhost:8000",
    # Add more origins as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    temp_image_path = "temp.jpg"
    with open(temp_image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Perform object detection using YOLO model
    results = model.predict(temp_image_path, conf=0.7)
    
    # Process results to match the required format
    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                "box": {
                    "x1": float(box.xyxy[0][0]),
                    "y1": float(box.xyxy[0][1]),
                    "x2": float(box.xyxy[0][2]),
                    "y2": float(box.xyxy[0][3]),
                },
                "confidence": float(box.conf[0]),
                "class": int(box.cls[0]),
                "name": model.names[int(box.cls[0])]
            })
    
    # Prepare the response
    response = {
        "images": [
            {
                "results": detections,
                "shape": [result.orig_shape[0], result.orig_shape[1]],  # (height, width)
                "speed": {
                    "inference": results[0].speed["inference"],  # Accessing the correct speed dictionary
                    "postprocess": results[0].speed["postprocess"],  # Placeholder
                    "preprocess": results[0].speed["preprocess"],  # Placeholder
                }
            }
        ],
        "metadata": {
            "imageCount": 1,
            "model": "final.pt",
            "version": {
                "python": "3.x.x",  # Replace with actual version
                "torch": "x.x.x",   # Replace with actual version
                "ultralytics": "x.x.x"  # Replace with actual version
            }
        }
    }
    
    # Remove the temporary file
    os.remove(temp_image_path)
    
    # Return the formatted JSON response
    return JSONResponse(content=response)

@app.get('/')
def hello_world():
    return {"data": "The detection model is RUNNING!"}

