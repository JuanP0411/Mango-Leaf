import os
import io
from fastapi import APIRouter, FastAPI, HTTPException, WebSocket,  UploadFile
from fastapi.responses import JSONResponse
import uuid
from tensorflow import keras
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input

MODEL_PATH='mango_sickness_classifier.h5'
router = APIRouter()

model = keras.models.load_model(MODEL_PATH, compile=False)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001, clipvalue=None),loss='categorical_crossentropy')
class_label = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']
IMAGEDIR = 'images/'

def process_image(file, model, class_labels) -> str:
    try:
        # Load and preprocess the image using keras.preprocessing.image
        img = Image.open(io.BytesIO(file))
        img = img.convert('RGB')  # Ensure the image is in RGB mode
        img = img.resize((256, 256))  # Resize the image
        img_array = image.img_to_array(img)
        
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)

        predicted_class_label = class_labels[predicted_class_index]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return predicted_class_label

@router.post('/get-predict')
async def get_and_predict(file: UploadFile):
    
    image_bytes = await file.read()
    global model  
    result = process_image(image_bytes, model=model,class_labels=class_label)
    print(result)

    return result

