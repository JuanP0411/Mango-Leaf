import os
from fastapi import APIRouter, FastAPI, HTTPException, WebSocket,  UploadFile
from fastapi.responses import JSONResponse
import uuid
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from dotenv import load_dotenv

load_dotenv()
MODEL_PATH=os.getenv('MODEL_PATH')
router = APIRouter()

model = keras.models.load_model(MODEL_PATH)
class_label = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']
IMAGEDIR = 'images/'


def predict_and_display(image_path, model, class_labels):

    img = image.load_img(image_path, target_size=(256, 256))
    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)

    predicted_class_label = class_labels[predicted_class_index]


    return predicted_class_label



@router.post("/image")
async def upload_image(file: UploadFile):

    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()

    with open(f"{IMAGEDIR}{file.filename}","wb") as f :
        f.write(contents)



    return {"filename": file.filename}

@router.get('/predict')
def predict(filename):
    global model  
    image_path = f"{IMAGEDIR}{filename}"
    result = predict_and_display(image_path= image_path, model=model,class_labels=class_label)
    print(result)

    return result

