from io import BytesIO
from typing import List
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile,status
#from model import load_model, predict, prepare_image
from .model import load_model2,prepare_image,predict
from PIL import Image
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import run
import aiofiles
from fastapi.responses import JSONResponse


import os

#----------------------
app = FastAPI()

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(    # CORSMiddleware which essentially allows us to access the API in a different host.
    CORSMiddleware, 
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = methods,
    allow_headers = headers    
)

#------------------------------

model2 = load_model2()


#----------------------------------

@app.get("/")     # tester l'API
async def root():
    return {"message": "Welcome to the RD Detection API!"}


#-----------------------------------



# Define the response JSON
class Prediction(BaseModel):      # cette classe hérite la classe BaseModel de pydantic. Cela nous permet de définir un schéma de réponse pour notre API
    filename: str                  # On retourne le nom du fichier, son type et les prédictions du modèle.
    content_type: str
    predictions: List[dict] = []

#-----------------
@app.post("/predict", response_model=Prediction)

async def prediction(file: UploadFile = File(...)):

    # Ensure that the file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    else :

    
        content = await file.read()
        image = Image.open(BytesIO(content)).convert("RGB")

        # preprocess the image and prepare it for classification
        image = prepare_image(image, target=(64, 64))
        response = predict(image, model2)


        # return the response as a JSON
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "predictions": response,
        }


@app.post("/upload_file/", response_description="", response_model = "")
async def result(file:UploadFile = File(...)):
    try:
        async with aiofiles.open(file.filename, 'wb') as out_file:
            content = await file.read()  # async read
            await out_file.write(content)  # async write

    except Exception as e:
        return JSONResponse(
            status_code = status.HTTP_400_BAD_REQUEST,
            content = { 'message' : str(e) }
            )
    else:
        return JSONResponse(
            status_code = status.HTTP_200_OK,
            content = {"result":'success'}
            )



    


 












if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000,workers=2)
   # port = int(os.environ.get('PORT', 5000))

#run(app, host="0.0.0.0", port=port)

