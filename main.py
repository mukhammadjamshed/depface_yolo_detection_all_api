from fastapi import FastAPI, File
from fastapi import UploadFile

from fastapi.responses import HTMLResponse
from deepface import DeepFace
from fastapi.staticfiles import StaticFiles
from retinaface import RetinaFace
import uuid


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.post("/deepface/")
async def create_files(file: UploadFile=File(...)):

    filename = f"static/{uuid.uuid4().hex}.jpeg"


    with open(filename, "wb") as f:
        f.write(file.file.read())

    print(file.filename, filename)

    obj = DeepFace.analyze(img_path=filename, actions = ['age', 'gender', 'race', 'emotion'])

    return {"result": obj}

@app.post("/facenum/")
async def create_files(file: UploadFile=File(...)):

    filename = f"static/{uuid.uuid4().hex}.jpeg"


    with open(filename, "wb") as f:
        f.write(file.file.read())

    print(file.filename, filename)

    resp = RetinaFace.detect_faces(filename)
    
    #num_of_faces = len(list(resp.keys()))

    return {len(resp)}