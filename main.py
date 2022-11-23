from fastapi import FastAPI, File
from fastapi import UploadFile

from fastapi.responses import HTMLResponse
from deepface import DeepFace
from fastapi.staticfiles import StaticFiles
from retinaface import RetinaFace
import uuid
import numpy as np
import cv2
import os
from collections import Counter

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

    #----------------------------------------------------

@app.post("/yolo/")
async def create_files(file: UploadFile=File(...)):

    filename = f"static/{uuid.uuid4().hex}.jpeg"

    with open(filename, "wb") as f:
        f.write(file.file.read())

    print(file.filename, filename)

    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    #save all the names in file o the list classes
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    #get layers of the network
    layer_names = net.getLayerNames()
    #Determine the output layer names from the YOLO model 
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
    # print("YOLO LOADED")


    #---------------------------------------------------------------#
     # Capture frame-by-frame
    img = cv2.imread(filename)
    #     img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

        # USing blob function of opencv to preprocess image
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
         swapRB=True, crop=False)
        #Detecting objects
    net.setInput(blob)
    outs = net.forward(output_layers)

        # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                    # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                    # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    labels_list= []    
    #We use NMS function in opencv to perform Non-maximum Suppression
    #we give it score threshold and nms threshold as arguments.
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y -5),cv2.FONT_HERSHEY_SIMPLEX,
      1/2, color, 2)


            labels_list.append(label)


    count_elements = dict(Counter(labels_list))
    print(count_elements)

        
    # cv2.imshow("Image",img)
    # cv2.waitKey(0)


    return {"Count of elements": count_elements}