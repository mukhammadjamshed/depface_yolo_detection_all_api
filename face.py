from deepface import DeepFace

from retinaface import RetinaFace

resp = RetinaFace.detect_faces("img2.jpg")

analysis = DeepFace.analyze(img_path = "img.jpg", actions = 
["age", "gender", "emotion", "race"]) 

num_of_faces = len(list(resp.keys()))

print(analysis)
print("face", num_of_faces)

