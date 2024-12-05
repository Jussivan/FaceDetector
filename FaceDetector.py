from mtcnn import MTCNN
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

detector = MTCNN()
model = load_model('model.h5')

def preprocess_face(image, box):
    x, y, width, height = box
    face = image[y:y+height, x:x+width]
    face = cv2.resize(face, (160, 160))
    face = face / 255.0
    return np.expand_dims(face, axis=0)

def recognize_face(embedding, known_embeddings, labels):
    distances = np.linalg.norm(known_embeddings - embedding, axis=1)
    min_distance_idx = np.argmin(distances)
    return labels[min_distance_idx]

image_path = "image.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
faces = detector.detect_faces(image_rgb)

known_embeddings = np.array([[...]])
labels = ["Pessoa 1", "Pessoa 2", "Pessoa 3"]

recognized_names = []
for face in faces:
    face_img = preprocess_face(image_rgb, face['box'])
    embedding = model.predict(face_img)
    recognized_name = recognize_face(embedding, known_embeddings, labels)
    recognized_names.append(recognized_name)

for face, name in zip(faces, recognized_names):
    x, y, width, height = face['box']
    cv2.rectangle(image_rgb, (x, y), (x+width, y+height), (255, 0, 0), 2)
    cv2.putText(image_rgb, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

plt.imshow(image_rgb)
