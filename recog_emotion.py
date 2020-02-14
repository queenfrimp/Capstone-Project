from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import face_recognition
#import keras
from tensorflow.keras.models import load_model
import cv2

#Face Recognition
image1 = Image.open("040wrmpyTF5l.jpg")
image_array1 = np.array(image1)
plt.imshow(image_array1)

image = face_recognition.load_image_file("040wrmpyTF5l.jpg")

face_locations = face_recognition.face_locations(image)

top, right, bottom, left = face_locations[0]
face_image1 = image[top:bottom, left:right]
plt.imshow(face_image1)
image_save = Image.fromarray(face_image1)
image_save.save("image_1.jpg")

top, right, bottom, left = face_locations[1]
face_image2 = image[top:bottom, left:right]
plt.imshow(face_image2)
image_save = Image.fromarray(face_image2)
image_save.save("image_2.jpg")

image1 = Image.open("index2.jpeg")
image_array1 = np.array(image1)
plt.imshow(image_array1)

image2 = Image.open("index1.jpg")
image_array2 = np.array(image2)
plt.imshow(image_array2)

image1 = face_recognition.load_image_file("index1.jpg")
image2 = face_recognition.load_image_file("index2.jpeg")
 
encoding_1 = face_recognition.face_encodings(image1)[0]

encoding_2 = face_recognition.face_encodings(image1)[0]

results = face_recognition.compare_faces([encoding_1], encoding_2,tolerance=0.50)

print(results)

image1 = face_recognition.load_image_file("index1.jpg")
image2 = face_recognition.load_image_file("rajeev.jpg")
 
encoding_1 = face_recognition.face_encodings(image1)[0]

encoding_2 = face_recognition.face_encodings(image2)[0]

result = face_recognition.compare_faces([encoding_1], encoding_2,tolerance=0.50)

print(result)

#Emotion detection
emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}

face_image  = cv2.imread("39.jpg")
plt.imshow(face_image)
print (face_image.shape)

#Resizing the image
face_image = cv2.resize(face_image, (48,48))
face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])

#Load the model trained for detecting emotions of a face
model = load_model("emotion_detector_models/model_v6_23.hdf5")
print(face_image.shape)
predicted_class = np.argmax(model.predict(face_image))

#Successful Emotion detection
#Predicted label (or emotion)
label_map = dict((v,k) for k,v in emotion_dict.items()) 
predicted_label = label_map[predicted_class]
print(predicted_label)
