from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import face_recognition
#import keras
from tensorflow.keras.models import load_model
import cv2

#Emotion detection
emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}

face_image  = cv2.imread("39.jpg") #This image is one that expresses Surprise
plt.imshow(face_image)
print (face_image.shape)
plt.show() #This displays the image on an x-y axis

#Resizing the image
face_image = cv2.resize(face_image, (48,48))
face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])
face_landmarks_list = face_recognition.face_landmarks(face_image)
print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))

#Load the model trained for detecting emotions of a face
model = load_model("emotion_detector_models/model_v6_23.hdf5")
print(face_image.shape)
predicted_class = np.argmax(model.predict(face_image))

#Predicted label (or emotion)
label_map = dict((v,k) for k,v in emotion_dict.items()) 
predicted_label = label_map[predicted_class]
print(predicted_label)
