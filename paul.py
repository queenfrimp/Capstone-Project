####import cv2
####import numpy as np
####video_capture = cv2.VideoCapture(0)
####facecascade = cv2.CascadeClassifier("/Users/queenfrimpong/capstone/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_default.xml")
####def crop_face(gray, face): #Crop the given face
####    for (x, y, w, h) in face:
####        faceslice = gray[y:y+h, x:x+w]
####    return faceslice
####while True:
####    ret, frame = video_capture.read() #Grab frame from webcam. Ret is 'true' if the frame was successfully grabbed.
####    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert image to grayscale to improve detection speed and accuracy
####    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
####    clahe_image = clahe.apply(gray)
####    #Run classifier on frame
####    face = facecascade.detectMultiScale(clahe_image, scaleFactor=1.1, minNeighbors=15, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)
####    for (x, y, w, h) in face: #Draw rectangle around detected faces
####        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2) #draw it on "frame", (coordinates), (size), (RGB color), thickness 2
####    if len(face) == 1: #Use simple check if one face is detected, or multiple (measurement error unless multiple persons on image)
####        faceslice = crop_face(gray, face) #slice face from image
####        cv2.imshow("detect", faceslice) #display sliced face
####    else:
####        print("no/multiple faces detected, passing over frame")
####    cv2.imshow("webcam", frame) #Display frame
####    if cv2.waitKey(1) & 0xFF == ord('q'): #imshow expects a termination definition to work correctly, here it is bound to key 'q'
####        break

import sys
import dlib
import cv2

detector = dlib.get_frontal_face_detector()
cam = cv2.VideoCapture(0)
color_green = (0,255,0)
line_width = 3
while True:
    ret_val, img = cam.read()
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dets = detector(rgb_image)
    for det in dets:
        cv2.rectangle(img,(det.left(), det.top()), (det.right(), det.bottom()), color_green, line_width)
    cv2.imshow('my webcam', img)
    if cv2.waitKey(1) == 27:
        break  # esc to quit
cv2.destroyAllWindows()
