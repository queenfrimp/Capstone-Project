#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to find frontal human faces in an image.  In
#   particular, it shows how you can take a list of images from the command
#   line and display each on the screen with red boxes overlaid on each human
#   face.
#
#   You can run this program on images and see the detections by executing the
#   following command:
#       python3 face_detect.py png2jpg/*.jpg
#   Remember to run from project folder


import sys
import cv2
import dlib
from sklearn.svm import SVC

#Face Detection
detector = dlib.get_frontal_face_detector() #Face detector
win = dlib.image_window()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever you named the downloaded file

for f in sys.argv[1:]:
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)
    
#Facial landmarking begins
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)
    detections = detector(clahe_image, 1) #Detect the faces in the image
    xlist = []
    ylist = []
    landmarks = []
    for k,d in enumerate(detections): #For each detected face
        shape = predictor(clahe_image, d) #Get coordinates
        for i in range(1,68): #There are 68 landmark points on each face
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
            cv2.circle(img, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2) #For each point, draw a red circle with thickness2 on the original frame
        for x, y in zip(xlist, ylist): #Store all landmarks in one list in the format x1,y1,x2,y2,etc.
            landmarks.append(x)
            landmarks.append(y)
            
    #Printing the stored features that have been extracted
    if len(detections) > 0:
        print("Landmarks:", landmarks)
    else: #If no faces are detected, return error message to other function to handle
        print("Error")
    
    #cv2.imshow("Image", img) #Display the frame
    if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
       break
#Facial landmarking ends
    
    dets = detector(clahe_image, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for i, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            i, d.left(), d.top(), d.right(), d.bottom()))

    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(dets)
    dlib.hit_enter_to_continue()
    #recognize_emotion()

if (len(sys.argv[1:]) > 0):
    img = dlib.load_rgb_image(sys.argv[1])
    dets, scores, idx = detector.run(img, 1, -1)
    for i, d in enumerate(dets):
        print("Detection {}, score: {}, face_type:{}".format(d, scores[i], idx[i]))

