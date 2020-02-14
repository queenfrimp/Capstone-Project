#Import required modules
import sys
import cv2
import dlib
from sklearn.svm import SVC

#Set up some required objects
#video_capture = cv2.VideoCapture(0) #Webcam object
detector = dlib.get_frontal_face_detector() #Face detector
win = dlib.image_window()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Landmark identifier. Set the filename to whatever you named the downloaded file

for f in sys.argv[1:]:
    print("Processing file: {}".format(f))
    img = dlib.load_rgb_image(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)
    detections = detector(clahe_image, 1) #Detect the faces in the image

    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(clahe_image, d) #Draw Facial Landmarks with the predictor class
        landmarks = []
        xlist = []
        ylist = []
        for i in range(1,68): #Store X and Y coordinates in two lists
            cv2.circle(img, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2) #For each point, draw a red circle with thickness2 on the original frame
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        for x, y in zip(xlist, ylist): #Store all landmarks in one list in the format x1,y1,x2,y2,etc.
            landmarks.append(x)
            landmarks.append(y)
    if len(detections) > 0:
        print("Landmarks:", landmarks)
    else: #If no faces are detected, return error message to other function to handle
        landmarks = "error"
        print("Error")
    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(detections)
    dlib.hit_enter_to_continue()
