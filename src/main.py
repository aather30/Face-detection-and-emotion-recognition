# importing libraries
import cv2
import dlib
import os
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# importing model
face_detector = dlib.get_frontal_face_detector()

# laoding model
model = load_model("../models/model.h5")

labels = [x for x in sorted(os.listdir("../dataset/"))]

# starting webcam
webcam = cv2.VideoCapture(0)

# looping through every frame
while True:
    # storing a single frame from the webcam and saving it in a variable
    _, frame = webcam.read()

    try: 
        # converting frame from bgr to grey and detecting the face as well
        face = face_detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1)[0]
        
        # creating a boundary of the face detected
        frame = cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 2)
        
        gray_face = cv2.cvtColor(frame[face.top():face.bottom(), face.left():face.right()], cv2.COLOR_BGR2GRAY)

        gray_face = cv2.cvtColor(gray_face, cv2.COLOR_GRAY2BGR)

        gray_face = preprocess_input(img_to_array(cv2.resize(gray_face, (224, 224))))

        gray_face = np.expand_dims(gray_face, axis = 0)

        predicted_emotion = labels[np.argmax(model.predict(gray_face))]


        cv2.putText(frame, "Predicted Emotion: " + predicted_emotion, (10, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)

        cv2.putText(frame, "Q - Quit", (10, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)

    except IndexError:
        print("[ERROR] No face detected!")

    # showing the bounded image
    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1)

    # exiting the loop when "Q" key is pressed
    if key == ord("q"):
        break