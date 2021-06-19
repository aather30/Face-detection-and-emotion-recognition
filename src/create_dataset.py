# importing libraries
import cv2
import dlib
import pandas as pd
import os

# importing model
face_detector = dlib.get_frontal_face_detector()

# getting the emotions from a csv file
emotions = pd.read_csv("../params/emotions.csv", header=None).to_numpy().flatten()

# make folders for dataset and models 
if not os.path.isdir("../dataset"):
    os.mkdir("../dataset")

if not os.path.isdir("../models"):
    os.mkdir("../models")

if ".DS_Store" in os.listdir("../dataset"):
    os.remove("../dataset/.DS_Store")

# making directories for every emotion
for emotion in emotions:
    
    if os.path.isdir("../dataset/" + emotion):
        continue

    # making a directory for every emotion
    os.mkdir("../dataset/" + emotion)

# making directories for every emotion
for emotion in emotions:
    
    if os.path.isdir("../params/sprites/" + emotion):
        continue

    # making a directory for every emotion
    os.mkdir("../params/sprites/" + emotion)

# starting webcam
webcam = cv2.VideoCapture(0)
webcam.set(3,640)
webcam.set(4,480)

emotion_path = ""   
emotion_str = ""
enter_text = ""

left = 500
top = 100
right = 900
bottom = 500
# looping through every frame
while True:
    # storing a single frame from the webcam and saving it in a variable
    _, frame = webcam.read()

    try: 
        # converting frame from bgr to grey and detecting the face as well
        #face = face_detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1)[0]
        
        # creating a boundary of the face detected
        frame = cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # showing menu on the screen
        cv2.putText(frame, "Select a number!", (10, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)

        for index, emotion in enumerate(emotions):
            option = str(index + 1) + ". " + emotion
            cv2.putText(frame, option, (10, 25+25*(index+1)),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)
  
        cv2.putText(frame, "Q - Quit", (10, 25+25 * (len(emotions) + 1)),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)
        cv2.putText(frame, emotion_str, (600, 25),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)
        cv2.putText(frame, enter_text, (600, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)
        

    except IndexError:
        print("[ERROR] No face detected!")

    # showing the bounded image
    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1)

    # exiting the loop when "Q" key is pressed
    if key == ord("q"):
        break
    elif (key - ord("0")) in range(1, len(emotions)+1):
        emotion_str = f"Make a {emotions[key - ord('0')-1]} face!"
        enter_text = "Press Enter to Capture."
        emotion_path = "../dataset/" + emotions[key - ord('0') - 1] + "/"
    elif key == 13 or key == 10:
        try:
            gray_face = cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
            cv2.imwrite(emotion_path + str(len(os.listdir(emotion_path))) + ".jpg", gray_face)

        except Exception:
            print("[ERROR] Could not save the picture!")# importing libraries