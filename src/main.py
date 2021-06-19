# importing libraries
import cv2
import dlib
import os
import numpy as np
import random
from PIL import Image
import threading

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
webcam.set(3,640)
webcam.set(4,480)

predicted_emotion = ""
prev_emotion = " "
ims = []
count = 0
anim_flag = False
thread_flag = True
show_frame = np.zeros((512,512))

# show menu
print("S - Play/Pause a Gif")
print("Q - Quit")

def animation():
    global count
    global ims
    
    if count >= len(ims):
        count = 0

    show_im = cv2.cvtColor(np.asarray(ims[count]),cv2.COLOR_RGBA2BGRA)
    count+=1
    
    return show_im

   
def detect_emotion(frame):
    # creating a boundary of the face detected
    frame = cv2.rectangle(frame, (500, 100), (900, 500), (0, 255, 0), 2)
    
    gray_face = cv2.cvtColor(frame[100:500, 500:900], cv2.COLOR_BGR2GRAY)

    gray_face = cv2.cvtColor(gray_face, cv2.COLOR_GRAY2BGR)

    gray_face = preprocess_input(img_to_array(cv2.resize(gray_face, (224, 224))))

    gray_face = np.expand_dims(gray_face, axis = 0)

    predicted_emotion = labels[np.argmax(model.predict(gray_face))]

    cv2.putText(frame, predicted_emotion, (600, 55),cv2.FONT_HERSHEY_SIMPLEX, 2,(255,255,255), 2)

    return predicted_emotion

def input_handler(inp):
    global predicted_emotion
    global anim_flag
    global ims

    ims = []

    if inp == "q":
        os._exit(1)
    
    elif inp == "s":
        sprite_im_path = "../params/sprites/" + predicted_emotion + "/"
        sprite_im_path += os.listdir(sprite_im_path)[random.randint(0,len(os.listdir(sprite_im_path))-1)]
        
        print(sprite_im_path)

        sprite_im = Image.open(sprite_im_path)
        
        for i in range(sprite_im.n_frames):
            sprite_im.seek(i)
            ims.append(sprite_im.convert("RGBA"))
        
        anim_flag =  not anim_flag

def wait_for_input(prompt='Enter Option:\n'):
    global thread_flag

    inp = input(prompt)
    input_handler(inp)

    thread_flag = True

def make_thread():
    global thread_flag

    
    thread_flag = False
    x = threading.Thread(target=wait_for_input, args=())
    x.start()


def main():
    global thread_flag
    global anim_flag
    global predicted_emotion
    global show_frame
    global prev_emotion
    # looping through every frame
    while True:
        # storing a single frame from the webcam and saving it in a variable
        _, frame = webcam.read()

        try: 
            # converting frame from bgr to grey and detecting the face as well
            #face = face_detector(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1)[0]
            #frame = cv2.rectangle(frame, (500, 100), (900, 500), (0, 255, 0), 2)
            #face = cv2.cvtColor(frame[100:500, 500:900], cv2.COLOR_BGR2GRAY)

            predicted_emotion = detect_emotion(frame)
            if prev_emotion != predicted_emotion:
                print(predicted_emotion)
            prev_emotion = predicted_emotion

            if thread_flag:
                make_thread()
            
            if anim_flag:

                show_frame = animation()

        except IndexError:
            print("[ERROR] No face detected!")
        # showing the bounded image
        cv2.imshow("Sprite", show_frame)
        cv2.imshow("Webcam", frame)
        cv2.waitKey(1)
   
main()


