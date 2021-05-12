# importing libraries
import cv2
import dlib
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

# importing model
face_detector = dlib.get_frontal_face_detector()
emotion_model = load_model("./models/emotion_model.hdf5")

# initializing dictionary
emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
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

        gray_face = cv2.resize(gray_face, (48, 48), interpolation=cv2.INTER_AREA)

        gray_face = gray_face.astype("float")/255.0

        gray_face = img_to_array(gray_face)

        gray_face = np.expand_dims(gray_face, axis=0)

        predicted_emotion = np.argmax(emotion_model.predict(gray_face))
        cv2.putText(frame, emotions[predicted_emotion], (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, str(predicted_emotion), (180, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    except IndexError:
        print("No face detected!")

    # showing the bounded image
    cv2.imshow("Webcam", frame)

    # exiting the loop when "esc" key is pressed
    if cv2.waitKey(1) == 27:
        break