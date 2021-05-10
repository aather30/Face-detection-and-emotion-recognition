# importing libraries
import cv2
import dlib

# importing model
face_detector = dlib.get_frontal_face_detector()
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

    except IndexError:
        print("No face detected!")

    # showing the bounded image
    cv2.imshow("Webcam", frame)

    # exiting the loop when "esc" key is pressed
    if cv2.waitKey(1) == 27:
        break