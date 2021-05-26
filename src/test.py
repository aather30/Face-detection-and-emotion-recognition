import tkinter as tk
import numpy as np
from PIL import Image, ImageSequence
import cv2
import time
import os
import matplotlib.pyplot as plt

# root = tk.Tk()

sprite_image_path = "/Users/aliather/Documents/Fiverr/no_13ody/Face-detection-and-emotion-recognition/params/sprites/sad/giphy.gif"

ims = []
# save frames

im = Image.open(sprite_image_path)


frames = im.n_frames
for i in range(frames):
    im.seek(i)
    ims.append(im.convert('RGB'))
    #im.save('{}.png'.format(i))

count = 0

def animation():
    global count

    im2 = ims[count]
    
    cv2.imshow("hi",np.asarray(im2)[:,:,::-1])
    

    count+=1

    if count == len(ims):
        count = 0
  
    
while(True):
    time.sleep(0.1)
    animation()
    key = cv2.waitKey(1)

    if key == ord("q"):
        break
