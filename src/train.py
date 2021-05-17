# importing libraries
import os
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, AveragePooling2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Model

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.backend import binary_crossentropy


def main():
    path = "../dataset/"
    epochs = 20
    # reading the images in to a single array
    x, y = reading_images(path)

    x = np.array(x, dtype="float32")
    y = np.array(y)

    lb = LabelBinarizer()

    # one hot encoding the labels
    y = lb.fit_transform(y)

    # splitting the data
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=1, shuffle=True)

    # data augmentation
    dg  = ImageDataGenerator(zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, fill_mode="nearest")
    
    face_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

    # defining main model
    main_model = face_model.output
    main_model = AveragePooling2D(pool_size=(7, 7))(main_model)

    main_model = Flatten()(main_model)

    main_model = Dense(128, activation="relu")(main_model)

    main_model = Dropout(0.5)(main_model)

    main_model = Dense(len(os.listdir(path)), activation="softmax")(main_model)

    model = Model(inputs=face_model.input, outputs=main_model)

    for layer in face_model.layers:
        layer.trainable = False
    
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    history = model.fit(dg.flow(train_x, train_y, batch_size=10), epochs = epochs, validation_data=(test_x, test_y))

    plt.figure()
    plt.plot(np.arange(0, epochs), history.history["loss"], label="Training Loss")
    plt.plot(np.arange(0, epochs), history.history["val_loss"], label="Validation Loss")

    plt.plot(np.arange(0, epochs), history.history["accuracy"], label="Training Accuracy")  
    plt.plot(np.arange(0, epochs), history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Loss and Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower right")
    plt.show()
    model.save("../models/model.h5")
def reading_images(path):
    images = []
    image_labels = []
    labels = [x for x in os.listdir(path)]
    for label in labels:
        for image in os.listdir(path + label + "/"):
            images.append(preprocess_input(img_to_array(load_img(path + label + "/" + image, target_size=(224, 224)))))
            image_labels.append(label)

    return images, image_labels

main()