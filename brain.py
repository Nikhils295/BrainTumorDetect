import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os  # interacting with the operating system
for dirname, _, filenames in os.walk('C:\Users\NIKHIL SAXENA\Desktop'):  # iterate over files
    for filename in filenames:  # iterate over filenames
        print(os.path.join(dirname, filename))  # print full file paths

import keras  # deep learning library
from keras.models import Sequential  # for building sequential models
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout  # layers for the model
from sklearn.metrics import accuracy_score  # for evaluating accuracy

import ipywidgets as widgets  # for interactive widgets
import io  # for input/output operations
from PIL import Image  # image processing
import tqdm  # progress bar
from sklearn.model_selection import train_test_split  # splitting data
import cv2  # OpenCV for image processing
from sklearn.utils import shuffle  # shuffling the data
import tensorflow as tf  # TensorFlow library

# Data preparation
X_train = []  # initialize empty list for training images
Y_train = []  # initialize empty list for training labels
image_size = 150  # set image size
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']  # list of labels

for i in labels:  # iterate over labels
    folderPath = os.path.join('"C:\Users\NIKHIL SAXENA\Desktop\Brain_Tumor\Brain_Tumor\Training"', i)  # path to training folder
    for j in os.listdir(folderPath):  # iterate over images in folder
        img = cv2.imread(os.path.join(folderPath, j))  # read image
        img = cv2.resize(img, (image_size, image_size))  # resize image
        X_train.append(img)  # append image to training data
        Y_train.append(i)  # append label to training labels

for i in labels:  # iterate over labels
    folderPath = os.path.join('"C:\Users\NIKHIL SAXENA\Desktop\Brain_Tumor\Brain_Tumor\Testing"', i)  # path to testing folder
    for j in os.listdir(folderPath):  # iterate over images in folder
        img = cv2.imread(os.path.join(folderPath, j))  # read image
        img = cv2.resize(img, (image_size, image_size))  # resize image
        X_train.append(img)  # append image to training data
        Y_train.append(i)  # append label to training labels

X_train = np.array(X_train)  # convert training data to numpy array
Y_train = np.array(Y_train)  # convert training labels to numpy array

X_train, Y_train = shuffle(X_train, Y_train, random_state=101)  # shuffle the data
X_train.shape  # get the shape of training data

X_train, X_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.1, random_state=101)  # split data

# One-hot encoding
y_train_new = []  # initialize empty list for new training labels
for i in y_train:  # iterate over training labels
    y_train_new.append(labels.index(i))  # convert label to index
y_train = y_train_new  # update training labels
y_train = tf.keras.utils.to_categorical(y_train)  # one-hot encode training labels

y_test_new = []  # initialize empty list for new testing labels
for i in y_test:  # iterate over testing labels
    y_test_new.append(labels.index(i))  # convert label to index
y_test = y_test_new  # update testing labels
y_test = tf.keras.utils.to_categorical(y_test)  # one-hot encode testing labels

# Model building
model = Sequential()  # initialize sequential model

# Adding a convolutional layer with 32 filters, each of size 3x3, and ReLU activation function
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))

# Add another convolutional layer with 64 filters of size 3x3
model.add(Conv2D(64, (3, 3), activation='relu'))

# Add a max pooling layer with a pool size of 2x2. This reduces the spatial dimensions of the output.
model.add(MaxPooling2D(2, 2))

# Add a dropout layer with a dropout rate of 0.3
model.add(Dropout(0.3))

# Add another convolutional layer with 64 filters of size 3x3 and ReLU activation function
model.add(Conv2D(64, (3, 3), activation='relu'))

# Add another convolutional layer with 64 filters of size 3x3 and ReLU activation function
model.add(Conv2D(64, (3, 3), activation='relu'))

# Add another dropout layer with a dropout rate of 0.3
model.add(Dropout(0.3))

# Add another max pooling layer with a pool size of 2x2
model.add(MaxPooling2D(2, 2))

# Add another dropout layer with a dropout rate of 0.3
model.add(Dropout(0.3))

# Add another convolutional layer with 128 filters of size 3x3 and ReLU activation function
model.add(Conv2D(128, (3, 3), activation='relu'))

# Add another convolutional layer with 128 filters of size 3x3 and ReLU activation function
model.add(Conv2D(128, (3, 3), activation='relu'))

# Add another convolutional layer with 128 filters of size 3x3 and ReLU activation function
model.add(Conv2D(128, (3, 3), activation='relu'))

# Add another max pooling layer with a pool size of 2x2
model.add(MaxPooling2D(2, 2))

# Add another dropout layer with a dropout rate of 0.3
model.add(Dropout(0.3))

# Add another convolutional layer with 128 filters of size 3x3 and ReLU activation function
model.add(Conv2D(128, (3, 3), activation='relu'))

# Add another convolutional layer with 256 filters of size 3x3 and ReLU activation function
model.add(Conv2D(256, (3, 3), activation='relu'))

# Add another max pooling layer with a pool size of 2x2
model.add(MaxPooling2D(2, 2))

# Add another dropout layer with a dropout rate of 0.3
model.add(Dropout(0.3))

# Add a flatten layer to convert the 2D output to a 1D vector
model.add(Flatten())

# Add a dense (fully connected) layer with 512 units and ReLU activation function
model.add(Dense(512, activation='relu'))

# Add another dense (fully connected) layer with 512 units and ReLU activation function
model.add(Dense(512, activation='relu'))

# Add another dropout layer with a dropout rate of 0.3
model.add(Dropout(0.3))

# Add the output layer with 4 units (one for each class) and softmax activation function
# Softmax activation function is used for multi-class classification
model.add(Dense(4, activation='softmax'))
model.summary()  # print model summary

# Model compilation and training
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])  # compile model

history = model.fit(X_train, y_train, epochs=20, validation_split=0.1)  # train model

import matplotlib.pyplot as plt  # for plotting
import seaborn as sns  # for visualization

# Save the model
model.save("brain_tumor_classification_model.h5")  # Save the model to a file

# Plot accuracy
acc = history.history['accuracy']  # get training accuracy
val_acc = history.history['val_accuracy']  # get validation accuracy
epochs = range(len(acc))  # get epochs
fig = plt.figure(figsize=(14, 7))  # create figure
plt.plot(epochs, acc, 'r', label="Training Accuracy")  # plot training accuracy
plt.plot(epochs, val_acc, 'b', label="Validation Accuracy")  # plot validation accuracy
plt.legend(loc='upper left')  # add legend
plt.show()

# Plot loss
loss = history.history['loss']  # get training loss
val_loss = history.history['val_loss']  # get validation loss
epochs = range(len(loss))  # get number of epochs
fig = plt.figure(figsize=(14, 7))  # create figure
plt.plot(epochs, loss, 'r', label="Training loss")  # plot training loss
plt.plot(epochs, val_loss, 'b', label="Validation loss")  # plot validation loss
plt.legend(loc='upper left')  # add legend
plt.show()  # show plot

# Image prediction
img = cv2.imread('C:\Users\NIKHIL SAXENA\Desktop\Brain_Tumor\Brain_Tumor\Training\pituitary_tumor')  # read image
img = cv2.resize(img, (150, 150))  # resize image
img_array = np.array(img)  # convert to array
img_array.shape  # get shape of array

img_array = img_array.reshape(1, 150, 150, 3)  # reshape array
img_array.shape  # get shape of array

from tensorflow.keras.preprocessing import image  # image preprocessing
img = image.load_img('C:\Users\NIKHIL SAXENA\Desktop\Brain_Tumor\Brain_Tumor\Training\pituitary_tumor')  # load image
plt.imshow(img, interpolation='nearest')  # show image
plt.show()  # display plot

# Predict class
a = model.predict(img_array)  # predict image class
indices = a.argmax()  # get predicted class index

# Output tumor name
tumor_name = labels[indices]  # get tumor name from labels
print(f"The predicted tumor type is: {tumor_name}")  # print tumor name
