from keras.utils import to_categorical                                               #imports
from keras_preprocessing.image import load_img
from keras.models import Sequential
from keras.applications import MobileNetV2, ResNet152, VGG16, EfficientNetB0, InceptionV3
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
import os
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

def createdataframe(dir):                                                                 #function accepts directory and returns two lists
    image_paths = []                                                                      #list to store image paths present in directory
    labels = []                                                                           #to store labels of images(sub folders in directory) in same order as the image paths
    for label in os.listdir(dir):                                                         #iterates through sub folders in directory's list of contents
        for imagename in os.listdir(os.path.join(dir, label)):                            #iterates through every image path in dir/label
            image_paths.append(os.path.join(dir, label, imagename))                       #adds dir/label/imagename to end of list
            labels.append(label)                                                          #appends corresponding label to label - this means image_paths and labels woukd have same no of elements
        print(label, "completed")                                                         #gets printed after each label completes
    return image_paths, labels                                                            #returns the two lists 

def extract_features(images):                                                             #function that takes in a list of image_paths
    features = []                                                                         #list to store features of the images - breaking down image into numbers
    for image in tqdm(images):                                                            #iterates through input images - tqdm provides a progress bar
        img = load_img(image, target_size=(236, 236))                                     #for given image, stores a 236x236x3 list like thing in img - storing a rgb list for every pixel
        img = np.array(img)                                                               
        features.append(img)                                                              #adds np array of img to features - this happens for all in images
    features = np.array(features)                                                         #converts features list to an nparray
    features = features.reshape(features.shape[0], 236, 236, 3)  # Reshape all images in one go
    return features    #returns the nparray features

TRAIN_DIR = "/kaggle/input/dataset/Train"

train = pd.DataFrame()
train['image'], train['label'] = createdataframe(TRAIN_DIR)                               #stores image paths and labels as a two series of the train dataframe - 'image' and 'label'

train_features = extract_features(train['image'])                                         #uses the 'image' series of the train dataframe to store nparray of features of training images in train_features

x_train = train_features / 255.0                                                          #x_train contains training features with all numbers between 0 and 1, since any rgb value is between 0 and 255

le = LabelEncoder()                                                                       
le.fit(train['label'])                                                                    #label encoder object le encodes labels in series 'label' which contains either 'AI' or 'Real'
y_train = le.transform(train['label'])                                                    #this stores the transformed version of the 'label' series in y_train
y_train = to_categorical(y_train, num_classes=2)                                          #this 'hot-encodes' the encoded label series, where each encoded label is replaced by a length 2 list where
                                                                                          #value at position of label is 1, everywhere else is 0

model = Sequential()                                                                      #model object using keras' Sequential API
# Convolutional layers
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(236, 236, 3)))   #adds CNN filters to our model, this one adds 32 filters, with relu function, and shape of each image being (236,236,3)
model.add(MaxPooling2D(pool_size=(2, 2)))                                                 #picks out maximum values from the image in patches of 2x2, leaving us with less numbers and 
                                                                                          #preventing model to get too obsessed with specific features of training data(overfitting)

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))                             #adds another CNN layer with 512 filters, similarly below   
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(1024, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())                                                                      #flattens the image  
model.add(Dense(1024, activation='relu'))                                                 #now makes a regular layer of 1024 neurons
model.add(Dropout(0.3))                                                                   #dropouts 30% of outputs, prevents overfitting
model.add(Dense(2048, activation='relu'))                                                 #another layer of 2048 neurons   
model.add(Dense(2, activation='softmax'))                                                 #output layer with 2 neurons, with softmax activations - 2 values that add to one   

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])    #adam is an algorithm for grad desc(something I dont understad yet), loss type and metrics specified
model.fit(x=x_train, y=y_train, batch_size=25, epochs=20)                                 #training command   
