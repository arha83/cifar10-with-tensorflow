import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, losses
import cv2 as cv
import numpy as np




# data classes:

'''
airplane : 0
automobile : 1
bird : 2
cat : 3
deer : 4
dog : 5
frog : 6
horse : 7
ship : 8
truck : 9
'''
classes= ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# loading dataset:
'''
print('### loading dataset...')
(trainImages, trainLabels), (testImages, testLables)= datasets.cifar10.load_data()
# normalizing dataset:
print('### resizing dataset...')
trainImages, testImages= trainImages/255.0, testImages/255.0

# desgining tme model:
print('### building model...')
model= models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
# loading model:
print('### loading model...')
model= models.load_model(os.path.dirname(os.path.abspath(__file__))+'\\myModel')
'''
# training:
'''
print('### training...')
model.compile(optimizer='adam', loss=losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
model.fit(trainImages, trainLabels, epochs=10, validation_data=(testImages, testLables))
model.save(os.path.dirname(os.path.abspath(__file__))+'\\myModel')
'''

model= models.load_model(os.path.dirname(os.path.abspath(__file__))+'\\myModel')
image= cv.imread('C:\\Users\\amirr\\Desktop\\cifar10\\ship.jpg')
image= cv.cvtColor(image, cv.COLOR_BGR2RGB)
pre= np.expand_dims(image, 0)
predictions= model.predict([pre])
maxi= np.where(predictions[0] == np.amax(predictions[0]))[0][0]
for i in range(10):
    print(classes[i], ': ', predictions[0][i], sep='')
print('\nclosest class:',classes[maxi],'\n\n')