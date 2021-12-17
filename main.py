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
print('### loading dataset...')
(trainImages, trainLabels), (testImages, testLables)= datasets.cifar10.load_data()
# normalizing dataset:
print('### resizing dataset...')
trainImages, testImages= trainImages/255.0, testImages/255.0

ch1= input('### do you want new model? (y/n) ')
if ch1 == 'y':
    # desgining the model:
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

else:
    # loading model:
    print('### loading model...')
    model= models.load_model(os.path.dirname(os.path.abspath(__file__))+'\\myModel')
print(model.summary())

# training:
ch2= 'y'
if ch1 != 'y': ch2= input('### do you want to train model? (y/n) ')
if ch2 == 'y' or ch1 == 'y':
    epochs= int(input('enter the number of epochs for training: '))
    print('### training...')
    model.compile(optimizer='adam', loss=losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    model.fit(trainImages, trainLabels, epochs=epochs, validation_data=(testImages, testLables))
    model.save(os.path.dirname(os.path.abspath(__file__))+'\\myModel')

# predicting:
while True:
    path= input('enter the image path (enter q to exit)\n')
    if path == 'q': break
    image= cv.imread(path)
    image= cv.cvtColor(image, cv.COLOR_BGR2RGB)
    pre= np.expand_dims(image, 0)
    predictions= model.predict([pre])
    maxi= np.where(predictions[0] == np.amax(predictions[0]))[0][0]
    for i in range(10): print(classes[i], ': ', predictions[0][i], sep='')
    print('\nclosest class:',classes[maxi],'\n\n')

print('\n\n')