import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, losses, callbacks



# data classes:
classes= ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# loading dataset:
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

# training:
print('### training...')
model.compile(optimizer='adam', loss=losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
model.fit(trainImages, trainLabels, epochs=1, validation_data=(testImages, testLables))
model.save(os.path.dirname(os.path.abspath(__file__))+'\\myModel')
