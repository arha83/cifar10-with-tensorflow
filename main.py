import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models



# data classes:
classes= ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# loading dataset:
(trainImages, trainLabels), (testImages, testLables)= datasets.cifar10.load_data()
# normalizing dataset:
trainImages, testImages= trainImages/255.0, testImages/255.0

# desgining tme model:
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
