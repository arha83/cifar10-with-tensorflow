import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models


# data classes:
classes= ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# loading dataset:
(trainImages, trainLabels), (testImages, testLables)= datasets.cifar10.load_data()
# normalizing dataset:
trainImages, testImages= trainImages/255.0, testImages/255.0

