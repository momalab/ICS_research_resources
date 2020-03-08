import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split, StratifiedKFold
import os
from keras import applications
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from keras.layers import BatchNormalization
import os
from keras.callbacks import ModelCheckpoint
#import cv2
from shutil import copyfile
shape = (224,224,3)
model_5 = Sequential()
base_model = applications.VGG16(include_top=False,weights='imagenet', input_shape=shape)
for layer in base_model.layers:
    model_5.add(layer)
    
for mLayer in model_5.layers:
    mLayer.trainable=False

model_5.add(Flatten())
model_5.add(Dense(224, activation='relu'))
model_5.add(Dropout(0.15))
model_5.add(Dense(128, activation='relu'))
model_5.add(Dropout(0.15))
model_5.add(Dense(3, activation='sigmoid'))
opt = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)

model_5.compile(loss='categorical_crossentropy',
              optimizer = opt,
              metrics=['accuracy'])

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'Medi_VGG16_{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)
callbacks=[checkpoint]

model_5.summary()

batch_size = 16
path = os.getcwd()

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        width_shift_range=0.1,
        height_shift_range=0.1)


train_generator = train_datagen.flow_from_directory(
        path+'/Train',  # this is the target directory
        target_size=(224,224),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

val_datagen = ImageDataGenerator(
        rescale=1./255)


val_generator = val_datagen.flow_from_directory(
        path+'/Test',  # this is the target directory
        target_size=(224,224),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

model_5.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,validation_data=val_generator,validation_steps=100,
        epochs=100,callbacks=callbacks)
