import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split, StratifiedKFold
import os
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

shape = (256,256,3)

model_1b = Sequential()
model_1b.add(Conv2D(32, (3, 3), input_shape=shape,padding='same'))
model_1b.add(Conv2D(32, (3, 3), input_shape=shape,padding='same'))
model_1b.add(Activation('relu'))
model_1b.add(MaxPooling2D(pool_size=(2, 2)))
model_1b.add(Dropout(0.1))

model_1b.add(Conv2D(64, (3, 3), input_shape=shape,padding='same'))
model_1b.add(Activation('relu'))
model_1b.add(MaxPooling2D(pool_size=(2, 2)))
model_1b.add(Dropout(0.1))
model_1b.add(Flatten())
model_1b.add(Dense(64))
model_1b.add(Activation('relu'))
model_1b.add(Dropout(0.1))
model_1b.add(Dense(3))
model_1b.add(Activation('sigmoid'))
opt = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
model_1b.compile(loss='categorical_crossentropy',
              optimizer = opt,
              metrics=['accuracy'])

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'NDSS_3layers_orig.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)
callbacks=[checkpoint]
model_1b.summary()
#Load Data
batch_size = 16
path = os.getcwd()

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255)
        #width_shift_range=0.1,
        #height_shift_range=0.1,
        #zoom_range=0.2,
        #horizontal_flip=True)


train_generator = train_datagen.flow_from_directory(
        path+'/Train',  # this is the target directory
        target_size=(256,256),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

val_datagen = ImageDataGenerator(
        rescale=1./255)


val_generator = val_datagen.flow_from_directory(
        path+'/Test',  # this is the target directory
        target_size=(256,256),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

model_1b.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,validation_data=val_generator,validation_steps=100,
        epochs=100,callbacks=callbacks)
