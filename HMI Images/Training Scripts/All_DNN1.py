#!/usr/bin/env python
# coding: utf-8

# In[32]:


import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
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
#import cv2
from shutil import copyfile


# In[39]:


# CurrentPath = os.getcwd()
# DatasetPath = os.path.join(CurrentPath,'All')
# Labels = os.listdir(DatasetPath)
# for label in Labels:
#     LabelPath = os.path.join(DatasetPath,label)
#     num_files = os.listdir(LabelPath)
#     training_num = int(0.8*len(num_files))
#     perm_vector = np.random.permutation(len(num_files))
#     training_files = perm_vector[:training_num]
#     testing_files = perm_vector[training_num:]
    
#     #Training


#     if not(os.path.exists(os.path.join(CurrentPath,'Train',label))):
#         os.mkdir(os.path.join(CurrentPath,'Train',label))
#     for i in training_files:       
#         copyfile(os.path.join(LabelPath,num_files[i]),os.path.join(CurrentPath,'Train',label,num_files[i]))
        
#     #Testing
#     if not(os.path.exists(os.path.join(CurrentPath,'Test',label))):
#         os.mkdir(os.path.join(CurrentPath,'Test',label))
#     for i in testing_files:       
#         copyfile(os.path.join(LabelPath,num_files[i]),os.path.join(CurrentPath,'Test',label,num_files[i]))


# Refer to paper for details in the markdown cells

# ### Experiment 1: Image size

# In[48]:


shape = (128,128,3)

model_1a = Sequential()
model_1a.add(Conv2D(32, (3, 3), input_shape=shape,padding='same'))
model_1a.add(Activation('relu'))
model_1a.add(MaxPooling2D(pool_size=(2, 2)))

model_1a.add(Conv2D(32, (3, 3), input_shape=shape,padding='same'))
model_1a.add(Activation('relu'))
model_1a.add(MaxPooling2D(pool_size=(2, 2)))

model_1a.add(Flatten())
model_1a.add(Dense(64))
model_1a.add(Activation('relu'))
model_1a.add(Dense(3))
model_1a.add(Activation('sigmoid'))
opt = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
model_1a.compile(loss='categorical_crossentropy',
              optimizer = opt,
              metrics=['accuracy'])



model_1a.summary()
#Load Data
batch_size = 16
path = os.getcwd()

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255)
        #,shear_range=0.2,
        #zoom_range=0.2,
        #horizontal_flip=True)


train_generator = train_datagen.flow_from_directory(
        path+'/Train',  # this is the target directory
        target_size=(128,128),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

val_datagen = ImageDataGenerator(
        rescale=1./255)


val_generator = train_datagen.flow_from_directory(
        path+'/Test',  # this is the target directory
        target_size=(128,128),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

model_1a.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,validation_data=val_generator,validation_steps=100,
        epochs=50)


# In[ ]:


shape = (256,256,3)

model_1b = Sequential()
model_1b.add(Conv2D(32, (3, 3), input_shape=shape,padding='same'))
model_1b.add(Activation('relu'))
model_1b.add(MaxPooling2D(pool_size=(2, 2)))

model_1b.add(Conv2D(32, (3, 3), input_shape=shape,padding='same'))
model_1b.add(Activation('relu'))
model_1b.add(MaxPooling2D(pool_size=(2, 2)))

model_1b.add(Flatten())
model_1b.add(Dense(64))
model_1b.add(Activation('relu'))
model_1b.add(Dense(3))
model_1b.add(Activation('sigmoid'))
opt = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
model_1b.compile(loss='categorical_crossentropy',
              optimizer = opt,
              metrics=['accuracy'])



model_1b.summary()
#Load Data
batch_size = 16
path = os.getcwd()

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255)
        #,shear_range=0.2,
        #zoom_range=0.2,
        #horizontal_flip=True)


train_generator = train_datagen.flow_from_directory(
        path+'/Train',  # this is the target directory
        target_size=(256,256),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

val_datagen = ImageDataGenerator(
        rescale=1./255)


val_generator = train_datagen.flow_from_directory(
        path+'/Test',  # this is the target directory
        target_size=(256,256),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

model_1b.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,validation_data=val_generator,validation_steps=100,
        epochs=50)


# In[ ]:


shape = (512,512,3)

model_1c = Sequential()
model_1c.add(Conv2D(32, (3, 3), input_shape=shape,padding='same'))
model_1c.add(Activation('relu'))
model_1c.add(MaxPooling2D(pool_size=(2, 2)))

model_1c.add(Conv2D(32, (3, 3), input_shape=shape,padding='same'))
model_1c.add(Activation('relu'))
model_1c.add(MaxPooling2D(pool_size=(2, 2)))

model_1c.add(Flatten())
model_1c.add(Dense(64))
model_1c.add(Activation('relu'))
model_1c.add(Dense(3))
model_1c.add(Activation('sigmoid'))
opt = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
model_1c.compile(loss='categorical_crossentropy',
              optimizer = opt,
              metrics=['accuracy'])



model_1c.summary()
#Load Data
batch_size = 16
path = os.getcwd()

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255)
        #,shear_range=0.2,
        #zoom_range=0.2,
        #horizontal_flip=True)


train_generator = train_datagen.flow_from_directory(
        path+'/Train',  # this is the target directory
        target_size=(512,512),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

val_datagen = ImageDataGenerator(
        rescale=1./255)


val_generator = train_datagen.flow_from_directory(
        path+'/Test',  # this is the target directory
        target_size=(512,512),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

model_1c.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,validation_data=val_generator,validation_steps=100,
        epochs=50)


# In[ ]:




