import numpy as np
import sklearn
import sklearn.datasets as skd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn.feature_selection import mutual_info_classif,chi2,f_classif,f_regression,mutual_info_regression,SelectPercentile,SelectFpr,SelectFdr,SelectFwe,GenericUnivariateSelect
from sklearn.feature_selection import SelectKBest
import os
from string import digits
import re
import os.path
from sklearn.metrics import precision_recall_fscore_support,average_precision_score,precision_recall_curve

'''
Evaluation image test data
'''
import numpy as np
import tensorflow as tf
import keras
import os
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

shape = (256,256,3)
vgg=keras.applications.VGG16(include_top=False,weights='imagenet', input_shape=shape)
Model=keras.models.Sequential()
for capa in vgg.layers:
    Model.add(capa)
Model.layers.pop()

Model.add(Flatten())
Model.add(Dense(256, activation='relu'))
Model.add(Dropout(0.2))
Model.add(Dense(128, activation='relu'))
Model.add(Dropout(0.2))
Model.add(Dense(3, activation='softmax')) 
for layer in Model.layers:
    layer.trainable=False

Model.load_weights('NDSS_VGG16_078.h5')
opt = keras.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
Model.compile(loss='categorical_crossentropy',
              optimizer = opt,
              metrics=['accuracy'])
val_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 94
path = os.getcwd()
val_generator = val_datagen.flow_from_directory(
        path+'/Test',  # this is the target directory
        target_size=(256,256),  # all images will be resized to 256x256
        batch_size=batch_size,
        class_mode='categorical',shuffle=False)
Xts,yts = val_generator.next()
#print (yts)
scores = Model.evaluate(Xts,yts)
print ("Accuracy on images: ", scores[1])

'''
Evaluation on OCR test data
'''
fh = open('TransTextClean_Train.txt','r')
lines = fh.readlines()
fh.close()
NewLines = []
Labels = []
for num,line in enumerate(lines):
    temp = (line.strip()).split(',')
    temp2 = ' '.join(temp[1:])
    remove_digits = str.maketrans('', '', digits)
    temp3 = temp2.translate(remove_digits)
    temp4 = re.sub('[^A-Za-z0-9]+', ' ', temp3)
    NewLines.append(temp4)

    if 'Chemical_Sector' in temp[0] or 'chemical_sector' in temp[0]:
        Labels.append('Chemical_Sector')
    elif 'Energy_Sector' in temp[0] or 'energy_sector' in temp[0]:
        Labels.append('Energy_Sector')
    elif 'Water_and_Wastewater_Systems_Sector' in temp[0]:
        Labels.append('Water_and_Wastewater_Systems_Sector')
    else:
        print ("Line no.: ",num)
        print(temp[0])
Xtr_text = NewLines
ytr_text = Labels
    




fh = open('TransTextClean_Test.txt','r')
lines = fh.readlines()
fh.close()
NewLines = []
Labels = []
for num,line in enumerate(lines):
    temp = (line.strip()).split(',')
    temp2 = ' '.join(temp[1:])
    remove_digits = str.maketrans('', '', digits)
    temp3 = temp2.translate(remove_digits)
    temp4 = re.sub('[^A-Za-z0-9]+', ' ', temp3)
    NewLines.append(temp4)
    if 'Chemical_Sector' in temp[0] or 'chemical_sector' in temp[0]:
        Labels.append('Chemical_Sector')
    elif 'Energy_Sector' in temp[0] or 'energy_sector' in temp[0]:
        Labels.append('Energy_Sector')
    elif 'Water_and_Wastewater_Systems_Sector' in temp[0]:
        Labels.append('Water_and_Wastewater_Systems_Sector')
    else:
        print ("Line no.: ",num)
        print(temp[0])
Xts_text = NewLines
yts_text = Labels

CV = CountVectorizer()
x_train = CV.fit_transform(Xtr_text)
x_test = CV.transform(Xts_text)

from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
y_train = encoder.fit_transform(ytr_text)
y_train = np.where (y_train==1)[1]
y_test = encoder.fit_transform(yts_text)
y_test = np.where (y_test==1)[1]

from sklearn import preprocessing
binarizer = preprocessing.Binarizer(threshold=0.0)
k=1000
N = SelectKBest(chi2, k=k)
N.fit(x_train, y_train)
x_train_transformed = N.transform(x_train)
xb_train=binarizer.transform(x_train_transformed)

x_test_transformed = N.transform(x_test)
xb_test=binarizer.transform(x_test_transformed)  

import pickle
mnb = pickle.load(open('MNB.sav', 'rb'))

ac1=mnb.score(xb_test,y_test)
print ("Test accuracy for Chi-square at k = ",str(1000)," is ",ac1)


