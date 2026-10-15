import csv
import pandas as pd
import scipy.io
import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# Change to the directory
import os
# Already in the correct directory

# Load data as in Healthy_Unhealthy_Classification_clean.py
data = np.loadtxt('dataa/healthy_unhealthy1.csv', delimiter=',')
X = data[:, :1024]  # Features
y = data[:, -1]     # Labels (0 healthy, 1 unhealthy)
X = X.reshape((X.shape[0], 1024, 1))

# Split data
x_train1, x_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Squeeze to 2D for the models
x_train1 = x_train1.squeeze()
x_test1 = x_test1.squeeze()

import tensorflow.keras.layers as tfl
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy

def create_model(param):
    model = Sequential()
    model.add(tfl.Conv1D(filters=32,
                    kernel_size=7,
                    strides=1,
                    padding='same',
            input_shape=(1024,1)))
    model.add(tfl.Conv1D(filters=8,
                    kernel_size=3,
                    strides=1,
                    padding='same', activation='relu'))
    model.add(tfl.MaxPool1D(pool_size= 16,
                       padding='same'))
    model.add(tfl.Conv1D(filters=16,
                    kernel_size=2,
                    strides=1,
                    padding='same'))
    model.add(tfl.Dropout(0.4))
    model.add(tfl.Conv1D(filters=8,
                    kernel_size=2,
                    strides=1,
                    padding='same',activation='relu'))
    model.add(tfl.MaxPool1D(pool_size= 4,
                       padding='same'))
    model.add(tfl.Flatten())
    model.add(tfl.Dense(param, activation='relu'))
    model.add(tfl.Dense(1, activation='sigmoid'))
    model.compile(optimizer = 'adam',
                  loss='binary_crossentropy',
              metrics=['accuracy'])
    return model

from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow import keras
import tensorflow.keras.layers as tfl
from tensorflow.keras.layers import *
from tensorflow.keras.utils import *
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D

kfold = KFold(n_splits=5, shuffle=True, random_state=1)
fold_no = 1
acc_per_fold1 = []
loss_per_fold1 = []

for train, test in kfold.split(x_train1, y_train1):
    model1 = Sequential()
    model1.add(keras.Input(shape=(1024,1)))
    model1.add(Conv1D(filters=32,
                        kernel_size=7,
                        strides=1,
                        padding='same'))
    model1.add(Conv1D(filters=16,
                        kernel_size=9,
                        strides=1,
                        padding='same'))
    # model.add(MaxPooling1D(pool_size= 16,
                          #  padding='same'))
    model1.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'))
    # model.add(MaxPooling1D(pool_size=8))
    model1.add(LSTM(100))                               ############### parameter
    model1.add(Dense(1, activation='sigmoid'))
    model1.compile(optimizer = 'adam',
                  loss='binary_crossentropy',
              metrics=['accuracy'])

    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # Fit data to model
    history1 = model1.fit(x_train1[train], y_train1[train],
                batch_size=75,
                epochs=50)

    # Generate generalization metrics
    scores1 = model1.evaluate(x_train1[test], y_train1[test], verbose=0)
    print(f'Score for fold {fold_no}: {model1.metrics_names[0]} of {scores1[0]}; {model1.metrics_names[1]} of {scores1[1]*100}%')
    acc_per_fold1.append(scores1[1] * 100)
    loss_per_fold1.append(scores1[0])

    # Increase fold number
    fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold1)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold1[i]} - Accuracy: {acc_per_fold1[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold1)} (+- {np.std(acc_per_fold1)})')
print(f'> Loss: {np.mean(loss_per_fold1)}')
print('------------------------------------------------------------------------')

np.savetxt('dataa/hyper_opti_lstm_param100_50.csv', acc_per_fold1, delimiter=',')

kfold = KFold(n_splits=5, shuffle=True, random_state=1)
fold_no = 1
acc_per_fold2 = []
loss_per_fold2 = []

for train, test in kfold.split(x_train1,y_train1):
    model2 = Sequential()
    model2.add(keras.Input(shape=(1024,1)))
    model2.add(Conv1D(filters=32,
                        kernel_size=7,
                        strides=1,
                        padding='same'))
    model2.add(Conv1D(filters=16,
                        kernel_size=9,
                        strides=1,
                        padding='same'))
    # model.add(MaxPooling1D(pool_size= 16,
                          #  padding='same'))
    model2.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'))
    # model.add(MaxPooling1D(pool_size=8))
    model2.add(LSTM(16))
    model2.add(Dense(1, activation='sigmoid'))
    model2.compile(optimizer = 'adam',
                  loss='binary_crossentropy',
              metrics=['accuracy'])

    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # Fit data to model
    history2 = model2.fit(x_train1[train], y_train1[train],
                batch_size=75,
                epochs=50)

    # Generate generalization metrics
    scores2 = model2.evaluate(x_train1[test], y_train1[test], verbose=0)
    print(f'Score for fold {fold_no}: {model2.metrics_names[0]} of {scores2[0]}; {model2.metrics_names[1]} of {scores2[1]*100}%')
    acc_per_fold2.append(scores2[1] * 100)
    loss_per_fold2.append(scores2[0])

    # Increase fold number
    fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold2)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold2[i]} - Accuracy: {acc_per_fold2[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold2)} (+- {np.std(acc_per_fold2)})')
print(f'> Loss: {np.mean(loss_per_fold2)}')
print('------------------------------------------------------------------------')

np.savetxt('dataa/hyper_opti_lstm_param16_50.csv', acc_per_fold2, delimiter=',')

kfold = KFold(n_splits=5, shuffle=True, random_state=1)
fold_no = 1
acc_per_fold3 = []
loss_per_fold3 = []

for train, test in kfold.split(x_train1,y_train1):
    model3 = Sequential()
    model3.add(keras.Input(shape=(1024,1)))
    model3.add(Conv1D(filters=32,
                        kernel_size=7,
                        strides=1,
                        padding='same'))
    model3.add(Conv1D(filters=16,
                        kernel_size=9,
                        strides=1,
                        padding='same'))
    # model.add(MaxPooling1D(pool_size= 16,
                          #  padding='same'))
    model3.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'))
    # model.add(MaxPooling1D(pool_size=8))
    model3.add(LSTM(128))
    model3.add(Dense(1, activation='sigmoid'))
    model3.compile(optimizer = 'adam',
                  loss='binary_crossentropy',
              metrics=['accuracy'])

    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # Fit data to model
    history3 = model3.fit(x_train1[train], y_train1[train],
                batch_size=75,
                epochs=50)

    # Generate generalization metrics
    scores3 = model3.evaluate(x_train1[test], y_train1[test], verbose=0)
    print(f'Score for fold {fold_no}: {model3.metrics_names[0]} of {scores3[0]}; {model3.metrics_names[1]} of {scores3[1]*100}%')
    acc_per_fold3.append(scores3[1] * 100)
    loss_per_fold3.append(scores3[0])

    # Increase fold number
    fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold3)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold3[i]} - Accuracy: {acc_per_fold3[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold3)} (+- {np.std(acc_per_fold3)})')
print(f'> Loss: {np.mean(loss_per_fold3)}')
print('------------------------------------------------------------------------')

np.savetxt('dataa/hyper_opti_lstm_param128_50.csv', acc_per_fold3, delimiter=',')

kfold = KFold(n_splits=5, shuffle=True, random_state=1)
fold_no = 1
acc_per_fold4 = []
loss_per_fold4 = []

for train, test in kfold.split(x_train1,y_train1):
    model4 = Sequential()
    model4.add(keras.Input(shape=(1024,1)))
    model4.add(Conv1D(filters=32,
                        kernel_size=7,
                        strides=1,
                        padding='same'))
    model4.add(Conv1D(filters=16,
                        kernel_size=9,
                        strides=1,
                        padding='same'))
    # model.add(MaxPooling1D(pool_size= 16,
                          #  padding='same'))
    model4.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'))
    # model.add(MaxPooling1D(pool_size=8))
    model4.add(LSTM(256))
    model4.add(Dense(1, activation='sigmoid'))
    model4.compile(optimizer = 'adam',
                  loss='binary_crossentropy',
              metrics=['accuracy'])

    # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')

    # Fit data to model
    history4 = model4.fit(x_train1[train], y_train1[train],
                batch_size=75,
                epochs=50)

    # Generate generalization metrics
    scores4 = model4.evaluate(x_train1[test], y_train1[test], verbose=0)
    print(f'Score for fold {fold_no}: {model4.metrics_names[0]} of {scores4[0]}; {model4.metrics_names[1]} of {scores4[1]*100}%')
    acc_per_fold4.append(scores4[1] * 100)
    loss_per_fold4.append(scores4[0])

    # Increase fold number
    fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold4)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold4[i]} - Accuracy: {acc_per_fold4[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold4)} (+- {np.std(acc_per_fold4)})')
print(f'> Loss: {np.mean(loss_per_fold4)}')
print('------------------------------------------------------------------------')

np.savetxt('dataa/hyper_opti_lstm_param256_50.csv', acc_per_fold4, delimiter=',')

import matplotlib.pyplot as plt
import numpy as np

# Creating dataset
data_4 = np.array(acc_per_fold4)
data_1 = np.array(acc_per_fold1)
data_2 = np.array(acc_per_fold2)
data_3 = np.array(acc_per_fold3)

data = [data_2.flatten(), data_3.flatten(), data_1.flatten(), data_4.flatten()]

fig = plt.figure()

# Creating axes instance
ax = fig.add_axes([0, 0, 1, 1])
#plt.ylim([60,85])

txtstr = f' param 16 acc = {round(np.mean(acc_per_fold2),2)} (+- {round(np.std(acc_per_fold2),2)}) \n param 64 acc = {round(np.mean(acc_per_fold3),2)} (+- {round(np.std(acc_per_fold3),2)}) \n param 100 acc = {round(np.mean(acc_per_fold1),2)} (+- {round(np.std(acc_per_fold1),2)}) \n param 256 acc = {round(np.mean(acc_per_fold4),2)} (+- {round(np.std(acc_per_fold4),2)})'

plt.text(0.452,0.045,txtstr,fontsize=13, transform=plt.gcf().transFigure, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 10})

bp = ax.boxplot(data, labels=[16, 128, 100, 256], showmeans=True)

plt.xlabel('LSTM Units')
plt.ylabel('Accuracy')
plt.grid()

plt.savefig("hyper_opti.pdf", bbox_inches = 'tight')
plt.show()

acc_per_fold3 = np.loadtxt('dataa/hyper_opti_lstm_param128_50.csv', delimiter=',')
acc_per_fold1 = np.loadtxt('dataa/hyper_opti_lstm_param100_50.csv', delimiter=',')
acc_per_fold2 = np.loadtxt('dataa/hyper_opti_lstm_param16_50.csv', delimiter=',')
acc_per_fold4 = np.loadtxt('dataa/hyper_opti_lstm_param256_50.csv', delimiter=',')
