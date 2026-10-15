from google.colab import drive\n
drive.mount('/content/drive')

import csv\n
import pandas as pd\n
import scipy.io\n
import numpy as np\n
import copy\n
\n
\n
import matplotlib.pyplot as plt\n
import h5py\n
import scipy\n
from PIL import Image\n
from scipy import ndimage\n
\n
from tensorflow import keras\n
import tensorflow as tf\n
\n
%matplotlib inline\n
%load_ext autoreload\n
%autoreload 2

!pip install watermark\n
%load_ext watermark\n
\n
%watermark --iversions 

data = np.loadtxt('/content/drive/MyDrive/ML/H&U_classification/dataset/healthy_unhealthy1.csv', delimiter=',')

from sklearn.model_selection import train_test_split \n
\n
## healthy 0 unhealthy 1\n

# GRADED FUNCTION: convolutional_model\n
import tensorflow as tf\n
from keras.callbacks import EarlyStopping\n
import tensorflow.keras.layers as tfl\n
import matplotlib.pyplot as plt\n
from time import time\n
\n
def convolutional_model(input_shape, x_train,y_train,test_data):\n
    start = time()\n
    input_img = tf.keras.Input(shape=input_shape)\n
    ## CONV1D: 8 filters 4x4, stride of 1, padding 'SAME'\n
    Z1 = tfl.Conv1D(filters=32,\n
                    kernel_size=15,\n
                    strides=1,\n
                    padding='same')(input_img) \n
    Z11 = tfl.Conv1D(filters=16,\n
                    kernel_size=7,\n
                    strides=1,\n
                    padding='same')(Z1)\n
    ## RELU\n
    A1 = tfl.ReLU()(Z11)\n
    ## MAXPOOL: window 8x8, stride 8, padding 'SAME'\n
    P1 = tfl.MaxPool1D(pool_size= 16,\n
                       padding='same')(A1)\n
\n
\n
    Z11 = tfl.Conv1D(filters=5,\n
                    kernel_size=5,\n
                    strides=1,\n
                    padding='same',kernel_regularizer=tf.keras.regularizers.L2(0.01))(P1)\n
    ## RELU\n
    A1 = tfl.ReLU()(Z11)\n
    ## MAXPOOL: window 8x8, stride 8, padding 'SAME'\n
    P1 = tfl.MaxPool1D(pool_size= 4,\n
                       padding='same')(A1)\n
    ## CONV2D: 16 filters 2x2, stride 1, padding 'SAME'\n
    Z2 = tfl.Conv1D(filters=8,\n
                    kernel_size=5,\n
                    strides=1,\n
                    padding='same')(P1)\n
    Z2=tf.nn.dropout(Z2, rate = 0.3)\n
    ## RELU\n
    A2 = tfl.ReLU()(Z2)\n
    ## MAXPOOL: window 4x4, stride 4, padding 'SAME'\n
    P2 = tfl.MaxPool1D(pool_size=8,\n
                       padding='same')(A2)\n
    ## FLATTEN\n
    F = tfl.Flatten()(P2)\n
    ## Dense layer\n
\n
    op1 = tfl.Dense(units=15,\n
                       activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01))(F)\n
\n
    #op1= tf.nn.dropout(op1, rate=0.4)\n
\n
    op = tfl.Dense(units=6,\n
                        activation='relu')(op1)\n
          \n
    outputs = tfl.Dense(units=1,\n
                        activation='sigmoid')(op)\n
\n
    model = tf.keras.Model(inputs=input_img, outputs=outputs)  \n
\n
    #compile\n
    opt = tf.keras.optimizers.Adam( learning_rate=0.0015, beta_1=0.9, beta_2=0.999, epsilon=1e-07)\n
\n
    #compile\n
    model.compile(optimizer = 'adam',\n
                  loss='binary_crossentropy',\n

# GRADED FUNCTION: convolutional_model\n
import tensorflow as tf\n
from keras.callbacks import EarlyStopping\n
import tensorflow.keras.layers as tfl\n
import matplotlib.pyplot as plt\n
from keras.layers import Dropout\n
from keras.utils.vis_utils import plot_model\n
from keras.layers.merge import concatenate\n
from tensorflow import keras\n
from time import time\n
\n
def convolutional_model2(input_shape, x_train,y_train,test_data):\n
    i = 0\n

conv_model, history_m = convolutional_model2((1024,1), x_train1, y_train1, (x_test1, y_test1))

conv_model.save('/content/drive/MyDrive/ML/H&U_classification/skip/model_v1.h5')\n
np.save('/content/drive/MyDrive/ML/H&U_classification/skip/ver1_history.npy',history_m)\n
\n
# save testdata to csv file\n
np.savetxt('/content/drive/MyDrive/ML/H&U_classification/skip/xtest_data_ver1.csv', x_test1, delimiter=',')\n
np.savetxt('/content/drive/MyDrive/ML/H&U_classification/skip/ytest_data_ver1.csv', y_test1, delimiter=',')\n
np.savetxt('/content/drive/MyDrive/ML/H&U_classification/skip/xtrain_data_ver1.csv', x_train1, delimiter=',')\n
np.savetxt('/content/drive/MyDrive/ML/H&U_classification/skip/ytrain_data_ver1.csv', y_train1, delimiter=',')

from tensorflow import keras\n
model1 = keras.models.load_model(\
)\n

import matplotlib.pyplot as plt\n
from sklearn import metrics\n
\n
prob = model1.predict(x_test1,verbose = 2)\n
pred = (prob > 0.5).astype(\
)\n
\n
\n
#(loss, acc) = conv_model.evaluate(x_test1, y_test1)\n
#print(\
, acc*100)\n
\n
\n
# reduce to 1d array\n

import numpy\n
from keras.datasets import imdb\n
from keras.models import Sequential\n
from keras.layers import Dense\n
from keras.layers import LSTM\n
from keras.layers.embeddings import Embedding\n
from keras.preprocessing import sequence\n
# fix random seed for reproducibility\n
numpy.random.seed(7)\n
\n
# LSTM and CNN for sequence classification in the IMDB dataset\n
import numpy\n
from keras.datasets import imdb\n
from keras.models import Sequential\n
from keras.layers import Dense\n
from keras.layers import LSTM\n
from keras.layers.convolutional import Conv1D\n
from keras.layers.convolutional import MaxPooling1D\n
from keras.callbacks import EarlyStopping\n
\n
from keras.layers.embeddings import Embedding\n
from keras.preprocessing import sequence\n
from time import time\n
\n
# fix random seed for reproducibility\n
\n
start = time()\n
model = Sequential()\n
model.add(keras.Input(shape=(1024,1)))\n
model.add(Conv1D(filters=32,\n
                    kernel_size=7,\n
                    strides=1,\n
                    padding='same'))\n
model.add(Conv1D(filters=16,\n
                    kernel_size=9,\n
                    strides=1,\n
                    padding='same'))\n
# model.add(MaxPooling1D(pool_size= 16,\n
                      #  padding='same'))\n
model.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'))\n
# model.add(MaxPooling1D(pool_size=8))\n
model.add(LSTM(100))\n
model.add(Dense(1, activation='sigmoid'))\n

model.save('/content/drive/MyDrive/ML/H&U_classification/lstm/model_vA1.h5')\n
np.save('/content/drive/MyDrive/ML/H&U_classification/lstm/verA1_history.npy',history_m)\n
\n
# save testdata to csv file\n
np.savetxt('/content/drive/MyDrive/ML/H&U_classification/lstm/xtest_data_verA1.csv', x_test1, delimiter=',')\n
np.savetxt('/content/drive/MyDrive/ML/H&U_classification/lstm/ytest_data_verA1.csv', y_test1, delimiter=',')\n
np.savetxt('/content/drive/MyDrive/ML/H&U_classification/lstm/xtrain_data_verA1.csv', x_train1, delimiter=',')\n
np.savetxt('/content/drive/MyDrive/ML/H&U_classification/lstm/ytrain_data_verA1.csv', y_train1, delimiter=',')

import matplotlib.pyplot as plt\n
from sklearn import metrics\n
\n
# list all data in history\n
print(history.history.keys())\n
# summarize history for accuracy\n

from sklearn.metrics import roc_curve\n
from sklearn.metrics import auc\n
y_pred_keras = model1.predict(x_test1).ravel()\n
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test1, y_pred_keras)\n
auc_keras = auc(fpr_keras, tpr_keras)\n
print(auc_keras)\n
plt.plot(fpr_keras, tpr_keras, label='ROC curve')\n
plt.fill_between(fpr_keras, tpr_keras, 0, color='blue', alpha=.1, label='AUC = {:.3f}'.format(auc_keras))\n

import matplotlib.pyplot as plt\n
from sklearn import metrics\n
\n
prob = model1.predict(x_test1,verbose = 2)\n
pred = (prob > 0.5).astype(\
)\n
\n
\n
#(loss, acc) = conv_model.evaluate(x_test1, y_test1)\n
#print(\
, acc*100)\n
\n
\n
# reduce to 1d array\n

# GRADED FUNCTION: convolutional_model\n
from tensorflow import keras\n
import tensorflow.keras.layers as tfl\n
from tensorflow.keras.layers import *\n
from tensorflow.keras.utils import *\n
from time import time\n
\n
\n
# Architecture of the model\n
def inception_module(layer, f1, f2, f3, f4):\n
    conv1 = Conv1D(f1, 1, activation='relu', padding='same')(layer)\n
    conv2 = Conv1D(f1, 1, activation='relu', padding='same')(layer)\n
    conv3 = Conv1D(f1, 1, activation='relu', padding='same')(layer)\n
    maxp = MaxPool1D(pool_size=3, strides=1, padding='same')(layer)\n
\n
    conv2 = Conv1D(f2, 3, activation='relu', padding='same')(conv2)\n
    conv3 = Conv1D(f3, 5, activation='relu', padding='same')(conv3)\n
    conv4 = Conv1D(f4, 1, activation='relu', padding='same')(maxp)\n
\n

conv_model, history = build_incep((1024,1), x_train1, y_train1, (x_test1,y_test1))

model1 = tf.keras.models.load_model('/content/drive/MyDrive/ML/H&U_classification/lstm/model_v4.h5')\n
history=np.load('/content/drive/MyDrive/ML/H&U_classification/lstm/ver4_history.npy',allow_pickle='TRUE').item()\n
\n
x_test1 = np.loadtxt('/content/drive/MyDrive/ML/H&U_classification/lstm/xtest_data_ver4.csv', delimiter=',')\n
y_test1 = np.loadtxt('/content/drive/MyDrive/ML/H&U_classification/lstm/ytest_data_ver4.csv', delimiter=',')

import matplotlib.pyplot as plt\n
from sklearn import metrics\n
\n
# list all data in history\n
#print(history.history.keys())\n
# summarize history for accuracy\n

from sklearn.metrics import roc_curve\n
from sklearn.metrics import auc\n
y_pred_keras = model1.predict(x_test1).ravel()\n
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test1, y_pred_keras)\n
auc_keras = auc(fpr_keras, tpr_keras)\n
print(auc_keras)\n
plt.plot(fpr_keras, tpr_keras, label='ROC curve')\n
plt.fill_between(fpr_keras, tpr_keras, 0, color='blue', alpha=.1, label='AUC = {:.3f}'.format(auc_keras))\n

labels=[\
, \

model1 = tf.keras.models.load_model('/content/drive/MyDrive/ML/H&U_classification/lstm/model_vB1.h5')\n
history=np.load('/content/drive/MyDrive/ML/H&U_classification/lstm/verB1_history.npy',allow_pickle='TRUE').item()\n
\n
x_test1 = np.loadtxt('/content/drive/MyDrive/ML/H&U_classification/lstm/xtest_data_verB1.csv', delimiter=',')\n
y_test1 = np.loadtxt('/content/drive/MyDrive/ML/H&U_classification/lstm/ytest_data_verB1.csv', delimiter=',')

import matplotlib.pyplot as plt\n
from sklearn import metrics\n
\n
# summarize history for accuracy\n

from sklearn.metrics import roc_curve\n
from sklearn.metrics import auc\n
y_pred_keras = model1.predict(x_test1).ravel()\n
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test1, y_pred_keras)\n
auc_keras = auc(fpr_keras, tpr_keras)\n
print(auc_keras)\n
plt.plot(fpr_keras, tpr_keras, label='ROC curve')\n
plt.fill_between(fpr_keras, tpr_keras, 0, color='blue', alpha=.1, label='AUC = {:.3f}'.format(auc_keras))\n

labels=[\
, \

model1 = tf.keras.models.load_model('/content/drive/MyDrive/ML/H&U_classification/lstm/model_vA1.h5')\n
history=np.load('/content/drive/MyDrive/ML/H&U_classification/lstm/verA1_history.npy',allow_pickle='TRUE').item()\n
\n
x_test1 = np.loadtxt('/content/drive/MyDrive/ML/H&U_classification/lstm/xtest_data_verA1.csv', delimiter=',')\n
y_test1 = np.loadtxt('/content/drive/MyDrive/ML/H&U_classification/lstm/ytest_data_verA1.csv', delimiter=',')

import matplotlib.pyplot as plt\n
from sklearn import metric\n
\n
# summarize history for accuracy\n

from sklearn.metrics import roc_curve\n
from sklearn.metrics import auc\n
y_pred_keras = model1.predict(x_test1).ravel()\n
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test1, y_pred_keras)\n
auc_keras = auc(fpr_keras, tpr_keras)\n
print(auc_keras)\n
plt.plot(fpr_keras, tpr_keras, label='ROC curve')\n
plt.fill_between(fpr_keras, tpr_keras, 0, color='blue', alpha=.1, label='AUC = {:.3f}'.format(auc_keras))\n

labels=[\
, \



