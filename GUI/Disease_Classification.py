from google.colab import drive\n
drive.mount('/content/drive')

import csv\n
import pandas as pd\n
import scipy.io\n
import numpy as np\n
import copy\n
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





from sklearn.model_selection import train_test_split \n
from sklearn.preprocessing import OneHotEncoder\n
\n

# GRADED FUNCTION: convolutional_model\n
import tensorflow as tf\n
from keras.callbacks import EarlyStopping\n
import tensorflow.keras.layers as tfl\n
import matplotlib.pyplot as plt\n
from keras.layers import Dropout\n
from keras.utils.vis_utils import plot_model\n
from keras.layers.merge import concatenate\n
from time import time\n
\n
def disease_classification(input_shape, x_train,y_train,test_data):\n
    \
\
,
        
,
        
,
        
,
        
,
        
,
        
,
        
,
        
\
\n
\n
    start = time()\n
\n
    input_img = tf.keras.Input(shape=input_shape)\n
    ## CONV1D: 8 filters 4x4, stride of 1, padding 'SAME'\n
    Z1 = tfl.Conv1D(filters=25,\n
                    kernel_size=15,\n
                    strides=1,\n
                    padding='same')(input_img) \n
    Z11 = tfl.Conv1D(filters=25,\n
                    kernel_size=9,\n
                    strides=1,\n
                    padding='same')(Z1)\n
    #Z11op=tf.nn.dropout(Z11, rate = 0.2)\n
    ## RELU\n
    A1 = tfl.ReLU()(Z11)\n
    ## MAXPOOL: window 8x8, stride 8, padding 'SAME'\n
    P1 = tfl.MaxPooling1D(pool_size= 5,\n
                       padding='same')(A1)\n
   \n
    ## CONV2D: 16 filters 3x3, stride 1, padding 'SAME'\n
    Z2 = tfl.Conv1D(filters=15,\n
                    kernel_size=15,\n
                    strides=1,\n
                    padding='same')(P1)\n
    Z21 = tfl.Conv1D(filters=15,\n
                    kernel_size=9,\n
                    strides=1,\n
                    padding='same')(Z2)\n
    #Z21op=tf.nn.dropout(Z21, rate = 0.2)\n
    ## RELU\n
    A2 = tfl.ReLU()(Z21)\n
    ## MAXPOOL: window 4x4, stride 4, padding 'SAME'\n
    P2 = tfl.MaxPooling1D(pool_size=5,\n
                       padding='same')(A2)\n
    \n
\n
    Z3 = tfl.Conv1D(filters=10,\n
                    kernel_size=15,\n
                    strides=1,\n
                    padding='same')(P2)\n
    Z31 = tfl.Conv1D(filters=10,\n
                    kernel_size=9,\n
                    strides=1,\n
                    padding='same')(Z3)\n
    Z31=tf.nn.dropout(Z31, rate = 0.2)\n
    ## RELU\n
    A3 = tfl.ReLU()(Z31)\n
    ## MAXPOOL: window 4x4, stride 4, padding 'SAME'\n
    P3 = tfl.MaxPooling1D(pool_size=5,\n
                       padding='same')(A3)\n
\n
\n
    ## FLATTEN\n
    F = tfl.Flatten()(P3)\n
    ## Dense layer\n
    op = tfl.Dense(units=128,\n
                        activation='relu')(F)\n
    op=tf.nn.dropout(op, rate = 0.2)\n
\n
    op1 = tfl.Dense(units=48,\n
                        activation='relu')(op)\n
\n
    op2 = tfl.Dense(units=16,\n
                        activation='relu')(op1)\n
          \n
    outputs = tfl.Dense(units=5,\n
                        activation='softmax')(op2)\n
\n
    model = tf.keras.Model(inputs=input_img, outputs=outputs)\n
\n
    #compile\n
    model.compile(optimizer = 'adam',\n
                  loss='categorical_crossentropy',\n

from tensorflow import keras\n
from keras.callbacks import EarlyStopping\n
from tensorflow import keras\n
import tensorflow.keras.layers as tfl\n
from tensorflow.keras.layers import *\n
from tensorflow.keras.utils import *\n
from time import time\n
import tensorflow.keras.layers as tfl\n
import matplotlib.pyplot as plt\n
from keras.layers import Dropout\n
from keras.utils.vis_utils import plot_model\n
from keras.layers.merge import concatenate\n
from time import time\n
\n
def disease_classification2(input_shape, x_train,y_train,test_data):\n
    \
\
,
        
,
        
,
        
,
        
,
        
,
        
,
        
,
        
\
\n
\n
    start = time()\n
\n
    input_img = tf.keras.Input(shape=input_shape)\n
    ## CONV1D: 8 filters 4x4, stride of 1, padding 'SAME'\n
    Z1 = tfl.Conv1D(filters=100,\n
                    kernel_size=25,\n
                    strides=1,\n
                    padding='same')(input_img) \n
    M1 = MaxPool1D()(Z1)\n
    Z2 = tfl.Conv1D(filters=80,\n
                    kernel_size=40,\n
                    strides=1,\n
                    padding='same')(M1) \n
\n
    M2 = MaxPool1D()(Z2)\n
    Z3 = tfl.Conv1D(filters=60,\n
                    kernel_size=30,\n
                    strides=1,\n
                    padding='same')(M2) \n
\n
    M3 = MaxPool1D()(Z3)\n
\n
    Z4 = tfl.Conv1D(filters=40,\n
                    kernel_size=20,\n
                    strides=1,\n
                    padding='same')(M3) \n
\n
    M4 = MaxPool1D()(Z4)\n
\n
    gru = GRU(40, return_sequences=True)(M4)\n
    gru1 = GRU(40, return_sequences=True)(gru)\n
\n
    gru1=tf.nn.dropout(gru1, rate = 0.25)\n
    \n
    F = tfl.Flatten()(gru1)\n
    ## Dense layer\n
  \n
\n
    \n
    outputs = tfl.Dense(units=5,\n
                        activation='softmax')(F)\n
\n
    model = tf.keras.Model(inputs=input_img, outputs=outputs)\n
\n
    #compile\n
    model.compile(optimizer = 'adam',\n
                  loss='categorical_crossentropy',\n

conv_model,history = disease_classification2((1024,1), x_train1, y_train1, (x_test1,y_test1))

# GRADED FUNCTION: convolutional_model\n
from tensorflow import keras\n
import tensorflow.keras.layers as tfl\n
from tensorflow.keras.layers import *\n
from tensorflow.keras.utils import *\n
from time import time\n
from keras.callbacks import ModelCheckpoint\n
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

incep_model = build_incep((1024,1), x_train1, y_train1, (x_test1,y_test1))

from sklearn import metrics\n
 \n
def analysis(actual, predict, classes):   \n

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
def skip_model(input_shape, x_train,y_train,test_data):\n
    i = 0\n

 model, history_m = skip_model((1024,1), x_train1, y_train1, (x_test1, y_test1))

from tensorflow import keras\n
model1 = keras.models.load_model(\
)

from sklearn import metrics\n
 \n
def analysis(actual, predict, classes):   \n

bal_all = np.loadtxt('/content/drive/MyDrive/FYP/combine_classification/dataset/both_phases.csv', delimiter=',')



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
model.add(Dense(5, activation='softmax'))\n

from tensorflow import keras\n
model1 = keras.models.load_model(\
)

from sklearn import metrics\n
 \n
def analysis(actual, predict, classes):   \n

from tensorflow import keras\n
model1 = keras.models.load_model(\
)\n
bal_all = np.loadtxt('/content/drive/MyDrive/FYP/combine_classification/dataset/B_phases.csv', delimiter=',')



\n
from sklearn import metrics\n
 \n
def analysis(actual, predict, classes):   \n

from tensorflow import keras\n
model = tf.keras.models.load_model('/content/drive/MyDrive/ML/Disease_classification/Dhruvi/LSTM/A/model_v1.h5')\n
bal_all = np.loadtxt('/content/drive/MyDrive/FYP/combine_classification/dataset/A_phases.csv', delimiter=',')



\n
from sklearn import metrics\n
 \n
def analysis(actual, predict, classes):   \n

