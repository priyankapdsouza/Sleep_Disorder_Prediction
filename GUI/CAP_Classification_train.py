import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, MaxPool1D, Dense, Flatten, Input, BatchNormalization, LSTM, Concatenate, GlobalAveragePooling1D
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Change to the script's directory to ensure relative paths work
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def lstm_model(input_shape, x_train, y_train, test_data, save_path):
    input_img = Input(shape=input_shape)
    lstm1 = LSTM(64, return_sequences=True, kernel_regularizer=l2(0.001))(input_img)
    lstm1 = BatchNormalization()(lstm1)
    lstm2 = LSTM(32, kernel_regularizer=l2(0.001))(lstm1)
    lstm2 = BatchNormalization()(lstm2)
    dense1 = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(lstm2)
    dense2 = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(dense1)
    outputs = Dense(1, activation='sigmoid')(dense2)

    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(save_path, monitor='val_accuracy', save_best_only=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-6)
    history = model.fit(x_train, y_train, epochs=30, batch_size=128, validation_data=test_data, callbacks=[early_stop, checkpoint, lr_scheduler])
    return model, history

def inception_block(x, filters):
    branch1 = Conv1D(filters, 1, padding='same', activation='relu')(x)
    branch2 = Conv1D(filters, 1, padding='same', activation='relu')(x)
    branch2 = Conv1D(filters, 3, padding='same', activation='relu')(branch2)
    branch3 = Conv1D(filters, 1, padding='same', activation='relu')(x)
    branch3 = Conv1D(filters, 5, padding='same', activation='relu')(branch3)
    branch4 = MaxPool1D(3, strides=1, padding='same')(x)
    branch4 = Conv1D(filters, 1, padding='same', activation='relu')(branch4)
    return Concatenate()([branch1, branch2, branch3, branch4])

def inception_model(input_shape, x_train, y_train, test_data, save_path):
    input_img = Input(shape=input_shape)
    x = Conv1D(32, 7, strides=2, padding='same', activation='relu')(input_img)
    x = MaxPool1D(3, strides=2, padding='same')(x)
    x = inception_block(x, 64)
    x = inception_block(x, 128)
    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint(save_path, monitor='val_accuracy', save_best_only=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-6)
    history = model.fit(x_train, y_train, epochs=30, batch_size=128, validation_data=test_data, callbacks=[early_stop, checkpoint, lr_scheduler])
    return model, history

# Train healthy CAP models
print("Training Healthy CAP models...")
data = np.loadtxt('dataa/healthy_unhealthy1.csv', delimiter=',')
healthy_data = data[data[:, -1] == 0]  # Filter healthy (label 0)
X = healthy_data[:, :1024].reshape((healthy_data.shape[0], 1024, 1))
y = healthy_data[:, 1024]  # Phase: 0=B, 1-3=A, but for binary, map 0 to 0, others to 1
y = (y > 0).astype(int)  # 0 if B, 1 if A
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training LSTM Healthy CAP model...")
lstm_model((1024,1), x_train, y_train, (x_test, y_test), 'GUI/lstm_healthy_cap.h5')

print("Training Inception Healthy CAP model...")
inception_model((1024,1), x_train, y_train, (x_test, y_test), 'GUI/inception_healthy_cap.h5')

# Train disease CAP models
print("Training Disease CAP models...")
data = np.loadtxt('dataa/both_phases.csv', delimiter=',')
diseases = ['INS', 'NFLE', 'NARCO', 'RBD', 'PLM']
disease_labels = [0, 1, 2, 3, 4]
lstm_files = ['GUI/lstm_ins_cap.h5', 'GUI/lstm_nfle_cap.h5', 'GUI/lstm_narco_cap.h5', 'GUI/lstm_rbd_cap.h5', 'GUI/lstm_plm_cap.h5']
inception_files = ['GUI/inception_ins_cap.h5', 'GUI/inception_nfle_cap.h5', 'GUI/inception_narco_cap.h5', 'GUI/inception_rbd_cap.h5', 'GUI/inception_plm_cap.h5']

for label, name, lstm_file, inception_file in zip(disease_labels, diseases, lstm_files, inception_files):
    print(f"Training {name} CAP models...")
    disease_data = data[data[:, -1] == label]
    X = disease_data[:, :1024].reshape((disease_data.shape[0], 1024, 1))
    y = disease_data[:, 1024]
    y = (y > 0).astype(int)  # 0 if B, 1 if A
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training LSTM {name} CAP model...")
    lstm_model((1024,1), x_train, y_train, (x_test, y_test), lstm_file)

    print(f"Training Inception {name} CAP model...")
    inception_model((1024,1), x_train, y_train, (x_test, y_test), inception_file)

print("All CAP models trained and saved.")
