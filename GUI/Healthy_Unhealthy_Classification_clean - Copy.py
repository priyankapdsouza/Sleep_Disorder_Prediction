import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, MaxPool1D, GRU, Dense, Flatten, Add, concatenate, Input
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization

# Change to the script's directory to ensure relative paths work
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load data
data = np.loadtxt('Sleep-disorder-Classification/dataa/healthy_unhealthy1.csv', delimiter=',')
X = data[:, :1024]  # Features
y = data[:, -1]     # Labels (0 healthy, 1 unhealthy)

# Reshape X to (samples, 1024, 1)
X = X.reshape((X.shape[0], 1024, 1))

# Split data
x_train1, x_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def augment_data(x, y):
    # Gaussian noise
    noise = np.random.normal(0, 0.01, x.shape)
    x_aug = x + noise
    # Scaling
    scale_factor = np.random.uniform(0.9, 1.1, size=(x.shape[0], 1, 1))
    x_aug = x_aug * scale_factor
    # Shifting
    shift = np.random.normal(0, 0.05, size=(x.shape[0], 1, 1))
    x_aug = x_aug + shift
    # Time warping (stretch/compress)
    warp_factor = np.random.uniform(0.8, 1.2, size=(x.shape[0],))
    x_aug_warped = np.zeros_like(x_aug)
    for i in range(x_aug.shape[0]):
        indices = np.linspace(0, x_aug.shape[1]-1, int(x_aug.shape[1] * warp_factor[i]))
        indices = np.clip(indices, 0, x_aug.shape[1]-1).astype(int)
        x_aug_warped[i, :, 0] = np.interp(np.arange(x_aug.shape[1]), indices, x_aug[i, indices, 0])
    return x_aug_warped, y

# CNN Model
def cnn_model(input_shape, x_train, y_train, test_data):
    input_img = Input(shape=input_shape)
    Z1 = Conv1D(filters=32, kernel_size=15, strides=1, padding='same', kernel_regularizer=l2(0.001))(input_img)
    Z1 = BatchNormalization()(Z1)
    Z11 = Conv1D(filters=16, kernel_size=7, strides=1, padding='same', kernel_regularizer=l2(0.001))(Z1)
    Z11 = BatchNormalization()(Z11)
    A1 = tf.keras.layers.ReLU()(Z11)
    P1 = tf.keras.layers.MaxPooling1D(pool_size=16, padding='same')(A1)

    Z11 = Conv1D(filters=5, kernel_size=5, strides=1, padding='same', kernel_regularizer=l2(0.001))(P1)
    Z11 = BatchNormalization()(Z11)
    A1 = tf.keras.layers.ReLU()(Z11)
    P1 = tf.keras.layers.MaxPooling1D(pool_size=4, padding='same')(A1)

    Z2 = Conv1D(filters=8, kernel_size=5, strides=1, padding='same', kernel_regularizer=l2(0.001))(P1)
    Z2 = BatchNormalization()(Z2)
    Z2 = tf.nn.dropout(Z2, rate=0.5)
    A2 = tf.keras.layers.ReLU()(Z2)
    P2 = tf.keras.layers.MaxPooling1D(pool_size=8, padding='same')(A2)

    F = tf.keras.layers.Flatten()(P2)
    op1 = Dense(units=15, activation='relu', kernel_regularizer=l2(0.001))(F)
    op = Dense(units=6, activation='relu', kernel_regularizer=l2(0.001))(op1)
    outputs = Dense(units=1, activation='sigmoid')(op)

    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    x_train_aug, y_train_aug = augment_data(x_train, y_train)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
    checkpoint = ModelCheckpoint('Sleep-disorder-Classification/GUI/model_cnn_HU_new.h5', monitor='val_accuracy', save_best_only=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=10, min_lr=1e-6)
    history = model.fit(x_train_aug, y_train_aug, epochs=100, batch_size=64, validation_data=test_data, callbacks=[early_stop, checkpoint, lr_scheduler])
    return model, history

# Skip Connection Model
def skip_model(input_shape, x_train, y_train, test_data):
    input_img = Input(shape=input_shape)
    layer = Conv1D(64, 7, activation='relu', padding='same', kernel_regularizer=l2(0.001))(input_img)
    layer = BatchNormalization()(layer)
    layer = MaxPool1D(pool_size=3, strides=2, padding='same')(layer)
    skip1 = layer
    layer = Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(layer)
    layer = BatchNormalization()(layer)
    layer = Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPool1D(pool_size=3, strides=2, padding='same')(layer)
    skip1 = Conv1D(128, 1, strides=2, padding='same', kernel_regularizer=l2(0.001))(skip1)
    layer = Add()([layer, skip1])
    skip2 = layer
    layer = Conv1D(256, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(layer)
    layer = BatchNormalization()(layer)
    layer = Conv1D(256, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPool1D(pool_size=3, strides=2, padding='same')(layer)
    skip2 = Conv1D(256, 1, strides=2, padding='same', kernel_regularizer=l2(0.001))(skip2)
    layer = Add()([layer, skip2])
    skip3 = layer
    layer = Conv1D(512, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(layer)
    layer = BatchNormalization()(layer)
    layer = Conv1D(512, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(layer)
    layer = BatchNormalization()(layer)
    layer = MaxPool1D(pool_size=3, strides=2, padding='same')(layer)
    skip3 = Conv1D(512, 1, strides=2, padding='same', kernel_regularizer=l2(0.001))(skip3)
    layer = Add()([layer, skip3])
    layer = tf.nn.dropout(layer, rate=0.6)
    layer = tf.keras.layers.GlobalMaxPooling1D()(layer)
    layer = Dense(1000, activation='relu', kernel_regularizer=l2(0.001))(layer)
    layer = tf.nn.dropout(layer, rate=0.6)
    outputs = Dense(1, activation='sigmoid')(layer)
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    x_train_aug, y_train_aug = augment_data(x_train, y_train)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
    checkpoint = ModelCheckpoint('Sleep-disorder-Classification/GUI/model_skip_HU_new.h5', monitor='val_accuracy', save_best_only=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=10, min_lr=1e-6)
    history = model.fit(x_train_aug, y_train_aug, epochs=100, batch_size=64, validation_data=test_data, callbacks=[early_stop, checkpoint, lr_scheduler])
    return model, history

# LSTM Model
def lstm_model(input_shape, x_train, y_train, test_data):
    model = tf.keras.Sequential([
        Input(shape=input_shape),
        Conv1D(filters=32, kernel_size=7, strides=1, padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        Conv1D(filters=16, kernel_size=9, strides=1, padding='same', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        Conv1D(filters=32, kernel_size=5, padding='same', activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.LSTM(100, kernel_regularizer=l2(0.001)),
        tf.keras.layers.Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    x_train_aug, y_train_aug = augment_data(x_train, y_train)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
    checkpoint = ModelCheckpoint('Sleep-disorder-Classification/GUI/model_lstm_HU_new.h5', monitor='val_accuracy', save_best_only=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=10, min_lr=1e-6)
    history = model.fit(x_train_aug, y_train_aug, epochs=100, batch_size=64, validation_data=test_data, callbacks=[early_stop, checkpoint, lr_scheduler])
    return model, history

# Inception Model
def inception_module(layer, f1, f2, f3, f4):
    conv1 = Conv1D(f1, 1, activation='relu', padding='same', kernel_regularizer=l2(0.001))(layer)
    conv2 = Conv1D(f1, 1, activation='relu', padding='same', kernel_regularizer=l2(0.001))(layer)
    conv3 = Conv1D(f1, 1, activation='relu', padding='same', kernel_regularizer=l2(0.001))(layer)
    maxp = MaxPool1D(pool_size=3, strides=1, padding='same')(layer)
    conv2 = Conv1D(f2, 3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(conv2)
    conv3 = Conv1D(f3, 5, activation='relu', padding='same', kernel_regularizer=l2(0.001))(conv3)
    conv4 = Conv1D(f4, 1, activation='relu', padding='same', kernel_regularizer=l2(0.001))(maxp)
    return concatenate([conv1, conv2, conv3, conv4], axis=-1)

def inception_model(input_shape, x_train, y_train, test_data):
    input_img = Input(shape=input_shape)
    layer = Conv1D(32, 7, activation='relu', padding='same', kernel_regularizer=l2(0.001))(input_img)
    layer = BatchNormalization()(layer)
    layer = MaxPool1D(pool_size=3, strides=2, padding='same')(layer)
    layer = inception_module(layer, 32, 48, 64, 8)
    layer = inception_module(layer, 64, 64, 96, 16)
    layer = MaxPool1D(pool_size=3, strides=2, padding='same')(layer)
    layer = inception_module(layer, 96, 48, 104, 8)
    layer = MaxPool1D(pool_size=3, strides=2, padding='same')(layer)
    layer = tf.nn.dropout(layer, rate=0.6)
    layer = tf.keras.layers.GlobalMaxPooling1D()(layer)
    layer = Dense(500, activation='relu', kernel_regularizer=l2(0.001))(layer)
    layer = tf.nn.dropout(layer, rate=0.6)
    outputs = Dense(1, activation='sigmoid')(layer)
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    x_train_aug, y_train_aug = augment_data(x_train, y_train)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)
    checkpoint = ModelCheckpoint('Sleep-disorder-Classification/GUI/model_incep_HU_new.h5', monitor='val_accuracy', save_best_only=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=10, min_lr=1e-6)
    history = model.fit(x_train_aug, y_train_aug, epochs=100, batch_size=64, validation_data=test_data, callbacks=[early_stop, checkpoint, lr_scheduler])
    return model, history

# Train models
print("Training CNN model...")
cnn_model((1024,1), x_train1, y_train1, (x_test1, y_test1))

print("Training Skip Connection model...")
skip_model((1024,1), x_train1, y_train1, (x_test1, y_test1))

print("Training LSTM model...")
lstm_model((1024,1), x_train1, y_train1, (x_test1, y_test1))

print("Training Inception model...")
inception_model((1024,1), x_train1, y_train1, (x_test1, y_test1))

# Add confusion matrix and plots after training
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load the best model (assuming skip model is the best)
model = tf.keras.models.load_model('Sleep-disorder-Classification/GUI/model_skip_HU_new.h5')

# Predict on test set
y_pred = (model.predict(x_test1) > 0.5).astype(int).flatten()
y_true = y_test1.astype(int)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification report
report = classification_report(y_true, y_pred, target_names=['Healthy', 'Unhealthy'])
print("Classification Report:")
print(report)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Healthy', 'Unhealthy'], rotation=45)
plt.yticks(tick_marks, ['Healthy', 'Unhealthy'])
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('Sleep-disorder-Classification/GUI/confusion_matrix_HU.png')
plt.show()



print("All models trained and saved. Confusion matrix, classification report, and plots generated.")
