##=========================================================
#
#  multiscale CNN Bi-LSTM network 
#
#  author: shaokuiWang, JunyeLin
#
#==========================================================

from sklearn import preprocessing
import tensorflow as tf
import time
import keras
from keras.models import load_model, Model
import keras.optimizers as opt
import keras.losses
import keras.metrics
from keras.metrics import Recall, Precision, SensitivityAtSpecificity
import keras.backend as K
import keras.callbacks
import numpy as np
import scipy.io
from numpy import random
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization, Dropout, Dense, Bidirectional, LSTM, LeakyReLU, ReLU, concatenate

# Load data from MATLAB files
data_e = scipy.io.loadmat('E_P_ZJ.mat').get('data_e')
data_p = scipy.io.loadmat('E_P_ZJ.mat').get('data_p')
labels = scipy.io.loadmat('E_P_ZJ.mat').get('label')[0]

data_e_n = scipy.io.loadmat('E_P_ZJ_p.mat').get('data_e_n')
data_p_n = scipy.io.loadmat('E_P_ZJ_p.mat').get('data_p_n')
labels_n = scipy.io.loadmat('E_P_ZJ_p.mat').get('label')[0]

# Set the first 408 labels to 0 for both sets
labels[0:408] = 0
labels_n[0:408] = 0

# Data augmentation: stack original, additional, and negative signals
data_e = np.vstack((data_e, data_e_n, -data_e))
data_p = np.vstack((data_p, data_p_n, -data_p))
labels = np.concatenate((labels, labels_n, labels))

# Shuffle data indices
num_samples = len(labels)
indices = np.arange(num_samples)
random.seed(int(time.time()))
random.shuffle(indices)

# Preprocess data: scale to [0,1] and apply one-hot encoding to labels
train_data_ecg = preprocessing.minmax_scale(data_e, axis=-1, feature_range=(0, 1))[indices]
train_data_pcg = preprocessing.minmax_scale(data_p, axis=-1, feature_range=(0, 1))[indices]
train_labels = to_categorical(labels)[indices]
print(train_labels[0:200])
print(train_labels.shape)


def MultiScaleCNNFusionBiLSTM():
    """
    Build a multi-scale CNN fusion network for ECG and PCG signals, followed by a Bi-LSTM and Dense layers.
    The architecture corresponds to the multi-scale and feature fusion framework described in our paper.
    """

    ########################### ECG Branch ###########################
    # ECG input: shape (2500, 1); note that ECG sampling rate is 500Hz (thus kernel sizes are scaled by 1/4 relative to PCG)
    ecg_input = Input(shape=(2500, 1), name="ECG_Input")

    # Layer 1: Convolutional layer with kernel size 50 (i.e., 200/4) and stride 2 --> output shape ~ (1250, 64)
    ecg_conv1 = Conv1D(filters=64, kernel_size=50, strides=2, padding='same', activation='relu', name="ECG_Conv1")(ecg_input)
    ecg_bn1 = BatchNormalization(name="ECG_BN1")(ecg_conv1)

    # Layer 2: MaxPooling with pool size 5 and stride 2 --> output shape ~ (625, 64), followed by dropout
    ecg_pool1 = MaxPooling1D(pool_size=5, strides=2, padding='same', name="ECG_Pool1")(ecg_bn1)
    ecg_drop1 = Dropout(0.5, name="ECG_Dropout1")(ecg_pool1)

    # Layer 3: Convolutional layer with kernel size 25 (i.e., 100/4) and stride 2 --> output shape ~ (313, 128)
    ecg_conv2 = Conv1D(filters=128, kernel_size=25, strides=2, padding='same', activation='relu', name="ECG_Conv2")(ecg_drop1)
    ecg_bn2 = BatchNormalization(name="ECG_BN2")(ecg_conv2)

    # Layer 4: MaxPooling with pool size 5 and stride 2 --> output shape ~ (157, 128), followed by dropout
    ecg_pool2 = MaxPooling1D(pool_size=5, strides=2, padding='same', name="ECG_Pool2")(ecg_bn2)
    ecg_drop2 = Dropout(0.5, name="ECG_Dropout2")(ecg_pool2)

    # Layer 5 & 6 (Multi-scale convolution):
    # Small-scale branch: kernel size 25 (100/4) with stride 1
    ecg_conv_small = Conv1D(filters=128, kernel_size=25, strides=1, padding='same', activation='relu', name="ECG_Conv_Small")(ecg_drop2)
    ecg_bn_small = BatchNormalization(name="ECG_BN_Small")(ecg_conv_small)
    # Large-scale branch: kernel size 100 (400/4) with stride 1
    ecg_conv_large = Conv1D(filters=128, kernel_size=100, strides=1, padding='same', activation='relu', name="ECG_Conv_Large")(ecg_drop2)
    ecg_bn_large = BatchNormalization(name="ECG_BN_Large")(ecg_conv_large)

    # Layer 7: Feature fusion via element-wise addition of multi-scale features --> output shape remains (157, 128)
    ecg_fused = keras.layers.add([ecg_bn_small, ecg_bn_large], name="ECG_MultiScale_Add")

    # Layer 8: Additional MaxPooling to integrate features; pool size 3, stride 1, followed by dropout
    ecg_pool3 = MaxPooling1D(pool_size=3, strides=1, padding='same', name="ECG_Pool3")(ecg_fused)
    ecg_drop3 = Dropout(0.5, name="ECG_Dropout3")(ecg_pool3)

    # Layer 9: Bi-LSTM layer with 256 units (bidirectional gives 512 features) with dropout
    ecg_bilstm = Bidirectional(LSTM(256, return_sequences=True), name="ECG_BiLSTM")(ecg_drop3)
    ecg_drop4 = Dropout(0.5, name="ECG_Dropout4")(ecg_bilstm)

    # Layer 10: Dense layer to reduce dimensionality to 256 with LeakyReLU activation and BN
    ecg_dense1 = Dense(256, name="ECG_Dense1")(ecg_drop4)
    ecg_act1 = LeakyReLU(name="ECG_LeakyReLU1")(ecg_dense1)
    ecg_bn3 = BatchNormalization(name="ECG_BN3")(ecg_act1)

    # Layer 11: Final Dense layer to obtain 128 features with dropout
    ecg_dense2 = Dense(128, name="ECG_Dense2")(ecg_bn3)
    ecg_act2 = LeakyReLU(name="ECG_LeakyReLU2")(ecg_dense2)
    ecg_bn4 = BatchNormalization(name="ECG_BN4")(ecg_act2)

    # Define ECG model
    ecg_model = Model(ecg_input, ecg_bn4, name="ECG_Model")


    ########################### PCG Branch ###########################
    # PCG input: shape (10000, 1); PCG sampling rate is 2000Hz
    pcg_input = Input(shape=(10000, 1), name="PCG_Input")
    pcg_bn_input = BatchNormalization(name="PCG_BN_Input")(pcg_input)

    # Layer 1: Convolutional layer with kernel size 200 and stride 2 --> output shape ~ (1250, 64)
    pcg_conv1 = Conv1D(filters=64, kernel_size=200, strides=2, padding='same', activation='relu', name="PCG_Conv1")(pcg_bn_input)
    pcg_bn1 = BatchNormalization(name="PCG_BN1")(pcg_conv1)

    # Layer 2: MaxPooling with pool size 5 and stride 2 --> output shape ~ (625, 64), followed by dropout
    pcg_pool1 = MaxPooling1D(pool_size=5, strides=2, padding='same', name="PCG_Pool1")(pcg_bn1)
    pcg_drop1 = Dropout(0.5, name="PCG_Dropout1")(pcg_pool1)

    # Layer 3: Convolutional layer with kernel size 100 and stride 2 --> output shape ~ (313, 128)
    pcg_conv2 = Conv1D(filters=128, kernel_size=100, strides=2, padding='same', activation='relu', name="PCG_Conv2")(pcg_drop1)
    pcg_bn2 = BatchNormalization(name="PCG_BN2")(pcg_conv2)

    # Layer 4: MaxPooling with pool size 5 and stride 2 --> output shape ~ (157, 128), followed by dropout
    pcg_pool2 = MaxPooling1D(pool_size=5, strides=2, padding='same', name="PCG_Pool2")(pcg_bn2)
    pcg_drop2 = Dropout(0.5, name="PCG_Dropout2")(pcg_pool2)

    # Layer 5 & 6 (Multi-scale convolution):
    # Small-scale branch: kernel size 100 with stride 1
    pcg_conv_small = Conv1D(filters=128, kernel_size=100, strides=1, padding='same', activation='relu', name="PCG_Conv_Small")(pcg_drop2)
    pcg_bn_small = BatchNormalization(name="PCG_BN_Small")(pcg_conv_small)
    # Large-scale branch: kernel size 400 with stride 1
    pcg_conv_large = Conv1D(filters=128, kernel_size=400, strides=1, padding='same', activation='relu', name="PCG_Conv_Large")(pcg_drop2)
    pcg_bn_large = BatchNormalization(name="PCG_BN_Large")(pcg_conv_large)

    # Layer 7: Feature fusion via element-wise addition --> output shape remains (157, 128)
    pcg_fused = keras.layers.add([pcg_bn_small, pcg_bn_large], name="PCG_MultiScale_Add")
    pcg_bn_fused = BatchNormalization(name="PCG_BN_Fused")(pcg_fused)

    # Layer 8: Additional MaxPooling with pool size 3 and stride 1 --> output shape (157, 128)
    pcg_pool3 = MaxPooling1D(pool_size=3, strides=1, padding='same', name="PCG_Pool3")(pcg_bn_fused)

    # Scale up to ensure consistency with ECG data points (by factor of 4)
    pcg_scaled = Conv1D(filters=128, kernel_size=1, strides=4, padding='same', name="PCG_ScaleUp")(pcg_pool3)

    # Layer 9: Bi-LSTM with 256 units --> output shape (157, 512) after bidirectionality, with dropout
    pcg_bilstm = Bidirectional(LSTM(256, return_sequences=True), name="PCG_BiLSTM")(pcg_scaled)
    pcg_drop3 = Dropout(0.5, name="PCG_Dropout3")(pcg_bilstm)

    # Layer 10: Dense layer to reduce dimensionality to 256 with LeakyReLU and BN
    pcg_dense1 = Dense(256, name="PCG_Dense1")(pcg_drop3)
    pcg_act1 = LeakyReLU(name="PCG_LeakyReLU1")(pcg_dense1)
    pcg_bn3 = BatchNormalization(name="PCG_BN3")(pcg_act1)

    # Layer 11: Final Dense layer to obtain 128 features with dropout
    pcg_dense2 = Dense(128, name="PCG_Dense2")(pcg_bn3)
    pcg_act2 = LeakyReLU(name="PCG_LeakyReLU2")(pcg_dense2)
    pcg_bn4 = BatchNormalization(name="PCG_BN4")(pcg_act2)
    pcg_model = Model(pcg_input, pcg_bn4, name="PCG_Model")


    ########################### Fusion and CNN+Bi-LSTM Layers ###########################
    # Fusion: Concatenate ECG and PCG features along the feature dimension
    fused_features = concatenate([ecg_model.output, pcg_model.output], name="Feature_Concatenation")
    fused_bn = BatchNormalization(name="Fused_BN")(fused_features)

    # # Fusion CNN layer: Convolution with kernel size 5 and stride 2 --> output shape ~ (79, 256)
    # fusion_conv = Conv1D(filters=256, kernel_size=5, strides=2, padding='same', activation=ReLU(), name="Fusion_Conv")(fused_bn)
    # 
    # # Fusion pooling: MaxPooling with pool size 10 and stride 2 --> output shape ~ (40, 256)
    # fusion_pool = MaxPooling1D(pool_size=10, strides=2, padding='same', name="Fusion_Pool")(fusion_conv)
    # fusion_act = ReLU(name="Fusion_ReLU")(fusion_pool)
    # fusion_bn = BatchNormalization(name="Fusion_BN")(fusion_act)

    # Bi-LSTM layers and further CNN layers for temporal modeling
    fusion_bilstm1 = Bidirectional(LSTM(256, return_sequences=True), name="Fusion_BiLSTM1")(fused_bn)
    fusion_drop1 = Dropout(0.5, name="Fusion_Dropout1")(fusion_bilstm1)
    fusion_conv2 = Conv1D(filters=512, kernel_size=5, strides=2, padding='same', activation=ReLU(), name="Fusion_Conv2")(fusion_drop1)
    fusion_bn2 = BatchNormalization(name="Fusion_BN2")(fusion_conv2)
    fusion_bilstm2 = Bidirectional(LSTM(256, return_sequences=True), name="Fusion_BiLSTM2")(fusion_bn2)
    fusion_conv3 = Conv1D(filters=512, kernel_size=5, strides=1, padding='same', activation=ReLU(), name="Fusion_Conv3")(fusion_bilstm2)
    fusion_bilstm3 = Bidirectional(LSTM(512, return_sequences=True), name="Fusion_BiLSTM3")(fusion_conv3)
    fusion_drop2 = Dropout(0.5, name="Fusion_Dropout2")(fusion_bilstm3)
    fusion_conv4 = Conv1D(filters=512, kernel_size=5, strides=1, padding='same', activation=ReLU(), name="Fusion_Conv4")(fusion_drop2)
    fusion_bilstm4 = Bidirectional(LSTM(512, return_sequences=True), name="Fusion_BiLSTM4")(fusion_conv4)
    fusion_bilstm5 = Bidirectional(LSTM(512, return_sequences=True), name="Fusion_BiLSTM5")(fusion_bilstm4)
    fusion_bilstm6 = Bidirectional(LSTM(256, return_sequences=False), name="Fusion_BiLSTM6")(fusion_bilstm5)
    fusion_drop3 = Dropout(0.5, name="Fusion_Dropout3")(fusion_bilstm6)

    # Fully connected layers for final classification
    fc1 = Dense(1024, activation='relu', name="FC1")(fusion_drop3)
    fc2 = Dense(1024, activation='relu', name="FC2")(fc1)
    output_layer = Dense(2, activation='sigmoid', name="Output")(fc2)

    # Define the complete model
    model = Model(inputs=[ecg_model.input, pcg_model.input], outputs=output_layer, name="MultiScale_CNN_Fusion_BiLSTM")
    model.summary()

    # Compile the model with Adam optimizer and categorical crossentropy loss
    model.compile(optimizer=opt.Adam(lr=0.00001), loss=keras.losses.categorical_crossentropy,
                  metrics=["accuracy", Recall(), Precision(), SensitivityAtSpecificity(0.5232)])

    # Callbacks for model checkpointing and learning rate reduction
    checkpoint = keras.callbacks.ModelCheckpoint('ep_net_31.h5', monitor='val_accuracy', verbose=1,
                                                   save_best_only=True, mode='max', name="Model_Checkpoint")
    lr_reducer = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1,
                                                   mode='max', min_lr=1e-8, epsilon=0.0005, name="LR_Reducer")

    # Class weights for handling class imbalance
    class_weights = {0: 1, 1: 1, 2: 3.5}

    # Train the model using 80% training and 20% validation split
    history = model.fit([train_data_ecg, train_data_pcg], train_labels,
                        batch_size=100, epochs=50, validation_split=0.2,
                        callbacks=[lr_reducer, checkpoint])

    return history


# Train the model and plot accuracy curves
history = MultiScaleCNNFusionBiLSTM()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()