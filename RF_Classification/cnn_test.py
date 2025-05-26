import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import keras
import pickle as pk
from keras.models import Sequential
from keras.layers import Reshape, Dropout, Dense, Activation, BatchNormalization
from keras.layers import Conv2D, ZeroPadding2D, GlobalAveragePooling2D, Flatten
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix
import pdb

# ------------------- Data Loading -------------------

# Load the IQ data
iq_data = pk.load(open('gold.dat', 'rb'), encoding='latin1')  # CHANGE the path accordingly

print('Dataset imported')

snrs, modulation = map(lambda j: sorted(list(set(map(lambda x: x[j], iq_data.keys())))), [1, 0])

print(f'Modulation labels: {modulation}')
print(f'SNR values for each modulation: {snrs}')

x_data = []
label = []
for m in modulation:
    for snr in snrs:
        samples = iq_data[(m, snr)]
        x_data.append(samples)
        for l in np.arange(samples.shape[0]):
            label.append((m, snr))

x_stacked = np.vstack(x_data)
print(f'Dataset shape: {x_stacked.shape}')

# ------------------- Train/Test Split -------------------

np.random.seed(200)
N_samples = x_stacked.shape[0]
N_train = int(N_samples * 0.7)

train_Idx = np.random.choice(np.arange(N_samples), size=N_train, replace=False)
test_Idx = list(set(np.arange(N_samples)) - set(train_Idx))

x_train = x_stacked[train_Idx]
x_test = x_stacked[test_Idx]
print(x_train.shape, x_test.shape)

input_encode = lambda x: modulation.index(label[x][0])
y_list_train = np.array(list(map(input_encode, train_Idx)), dtype='float32')
y_list_test = np.array(list(map(input_encode, test_Idx)), dtype='float32')
print(len(y_list_train), len(y_list_test))

y_train = to_categorical(y_list_train, len(modulation))
y_test = to_categorical(y_list_test, len(modulation))

print('Number of Samples, height, width')
print(x_train.shape)

N, H, W = x_train.shape
N_test = x_test.shape[0]
C = 1

x_train = x_train.reshape(N, H, W, C)
x_test = x_test.reshape(N_test, H, W, C)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

plt.plot(x_train[0, 0, :], x_train[0, 1, :], '.')

# ------------------- CNN Model Definition -------------------

model = Sequential(name='CNN_Architecture')

model.add(ZeroPadding2D((0, 2), data_format='channels_last'))
model.add(Conv2D(64, (2, 3), activation='relu', data_format='channels_last', input_shape=(H, W, C), name='conv1'))
model.add(Dropout(0.5))
model.add(Conv2D(80, (1, 3), activation='relu', data_format='channels_last'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(modulation), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.build(input_shape=(None, H, W, C))
model.summary()

# ------------------- Training -------------------

epoch = 100
batch = 1024

checkpoint = ModelCheckpoint(
    filepath='CNN.weights.h5',  # Save locally in the HPC
    monitor='loss',
    save_best_only=True,
    mode='auto', 
    save_weights_only=True
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,  # <-- You can adjust patience (how many epochs to wait before stopping)
    restore_best_weights=True
)

# Train the model
start_run = model.fit(
    x_train, y_train,
    batch_size=batch,
    epochs=epoch,
    verbose=2,
    validation_data=(x_test, y_test),
    callbacks=[checkpoint, early_stopping]
)

# ------------------- Evaluation -------------------

model.load_weights('CNN.weights.h5')

y_pred = model.predict(x_test, batch_size=batch)
score = model.evaluate(x_test, y_test, verbose=0, batch_size=batch)
print(f"Test Loss and Accuracy: {score}")

# ------------------- Confusion Matrix -------------------

y_true_classes = np.argmax(y_test, axis=1)
y_pred_classes = np.argmax(y_pred, axis=1)

# cm = confusion_matrix(y_true_classes, y_pred_classes)
# print("Confusion Matrix:")
# print(cm)
history = start_run.history

# Find best validation accuracy
best_val_acc = max(history['val_accuracy'])
best_val_acc_epoch = history['val_accuracy'].index(best_val_acc) + 1  # +1 because epochs are 1-indexed

print(f"🏆 Best Validation Accuracy: {best_val_acc:.4f} at Epoch {best_val_acc_epoch}")
