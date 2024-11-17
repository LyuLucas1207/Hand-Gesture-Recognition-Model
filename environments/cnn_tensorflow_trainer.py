# How to run: python -m environments.cnn_tensorflow_trainer

import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the data
data_dict = pickle.load(open("./data/data.pickle", "rb"))

print(data_dict.keys())  # dict_keys(['data', 'labels'])

data = np.asarray(data_dict["data"])
labels = np.asarray(data_dict["labels"])

print(f"Data shape: {data.shape}")

# Reshape data for CNN
num_samples, feature_size = data.shape
data = data.reshape(num_samples, 6, 7, 1)  # Assuming features = 6x7
print(f"Data shape after reshaping: {data.shape}")

labels = to_categorical(labels)  # One-hot encode labels

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=np.argmax(labels, axis=1)
)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
)
datagen.fit(x_train)

# Model architecture
model = Sequential([
    Conv2D(64, (3, 3), activation="relu", padding="same", input_shape=(6, 7, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(256, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),
    
    Dense(labels.shape[1], activation="softmax"),
])


# Compile model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# Callbacks
early_stopping = EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
)

# Train model
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    validation_data=(x_test, y_test),
    epochs=50,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Evaluate model
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Final Model Accuracy: {accuracy * 100:.2f}%")

# Save the model
model.save("./models/cnn_advanced.h5")
