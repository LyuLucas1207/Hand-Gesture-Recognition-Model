import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Load the data
data_dict = pickle.load(open('./data/data.pickle', 'rb'))

print(data_dict.keys())  # dict_keys(['data', 'labels'])

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Reshape data for CNN (assuming 2D data)
# Replace (6, 7) with the dimensions of your feature if it fits logically
data = data.reshape(-1, 6, 7, 1)  # Example: Reshape (num_samples, height, width, channels)

# One-hot encode labels
labels = to_categorical(labels)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=np.argmax(labels, axis=1))

# Initialize the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(6, 7, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(labels.shape[1], activation='softmax')  # Output layer with softmax
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'CNN Accuracy: {accuracy * 100:.2f}%')

# Save the model
model.save('./models/cnn_tensorflow.h5')
