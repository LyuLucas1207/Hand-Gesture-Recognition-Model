# How to run: python -m environments.SVM_trainer

import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

import utils.modeltools as mt

# Load the data
data_dict = pickle.load(open('./data/data.pickle', 'rb'))

print(data_dict.keys())  # dict_keys(['data', 'labels'])

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize the SVM classifier
model = SVC(kernel='linear', probability=True)

# Train the model
model.fit(x_train, y_train)

# Make predictions
y_predict = model.predict(x_test)

# Evaluate the model
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

# Save the model
mt.save_model(model, './models/svm_model.p')
