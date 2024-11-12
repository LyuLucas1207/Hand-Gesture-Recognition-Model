
# Hand Gesture Recognition Model

## Project Overview

This project aims to develop a hand gesture recognition model to identify numbers, letters, and other gestures to enhance communication. By manually collecting gesture data and applying data augmentation, we aim to build a robust model. Our classification approach will primarily leverage a Support Vector Machine (SVM), with evaluation metrics such as accuracy, precision, recall, and F1-score. Cross-validation will be used to ensure reliability.

## Dataset

We will create a custom dataset by recording hand gestures for the 26 letters (A-Z) and numbers (0-9) in sign language. This dataset will serve as the training data for the model, allowing it to recognize these sign language symbols. By creating our dataset, we gain control over data quality, consistency, and customization, which optimizes the model's performance for this specific task.

### Key Benefits of a Custom Dataset

- **Flexibility**: Ability to adjust data collection to our specific needs.
- **Standardization**: Ensures consistent quality across samples.
- **High Relevance**: Focuses on task-specific data to enhance the model’s accuracy.

## Model

The proposed model is a **Support Vector Machine (SVM)**, a supervised learning algorithm known for its effectiveness in classification tasks. SVM will classify the gesture data by finding the optimal hyperplane that separates different classes (i.e., letters and numbers).

### Model Features

- **Regularization**: To improve generalization and reduce overfitting.
- **Multi-class Classification**: SVM is suitable for complex gesture images and will handle our multi-class problem effectively.
  
## Evaluation Metrics

To evaluate our model's performance, we will focus on several metrics:

- **Accuracy**: The main evaluation metric, indicating the proportion of correctly classified gestures.
- **Confusion Matrix**: To visualize and understand misclassification patterns.
- **Precision, Recall, F1-score**: To handle any class imbalances and provide insights into model precision and robustness.
  
Additionally, we will assess **computational complexity**, including training and inference times, ensuring that the model is efficient for practical use.

## Proposed Solution

Our solution involves the following steps:

1. **Data Collection**: Manually collect hand gesture data for numbers, letters, and additional gestures using OpenCV.
2. **Data Augmentation**: Apply transformations to increase data variability and model robustness.
3. **Model Training**: Train an SVM classifier on the custom dataset.
4. **Evaluation**: Measure performance using the metrics above to ensure high accuracy and reliability.

This model aims to offer a reliable, efficient solution for hand gesture recognition, advancing communication capabilities.

---

**Note**: This project is part of a research and development effort and will continue to evolve as we iterate on data collection and model optimization.

# Project Structure

```bash
.   
├── .vscode/                            # VSCode configuration files directory
├── classes/                            # Model builder classes
│   ├── KNearestNeighborsBuilder.py     # Configuration builder for K-Nearest Neighbors (KNN) model
│   ├── LogisticRegressionBuilder.py    # Configuration builder for Logistic Regression model
│   ├── ModelTrainerBuilder.py          # Generic model trainer configuration builder
│   ├── RandomForestBuilder.py          # Configuration builder for Random Forest model
│   ├── SklearnNeuralNetworkBuilder.py  # Configuration builder for Neural Network (MLPClassifier) model
│   ├── SupportVectorMachineBuilder.py  # Configuration builder for Support Vector Machine (SVM) model
│   └── __pycache__/                    # Compiled Python cache files
├── data/                               # Data storage directory
│   └── data.pickle                     # Example dataset in pickle format
├── enums/                              # Enumerations and mappings
│   ├── labels_dict.py                  # Dictionary mapping for labels
│   ├── model_dict.py                   # Dictionary mapping for model selection
│   ├── __init__.py                     # Initialization module for enums package
│   └── __pycache__/                    # Compiled Python cache files
├── environments/                       # Training environments and workflows
│   ├── cnn_tensorflow_trainer.py       # TensorFlow CNN training workflow script
│   ├── simple_trainer.py               # A simple training workflow
│   ├── __init__.py                     # Initialization module for environments package
│   └── __pycache__/                    # Compiled Python cache files
├── images/                             # Directory for storing image resources
├── interface/                          # Interface layer scripts
│   ├── simple_classifier.py            # A simple classification interface
│   ├── __init__.py                     # Initialization module for interface package
│   └── __pycache__/                    # Compiled Python cache files
├── models/                             # Trained models storage
│   ├── svm.p                           # Serialized Support Vector Machine (SVM) model
├── output/                             # Output results directory
│   └── svm.txt                         # Results file for SVM model evaluation
├── test/                               # Test scripts directory
├── ui/                                 # User interface modules
│   ├── image_collection.py             # Module for collecting images
│   ├── image_depiction.py              # Module for describing images
│   ├── image_process.py                # Module for image processing
│   ├── image_showing.py                # Module for displaying images
│   ├── __init__.py                     # Initialization module for UI package
│   └── __pycache__/                    # Compiled Python cache files
├── utils/                              # Utility scripts
│   ├── datatools.py                    # Tools for data handling
│   ├── filetools.py                    # Tools for file operations
│   ├── graphtools.py                   # Tools for creating graphs and visualizations
│   ├── modeltools.py                   # Tools for model handling and utilities
│   ├── printtools.py                   # Tools for formatted console printing
│   ├── videotools.py                   # Tools for video processing
│   ├── __init__.py                     # Initialization module for utils package
│   └── __pycache__/                    # Compiled Python cache files
├── .gitignore                          # Git ignore file for excluding unwanted files
├── Hand.png                            # Example image file
├── License                             # Project license
├── README.md                           # Project documentation
└── requirements.txt                    # Python package dependencies
```

# Executable Files

## User Interface (ui/)

### Collect Gesture Images

```bash
python -m ui.image_collection
```

### Display Hand Gesture Landmarks

```bash
python -m ui.image_depiction
```

### Process Collected Images

```bash
python -m ui.image_process
```

### Show Collected Images

```bash
python -m ui.image_showing
```

## Inference Interface (interface/)

### Run Gesture Classifier

```bash
python -m interface.simple_classifier
```

## Model Training Environment (environments/)

### Train Simple Model

```bash
python -m environments.simple_trainer
```
