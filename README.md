
# CPEN355 Final Project -- Hand Gesture Recognition Model

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
├── data/                          # Contains data files used in the project
│   └── data.pickle                # Serialized data file
├── enums/                         # Enumerations and constants
│   ├── __init__.py                # Initialization file for the module
│   └── labels_dict.py             # Dictionary for mapping labels to gestures
├── environments/                  # Model training environments
│   ├── __init__.py                # Initialization file for the module
│   ├── random_forest_trainer.py   # Script to train the random forest model
│   └── may_have_more_model.py
├── images/                        # Directory to store images
├── interface/                     # Interface for inference
│   ├── __init__.py                # Initialization file for the module
│   └── inference_classifier.py    # Script to classify gestures using the trained model
├── models/                        # Directory to store pre-trained models
│   └── random_forest_model.p      # Pre-trained random forest model file
├── test/                          # Testing-related scripts and resources
│   ├── __init__.py                # Initialization file for the module
│   ├── images/                    # Test image directory
│   └── testfiletools.py           # Test cases for file utilities
├── ui/                            # User interface and data collection modules
│   ├── __init__.py                # Initialization file for the module
│   ├── image_collection.py        # Script to collect gesture images
│   ├── image_depiction.py         # Script to depict hand gestures
│   ├── image_process.py           # Script for processing images
│   └── image_showing.py           # Script for displaying images
├── utils/                         # Utility modules for various tasks
│   ├── __init__.py                # Initialization file for the module
│   ├── datatools.py               # Utilities for data-related operations
│   ├── filetools.py               # Utilities for file and directory operations
│   ├── graphtools.py              # Graph and visualization utilities
│   ├── modeltools.py              # Tools for saving and loading models
│   └── videotools.py              # Video processing utilities
├── .gitignore                     # Specifies intentionally untracked files to ignore
├── Hand.png                       # Example image of a hand gesture
├── License                        # Project license file
├── README.md                      # Project documentation file
├── requirements.txt               # Python dependencies for the project
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
python -m interface.inference_classifier
```

## Model Training Environment (environments/)

### Train Random Forest Model

```bash
python -m environments.random_forest_trainer
```
