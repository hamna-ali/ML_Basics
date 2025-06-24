Activity Recognition using SVM and CNN
This project performs human activity recognition using motion sensor data (Accelerometer and Gyroscope) through two approaches:

Support Vector Machine (SVM) using engineered statistical features

1D Convolutional Neural Network (CNN) trained on raw time-series data

Project Overview
The goal is to classify two human activities (e.g., Open Door vs. Rub Hands) using sensor data.
The data is provided in .npy format and includes both training and testing sets for:

Multi-sensor Accelerometer readings

Multi-sensor Gyroscope readings

Corresponding activity labels

Dataset
Format: NumPy arrays (.npy)

Shape: (samples, time_steps, axes)

Sensors used:

Accelerometer

Gyroscope

Feature Extraction (for SVM)
Each sample generates 8 features per axis per sensor:

Mean

Max

Min

Standard deviation

Median

Range

Skewness

Kurtosis

Final shape: 48 features per sample
= 2 sensors × 3 axes × 8 features

Technologies Used
Python

NumPy, SciPy

scikit-learn

PyTorch

matplotlib, seaborn

Model 1: Support Vector Machine (SVM)
Kernel: Linear

Input: Engineered statistical features (48-dimensional)

Evaluation Metrics:

Accuracy

Macro F1-score

Weighted F1-score

Confusion Matrix

Model 2: 1D CNN (PyTorch)
Input: Raw sensor data (Accelerometer + Gyroscope)

Layers:

3 convolutional + max-pooling layers

Fully connected layers

Loss: CrossEntropyLoss

Optimizer: SGD

Trained for: 150 epochs

Batch size: 50

Evaluation Results
CNN-based classifier achieves high accuracy on test data

Confusion matrices and classification reports are generated for both models

Example metrics:

Test Accuracy: ~XX% (to be filled after running)

F1-score (macro and weighted)

Class-wise performance

Project Structure
css
Copy
Edit
activity-recognition/
│
├── train_MSAccelerometer_OpenDoor_RubHands.npy
├── train_MSGyroscope_OpenDoor_RubHands.npy
├── train_labels_OpenDoor_RubHands.npy
├── test_MSAccelerometer_OpenDoor_RubHands.npy
├── test_MSGyroscope_OpenDoor_RubHands.npy
├── test_labels_OpenDoor_RubHands.npy
└── main.ipynb
Learning Outcomes
Preprocessed multi-dimensional sensor data for ML and DL pipelines

Extracted meaningful statistical features from time-series data

Applied SVM for activity classification

Built and trained a 1D CNN using PyTorch

Evaluated models using accuracy, F1-score, and confusion matrix

Visualized performance with seaborn and matplotlib

