# Activity Recognition using SVM and CNN

This project performs binary classification of human activities (e.g., *Open Door* vs. *Rub Hands*) using motion sensor data collected from Accelerometer and Gyroscope. Two classification approaches are implemented:

1. **Support Vector Machine (SVM)** with statistical feature extraction
2. **1D Convolutional Neural Network (CNN)** using raw time-series data in PyTorch

---

## 📂 Dataset

Preprocessed `.npy` files are used as input:

- **train_MSAccelerometer_OpenDoor_RubHands.npy**
- **train_MSGyroscope_OpenDoor_RubHands.npy**
- **train_labels_OpenDoor_RubHands.npy**
- **test_MSAccelerometer_OpenDoor_RubHands.npy**
- **test_MSGyroscope_OpenDoor_RubHands.npy**
- **test_labels_OpenDoor_RubHands.npy**

Each input shape: `(samples, time_steps, axes)`

---

## 🧮 Feature Engineering (for SVM)

For each sensor axis, the following 8 features are extracted:
- Mean
- Max
- Min
- Standard Deviation
- Median
- Range
- Skewness
- Kurtosis

Total: `2 sensors × 3 axes × 8 features = 48 features/sample`

---

## ⚙️ Technologies Used

- Python
- NumPy, SciPy
- scikit-learn
- PyTorch
- matplotlib, seaborn

---

## ✅ SVM Classifier

- **Input**: Extracted 48-dimensional feature vectors
- **Model**: `SVC(kernel='linear')`
- **Evaluation Metrics**:
  - Accuracy
  - Macro F1-Score
  - Weighted F1-Score
  - Confusion Matrix (with labels)

---

## ✅ CNN Classifier (PyTorch)

- **Input**: Raw concatenated sensor data
- **Architecture**:
  - 3 Conv1D layers + MaxPool1D
  - Flatten
  - 2 Fully Connected layers
- **Training**:
  - Loss: CrossEntropyLoss
  - Optimizer: SGD
  - Epochs: 150
  - Batch size: 50

---

## 📊 Evaluation

- Confusion matrix is plotted for both models
- Classification report includes:
  - Precision
  - Recall
  - F1-Score (per class and average)

---

## 📁 Project Structure

```
activity-recognition/
├── train_MSAccelerometer_OpenDoor_RubHands.npy
├── train_MSGyroscope_OpenDoor_RubHands.npy
├── train_labels_OpenDoor_RubHands.npy
├── test_MSAccelerometer_OpenDoor_RubHands.npy
├── test_MSGyroscope_OpenDoor_RubHands.npy
├── test_labels_OpenDoor_RubHands.npy
└── main.ipynb
```

---

## 📘 Learning Outcomes

- Processed motion sensor data from accelerometers and gyroscopes
- Engineered features for traditional ML classification
- Built a deep learning pipeline using PyTorch
- Understood the impact of feature engineering vs deep feature learning
- Applied evaluation metrics like accuracy and F1-score for imbalanced classes



