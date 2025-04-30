# **Credit Card Fraud Detection - NumPy Operations**

This project demonstrates the use of **NumPy** for implementing a custom **k-Nearest Neighbors (k-NN)** classifier, statistical analysis, and data processing on a **Credit Card Fraud Detection dataset** (`creditcard_2023.csv`).

It is developed as part of an assignment to showcase how **NumPy functions** can be used in practical **AI scenarios** without relying on higher-level libraries like pandas for core operations.

---

## **📁 Files**

- `Assignment_2.py` — Python script implementing a **NumPy-based k-NN algorithm** and comparing it with **scikit-learn’s** implementation. Also includes testing with **Naïve Bayes**.
- `creditcard_2023.csv` — Dataset containing anonymized transaction data, including **PCA-transformed features** and fraud labels.
- `graph.py` — Script used to generate and **visualize graphs** of performance trends.
- `Explanation & Documentation.docx` — Report explaining the **methodology**, **testing strategies**, and **results**.

---

## **🎯 Project Objective**

The main goals are to:

- Load and preprocess a large dataset using **NumPy**
- Implement a **custom k-NN classifier** from scratch using **vectorized operations**
- Compare performance with **scikit-learn’s k-NN** and **Naïve Bayes**
- Perform **basic statistical analysis** using NumPy (mean, standard deviation, etc.)
- Generate visual insights from results

This serves as a foundational example of building AI models and performing data manipulation using **NumPy**, showing how low-level libraries can be powerful tools for learning and prototyping.

---

## **🔧 Key NumPy Operations**

- **Distance Calculations**: Euclidean and Manhattan distances calculated using vectorized NumPy functions
- **Majority Voting**: Label prediction using `np.bincount()` and `argmax()`
- **Data Normalization**: Standardizing features using NumPy math operations
- **Matrix Manipulation**: Transpose, broadcasting, slicing, and filtering

---

## **📌 Sample Code Snippet**

```python
# Normalize dataset using NumPy
X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Compute Euclidean distances from a test sample
distances = np.sqrt(np.sum((X_train - x_test) ** 2, axis=1))

# Predict using majority vote
k_nearest = y_train[np.argsort(distances)[:k]]
prediction = np.bincount(k_nearest).argmax()
