# **Credit Card Fraud Detection using AI & NumPy**

This project demonstrates how **Artificial Intelligence** and **NumPy** can be used to build effective models for detecting **fraudulent credit card transactions**. It includes a **custom implementation of the k-Nearest Neighbors (k-NN)** algorithm using only NumPy, as well as comparisons with popular **AI models from scikit-learn** like **Naïve Bayes** and built-in k-NN.

Fraud detection is a high-stakes binary classification problem, and this project provides insight into **how AI models can be trained, tested, and validated** to classify transactions as either **legitimate** or **fraudulent**, even in the face of class imbalance and anonymized features.

---

## **📁 Files**

- `Assignment_2.py` — Core script implementing:
  - Custom **k-NN classifier using NumPy**
  - scikit-learn **k-NN** and **Naïve Bayes** models
  - Performance evaluation (Accuracy & F1 Score)
- `creditcard_2023.csv` — Real-world dataset of credit card transactions, including anonymized features and fraud labels.
- `graph.py` — Python script to generate and visualize **trends, accuracy comparisons**, and other insights.
- `Explanation & Documentation.docx` — Detailed report covering AI approach, testing methodology, observations, and conclusions.

---

## **🎯 Project Objective**

This project focuses on:

- **Classifying** credit card transactions as **fraudulent or not** (binary classification)
- Understanding the **inner workings of k-NN** by building it from scratch with **NumPy**
- Evaluating multiple AI models (k-NN and Naïve Bayes) using **real imbalanced data**
- Comparing results between **custom logic** and **prebuilt AI tools**
- Providing explainable metrics and graphs to justify AI model selection

---

## **🤖 AI Techniques & Models Used**

### ✅ 1. **Custom k-NN Classifier (NumPy-based)**
- Manually computes distances using Euclidean/Manhattan metrics
- Uses NumPy’s array and sorting functions for prediction logic
- Produces similar performance to scikit-learn’s model — validating the approach

### ✅ 2. **Scikit-learn k-NN**
- Hyperparameter tuning: `k`, weighting (`uniform` vs `distance`), distance metric (`p=1`, `p=2`)
- Uses `KNeighborsClassifier` and `cross_val_score` for robust evaluation

### ✅ 3. **Naïve Bayes Classifier**
- Fast, probabilistic model for comparison
- Highlights trade-offs between simplicity and predictive power

---

## **🧠 Classification Focus: Why AI for Fraud Detection?**

- Credit card fraud detection is **imbalanced and noisy** — perfect for AI-driven insights
- AI models can **learn patterns** even in anonymized, transformed features (like PCA)
- Your k-NN implementation and experiments show how **different configurations** (like `k=3` with distance weighting) lead to better fraud detection
- **Naïve Bayes**, while fast, struggles slightly due to its assumption of feature independence — demonstrating why **model selection matters in real AI applications**

---

## **📈 Performance Summary**

| Model                     | Accuracy | F1 Score |
|---------------------------|----------|----------|
| Custom k-NN (NumPy)       | 0.97     | 0.97     |
| Scikit-learn k-NN (best)  | 0.97     | 0.97     |
| Naïve Bayes (sklearn)     | 0.93     | 0.93     |

- Best performance was achieved with `k=3`, `weights='distance'`, and `p=1` (Manhattan Distance)
- F1 Score is emphasized due to **class imbalance**

---

## **📊 Sample NumPy Logic for AI Classification**

```python
# Normalize data
X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Compute distances from test sample
distances = np.sqrt(np.sum((X_train - x_test) ** 2, axis=1))

# Predict using majority voting
neighbors = y_train[np.argsort(distances)[:k]]
prediction = np.bincount(neighbors).argmax()
