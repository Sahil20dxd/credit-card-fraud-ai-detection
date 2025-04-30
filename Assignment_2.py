"""
The dataset contains numeric features related to credit card transactions, such as transaction amount, time, and anonymized features (V1, V2, ..., V28) resulting from PCA transformation. The last column represents the classification label (fraud or not fraud).

Author: Sahil Hirpara
"""

import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

def read_data(sample_size=1000):
    """
    Reads the CSV file and splits it into numeric data and labels.
    Only a specified number of samples are used for faster processing.

    Args:
        sample_size (int): Number of samples to use (default: 1000).

    Returns:
        headers (ndarray): Array of column headers.
        numeric_data (ndarray): Array of numeric data (all columns except the last).
        labels (ndarray): Array of labels (last column).
    """
    data = np.genfromtxt('creditcard_2023.csv', delimiter=',', skip_header=1)
    headers = np.genfromtxt('creditcard_2023.csv', delimiter=',', max_rows=1, dtype=str)

    # Subsample the dataset
    indices = np.random.choice(len(data), sample_size, replace=False)
    data = data[indices]

    numeric_data = data[:, :-1].astype(float)
    labels = data[:, -1].astype(int)
    return headers, numeric_data, labels

def euclidean_distance(x1, x2):
    """
    Calculates the Euclidean distance between two points.

    Args:
        x1 (ndarray): First point.
        x2 (ndarray): Second point.

    Returns:
        float: Euclidean distance between x1 and x2.
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn_predict(train_data, train_labels, test_point, k):
    """
    Predicts the label for a test data point using the k-NN algorithm.

    Args:
        train_data (ndarray): Training data.
        train_labels (ndarray): Training labels.
        test_point (ndarray): Test data point.
        k (int): Number of neighbors to consider.

    Returns:
        int: Predicted label for the test data.
    """
    distances = []
    for i in range(len(train_data)):
        dist = euclidean_distance(test_point, train_data[i])
        distances.append((dist, train_labels[i]))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    labels = [neighbor[1] for neighbor in neighbors]
    return max(set(labels), key=labels.count)

def main():
    """
    Main function to execute the smoke detection analysis.
    - Reads and preprocesses the data.
    - Tests custom k-NN implementation and compares it with scikit-learn's k-NN.
    - Tweaks scikit-learn's k-NN parameters and evaluates performance.
    - Evaluates Gaussian Naïve Bayes and compares it with k-NN.
    """
    # Read the data (using only 1000 samples for faster processing)
    headers, numeric_data, labels = read_data(sample_size=1000)

    # Normalize the data
    scaler = StandardScaler()
    numeric_data = scaler.fit_transform(numeric_data)

    # Split the data into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(numeric_data, labels, test_size=0.2, random_state=42)

    # Task 1: Test your k-NN implementation
    print("Task 1: Testing k-NN Implementation")
    k_values = [1, 3, 5]
    for k in k_values:
        predictions = []
        for test_point in test_data:
            pred_label = knn_predict(train_data, train_labels, test_point, k)
            predictions.append(pred_label)
        accuracy = accuracy_score(test_labels, predictions)
        print(f"k = {k}, Accuracy: {accuracy:.2f}")

    # Compare with scikit-learn's k-NN implementation
    print("\nComparing with scikit-learn's k-NN:")
    for k in k_values:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(train_data, train_labels)
        predictions = clf.predict(test_data)
        accuracy = accuracy_score(test_labels, predictions)
        print(f"k = {k}, Accuracy: {accuracy:.2f}")

    # Task 2: Tweaking scikit-learn's k-NN
    print("\nTask 2: Tweaking scikit-learn's k-NN")
    parameters = {
        'n_neighbors': [3, 5],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]  # 1 for Manhattan, 2 for Euclidean
    }

    for k in parameters['n_neighbors']:
        for weight in parameters['weights']:
            for p in parameters['p']:
                clf = KNeighborsClassifier(n_neighbors=k, weights=weight, p=p)
                results = cross_validate(clf, numeric_data, labels, scoring=["accuracy", "f1_macro"], cv=5, n_jobs=-1)
                avg_accuracy = results["test_accuracy"].mean()
                avg_f1 = results["test_f1_macro"].mean()
                print(f"k = {k}, weights = {weight}, p = {p}, Average Accuracy: {avg_accuracy:.2f}, Average F1: {avg_f1:.2f}")

    # Task 3: Naïve Bayes
    print("\nTask 3: Naïve Bayes")
    gnb = GaussianNB()
    results = cross_validate(gnb, numeric_data, labels, scoring=["accuracy", "f1_macro"], cv=5, n_jobs=-1)
    avg_accuracy = results["test_accuracy"].mean()
    avg_f1 = results["test_f1_macro"].mean()
    print(f"Naïve Bayes Average Accuracy: {avg_accuracy:.2f}, Average F1: {avg_f1:.2f}")

if __name__ == "__main__":
    main()