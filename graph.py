
"""
This file visualizes the relationship between transaction amount and fraud detection
in the Credit Card Fraud Detection Dataset (2023). It creates a scatter plot to show
how transaction amounts correlate with fraudulent transactions.

Author: Sahil Hirpara
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('creditcard_2023.csv')

# Plot Transaction Amount vs Fraud
plt.figure(figsize=(10, 6))
plt.scatter(df['Amount'], df['Class'], alpha=0.5, c=df['Class'], cmap='coolwarm')
plt.title('Transaction Amount vs Fraud', fontsize=16)
plt.xlabel('Transaction Amount', fontsize=14)
plt.ylabel('Fraud (0 = No, 1 = Yes)', fontsize=14)
plt.colorbar(label='Fraud')
plt.grid(True)
plt.show()