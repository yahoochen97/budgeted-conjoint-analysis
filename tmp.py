import numpy as np
import matplotlib.pyplot as plt

# Sample data (replace with your actual data)
num_samples = np.array([10, 20, 30, 40, 50])  # Sample sizes
accuracy_mean = np.array([0.75, 0.80, 0.85, 0.90, 0.92])  # Mean accuracy
accuracy_std = np.array([0.05, 0.03, 0.02, 0.02, 0.01])  # Standard deviation of accuracy

# Plotting
fig, ax = plt.subplots()
for k in range(4):
    ax.errorbar(4+4*np.arange(len(num_samples))+(k-1)*0.75, accuracy_mean, yerr=accuracy_std, fmt='o', capsize=5)
ax.set_xlabel('Number of Samples')
ax.set_ylabel('Model Accuracy')
ax.set_title('Model Accuracy vs. Number of Samples')

plt.show()