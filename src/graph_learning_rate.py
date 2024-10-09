import numpy as np
import matplotlib.pyplot as plt

# Cosine Annealing Learning Rate Scheduler parameters
initial_lr = 0.1  # initial learning rate
min_lr = 0.001  # minimum learning rate
total_epochs = 100  # total epochs

# Cosine annealing function for learning rate scheduling
epochs = np.arange(0, total_epochs)
lr_values = min_lr + (initial_lr - min_lr) * (1 + np.cos(np.pi * epochs / total_epochs)) / 2

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, lr_values, color='blue', marker='o', linestyle='-', label='Learning Rate')
plt.title('Cosine Annealing Learning Rate Scheduler', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Learning Rate', fontsize=12)
plt.grid(True)
plt.legend(loc='upper right')
plt.show()
