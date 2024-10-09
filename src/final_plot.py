import matplotlib.pyplot as plt

# Data
validation_losses =[1.6235685750961304, 1.088385629272461, 0.9385089658737182, 0.9655444165229797, 0.9125016822814941, 0.8232612754821778, 0.8076404512405395, 0.8540171035766602, 0.8900152311325074, 0.7134873148918152, 0.7692599022388458, 0.7960104480266571, 0.7973906769752502, 0.8895880443572998, 0.8286512593269348]

validation_accuracies = [0.4614, 0.6202, 0.6753, 0.6646, 0.6838, 0.7131, 0.7225, 0.7197, 0.7088, 0.7599, 0.7552, 0.7406, 0.75, 0.732, 0.7459]

poison_losses = [1.5095686588287354, 0.6005624294281006, 0.54374187707901, 0.5889435482025146, 0.7354066505432129, 0.3241671061515808, 0.32254784560203553, 0.3783647174835205, 0.3170328221321106, 0.3649318516254425, 0.27528802800178526, 0.5202020573616027, 0.22599874854087829, 0.29504060649871827]
poison_accuracies =  [ 0.637, 0.801, 0.793, 0.769, 0.789, 0.886, 0.887, 0.872, 0.9, 0.883, 0.901, 0.85, 0.932, 0.913]
# Plot
epochs = range(1, 15)

plt.figure(figsize=(10, 6))

# Validation Loss and Accuracy
plt.plot(epochs, validation_losses, label='Validation Loss', color='blue', marker='o')
plt.plot(epochs, validation_accuracies, label='Validation Accuracy', color='green', marker='x')

# Poison Loss and Accuracy
plt.plot(epochs, poison_losses, label='Poison Loss', color='red', marker='o', linestyle='--')
plt.plot(epochs, poison_accuracies, label='Poison Accuracy', color='purple', marker='x', linestyle='--')

# Labels and Title
plt.xlabel('Epochs')
plt.ylabel('Values')
plt.title('Validation and Poison Losses and Accuracies over Epochs with new ResNet18 Model instead of CNN for CIFAR-10 ')

# Legend
plt.legend()

# Display the plot
plt.show()
