# Task 2: Flatten images
from tensorflow.keras.datasets import mnist, fashion_mnist
import numpy as np

# Load datasets
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
(f_train_images, f_train_labels), (f_test_images, f_test_labels) = fashion_mnist.load_data()

# MNIST flatten
X_train = train_images.reshape(train_images.shape[0], -1).astype(np.float64)
X_test  = test_images.reshape(test_images.shape[0], -1).astype(np.float64)

# Fashion-MNIST flatten
Xf_train = f_train_images.reshape(f_train_images.shape[0], -1).astype(np.float64)
Xf_test  = f_test_images.reshape(f_test_images.shape[0], -1).astype(np.float64)


print("After Flattening:")
print("MNIST Train:", X_train.shape)
print("MNIST Test:", X_test.shape)
print("Fashion MNIST Train:", Xf_train.shape)
print("Fashion MNIST Test:", Xf_test.shape)
