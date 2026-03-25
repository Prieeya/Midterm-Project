# Task 1: Load datasets
from tensorflow.keras.datasets import mnist, fashion_mnist
import numpy as np

# MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Fashion-MNIST
(f_train_images, f_train_labels), (f_test_images, f_test_labels) = fashion_mnist.load_data()


print("Original Shapes:")
print("MNIST Train:", train_images.shape)
print("MNIST Test:", test_images.shape)
print("Fashion MNIST Train:", f_train_images.shape)
print("Fashion MNIST Test:", f_test_images.shape)