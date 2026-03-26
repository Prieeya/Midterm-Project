# Task 2: Flatten images + Standardization
from task_1 import *
import numpy as np

# Flatten MNIST
X_train = train_images.reshape(len(train_images), -1).astype("float64")
X_test  = test_images.reshape(len(test_images), -1).astype("float64")

# Flatten Fashion MNIST
Xf_train = f_train_images.reshape(len(f_train_images), -1).astype("float64")
Xf_test  = f_test_images.reshape(len(f_test_images), -1).astype("float64")

print("After Flattening:")
print("MNIST Train:", X_train.shape)
print("MNIST Test:", X_test.shape)
print("Fashion MNIST Train:", Xf_train.shape)
print("Fashion MNIST Test:", Xf_test.shape)