from task_1 import *
from task_2 import *

from task_3 import train_models
from task_4 import bagging_ensemble

def run_dataset(X_train, y_train, X_test, y_test, name):
    print(f"\n===== {name} =====\n")

    results, best_models = train_models(X_train, y_train, X_test, y_test)

    print("PCA:", results["PCA"])
    print("LDA:", results["LDA"])

    bagging_ensemble(best_models, X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    # MNIST
    run_dataset(X_train, train_labels, X_test, test_labels, "MNIST")

    # Fashion MNIST
    run_dataset(Xf_train, f_train_labels, Xf_test, f_test_labels, "Fashion MNIST")