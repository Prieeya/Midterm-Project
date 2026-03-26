from task_2 import *
from task_3 import train_models
from task_4 import bagging_ensemble

def run_dataset(X_train, y_train, X_test, y_test, name):

    print(f"\n===== {name} =====\n")

    results, best_models = train_models(X_train, y_train, X_test, y_test)

    print("\n--- PCA Results ---")
    for k, v in results["PCA"].items():
        print(f"PCA {k}: {v}")

    print("\n--- LDA Results ---")
    print(results["LDA"])

    print("\n--- Bagging ---")
    bag_acc, bag_time = bagging_ensemble(best_models, X_train, y_train, X_test, y_test)

    print(f"Bagging Accuracy: {bag_acc}")
    print(f"Bagging Time: {bag_time}")

if __name__ == "__main__":
    run_dataset(X_train, train_labels, X_test, test_labels, "MNIST")
    run_dataset(Xf_train, f_train_labels, Xf_test, f_test_labels, "Fashion-MNIST")