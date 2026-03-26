# Task 4: Manual Bagging (using multiple kernels)
import numpy as np
from sklearn.metrics import accuracy_score

def bagging_ensemble(models_dict, X_train, y_train, X_test, y_test, n_models=8):
    models = list(models_dict.values())
    predictions = []

    for i in range(n_models):
        # Bootstrap sampling
        indices = np.random.choice(len(X_train), len(X_train), replace=True)
        X_sample = X_train[indices]
        y_sample = y_train[indices]

        # Rotate models (linear, rbf, poly)
        model = models[i % len(models)]

        # Train
        model.fit(X_sample, y_sample)

        # Predict
        pred = model.predict(X_test)
        predictions.append(pred)

    # Majority Voting
    predictions = np.array(predictions)
    final_pred = []

    for i in range(predictions.shape[1]):
        values, counts = np.unique(predictions[:, i], return_counts=True)
        final_pred.append(values[np.argmax(counts)])

    final_pred = np.array(final_pred)

    acc = accuracy_score(y_test, final_pred)
    print(f"{n_models} models trained.")
    print("Bagging Accuracy:", acc)

    return acc