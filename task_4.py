import numpy as np
import time
from sklearn.metrics import accuracy_score
import copy

def bagging_ensemble(best_models, X_train, y_train, X_test, y_test, n_models=8):

    predictions = []
    kernels = list(best_models.keys())

    start = time.time()

    for i in range(n_models):
        idx = np.random.choice(len(X_train), len(X_train), replace=True)

        X_sample = X_train[idx]
        y_sample = y_train[idx]

        model = copy.deepcopy(best_models[kernels[i % len(kernels)]])
        model.fit(X_sample, y_sample)

        pred = model.predict(X_test)
        predictions.append(pred)

    total_time = time.time() - start

    predictions = np.array(predictions)

    final_pred = []
    for i in range(predictions.shape[1]):
        vals, counts = np.unique(predictions[:, i], return_counts=True)
        final_pred.append(vals[np.argmax(counts)])

    final_pred = np.array(final_pred)

    acc = accuracy_score(y_test, final_pred)

    return acc, total_time