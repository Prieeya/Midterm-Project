import time
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

def train_models(X_train, y_train, X_test, y_test):

    results = {"PCA": {}, "LDA": {}}
    best_models = {}

    # Correct kernel grids
    param_grid = [
        {"svc__kernel": ["linear"], "svc__C": [0.1, 1, 10]},
        {"svc__kernel": ["rbf"], "svc__C": [0.1, 1, 10], "svc__gamma": ["scale", "auto"]},
        {"svc__kernel": ["poly"], "svc__C": [0.1, 1, 10], "svc__gamma": ["scale"], "svc__degree": [2, 3, 4]},
    ]

    # ---------------- PCA ----------------
    for n in [50, 100, 200]:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n)),
            ("svc", SVC())
        ])

        start = time.time()
        grid = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        train_time = time.time() - start

        train_acc = grid.score(X_train, y_train)
        test_acc = grid.score(X_test, y_test)

        results["PCA"][n] = {
            "best_params": grid.best_params_,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "train_time": train_time
        }

        print(f"PCA {n} done")

    # ---------------- LDA ----------------
    lda_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lda", LDA()),  # reduces to 9 dims automatically
        ("svc", SVC())
    ])

    start = time.time()
    lda_grid = GridSearchCV(lda_pipe, param_grid, cv=3, n_jobs=-1)
    lda_grid.fit(X_train, y_train)
    lda_time = time.time() - start

    results["LDA"] = {
        "best_params": lda_grid.best_params_,
        "train_acc": lda_grid.score(X_train, y_train),
        "test_acc": lda_grid.score(X_test, y_test),
        "train_time": lda_time
    }

    # ---------------- Best Models for Bagging ----------------
    # Use best PCA (100) as required for kernel comparison
    for kernel in ["linear", "rbf", "poly"]:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=100)),
            ("svc", SVC(kernel=kernel))
        ])

        kernel_grid = [p for p in param_grid if p["svc__kernel"][0] == kernel]

        grid = GridSearchCV(pipe, kernel_grid, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)

        best_models[kernel] = grid.best_estimator_

    return results, best_models