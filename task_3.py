# Task 3: SVC with PCA and LDA
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def train_models(X_train, y_train, X_test, y_test):
    results = {}
    best_models = {}

    param_grid = {
        'svc__kernel': ['linear', 'rbf', 'poly'],
        'svc__C': [0.1, 1],
        'svc__gamma': ['scale'],
        'svc__degree': [2]
    }

    # -------- PCA --------
    pca_results = {}

    for n in [50, 100, 200]:
        pipe = Pipeline([
            ('pca', PCA(n_components=n)),
            ('svc', SVC())
        ])

        grid = GridSearchCV(pipe, param_grid, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)

        train_acc = grid.best_score_
        test_acc = grid.score(X_test, y_test)

        pca_results[n] = (grid.best_params_, train_acc, test_acc)

        print(f"PCA {n} Done")

    results["PCA"] = pca_results

    # -------- LDA --------
    lda_pipe = Pipeline([
        ('lda', LDA()),
        ('svc', SVC())
    ])

    lda_grid = GridSearchCV(lda_pipe, param_grid, cv=3, n_jobs=-1)
    lda_grid.fit(X_train, y_train)

    lda_train = lda_grid.best_score_
    lda_test = lda_grid.score(X_test, y_test)

    results["LDA"] = (lda_grid.best_params_, lda_train, lda_test)

    # -------- Extract best model PER kernel (IMPORTANT FIX) --------
    kernels = ['linear', 'rbf', 'poly']

    for kernel in kernels:
        pipe = Pipeline([
            ('pca', PCA(n_components=100)),  # fixed PCA size for consistency
            ('svc', SVC(kernel=kernel))
        ])

        grid = GridSearchCV(pipe, {
            'svc__C': [0.1, 1],
            'svc__gamma': ['scale'],
            'svc__degree': [2]
        }, cv=3, n_jobs=-1)

        grid.fit(X_train, y_train)
        best_models[kernel] = grid.best_estimator_

    return results, best_models