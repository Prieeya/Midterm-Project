import time
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


def train_models(X_train, y_train, X_test, y_test):
    results = {
        "PCA": {},
        "LDA": {},
        "kernel_comparison": {}
    }
    best_params_per_kernel = {}

    # Hyperparameter grids for each kernel (prefixed for Pipeline)
    param_grids = {
        "linear": {"svc__C": [0.1, 1, 10]},
        "rbf": {"svc__C": [0.1, 1, 10], "svc__gamma": ["scale", "auto"]},
        "poly": {"svc__C": [0.1, 1, 10], "svc__gamma": ["scale"], "svc__degree": [2, 3, 4]},
    }

    # ---------------- PCA Pipeline ----------------
    for n in [50, 100, 200]:
        results["PCA"][n] = {}
        
        # Measure dimensionality reduction time separately
        scaler_temp = StandardScaler()
        pca_temp = PCA(n_components=n)
        dim_start = time.time()
        X_train_scaled = scaler_temp.fit_transform(X_train)
        X_train_pca = pca_temp.fit_transform(X_train_scaled)
        dim_time = time.time() - dim_start
        X_test_scaled = scaler_temp.transform(X_test)
        X_test_pca = pca_temp.transform(X_test_scaled)
        
        for kernel in ["linear", "rbf", "poly"]:
            print(f"  Training PCA({n}) + {kernel} SVM Pipeline...", end=" ", flush=True)
            
            # Create the Pipeline: Standardization -> PCA -> SVC
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=n)),
                ('svc', SVC(kernel=kernel))
            ])
            
            # GridSearchCV with the pipeline
            grid = GridSearchCV(pipeline, param_grids[kernel], cv=3, n_jobs=1)
            
            start_time = time.time()
            grid.fit(X_train, y_train)
            total_time = time.time() - start_time
            
            # Extract best params (remove prefix for storage)
            best_params_clean = {k.replace('svc__', ''): v for k, v in grid.best_params_.items()}
            
            # Measure SVC-only training time with best params
            svc_start = time.time()
            svc_best = SVC(kernel=kernel, **best_params_clean)
            svc_best.fit(X_train_pca, y_train)
            svc_time = time.time() - svc_start
            
            train_acc = grid.score(X_train, y_train)
            test_acc = grid.score(X_test, y_test)
            train_error = 1 - train_acc
            test_error = 1 - test_acc
            
            results["PCA"][n][kernel] = {
                "best_params": best_params_clean,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "train_error": train_error,
                "test_error": test_error,
                "svc_train_time": svc_time,
                "dim_reduction_time": dim_time,
                "total_train_time": dim_time + svc_time
            }
            
            # Store best params for bagging (use PCA 100 as reference)
            if n == 100:
                best_params_per_kernel[kernel] = best_params_clean
            
            print(f"done ({total_time:.1f}s, acc={test_acc:.3f})", flush=True)
        
        print(f"PCA {n} complete!")

    # ---------------- LDA Pipeline ----------------
    # Measure LDA dimensionality reduction time separately
    scaler_lda = StandardScaler()
    lda_temp = LDA()
    dim_start = time.time()
    X_train_scaled_lda = scaler_lda.fit_transform(X_train)
    X_train_lda = lda_temp.fit_transform(X_train_scaled_lda, y_train)
    dim_time_lda = time.time() - dim_start
    X_test_scaled_lda = scaler_lda.transform(X_test)
    X_test_lda = lda_temp.transform(X_test_scaled_lda)
    
    n_components_lda = X_train_lda.shape[1]
    
    for kernel in ["linear", "rbf", "poly"]:
        print(f"  Training LDA + {kernel} SVM Pipeline...", end=" ", flush=True)
        
        # Create the Pipeline: Standardization -> LDA -> SVC
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('lda', LDA()),
            ('svc', SVC(kernel=kernel))
        ])
        
        # GridSearchCV with the pipeline
        grid = GridSearchCV(pipeline, param_grids[kernel], cv=3, n_jobs=1)
        
        start_time = time.time()
        grid.fit(X_train, y_train)
        total_time = time.time() - start_time
        
        # Extract best params (remove prefix for storage)
        best_params_clean = {k.replace('svc__', ''): v for k, v in grid.best_params_.items()}
        
        # Measure SVC-only training time with best params
        svc_start = time.time()
        svc_best = SVC(kernel=kernel, **best_params_clean)
        svc_best.fit(X_train_lda, y_train)
        svc_time = time.time() - svc_start
        
        train_acc = grid.score(X_train, y_train)
        test_acc = grid.score(X_test, y_test)
        train_error = 1 - train_acc
        test_error = 1 - test_acc
        
        results["LDA"][kernel] = {
            "n_components": n_components_lda,
            "best_params": best_params_clean,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "train_error": train_error,
            "test_error": test_error,
            "svc_train_time": svc_time,
            "dim_reduction_time": dim_time_lda,
            "total_train_time": dim_time_lda + svc_time
        }
        print(f"done ({total_time:.1f}s, acc={test_acc:.3f})", flush=True)
    
    print("LDA complete!")

    # ---------------- Kernel Comparison (PCA 100 only) ----------------
    for kernel in ["linear", "rbf", "poly"]:
        results["kernel_comparison"][kernel] = results["PCA"][100][kernel]

    # Build best models info for Task 4 (bagging)
    best_models = {}
    for kernel in ["linear", "rbf", "poly"]:
        best_models[kernel] = {
            "svc_params": best_params_per_kernel[kernel],
            "kernel": kernel
        }

    return results, best_models
