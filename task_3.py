import time
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

    param_grids = {
        "linear": {"C": [0.1, 1, 10]},
        "rbf": {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
        "poly": {"C": [0.1, 1, 10], "gamma": ["scale"], "degree": [2, 3, 4]},
    }

    # ---------------- PCA ----------------
    for n in [50, 100, 200]:
        results["PCA"][n] = {}
        
        # Fit scaler and PCA once for this n_components
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        pca = PCA(n_components=n)
        dim_start = time.time()
        X_train_pca = pca.fit_transform(X_train_scaled)
        dim_time = time.time() - dim_start
        X_test_pca = pca.transform(X_test_scaled)
        
        for kernel in ["linear", "rbf", "poly"]:
            print(f"  Training PCA {n} + {kernel} SVM...", end=" ", flush=True)
            grid_params = {"C": param_grids[kernel]["C"]}
            if kernel == "rbf":
                grid_params["gamma"] = param_grids[kernel]["gamma"]
            elif kernel == "poly":
                grid_params["gamma"] = param_grids[kernel]["gamma"]
                grid_params["degree"] = param_grids[kernel]["degree"]
            
            svc = SVC(kernel=kernel)
            
            svc_start = time.time()
            grid = GridSearchCV(svc, grid_params, cv=3, n_jobs=1)
            grid.fit(X_train_pca, y_train)
            svc_time = time.time() - svc_start
            
            train_acc = grid.score(X_train_pca, y_train)
            test_acc = grid.score(X_test_pca, y_test)
            train_error = 1 - train_acc
            test_error = 1 - test_acc
            
            results["PCA"][n][kernel] = {
                "best_params": grid.best_params_,
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
                best_params_per_kernel[kernel] = grid.best_params_
            
            print(f"done ({svc_time:.1f}s, acc={test_acc:.3f})", flush=True)
        
        print(f"PCA {n} complete!")

    # ---------------- LDA ----------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lda = LDA()
    dim_start = time.time()
    X_train_lda = lda.fit_transform(X_train_scaled, y_train)
    dim_time = time.time() - dim_start
    X_test_lda = lda.transform(X_test_scaled)
    
    n_components_lda = X_train_lda.shape[1]
    
    for kernel in ["linear", "rbf", "poly"]:
        print(f"  Training LDA + {kernel} SVM...", end=" ", flush=True)
        grid_params = {"C": param_grids[kernel]["C"]}
        if kernel == "rbf":
            grid_params["gamma"] = param_grids[kernel]["gamma"]
        elif kernel == "poly":
            grid_params["gamma"] = param_grids[kernel]["gamma"]
            grid_params["degree"] = param_grids[kernel]["degree"]
        
        svc = SVC(kernel=kernel)
        
        svc_start = time.time()
        grid = GridSearchCV(svc, grid_params, cv=3, n_jobs=1)
        grid.fit(X_train_lda, y_train)
        svc_time = time.time() - svc_start
        
        train_acc = grid.score(X_train_lda, y_train)
        test_acc = grid.score(X_test_lda, y_test)
        train_error = 1 - train_acc
        test_error = 1 - test_acc
        
        results["LDA"][kernel] = {
            "n_components": n_components_lda,
            "best_params": grid.best_params_,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "train_error": train_error,
            "test_error": test_error,
            "svc_train_time": svc_time,
            "dim_reduction_time": dim_time,
            "total_train_time": dim_time + svc_time
        }
        print(f"done ({svc_time:.1f}s, acc={test_acc:.3f})", flush=True)
    
    print("LDA complete!")

    # ---------------- Kernel Comparison (PCA 100 only) ----------------
    for kernel in ["linear", "rbf", "poly"]:
        results["kernel_comparison"][kernel] = results["PCA"][100][kernel]

    # Build best models for Task 4 (bagging)
    best_models = {}
    scaler = StandardScaler()
    pca = PCA(n_components=100)
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_pca = pca.fit_transform(X_train_scaled)
    
    for kernel in ["linear", "rbf", "poly"]:
        params = best_params_per_kernel[kernel]
        svc = SVC(kernel=kernel, **params)
        svc.fit(X_train_pca, y_train)
        
        best_models[kernel] = {
            "scaler": scaler,
            "pca": pca,
            "svc_params": params,
            "kernel": kernel
        }

    return results, best_models
