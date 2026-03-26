import numpy as np
import time
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def bagging_ensemble(best_models, X_train, y_train, X_test, y_test, n_models=8):
    results = {}
    
    # Create disjoint subsets of training data
    n_samples = len(X_train)
    indices = np.random.permutation(n_samples)
    subset_size = n_samples // n_models
    
    disjoint_subsets = []
    for i in range(n_models):
        start_idx = i * subset_size
        if i == n_models - 1:
            # Last subset gets remaining samples
            subset_indices = indices[start_idx:]
        else:
            subset_indices = indices[start_idx:start_idx + subset_size]
        disjoint_subsets.append(subset_indices)
    
    # Preprocess data once
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    pca = PCA(n_components=100)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Train ensemble for each kernel separately
    for kernel in ["linear", "rbf", "poly"]:
        print(f"  Bagging {kernel} ({n_models} models)...", end=" ", flush=True)
        model_info = best_models[kernel]
        svc_params = model_info["svc_params"]
        
        predictions = []
        
        start = time.time()
        
        for i in range(n_models):
            subset_idx = disjoint_subsets[i]
            X_subset = X_train_pca[subset_idx]
            y_subset = y_train[subset_idx]
            
            svc = SVC(kernel=kernel, **svc_params)
            svc.fit(X_subset, y_subset)
            
            pred = svc.predict(X_test_pca)
            predictions.append(pred)
        
        total_time = time.time() - start
        
        predictions = np.array(predictions)
        
        # Majority voting
        final_pred = []
        for i in range(predictions.shape[1]):
            vals, counts = np.unique(predictions[:, i], return_counts=True)
            final_pred.append(vals[np.argmax(counts)])
        
        final_pred = np.array(final_pred)
        
        acc = accuracy_score(y_test, final_pred)
        test_error = 1 - acc
        
        results[kernel] = {
            "accuracy": acc,
            "test_error": test_error,
            "train_time": total_time,
            "n_models": n_models,
            "subset_size": subset_size
        }
        print(f"done ({total_time:.1f}s, acc={acc:.3f})", flush=True)
    
    return results
