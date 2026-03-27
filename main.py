from task_2 import *
from task_3 import train_models
from task_4 import bagging_ensemble
import numpy as np


def run_dataset(X_train, y_train, X_test, y_test, name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}\n")

    results, best_models = train_models(X_train, y_train, X_test, y_test)

    # ---------------- PCA vs LDA Comparison ----------------
    print("\n--- PCA vs LDA Comparison ---")
    print(f"{'Method':<12} {'Dims':<6} {'Total Train Time (s)':<22} {'Test Error':<12}")
    print("-" * 55)
    
    for n in [50, 100, 200]:
        # Use average across kernels or pick one kernel for comparison
        avg_time = sum(results["PCA"][n][k]["total_train_time"] for k in ["linear", "rbf", "poly"]) / 3
        avg_error = sum(results["PCA"][n][k]["test_error"] for k in ["linear", "rbf", "poly"]) / 3
        print(f"{'PCA':<12} {n:<6} {avg_time:<22.4f} {avg_error:<12.4f}")
    
    lda_dims = results["LDA"]["linear"]["n_components"]
    avg_time = sum(results["LDA"][k]["total_train_time"] for k in ["linear", "rbf", "poly"]) / 3
    avg_error = sum(results["LDA"][k]["test_error"] for k in ["linear", "rbf", "poly"]) / 3
    print(f"{'LDA':<12} {lda_dims:<6} {avg_time:<22.4f} {avg_error:<12.4f}")

    # ---------------- Kernel Comparison (PCA n=100) ----------------
    print("\n--- Kernel Comparison (PCA n=100) ---")
    print(f"{'Kernel':<10} {'Best Params':<40} {'SVC Train Time (s)':<20} {'Test Error':<12}")
    print("-" * 85)
    
    for kernel in ["linear", "rbf", "poly"]:
        r = results["kernel_comparison"][kernel]
        params_str = str(r["best_params"])
        print(f"{kernel:<10} {params_str:<40} {r['svc_train_time']:<20.4f} {r['test_error']:<12.4f}")

    # ---------------- Kernel Comparison for All PCA Dimensions ----------------
    for n in [50, 100, 200]:
        print(f"\n--- Kernel Comparison (PCA n={n}) ---")
        print(f"{'Kernel':<10} {'Best Params':<40} {'SVC Train Time (s)':<20} {'Test Error':<12}")
        print("-" * 85)
        
        for kernel in ["linear", "rbf", "poly"]:
            r = results["PCA"][n][kernel]
            params_str = str(r["best_params"])
            print(f"{kernel:<10} {params_str:<40} {r['svc_train_time']:<20.4f} {r['test_error']:<12.4f}")

    # ---------------- Bagging Results ----------------
    print("\n--- Bagging Ensemble (Task 4) ---")
    bag_results = bagging_ensemble(best_models, X_train, y_train, X_test, y_test)
    
    print(f"\n{'Kernel':<10} {'Single SVC Error':<18} {'Bagging Error':<15} {'Single Time (s)':<16} {'Bagging Time (s)':<16}")
    print("-" * 80)
    
    for kernel in ["linear", "rbf", "poly"]:
        single_error = results["kernel_comparison"][kernel]["test_error"]
        single_time = results["kernel_comparison"][kernel]["svc_train_time"]
        bag_error = bag_results[kernel]["test_error"]
        bag_time = bag_results[kernel]["train_time"]
        print(f"{kernel:<10} {single_error:<18.4f} {bag_error:<15.4f} {single_time:<16.4f} {bag_time:<16.4f}")


if __name__ == "__main__":
    SAMPLE_SIZE = None
    
    if SAMPLE_SIZE:
        print(f"\n*** USING SUBSET OF {SAMPLE_SIZE} SAMPLES FOR FASTER TESTING ***\n")
        np.random.seed(42)
        
        # MNIST subset
        idx_train = np.random.choice(len(X_train), SAMPLE_SIZE, replace=False)
        idx_test = np.random.choice(len(X_test), SAMPLE_SIZE // 6, replace=False)
        X_train_sub = X_train[idx_train]
        y_train_sub = train_labels[idx_train]
        X_test_sub = X_test[idx_test]
        y_test_sub = test_labels[idx_test]
        
        # Fashion-MNIST subset
        idx_f_train = np.random.choice(len(Xf_train), SAMPLE_SIZE, replace=False)
        idx_f_test = np.random.choice(len(Xf_test), SAMPLE_SIZE // 6, replace=False)
        Xf_train_sub = Xf_train[idx_f_train]
        yf_train_sub = f_train_labels[idx_f_train]
        Xf_test_sub = Xf_test[idx_f_test]
        yf_test_sub = f_test_labels[idx_f_test]
        
        run_dataset(X_train_sub, y_train_sub, X_test_sub, y_test_sub, "MNIST")
        run_dataset(Xf_train_sub, yf_train_sub, Xf_test_sub, yf_test_sub, "Fashion-MNIST")
    else:
        run_dataset(X_train, train_labels, X_test, test_labels, "MNIST")
        run_dataset(Xf_train, f_train_labels, Xf_test, f_test_labels, "Fashion-MNIST")
