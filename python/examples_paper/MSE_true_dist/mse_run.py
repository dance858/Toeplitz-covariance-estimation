"""MSE experiment: covariance estimation error vs number of samples."""
import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.toeplitz_utils import random_toeplitz_cov
from MSE_true_dist.crb_true_dist import crb_true_dist
from MSE_true_dist.mse_compute import mse_compute


def main():
    np.random.seed(0)

    n = 16
    samples = np.arange(20, 201, 20)
    MC_runs = 5

    true_cov = random_toeplitz_cov(n)

    print(f"Running MSE experiment: n={n}, MC_runs={MC_runs}")
    results = mse_compute(samples, true_cov, MC_runs)

    # Compute CRB
    CRB = np.zeros(len(samples))
    for k, N in enumerate(samples):
        FIM = crb_true_dist(true_cov, N, n)
        CRB[k] = np.real(np.trace(np.linalg.inv(FIM)))

    # Print mean solve times
    print(f"\nMean solve times:")
    print(f"  NML: {results['NML_mean_time']:.2e} s")
    print(f"  AML: {results['AML_mean_time']:.2e} s")
    print(f"  DA:  {results['DA_mean_time']:.2e} s")

    # Plot
    plt.figure()
    plt.semilogy(samples, results["MSE_NML"], "-s", label="NML")
    plt.semilogy(samples, results["MSE_AML"], "-^", label="AML")
    plt.semilogy(samples, results["MSE_DA"], "-v", label="DA")
    plt.semilogy(samples, CRB, "--", label="CRB")
    plt.xlabel("N")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(os.path.dirname(__file__), "mse_true_dist.png"),
                dpi=150, bbox_inches="tight")
    print("Saved mse_true_dist.png")


if __name__ == "__main__":
    main()
