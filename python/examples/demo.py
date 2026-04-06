"""Demo: DOA estimation with NML Toeplitz covariance estimation.

Reproduces the MATLAB demo in matlab/examples/demo.m. Compares MUSIC
performance when using the sample covariance vs. the NML Toeplitz estimate,
and plots MSE against Cramér-Rao bounds.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
from nml import NMLSolver
from doa_utils import generate_ula_data, root_music, u_crb, s_crb


def main():
    # Array geometry
    theta_rad = np.array([-10, -5, 0, 5, 10]) * np.pi / 180
    m = 15                              # number of sensors
    M = len(theta_rad)                  # number of sources
    snr = 5                             # signal-to-noise ratio (dB)
    power_source = 1.0
    P = power_source * np.eye(M)        # source covariance
    sig2 = power_source * 10 ** (-snr / 10)  # noise variance
    wavelength = 1.0
    d = wavelength / 2                  # sensor spacing

    # Experiment parameters
    samples = np.arange(15, 501, 30)
    MC_runs = 2000

    # NML parameters
    tol = 1e-7
    beta = 0.7
    alpha = 0.05
    max_iter = 100

    # Create solver
    n = m - 1
    solver = NMLSolver(n, tol=tol, beta=beta, alpha=alpha, max_iter=max_iter)

    # Containers
    MSE_SC = np.zeros(len(samples))
    MSE_NML = np.zeros(len(samples))
    crb_sto = np.zeros(len(samples))
    crb_sto_uc = np.zeros(len(samples))
    total_solve_time = 0.0
    total_iter = 0

    for ii, K in enumerate(samples):
        print(f"Simulating K = {K}")
        for run in range(MC_runs):
            # Generate data
            Y, _ = generate_ula_data(power_source, sig2, d, m, K,
                                     wavelength, theta_rad)

            # Sample covariance
            sample_cov = (Y @ Y.conj().T) / K

            # NML estimate
            result = solver.solve(Y, verbose=False)
            x, y = result["x"], result["y"]
            first_col = np.concatenate([[2 * x[0]], x[1:] + 1j * y])
            nml_cov = toeplitz(first_col)

            total_solve_time += result["time"]
            total_iter += result["iter"]

            # Root-MUSIC (negate angles to match MATLAB convention)
            doa_nml = np.sort(-root_music(nml_cov, M, d, wavelength))
            doa_sc = np.sort(-root_music(sample_cov, M, d, wavelength))

            # Accumulate MSE
            MSE_SC[ii] += np.sum((theta_rad - doa_sc) ** 2)
            MSE_NML[ii] += np.sum((theta_rad - doa_nml) ** 2)

        MSE_SC[ii] /= M * MC_runs
        MSE_NML[ii] /= M * MC_runs
        crb_sto_uc[ii] = np.mean(np.diag(u_crb(P, theta_rad, sig2, m, d, K)))
        crb_sto[ii] = np.mean(np.diag(s_crb(P, theta_rad, sig2, m, d, K)))

    solver.free()

    avg_iter = total_iter / (MC_runs * len(samples))
    avg_time = total_solve_time / (MC_runs * len(samples))
    print(f"Average solve time (s): {avg_time:.2e}")
    print(f"Average iterations: {avg_iter:.1f}")

    # Plot
    plt.figure()
    plt.semilogy(samples, MSE_SC, '-x', label='MSE_SC')
    plt.semilogy(samples, MSE_NML, '-x', label='MSE_NML')
    plt.semilogy(samples, crb_sto_uc, '--', label='U-CRB')
    plt.semilogy(samples, crb_sto, '--', label='S-CRB')
    plt.xlabel('K')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.savefig('demo.pdf')
    

if __name__ == "__main__":
    main()
