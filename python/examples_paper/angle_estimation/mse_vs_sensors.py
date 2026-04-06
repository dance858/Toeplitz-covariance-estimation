"""DOA estimation MSE vs number of sensors."""
import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.doa_utils import generate_ula_data, root_music, u_crb, s_crb
from utils.estimates_cov import estimates_cov


def main():
    np.random.seed(1)

    # Array geometry
    theta_rad = np.array([-10, -5, 0, 5, 10]) * np.pi / 180
    M = len(theta_rad)
    snr = 10
    power_source = 1.0
    P = power_source * np.eye(M)
    sig2 = power_source * 10 ** (-snr / 10)
    wavelength = 1.0
    d = wavelength / 2

    # Experiment parameters
    sensors = np.arange(10, 31)
    MC_runs = 500
    K = 100

    # Containers
    MSE_SC = np.zeros(len(sensors))
    MSE_NML = np.zeros(len(sensors))
    MSE_AML = np.zeros(len(sensors))
    MSE_DA = np.zeros(len(sensors))
    crb_sto = np.zeros(len(sensors))
    crb_sto_uc = np.zeros(len(sensors))

    for ii, m in enumerate(sensors):
        print(f"Simulating m = {m}")
        for run in range(MC_runs):
            Y, _ = generate_ula_data(power_source, sig2, d, m, K,
                                     wavelength, theta_rad)

            sample_cov, da_out, aml_out, nml_out = estimates_cov(Y)

            doa_sc = np.sort(-root_music(sample_cov, M, d, wavelength))
            doa_nml = np.sort(-root_music(nml_out["estimate"], M, d, wavelength))
            doa_da = np.sort(-root_music(da_out["estimate"], M, d, wavelength))
            doa_aml = np.sort(-root_music(aml_out["estimate"], M, d, wavelength))

            MSE_SC[ii] += np.sum((theta_rad - doa_sc) ** 2)
            MSE_NML[ii] += np.sum((theta_rad - doa_nml) ** 2)
            MSE_DA[ii] += np.sum((theta_rad - doa_da) ** 2)
            MSE_AML[ii] += np.sum((theta_rad - doa_aml) ** 2)

        MSE_SC[ii] /= M * MC_runs
        MSE_NML[ii] /= M * MC_runs
        MSE_AML[ii] /= M * MC_runs
        MSE_DA[ii] /= M * MC_runs

        crb_sto[ii] = np.mean(np.diag(u_crb(P, theta_rad, sig2, m, d, K)))
        crb_sto_uc[ii] = np.mean(np.diag(s_crb(P, theta_rad, sig2, m, d, K)))

    # Plot
    plt.figure()
    plt.semilogy(sensors, MSE_SC, "-o", label="SC")
    plt.semilogy(sensors, MSE_NML, "-s", label="NML")
    plt.semilogy(sensors, MSE_AML, "-^", label="AML")
    plt.semilogy(sensors, MSE_DA, "-v", label="DA")
    plt.semilogy(sensors, crb_sto, "--", label="U-CRB")
    plt.semilogy(sensors, crb_sto_uc, "--", label="S-CRB")
    plt.xlabel("m")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(os.path.dirname(__file__), "mse_vs_sensors.png"),
                dpi=150, bbox_inches="tight")
    print("Saved mse_vs_sensors.png")


if __name__ == "__main__":
    main()
