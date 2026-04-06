"""DOA estimation MSE vs SNR."""
import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.doa_utils import generate_ula_data, root_music, u_crb, s_crb
from utils.estimates_cov import estimates_cov


def main():
    np.random.seed(0)

    # Array geometry
    theta_rad = np.array([-10, -5, 0, 5, 10]) * np.pi / 180
    m = 15
    M = len(theta_rad)
    power_source = 1.0
    P = power_source * np.eye(M)
    wavelength = 1.0
    d = wavelength / 2

    # Experiment parameters
    all_sig2 = 10.0 ** np.linspace(-2.5, 0, 10)
    MC_runs = 1000
    K = 200

    # Containers
    MSE_SC = np.zeros(len(all_sig2))
    MSE_NML = np.zeros(len(all_sig2))
    MSE_AML = np.zeros(len(all_sig2))
    MSE_DA = np.zeros(len(all_sig2))
    crb_sto = np.zeros(len(all_sig2))
    crb_sto_uc = np.zeros(len(all_sig2))
    snr = np.zeros(len(all_sig2))

    for ii, sig2 in enumerate(all_sig2):
        snr[ii] = 10 * np.log10(power_source / sig2)
        print(f"Simulating SNR = {snr[ii]:.0f} dB")
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
    plt.semilogy(snr, MSE_SC, "-o", label="SC")
    plt.semilogy(snr, MSE_NML, "-s", label="NML")
    plt.semilogy(snr, MSE_AML, "-^", label="AML")
    plt.semilogy(snr, MSE_DA, "-v", label="DA")
    plt.semilogy(snr, crb_sto, "--", label="U-CRB")
    plt.semilogy(snr, crb_sto_uc, "--", label="S-CRB")
    plt.xlabel("SNR (dB)")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(os.path.dirname(__file__), "mse_vs_snr.png"),
                dpi=150, bbox_inches="tight")
    print("Saved mse_vs_snr.png")


if __name__ == "__main__":
    main()
