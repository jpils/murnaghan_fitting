import numpy as np
from scipy.optimize import curve_fit

def bm_pressure_eos(v, v0, b0, b0_prime):
    """
    3rd-order Birch-Murnaghan Equation of State for Pressure.
    """
    eta = (v0 / v) ** (1 / 3)
    term1 = 1.5 * b0 * (eta**7 - eta**5)
    term2 = 1 + 0.75 * (b0_prime - 4) * (eta**2 - 1)
    return term1 * term2


def main():
    data = np.loadtxt("../best_params.csv", skiprows=1, delimiter=',')
    pressures = data[:,0]
    volumes = data[:, 1]
    c_a_ratios = data[:, 2]

    v0_guess = volumes[np.argmin(np.abs(pressures))]
    b0_guess = 1
    b0_prime_guess = 1

    initial_guesses = [v0_guess, b0_guess, b0_prime_guess]

    popt, pcov = curve_fit(bm_pressure_eos, volumes, pressures, p0=initial_guesses)

    perr = np.sqrt(np.diag(pcov))

    fit_p = bm_pressure_eos(volumes, popt[0], popt[1], popt[2])

    print("=== Birch-Murnaghan Fit Results ===")
    print(f"V0  (Equilibrium Volume):     {popt[0]:.4f} ± {perr[0]:.4f}")
    print(f"B0  (Bulk Modulus):          {popt[1]:.4f} ± {perr[1]:.4f}")
    print(f"B0' (Pressure Derivative):    {popt[2]:.4f} ± {perr[2]:.4f}")

    np.savetxt("../fit_pressure.csv", fit_p, delimiter=",")

    return

if __name__ == "__main__":
    main()
