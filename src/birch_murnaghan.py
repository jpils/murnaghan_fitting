import numpy as np
from scipy.optimize import curve_fit

def bm_energy_eos(v, e0, v0, b0_raw, b0_prime):
    """
    Outputs energy in eV when b0_raw is in eV/Å³.
    """
    f = ((v0 / v) ** (2 / 3)) - 1
    e_term = (9.0 * v0 * b0_raw / 16.0) * (
        f**3 * b0_prime + f**2 * (6.0 - 4.0 * (v0 / v)**(2 / 3))
    )
    return e0 + e_term


def bm_pressure_eos(v, v0, b0, b0_prime):
    """
    Outputs pressure in same units as b0.
    """
    eta = (v0 / v) ** (1 / 3)
    term1 = 1.5 * b0 * (eta**7 - eta**5)
    term2 = 1 + 0.75 * (b0_prime - 4) * (eta**2 - 1)
    return term1 * term2


def main():
    data = np.loadtxt("../best_params.csv", skiprows=1, delimiter=',')
    volumes = data[:, 1]
    energies_ev = data[:, 3]

    e0_guess = np.min(energies_ev)
    v0_guess = volumes[np.argmin(energies_ev)]
    b0_raw_guess = 1.0  
    b0_prime_guess = 4.0

    initial_guesses = [e0_guess, v0_guess, b0_raw_guess, b0_prime_guess]

    popt, pcov = curve_fit(
        bm_energy_eos, 
        volumes, 
        energies_ev, 
        p0=initial_guesses, 
    )
    
    e0_fit, v0_fit, b0_raw_fit, b0_prime_fit = popt
    perr = np.sqrt(np.diag(pcov))

    b0_kbar = b0_raw_fit * 1602.176634
    
    perr_b0_kbar = perr[2] * 1602.176634

    fit_p_kbar = bm_pressure_eos(volumes, v0_fit, b0_kbar, b0_prime_fit)

    print("=== Birch-Murnaghan Energy Fit Results ===")
    print(f"E0  (Equilibrium Energy):     {e0_fit:.4f} ± {perr[0]:.4f} eV")
    print(f"V0  (Equilibrium Volume):     {v0_fit:.4f} ± {perr[1]:.4f} Å^3")
    print(f"B0  (Bulk Modulus):          {b0_kbar:.4f} ± {perr_b0_kbar:.4f} kbar")
    print(f"B0' (Pressure Derivative):    {b0_prime_fit:.4f} ± {perr[3]:.4f}")

    np.savetxt(
        "../fit_pressure.csv", 
        fit_p_kbar, 
        delimiter=",", 
        header="fit_pressure(kbar)", 
        comments=''
    )
    print("\nFitted pressures saved to '../fit_pressure.csv'")

if __name__ == "__main__":
    main()
