# Proyecto proca v.1.0
# Mass and Energy 
# The referred equations can be found at https://arxiv.org/pdf/2412.06901.

import numpy as np

from scipy.integrate import quad, simpson
from scipy.interpolate import interp1d

#######################
## ENERGY'S EINGEVALUE
#######################
def energEng(r, sigtot, V0, gamma=0, kind='quadratic', fill_value="extrapolate"):
    """
    Calculates the energy eigenvalue (En) corresponding to the radial density profiles sigtot

    Parameters:
    - r: array-like, radial positions where `sigtot` is evaluated.
    - sigtot: array-like, density profiles as a function of `r`.
    - V0: float, value of u(r=0).
    - gamma: int parameter, 1 for radial polarization, 0 for the rest (default is 0).
    - kind: string, interpolation method for `sigtot` (default is 'quadratic').

    Returns:
    - En: float, calculated energy after integration.
    
    Note:
    For multifrequency cases:
    sigtot = sigs[0]**2 + sigs[1]**2 + sigs[2]**2
    """
    
    # Ensure r and sigtot are sorted
    if not all(r[i] < r[i+1] for i in range(len(r)-1)):
        raise ValueError("Input array `r` must be strictly increasing.")
    
    # Ensure that gamma=1 or gamma=0
    if gamma not in [0, 1]:
        raise ValueError("gamma value must be strictly 1 (for radial polarization) or 0 (for the rest).")
    
    # Interpolate sigtot
    sigF = interp1d(r, sigtot, kind=kind, fill_value=fill_value)

    # Define the integrand Af(r)
    Af = lambda r: r**(2 * gamma + 1) * sigF(r)
    
    # Integration limits
    rmin, rmax = r[0], r[-1]

    # Perform the numerical integration
    integral_result, error = quad(Af, rmin, rmax)
    
    if error > 1e-5:  # Large error margin, can be adjusted
        print(f"Warning: The integral may not have converged well. Error estimate: {error}")

    # Calculate energy eingevalue (En)
    En = V0 - integral_result  # energía: (2c^2 m)/Lambda  -> Lambda=4pi m^3/Mp^2
    return En

#######################
## MASS VALUE
#######################
def massVal(r, sigtot, gamma=0, fac=4*np.pi, 
            kind='quadratic', fill_value="extrapolate",
            integrationMetho='quad'):
    """
    Calculates the configuration mass based on the radial density profiles sigtot.

    Parameters:
    - r: array-like, radial positions where `sigtot` is evaluated.
    - sigtot: array-like, density profiles as a function of `r`.
    - gamma: int parameter, 1 for radial polarization, 0 for the rest (default is 0).
    - kind: string, interpolation method for `sigtot` (default is 'quadratic').
    - integrationMetho: 'quad', 'simpson', 'trapz'

    Returns:
    - Mas: float, calculated mass after integration.

    Note:
    For multifrequency cases:
    sigtot = sigs[0]**2 + sigs[1]**2 + sigs[2]**2
    """

    # Ensure r and sigtot are sorted
    if not all(r[i] < r[i+1] for i in range(len(r)-1)):
        raise ValueError("Input array `r` must be strictly increasing.")
    
    # Ensure that gamma is either 0 or 1
    if gamma not in [0, 1]:
        raise ValueError("gamma value must be 0 or 1.")
    
    if integrationMetho == 'quad':
        # Interpolate sigtot
        sigF = interp1d(r, sigtot, kind=kind, fill_value=fill_value)

        # Define the integrand Bf(r)
        Bf = lambda r: r**(2 * (gamma + 1)) * sigF(r)
        # Integration limits
        rmin, rmax = r[0], r[-1]
        # Perform the numerical integration
        integral_result, error = quad(Bf, rmin, rmax)

        if error > 1e-5:  # Large error margin, can be adjusted
            print(f"Warning: The integral may not have converged well. Error estimate: {error}")
    elif integrationMetho == 'simpson':
        integral_result = simpson(r**(2 * (gamma + 1))*sigtot, r)
    else:
        integral_result = np.trapz(r**(2 * (gamma + 1))*sigtot, r)
        
    # Calculate the mass (Mas)
    Mas = fac * integral_result  # masa: c*hb/(G*m*Lambda^(1/2))  -> Lambda=4pi m^3/Mp^2
    return Mas


########################################################
# IMPLEMENTATION OF THE ENERGY FUNCTIONAL
# Eqs. (11, 12, 25) # Equations (12a, 12b, 12c)
########################################################
def Tf(datos, gamma, rlim=None):
    # scaled variables
    r, sigma, _ = datos
    r = r.astype('float64')
    sigma = sigma.astype('float64')
    sigma = r**gamma * sigma

    sigmaF = interp1d(r, sigma, kind='quadratic') 
    dsigma = np.gradient(sigma, r)
    dsigmaF = interp1d(r, dsigma, kind='quadratic') 
    
    if rlim:
        rmin, rfin = rlim
    else: 
        rmin, rfin = r[0], r[-1]
    intf = lambda r: r**2 * (dsigmaF(r)**2 + 2 * gamma * sigmaF(r)**2 / r**2)
    Tval = 4 * np.pi * quad(intf, rmin, rfin)[0]
    Tval = Tval/2
    return Tval

def Ff(datos, gamma, rlim=None):  # Fs and Fn
    # scaled variables
    r, sigma, _ = datos
    r = r.astype('float64')
    sigma = sigma.astype('float64')
    sigma = r**gamma * sigma

    sigmaF = interp1d(r, sigma, kind='quadratic')
    
    if rlim:
        rmin, rfin = rlim
    else: 
        rmin, rfin = r[0], r[-1]
        
    intf = lambda r: r**2 * sigmaF(r)**4
    Fval = 4 * np.pi * quad(intf, rmin, rfin)[0]
    Fval = Fval/4.
    return Fval

def EnFuncion(datos, arg, rlim=None):
    LambT, gamma = arg
    Tfval = Tf(datos, gamma, rlim=rlim)
    Ffval = Ff(datos, gamma, rlim=rlim)
    
    Enf = - Tfval - 2 * LambT * Ffval
    return Enf, Tfval, Ffval


def find_nearest(array, value):
    """
    Encontrando el valor más cercano
    """
    n = [abs(i - value) for i in array]
    idx = n.index(min(n))
    #print(idx)
    return (array[idx], idx)