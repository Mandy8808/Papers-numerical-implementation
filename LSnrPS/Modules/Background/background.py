# Proyecto proca v.1.0
# Background
# The referred equations can be found at https://arxiv.org/pdf/2412.06901.

import os
import sys
import numpy as np
import warnings

from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp, quad

##############
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname("Modules/Plot/plot_conf.py"), ""))

# Add it to sys.path
sys.path.append(parent_dir)

# Now you can import the module
import plot_conf as cplot # Import from the parent folder (modules)

##############
parent_dir = os.path.abspath(os.path.join(os.path.dirname("Modules/EnergyMass/energy_mass.py"), ""))
sys.path.append(parent_dir)
import energy_mass as em


########################
#### System Equations 
########################

def system(r, yV, arg):
    r"""
    System of equations (52) for one scalar field. Note that we used the Ansatz: 
        \sigma = r^\gamma \sigma, with \gamma = 0 (multi-frequency, Lineal, Circular), \gamma = 1 (radial)
    in order to incorporated the radial case.

    Variables:
        yV = [p0, p1, u0, u1]  # System components
        arg = [sumpComp, LambT, gamma]  # System parameters

    Returns:
        [f0, f1, f2, f3]
    """

    # Ensure yV has exactly four elements
    if len(yV) != 4:
        raise ValueError("yV must contain exactly four elements: [p0, p1, u0, u1]")

    p0, p1, u0, u1 = yV
    sumpComp, LambT, gamma = arg

    # Validate parameters
    if sumpComp < 0:
        raise ValueError("sumpComp cannot be negative.")

    # If p0 is too large, return zeros to avoid instability
    if np.abs(p0) > 80:
        return np.array([0, 0, 0, 0])

    # Evaluate the system based on r
    if r > 0:
        f0 = p1
        f1 = LambT*sumpComp*p0*r**(2*gamma) - 2*(1 + gamma)*p1/r - u0*p0
        f2 = u1
        f3 = -r**(2*gamma)*sumpComp - 2*u1/r
    elif r == 0:
        f0 = p1
        f1 = (LambT*sumpComp*p0*r**(2*gamma) - u0*p0)/(2*gamma + 3)
        f2 = u1
        f3 = -r**(2*gamma)*sumpComp/3
    else:
        raise ValueError("r cannot be negative in this context.")

    return [f0, f1, f2, f3]

def systemMultifrequency(r, yV, arg):
    r"""
    Full System of equations (52), with the Ansatz 
    \sigma = r^\gamma \sigma, with \gamma = 0 (multi-frequency, Lineal, Circular), \gamma = 1 (radial)
    in order to incorporated the radial case.

    Parameters:
        r   : float - Radial coordinate
        yV  : array - System components of size [4 * numFields]
        arg : tuple - (numFields, LambT, gamma)

    Returns:
        Array of derivatives for each component
    """

    if len(arg) == 1:
        gamma = 0
        numFields = int(len(yV)/4)
        LambT, = arg
    elif len(arg) == 2:
        numFields = int(len(yV)/4)
        LambT, gamma = arg
    else:
        numFields, LambT, gamma = arg

    # Checking that for multifrequency case, gamma must be zero
    if numFields != 1 and gamma != 0:
        raise ValueError("gamma must be exactly zero for the multifrequency case.")

    # Ensuring yV has the correct number of elements
    if len(yV) != 4 * numFields:
        raise ValueError("yV must have exactly %d elements." % (4 * numFields))

    # Compute the sum of squared components
    sumpComp = np.sum(yV[:2*numFields:2]**2)
    
    # Preallocate result array
    valores = np.zeros(4 * numFields)
    
    # Handling extremely large values to prevent numerical instability
    if sumpComp > 1e6:
        warnings.warn("The density is extremely large and will cause numerical instabilities.")
        # raise ValueError("The density is extremely large and will cause numerical instabilities.")
        return valores

    # Compute derivatives for each component
    arg_Componente = [sumpComp, LambT, gamma]
    for componente in range(0, 2*numFields, 2):
        p0, p1 = yV[componente], yV[componente+1]
        u0, u1 = yV[2*numFields + componente], yV[2*numFields + componente+1]
        yV_Componente = [p0, p1, u0, u1]
        f0, f1, f2, f3 = system(r, yV_Componente, arg_Componente)
        valores[componente], valores[componente+1] = f0, f1
        valores[2*numFields + componente], valores[2*numFields + componente+1] = f2, f3
    
    return valores

def systemMultFreqTot(r, yVT, arg):
    """
    syst -> System of equation with the structure: f(r, yV, arg)
            yV = [x1, x2, ..., xn, dx1c1, dx2c1, ..., dxnc1, ..., dx1cn, dx2cn, ..., dxncn]
            arg: arguments that will be passed to the system

    argf -> arguments that will be passed to the system
    info -> Print some extra info

    yVT -> [p0x, p1x, p0y, p1y, p0z, p1z, u0x, u1x, u0y, u1y, u0z, u1z,\
           Duxp0x, Duxp1x, Duxp0y, Duxp1y, Duxp0z, Duxp1z, Duxu0x, Duxu1x, Duxu0y, Duxu1y, Duxu0z, Duxu1z,\
           Duyp0x, Duyp1x, Duyp0y, Duyp1y, Duyp0z, Duyp1z, Duyu0x, Duyu1x, Duyu0y, Duyu1y, Duyu0z, Duyu1z,\
           Duzp0x, Duzp1x, Duzp0y, Duzp1y, Duzp0z, Duzp1z, Duzu0x, Duzu1x, Duzu0y, Duzu1y, Duzu0z, Duzu1z]
    """

    yV = yVT[:12]
    f0to11 = systemMultifrequency(r, yV, arg)   # [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11]
    
    # Derivadas con respecto a u_i
    Mat = MatrizDXDu(r, yV, arg, info=False)
    yVDux = yVT[12:24]
    yVDuy = yVT[24:36]
    yVDuz = yVT[36:]
    
    f12tof23 = Mat@yVDux  # Dux [f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23]
    f24to35 = Mat@yVDuy  # Duy [f24, f25, f26, f27, f28, f29, f30, f31, f32, f33, f34, f35]
    f36to47 = Mat@yVDuz  # Duz  [f36, f37, f38, f39, f40, f41, f42, f43, f44, f45, f46, f47]
    
    yVOut = np.concatenate((f0to11, f12tof23, f24to35, f36to47))

    return yVOut

def MatrizDXDu(r, yV, arg, info=False):
    r"""
    Matriz creada a partir de computar d/dz(\partial X/\partial c_i)
    ver 2208.13221v1.pdf
    """
    #numFields, LambT, gamma = arg
    LambT, = arg
    p0x, p1x, p0y, p1y, p0z, p1z, u0x, u1x, u0y, u1y, u0z, u1z = yV

    sumpi = p0x**2 + p0y**2 + p0z**2

    # llenando matriz
    if r==0:
        Mat = np.zeros((12, 12))
    else:
        Mat = np.diag([0, -2/r]*6)  # lleno la diagonal
    
    diag1S = np.diag([1, 0]*6, k=1)[:-1,:-1]
    Mat += diag1S
    
    temp1 = np.array([-p0x, -p0y, -p0z])
    Mat[7::2, :5:2] = 2*temp1
    Mat[1, 6] = temp1[0]
    Mat[3, 8] = temp1[1]
    Mat[5, 10] = temp1[2]

    t1, t2, t3 = 2*LambT*p0y*p0x, 2*LambT*p0z*p0x, 2*LambT*p0z*p0y
    temp2 = [
        [LambT*(sumpi+2*p0x**2)-u0x, t1, t2],  
        [t1, LambT*(sumpi+2*p0y**2)-u0y, t3],
        [t2, t3, LambT*(sumpi+2*p0z**2)-u0z]]
    Mat[1:6:2, :5:2] = temp2

    if r==0:
        #np.fill_diagonal(Mat, 0)
        Mat = Mat/3

    if info:
        print(Mat)

    return Mat

#############################################################
### COMBINING THE NUMERICAL SOLUTION WITH THE ASYMPTOTIC ONE
#############################################################
def mainExt(datos, mult=False, rmin=0, Nptos=2000, fac=1, info=False,
            met='DOP853', metExp=1, Rtol=1e-09, Atol=1e-10, deltaExt=100, indF=-160,
            xscale='linear', yscale='linear', sigfilt=1e-28):
    """ 
    Function to numerically solve and extend profiles.
    
    IMPORTANT: fac=4*np.pi give the "True" mass, but for the extention we need to used fac=1
    """
    
    # Solving numerically using datos
    En, Mas, rD, perfSig, perfdSig, perfU, perfdU, _, gamma = profilesFromSolut(
        datos, mult=mult, rmin=rmin, Nptos=Nptos, met=met, Rtol=Rtol, Atol=Atol, fac=fac, info=info)
    
    numFields = len(perfSig)
        
    # filtering data
    sigtot = sum([perfSig[i]**2 for i in range(numFields)])
    ind = sigtot/sigtot[0] > sigfilt
    rD = rD[ind]
    perfSig = [Sig[ind] for Sig in perfSig]
    perfdSig = [dSig[ind] for dSig in perfdSig]
    perfU = [fU[ind] for fU in perfU]
    perfdU = [dfU[ind] for dfU in perfdU]
    
    # Making a datosOrg matrix array
    datosOrg = np.empty((numFields, 6), dtype=object)  # Using object dtype to hold lists/arrays
    
    temp = [En, Mas, perfSig, perfdSig, perfU, perfdU]
    for i in range(6):
        datosOrg[:, i] = temp[i]
    
    # Extending profiles
    dataEx = []
    for i in range(numFields):
        Ei, Mi, sD, dsD, uD, duD = datosOrg[i]
        
        if np.allclose(sD, 0):  # Check if sD is all zeros
            continue  # Avoid extending a null profile
        
        Ext = (sD[0] * rD[-1]) + deltaExt
        Np = int(Ext / 2)
        
        rDnew, sDnew, dsDnew, uDnew, duDnew = extend(
            gamma=gamma, rD=rD, sD=sD, dsD=dsD, uD=uD, duD=duD,
            Ext=Ext, En=Ei, Mas=Mi, met=metExp, Np=Np, inf=info, indF=indF
        )
        
        if info:
            # Plotting profiles for debugging/visualization
            perf1 = [rDnew, sDnew, dsDnew, uDnew, duDnew]
            perf2 = [rD, sD, dsD, uD, duD]
            cplot.plotUsingDiscSol(perf1, perf2=perf2, xscale=xscale, yscale=yscale)
        
        dataEx.append([Ei, rDnew, sDnew, dsDnew, uDnew, duDnew])
    
    return dataEx

def extend(gamma, rD, sD, dsD, uD, duD, Ext, En=None, Mas=None,
           met=1, Np=1000, inf=False, fac=1, indF=-160):
    """ 
    Extend radial data by an additional segment using one of two methods.
    
    Parameters:
    - rD: Radial data (initial).
    - sD: Profile data for the initial segment.
    - dsD: Derivative of profile data for the initial segment.
    - uD: Another data set to extend.
    - duD: Derivative of uD.
    - Ext: Length to extend the radial data.
    - En: Energy, required only for `met == 2` (optional).
    - Mas: Mass, required only for `met == 2` (optional).
    - met: Method selector (1 for Met1, 2 for Met2).
    - Np: Number of points for the extension.
    - inf: Information flag to control verbosity (for Met2).
    
    Returns:
    - rDnew: Extended radial data.
    - sDnew: Extended profile data.
    - dsDnew: Extended derivative of profile data.
    - uDnew: Extended data.
    - duDnew: Extended derivative of uD.
    
    IMPORTANT: fac=4*np.pi give the "True" mass, but for the extention we need to used fac=1
    """
    
    # Ensure `En` and `Mas` are provided when using method 2
    if En is None or Mas is None:
        warnings.warn("En and Mas are not provided. We computed them from the field profile.\n IMPORTANT: for multifrequency, this procedure is not correct.")
        
        if gamma not in [0, 1]:
            raise ValueError("Invalid value for `gamma`. It must be 0 or 1.")
        
        sigtot = sD**2
        V0 = uD[0]
        
        # Compute `En` and `Mas`
        En = em.energEng(rD, sigtot, V0, gamma=gamma, kind='quadratic', fill_value="extrapolate")
        Mas = em.massVal(rD, sigtot, gamma=gamma, fac=fac, kind='quadratic', fill_value="extrapolate")
    
    # Select method based on `met`
    if met == 2:
        rDnew, sDnew, dsDnew, uDnew, duDnew = extend_Met2(En, Mas, rD[:indF], sD[:indF], dsD[:indF], uD[:indF], duD[:indF], Ext,
                                                           Np=Np, info=inf)
    else:
        rDnew, sDnew, dsDnew, uDnew, duDnew = extend_Met1(En, Mas, rD[:indF], sD[:indF], dsD[:indF], uD[:indF], duD[:indF], Ext,
                                                           Np=Np)
    
    return rDnew, sDnew, dsDnew, uDnew, duDnew


def sParam(r, S, B=0):
    """
    Computes parameters C, k, and s based on the last two elements of r and S.
    
    Parameters:
        r (array-like): A sequence of radial values.
        S (array-like): A sequence of corresponding function values.
        B (float): Mass of the configuration. When B=0 the result correspond to met=1, else met=2

    Returns:
        tuple: (C, k, s), where
            - C is a scaling constant,
            - k is a decay/growth rate
    """
    # Extracting the last two elements
    r1, r2 = r[-2], r[-1]
    yr1, yr2 = S[-2], S[-1]

    # Avoid division by zero
    if yr2 == 0 or r2 == 0:
        raise ValueError("Division by zero encountered in logarithm computation.")

    # Compute k and C
    ratio = yr1 * r1 / (yr2 * r2)
    
    if ratio <= 0:
        #raise ValueError("Logarithm of a non-positive number encountered.")
        print(ValueError("Logarithm of a non-positive number encountered."))

    k = np.real(np.log(np.abs(ratio)))
    
    # Notice that if B=0 the result correspond to the met=1 else met=2
    s = np.exp(-k * r1)/r1**(1 - B/(2 * k))
    C = yr1/s  # C = yr1/s
    
    return C, k

def Asy_uProf(r, A, B):
    """
    Computes the asymptotic potential profile: U = A + B/r and its derivative dy.

    Parameters:
        r (float or np.ndarray): Radial coordinate (must be nonzero).
        A (float): Constant term (eingenvalue of the energy).
        B (float): Mass of the configuration.

    Returns:
        tuple: (y, dy), where
            - y  = A + B/r
            - dy = derivative of y with respect to r
    """
    # Avoid division by zero
    if np.any(r == 0):
        raise ValueError("r must be nonzero to avoid division by zero.")

    # Compute y and dy
    y = A + B/r
    dy = -B/r**2

    return y, dy

def Asy_sProf(r, C, k):
    """
    Computes the asymptotic field profile: sigma = C*exp(-k*r)/r and its derivative dy.

    Parameters:
        r (float or np.ndarray): Radial coordinate (must be nonzero).
        C (float): Scaling constant.
        k (float): Decay/growth rate.

    Returns:
        tuple: (y, dy), where
            - y  = C * exp(-k*r) / r
            - dy = derivative of y with respect to r
    """
    # Avoid division by zero
    if np.any(r == 0):
        raise ValueError("r must be nonzero to avoid division by zero.")

    # Compute y and dy
    exp_term = np.exp(-k * r)
    y = C * exp_term / r
    dy = -(C * exp_term * (1 + k * r)) / r**2

    return y, dy

def Asy_sProf_v2(r, C, En, M):
    """
    Computes a modified decaying exponential function with an additional power-law factor.

    Parameters:
    - r: float or np.array, radial coordinate (must be > 0 to avoid division errors)
    - C: float, amplitude constant
    - En: float, energy parameter (must be <= 0 to avoid sqrt of negative number)
    - M: float, mass parameter

    Returns:
    - y: computed function value
    - dy: first derivative
    """
    if En > 0:
        raise ValueError("En must be non-positive (≤ 0) to avoid complex values.")

    if np.any(r <= 0):
        raise ValueError("r must be strictly positive to prevent division errors.")

    # Compute y and dy
    k = np.sqrt(-En)
    exp_factor = np.exp(-k * r)
    power_factor = np.power(r, 1 - M/(2 * k))
    power_factor_derivada = np.power(r, -2 + M/(2 * k))

    y = C * exp_factor / power_factor
    dy = C * exp_factor * power_factor_derivada * (M - 2*k*(1 + k*r))/(2*k)
    return y, dy

def extend_Met1(En, Mas, rD, sD, dsD, uD, duD, Ext,
                Np=1000):
    """ 
    Extends the radial profile of a field using asymptotic approximations.

    Parameters:
        En (float): Energy parameter.
        Mas (float): Mass parameter.
        rD (np.ndarray): Array of radial points.
        sD (np.ndarray): Scalar field values at rD.
        dsD (np.ndarray): Derivative of scalar field at rD.
        uD (np.ndarray): Function U values at rD.
        duD (np.ndarray): Derivative of U at rD.
        Ext (float): Extension length for the radial coordinate.
        Np (int, optional): Number of new points for extension (default: 1000).

    Returns:
        tuple: Extended arrays (rDnew, sDnew, dsDnew, uDnew, duDnew).
    """
    
    # Validate input
    if len(rD) == 0 or np.isnan(rD[-1]):
        raise ValueError("rD must be a non-empty array with valid numerical values.")
    
    # Generate extended radial points
    rad = np.linspace(rD[-1], rD[-1] + Ext, Np)

    # Compute asymptotic parameters
    Ap, k = sParam(rD, sD)

    # Compute asymptotic profiles
    sExt, dsExt = Asy_sProf(rad, Ap, k)
    uExt, duExt = Asy_uProf(rad, En, Mas)

    # Concatenate with existing data
    rDnew = np.concatenate((rD[:-1], rad))
    sDnew = np.concatenate((sD[:-1], sExt))
    dsDnew = np.concatenate((dsD[:-1], dsExt))
    uDnew = np.concatenate((uD[:-1], uExt))
    duDnew = np.concatenate((duD[:-1], duExt))
    
    return rDnew, sDnew, dsDnew, uDnew, duDnew

def extend_Met2(En, Mas, rD, sD, dsD, uD, duD, Ext,
                Np=1000, info=False):
    """ 
    Extends the radial profile of a field using asymptotic approximations.

    Parameters:
        En (float): Energy parameter.
        Mas (float): Mass parameter.
        rD (np.ndarray): Array of radial points.
        sD (np.ndarray): Scalar field values at rD.
        dsD (np.ndarray): Derivative of scalar field at rD.
        uD (np.ndarray): Function U values at rD.
        duD (np.ndarray): Derivative of U at rD.
        Ext (float): Extension length for the radial coordinate.
        Np (int, optional): Number of new points for extension (default: 1000).
        info: Print the energy compute by two via

    Returns:
        tuple: Extended arrays (rDnew, sDnew, dsDnew, uDnew, duDnew).
    """
    
    # Validate input
    if len(rD) == 0 or np.isnan(rD[-1]):
        raise ValueError("rD must be a non-empty array with valid numerical values.")
    
    # Generate extended radial points
    rad = np.linspace(rD[-1], rD[-1] + Ext, Np)

    # Compute asymptotic parameters
    Ap, k = sParam(rD, sD, B=Mas)
    if info:
        print('En = ', En, ' from k => En =', -k**2)

    # Compute asymptotic profiles
    sExt, dsExt = Asy_sProf_v2(rad, Ap, En, Mas)
    uExt, duExt = Asy_uProf(rad, En, Mas)

    # Concatenate with existing data
    rDnew = np.concatenate((rD[:-1], rad))
    sDnew = np.concatenate((sD[:-1], sExt))
    dsDnew = np.concatenate((dsD[:-1], dsExt))
    uDnew = np.concatenate((uD[:-1], uExt))
    duDnew = np.concatenate((duD[:-1], duExt))
    
    return rDnew, sDnew, dsDnew, uDnew, duDnew

##############################################
### COMPUTANDO PERFILES FROM A DATA SOLUTION
##############################################
def profilesFromSolut(datos, mult=False, rmin=0, fac=4*np.pi, Nptos=2000, info=False,
                      met='DOP853', Rtol=1e-09, Atol=1e-10):  # 'RK45'
    """
    Generates discrete profiles from a numerical solution.
    
    Parameters:
    -----------
    datos : tuple
        Solution data, with different structures depending on the `mult` value.
    mult : bool, optional
        Indicates whether it involves multiple fields (default `False`).
    rmin : float, optional
        Minimum integration radius (default `0`).
    Nptos : int, optional
        Number of points in the discretization (default `2000`).
    info : bool, optional
        If `True`, prints the mass and energy values.

    Returns:
    --------
    dict with the following keys:
        - "energy": Computed energy.
        - "mass": Computed mass.
        - "profiles_f": Profiles of `f`.
        - "profiles_df": Derivatives of `f`.
        - "profiles_u": Profiles of `u`.
        - "profiles_du": Derivatives of `u`.
        - "LambT": LambdaT parameter.
        - "gamma": Gamma parameter.
    """

    # Extracting data
    if mult:
        LambT, nodes, f0 = datos[-1], datos[-2], datos[-3]
        numFields = len(nodes)
        u0, rmax = datos[:numFields], datos[numFields]
        gamma = 0
    else:
        f0, rmax, gamma, LambT, nodes, _, met, Rtol, Atol, u0 = datos
        numFields = 1
    
    # Initial conditions setup
    V0 = np.zeros(4 * numFields)
    V0[:2 * numFields:2] = f0
    V0[2 * numFields::2] = u0

    rspan = np.linspace(rmin, rmax, Nptos)
    arg = [numFields, LambT, gamma]

    sol = solve_ivp(systemMultifrequency, [rmin, rmax], V0, t_eval=rspan,
                     args=[arg], method=met, rtol=Rtol, atol=Atol)

    # Extracting profiles
    perfSig = [sol.y[i] for i in range(0, 2 * numFields, 2)]
    perfdSig = [sol.y[i] for i in range(1, 2 * numFields, 2)]
    perfU = [sol.y[i] for i in range(2 * numFields, 4 * numFields, 2)]
    perfdU = [sol.y[i] for i in range(2 * numFields + 1, 4 * numFields, 2)]

    # Computing energy and mass
    rD = sol.t
    sigtot = sum([perfSig[i]**2 for i in range(numFields)])
    energy, mass = [], []
    for V0, sig in zip(V0[2 * numFields::2], perfSig):
        En = em.energEng(rD, sigtot, V0, gamma=gamma, kind='quadratic', fill_value="extrapolate")
        Mas = em.massVal(rD, sigtot, gamma=gamma, fac=fac, kind='quadratic', fill_value="extrapolate")
        energy.append(En)
        mass.append(Mas)

    if info:
        print(f"Mass values: {mass}")
        print(f"Eigenvalues of energy: {energy}")

    return energy, mass, rD, perfSig, perfdSig, perfU, perfdU, LambT, gamma

##########################
## SHOOTING METHODOLOGIES
##########################

def shoot(imin: float, imax: float) -> float:
    """
    Compute the midpoint between imin and imax.

    Parameters:
    - imin (float): Lower bound.
    - imax (float): Upper bound.

    Returns:
    - float: The midpoint between imin and imax.
    """
    if imin > imax:
        raise ValueError("imin should be less than or equal to imax")
    
    return (imin + imax)/2

def freq_shoot(events, nodo, u0, iInterv, rTemp, inv=False):
    """
    Adjusts U bounds based on event crossings.
    
    Parameters:
    - events: list of arrays, where events[0] and events[1] represent different event crossings.
    - nodo: int, number of expected crossings.
    - u0: float, current u guess.
    - iInterv: tuple (imin, imax), the interval of frequency adjustment.
    - rTemp: float, last recorded event.
    - inv: bool, flag to determine inversion logic.

    Returns:
    updated the u interval and last event crossing.
    -> [umin, umax], rTemp
    """
    imin, imax = iInterv
    events0, events1 = events  # Unpacking for clarity
    i0 = u0  # Extract first value if it's in a list/array

    if events0.size == nodo and events1.size == nodo + 1:
        return [imin, imax], rTemp
    else:
        # Determine whether to increase or decrease the frequency bounds
        target_event = events0 if events0.size > nodo else events1
        
    if inv:
        imin, imax = (i0, imax) if target_event is events0 else (imin, i0)
    else:
        imin, imax = (imin, i0) if target_event is events0 else (i0, imax)

    rTemp = target_event[-1]  # Last crossing event update

    return [imin, imax], rTemp


def identify(events, nodos, p0Data):
    """
    Identifies which profile's U bound should be modified based on event crossings.

    Parameters:
    - events: list of NumPy arrays, where each index corresponds to a detected event.
    - nodos: list of expected number of crossings for each component.
    - p0Data: list of bools indicating which components are considered in the analysis.

    Returns:
    - sigModif: list of bools indicating which signals should be modified.
    """
    numFields = len(nodos)
    name = [str(i) for i in range(0, 2 * numFields, 2)]
    dicNod = dict(zip(name, nodos))

    # Identify components that are not zeros
    indices = np.array(list(map(int, dicNod.keys())), dtype=int)
    ind = np.array(p0Data, dtype=bool)  # Ensuring a boolean mask
    posit = indices[ind]
  
    valR = [np.inf] * numFields  # Initialize with infinity
    for i in posit:
        valtemp = []
        nodo = dicNod[str(i)]
                
        numNod = len(events[i])  # Size of current event
        numdSig = len(events[i+1]) if i+1 < len(events) else 0
        
        if (numNod == nodo) and (numdSig == nodo + 1):
            valtemp.append(0)
        elif numNod == nodo:
            if numdSig < nodo + 1:
                valtemp.append(events[i+1][nodo-1] if numdSig > 0 else np.inf)
            elif numdSig > nodo + 1:
                valtemp.append(events[i+1][nodo])
        elif numNod > nodo:
            valtemp.append(events[i][nodo])
        else:
            valtemp.append(events[i][-1] if numNod != 0 else 0)

        # Dynamically assign to valR based on index position
        valR[i // 2] = min(valtemp)

    # Determine the signal to modify
    sigModif = [False] * numFields
    test = np.min(valR)
    for i in range(numFields):
        if valR[i] == test:
            sigModif[i] = True
            break

    return sigModif

########################################################################
def freq_shoot2(events, nodo, i0, iInterv, rTemp):
    """
    """
    imin, imax = iInterv[0]
    events = events[0]
    i0 = i0[0]

    if events[0].size == nodo and events[1].size == nodo+1:
        return [imin, imax], rTemp
    elif events[1].size > nodo+1:
        if events[0].size > nodo:  # dos veces por nodo
            imax = i0
            rTemp = events[0][-1]
        else:  # si pasa por cero más veces que 2*nodos se aumenta la w, sino se disminuye
            imin = i0
            rTemp = events[1][-1]
    elif events[1].size <= nodo+1:
        if events[0].size > nodo:  # dos veces por nodo
            imax = i0
            rTemp = events[0][-1]
        else:
            imin = i0
            rTemp = events[1][-1]
    return [imin, imax], rTemp

def identify2(events, nodos, p0Data, info=False):
    """
    """
    dicNod = {'0': nodos[0], '2': nodos[1], '4': nodos[2]}

    # identificando que componentes no son ceros
    # cuando una componente se tomó como cero y se excluye del análisis
    indices = np.fromiter(map(int, dicNod.keys()), dtype=int)
    ind = list(map(bool, p0Data))
    posit = indices[ind]

    valR = [np.infty, np.infty, np.infty]
    for i in posit:
        #if info:
        #    print(events[i], events[i+1])
        
        valtemp = []
        nodo = dicNod[str(i)]
        numNod = events[i].size; numdSig = events[i+1].size
        if numNod == nodo and numdSig == nodo+1:
            valtemp.append(0)
        elif numNod == nodo:
            if numdSig < nodo+1:
                valtemp.append(events[i+1][nodo-1])
            elif numdSig > nodo+1:
                valtemp.append(events[i+1][nodo])
        elif numNod > nodo:
            valtemp.append(events[i][nodo])
        else:
            if numNod != 0:
                valtemp.append(events[i][-1])
            else:
                valtemp.append(0)

        if i==0:
            valR[0] = min(valtemp)
        elif i==2:
            valR[1] = min(valtemp)
        elif i==4:
            valR[2] = min(valtemp)
    
    # print(valR)
    sigModif = [False, False, False]
    test = np.min(valR)
    for i in range(3):
        if valR[i]==test:
            sigModif[i]=True
            break

    return sigModif

def MultFreq_solveG2_vO(Ini0, Uintrs, rmax, rmin=0, LambT=1, nodos=[0, 0, 0], 
                     met='RK45', Rtol=1e-09, Atol=1e-10, lim=1e-6, info=False,
                     klim=500, outval=13, delta=0.4): #'DOP853''LSODA'
    """
    In:
    Uintx -> [Umin, Umax]
    Uinty -> [Umin, Umax]
    Uintz -> [Umin, Umax]
    Ini0 -> [p0x, p1x, p0y, p1y, p0z, p1z, u1x, u1y, u1z]
    rmax, rmin -> 
    LambT -> +1 Repulsive case, -1 Atractive case
    nodos -> [nodos_p0x, nodos_p0y, nodos_p0z]

    Orden de las variables
    [phix, phix', phiy, phiy', phiz, phiz', ux, ux', uy, uy', uz, uz'] -> [p0x, p1x, p0y, p1y, p0z, p1z, u0x, u1x, u0y, u1y, u0z, u1z]
    """

    nodos = np.array(nodos)
    p0x, p1x, p0y, p1y, p0z, p1z, u1x, u1y, u1z = Ini0

    p0Data = [p0x, p0y, p0z]
    Uintx, Uinty, Uintz = Uintrs
    Uminx, Umaxx = Uintx
    Uminy, Umaxy = Uinty
    Uminz, Umaxz = Uintz

    print('Finding a profile with nx, ny, nz', nodos, 'nodes')

    # Events
    def Sigx(r, U, arg): return U[0]
    def dSigx(r, U, arg): return U[1]
    def Sigy(r, U, arg): return U[2]
    def dSigy(r, U, arg): return U[3]
    def Sigz(r, U, arg): return U[4]
    def dSigz(r, U, arg): return U[5]
    Sigx.direction = 0; dSigx.direction = 0
    Sigy.direction = 0; dSigy.direction = 0
    Sigz.direction = 0; dSigz.direction = 0

    # ordenando de mayor a menor para la iteracion
    k = 0
    
    arg = [LambT]
    Uintrs = np.array([[Uminx, Umaxx], [Uminy, Umaxy], [Uminz, Umaxz]])
    sigModifold = None
    UintrOrig = np.copy(Uintrs)
    out = 0
    while True:
        u0 = np.array([shoot(*i) for i in Uintrs])
        V0 = [p0x, p1x, p0y, p1y, p0z, p1z, u0[0], u1x, u0[1], u1y, u0[2], u1z]
        
        sol = solve_ivp(systemMultifrequency, [rmin, rmax], V0, events=(Sigx, dSigx, Sigy, dSigy, Sigz, dSigz),
                         args=(arg,), method=met,  rtol=Rtol, atol=Atol)
  
        eventos = np.array([[sol.t_events[0], sol.t_events[1]],
                   [sol.t_events[2], sol.t_events[3]],
                   [sol.t_events[4], sol.t_events[5]]], dtype=object)
        sigModif = identify2(sol.t_events, nodos, p0Data, info=info)

        if info:
            print(sigModif)

        iInterv, rTemp = freq_shoot2(eventos[sigModif], nodos[sigModif], u0[sigModif], Uintrs[sigModif], rmax)
        
        if abs((iInterv[1]-iInterv[0])/2) <= lim:
            if info:
                print(out)
                print('Maxima precisión alcanzada: U0x = ', V0[6], ' U0y = ', V0[8], ' U0z = ', V0[10], 'radio', rTemp)
            
            if out==outval:
                print('Maxima precisión alcanzada: U0x = ', V0[6], ' U0y = ', V0[8], ' U0z = ', V0[10], 'radio', rTemp)
                u0 = [V0[6], V0[8], V0[10]]
                return u0, rTemp, sol.t_events[::2]
            else:
                #Uintrs = np.copy(UintrOrig) # reinicio los que  ya no son iguales
                Uintrs[sigModif] = [iInterv[0]-delta, iInterv[1]+delta]
                out += 1
        else:
            Uintrs[sigModif] = iInterv

        if info:
            print(Uintrs)

        if np.all(np.array([shoot(*i) for i in Uintrs])==u0):
            print('Found: U0x = ', V0[6], ' U0y = ', V0[8], ' U0z = ', V0[10], 'radio', rTemp)
            u0 = [V0[6], V0[8], V0[10]]
            return u0, rTemp, sol.t_events[::2]
        
        if k==klim:
            print('loop limit reached')
            break
            
        k += 1
########################################################################


def MultFreq_solveG2(Ini0, numFields, Uintrs, rmax, rmin=0, LambT=0, gamma=0, nodos=None,
                     met='RK45', Rtol=1e-09, Atol=1e-10, lim=1e-14, info=False,
                     klim=500, outval=13, delta=0.4, part=None, inv=False):
    """
    Multi-frequency solver using solve_ivp.
    """
    
    # Ensuring gamma is zero for multifrequency case
    if numFields != 1 and gamma != 0:
        raise ValueError("gamma must be exactly zero for the multifrequency case.")
    
    # Validating initial conditions
    if len(Ini0) != 3 * numFields:
        raise ValueError(f"Ini0 must have exactly {3 * numFields} elements.")
    
    if len(Uintrs) != numFields:
        raise ValueError(f"Uintrs must have exactly {numFields} intervals.")
    
    if sum([len(Uintrs[i]) == 2 for i in range(numFields)]) != numFields:
        raise ValueError("Each Uintrs component must have exactly 2 extremal values.")
        
    
    Uintrs = np.array(Uintrs, dtype=object)  # Ensure it's modifiable
    arg = [numFields, LambT, gamma]
    
    # Setting node values
    if nodos is None:
        nodos = np.zeros(numFields, dtype=np.int8)
    else:
        if len(nodos) != numFields:
            raise ValueError(f"The node number must have exactly {numFields} elements.")
        nodos = np.array(nodos, dtype=np.int8)
    
    print('Finding a profile with nodes:', nodos)
    
    # Define event functions based on numFields
    def Sigx(r, U, arg): return U[0]
    def dSigx(r, U, arg): return U[1]
    
    Sigx.terminal = False
    dSigx.terminal = False
    
    FindEvents = [Sigx, dSigx]

    if numFields == 2:
        def Sigy(r, U, arg): return U[2]
        def dSigy(r, U, arg): return U[3]
        
        Sigy.terminal = False
        dSigy.terminal = False
        FindEvents.extend([Sigy, dSigy])

    if numFields == 3:
        def Sigz(r, U, arg): return U[4]
        def dSigz(r, U, arg): return U[5]
        
        Sigz.terminal = False
        dSigz.terminal = False
        FindEvents.extend([Sigz, dSigz])

    # Initial conditions setup
    V0 = np.zeros(4 * numFields)
    V0[:2 * numFields] = Ini0[:2 * numFields]
    V0[2 * numFields + 1::2] = Ini0[2 * numFields:]
    
    # Track the central amplitude values
    p0Data = [Ini0[i] for i in range(0, 2 * numFields, 2)]
        
    k = 0
    out = 0
    while k < klim:
        u0 = np.array([shoot(*i) for i in Uintrs])
        V0[2 * numFields::2] = u0  # Updating boundary conditions
        
        # Solving ODE
        sol = solve_ivp(systemMultifrequency, [rmin, rmax], V0, events=FindEvents,
                        args=(arg,), method=met, rtol=Rtol, atol=Atol)

        if info:
            cplot.plotUsingSol(sol)

        # Extracting event times
        eventos = np.array([sol.t_events[i:i+2] for i in range(0, 2*numFields, 2)], dtype=object)
        
        if part:
            even, shot = part
            iInterv, rTemp = freq_shoot(eventos[even], nodos[even], u0[shot], Uintrs[shot], rmax)
            sigModif = shot
        else:
            sigModif_bool = identify(sol.t_events, nodos, p0Data)
            sigModif = np.argmax(sigModif_bool)  # Convert boolean list to index
            
            iInterv, rTemp = freq_shoot(eventos[sigModif], nodos[sigModif], u0[sigModif], Uintrs[sigModif], rmax, inv=inv)

        if info:
            print("Modifying:", sigModif)
            
        
        if abs((iInterv[1] - iInterv[0]) / 2) <= lim:
            if info:
                print(out, '->>', iInterv[1], iInterv[0])
                print('Precision reached: U0 =', u0, 'radius', rTemp)
            
            if out == outval:
                print('Final precision: U0 =', u0, 'radius', rTemp)
                return u0, rTemp, sol.t_events[::2]
            else:
                Uintrs[sigModif] = [iInterv[0] - delta, iInterv[1] + delta]
                out += 1
        else:
            Uintrs[sigModif] = iInterv

        if info:
            print("Updated Uintrs:", Uintrs)

        if np.all(np.array([shoot(*i) for i in Uintrs]) == u0):
            print('Solution found: U0 =', u0, 'radius', rTemp)
            return u0, rTemp, sol.t_events[::2]

        k += 1

    print('Loop limit reached')
    return None

#################################
# Metodología articulo 2208.13221v1.pdf
# fitting -> encuentra los valores de los parametros c_k que cumplen con las condiciones a la derecha impuestas
# algebSyst -> devuelve el resultado de resolver la ecuacion (22) de 2208.13221v1.pdf. 
#              NOTAR que se implementó remNul=True para remover la columna nula en caso de ponerse alguna componente nula
#################################
def fitting(syst, V0, indck, indXc, BCind, inddXc, limit, argf=None, info=False,
            tol=1e-14, met='RK45', Rtol=1e-07, Atol=1e-8, npt=100, klim=500):
    """
    syst -> System of equation with the structure: f(r, yV, arg)
            yV = [x1, x2, ..., xn, dx1c1, dx2c1, ..., dxnc1, ..., dx1cn, dx2cn, ..., dxncn]
            arg: arguments that will be passed to the system

    limit -> iteration limit: [rmin, rmax]

    V0 -> Vector with the Boundary conditions (BC). 
          For example: V0 = [x1(0), c1, c2,
                             dx1c1(0), dx2c1(0), dx3c1(0),
                             dx1c2(0), dx2c2(0), dx3c2(0)]
          In this case, c1, c2 are the initial guesses or seeds for x2(0) and x3(0) respectively
    
    BCind -> A vector with the BC values at the right side which are used to fit the c1, ..., cn values

    indck -> A boolean vector with only True values on the c's positions on the vector V0. 
            From the previous example we have:
            indck = [False, True, True,
                     False, False, False,
                     False, False, False]
    
    indXc -> A boolean vector with only True values on the position in yV associated to the xi variables
            with BC values at the right side.
            
            For example, if we have a system with the structure:
              yV = [x1, x2, x3, dx1c1, dx2c1, dx3c1, dx1c2, dx2c2, dx3c2]
            the BC:
              V0 = [x1(0), c1, c2, 0, 0, 0, 0, 0, 0]
            and:
              BCind = [x1(1), x2(1)]  notice that the x1 and x2 variables have associated a BC at the right side
            
            For this example:
             indXc = [True, True, False, False, False, False, False, False, False]
            the True corresponds to the respective positions in yV of the variables x1, x2
    
    inddXc -> A boolean vector with only True values on the position in yV associated to the derivatives of xi variables
            with BC values at the right side. From the previous example we have:
            inddXc = [False, False, False, True, True, False, True, True, False]
    
    argf -> arguments that will be passed to the system
    info -> Print some extra info
    """

    # discrete points
    rmin, rmax = limit
    rspan = np.linspace(rmin, rmax, npt)
    V0 = np.array(V0, dtype='float64')

    k = 0
    while True:
        sol = solve_ivp(syst, [rmin, rmax], V0, t_eval=rspan, args=[argf], method=met, rtol=Rtol, atol=Atol)
 
        Xbc = (sol.y[indXc])[:,-1]
        if np.all(np.abs(Xbc-BCind)<tol):
            print('Error', np.abs(Xbc-BCind), 'k', k, 'ck', V0[indck])
            break
        else:
            temp = (sol.y[inddXc])[:,-1]
            # notar que es necesario la traspuesta para realizar el producto de la forma adecuada: e.g. dx1*c1+dx2*c2 ...
            dXc = np.transpose(temp.reshape((np.sum(indXc), np.sum(indck))))
            arg = [dXc, Xbc, BCind, V0[indck]]
            ck = algebSyst(arg, info=info)

        V0[indck] = ck

        if k==klim:
            print('Stop', np.abs(Xbc-BCind), 'k', k, 'ck', V0[indck])
            break
        k += 1
    return V0

def algebSyst(arg, remNul=True, info=False):
    """
    Resuelve la ecuacion 22 de 2208.13221v1.pdf
    """
    dXc, Xbc, Xb, ck = arg

    MI = np.array(dXc, dtype='float64')
    MD = np.array(dXc@ck - (Xbc - Xb), dtype='float64')
    
    if remNul:  # remueve las componentes asociados al vector nulo en caso de tener
        test = MD!=0
        MI = MI[:, test][test]
        MD = MD[test]

        ck1 = np.zeros(len(test))
        temp = np.linalg.solve(MI, MD)
        ck1[test] = temp
    else:
        ck1 = np.linalg.solve(MI, MD)

    if info:
        temp = np.allclose(np.dot(MI, ck1), MD)
        print(temp)
    
    return ck1
