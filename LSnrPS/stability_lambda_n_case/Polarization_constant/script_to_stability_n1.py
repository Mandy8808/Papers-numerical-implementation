# script to stability

# importing the modules
import Modules.SpectralMeth as sp
import numpy as np

from scipy.interpolate import interp1d


# loading the profiles
Family_cte_polarization_LPn1 = np.load("background_data_profiles_lambda_n_case/Family_cte_polarization_LPn1.npy", allow_pickle=True)
# Family_cte_polarization_LNn1 = np.load("background_data_profiles_lambda_n_case/Family_cte_polarization_LNn1.npy", allow_pickle=True)

# n = 1
s0 = Family_cte_polarization_LPn1
# s0M = Family_cte_polarization_LNn1

# computing the spectrum
data_real_save = []

polariz = 'linear'

Jval = [0, 1, 2, 3]
alp = 0.0
ln = 1.
# ln = -1.

lamV = [ln, 0] # ln, ls
for cont, s in enumerate(s0): 
# for cont, s in enumerate(s0M): 
    Ei, rDnew, sDnew, dsDnew, uDnew, duDnew = s
    print('==> ', sDnew[0])
    datFunc = [
        [interp1d(rDnew, sDnew)],
        [interp1d(rDnew, uDnew)],
    ]

    if cont < 3:
        rMax = 8000
        Nptos = 1350
    elif cont < 15:
        rMax = 1500
        Nptos = 1250
    elif cont < 15:
        rMax = 750
        Nptos = 1000
    else:
        rMax = 450
        Nptos = 800

    # Computing the spectrum
    lambdaVal, dataExtra = sp.LamJval(datFunc, rMax, lamV, alp, polariz, 
            Nptos=Nptos, Jval=Jval, info=False, fplot=False, real=False)

    # Save data
    for ind, [Jval_temp, data] in enumerate(dataExtra):
        rdis, eingVal, eigVect = data
        ind = np.abs(np.real(eingVal)) >= 1e-06 # filtrando los autovalores reales
        data_real_save.append([Jval_temp, float(sDnew[0]), rdis if sum(ind)!=0 else [], eingVal[ind], eigVect[:, ind]])

# Save data       
np.save("Real_Eingevalues_cte_polarization_LPn1", np.array(data_real_save, dtype=object))
# np.save("Real_Eingevalues_cte_polarization_LNn1", np.array(data_real_save, dtype=object))