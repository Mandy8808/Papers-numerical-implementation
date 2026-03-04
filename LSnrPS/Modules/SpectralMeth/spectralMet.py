# module to compute the spectro asociated to the matrix (operator) Mij
# by a SPECTRAL METHOD

# LOADING MODULES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.linalg import eig


####### CHEBISHEV POLYNOMIAL DIFFERENTIATION MATRIX
#################################################################
def cheb(op):
    '''Chebyshev polynomial differentiation matrix.
       Ref.: Trefethen's 'Spectral Methods in MATLAB' book.
       N - size of diff matrix - op+1 where op is polynomial order.
    '''
    N = op + 1
    if N == 1:
        return np.array([[0.0]]), np.array([1.0])  # Edge case for low order

    # Creating the Chebyshev points
    j = np.arange(N)
    x = np.cos(j * np.pi / (N - 1))

    # Weights for differentiation matrix
    c = np.ones(N)
    c[0], c[-1] = 2.0, 2.0  # Special weights for first and last points
    c *= (-1.0)**j
    c = c.reshape(N, 1)

    # Compute matrix entries
    X = np.tile(x.reshape(N, 1), (1, N))
    dX = X - X.T  # Difference matrix

    D = np.dot(c, (1.0/c).T) / (dX + np.eye(N))  # Compute off-diagonal elements
    np.fill_diagonal(D, 0)  # Zero diagonal elements first

    # Compute diagonal elements
    D[np.diag_indices(N)] = -D.sum(axis=1)

    return D, x
 
def cheb2(op):
    '''Chebushev polynomial differentiation matrix.
       Ref.: Trefethen's 'Spectral Methods in MATLAB' book.
       N - size of diff matrix - op+1 where op is polynomial order.
    '''

    N = op + 1
    # creating the Chebyshev points
    j = np.arange(N)
    x = np.cos(j*np.pi/(N-1))

    #  off-diagonal entries -> Dij
    Dtemp = np.ones(N)
    Dtemp[0], Dtemp[N-1] = 2.0, 2.0
    Dtemp *= (-1.0)**j
    Dtemp = Dtemp.reshape(N, 1)
    Dtemp = np.dot(Dtemp, (1.0/Dtemp).T)

    X = np.tile(x.reshape(N, 1), (1, N))
    dX = X - X.T  # distance difference (dX is a matrix)

    Dtemp = Dtemp/(dX+np.eye(N))  # eye Return a N+1 array with ones on the diagonal and zeros elsewhere.

    # diagonal entries
    Dii = np.diag(Dtemp.sum(axis=1))  # sum by the row, and construct a diagonal array

    # Dn
    D = Dtemp - Dii

    return D, x

####### SCALING OPERATORS
########################################
def chevQuant(util, info=False):
   """
   Map between [-1, 1] -> [0, rMax]
   -> radii
   -> differentiation matrix
   """
   # Unpacking tuple
   _, Nptos, rMax = util
    
   # Compute the Chebyshev differentiation matrix
   #D_chev, x_chev = cheb(Nptos - 1)  # Nptos = order + 1, so pass Nptos-1
   D_chev, x_chev = cheb2(Nptos)  # Nptos = order + 1, so pass Nptos-1
    
   # Scaling from [-1, 1] to [0, rMax] using x_chev = 2(r/rMax) - 1
   r_dis = (1. - x_chev) * (rMax / 2.)

   if info:
      # Checking rescaling
      print('Checking the distance scaling:', np.isclose(r_dis[0], 0.), np.isclose(r_dis[-1], rMax))
    
   return r_dis, D_chev, x_chev

####### COMPUTING THE BACKGROUND OPERATORS
################################################
def backgroundOper(datFunc, util, lamV, alp, info=False):
   """
   Computes the background operators:
   Sigma0, Ueff, TrianJInv, Rmatriz, D2i_chev, r_dis_inner, lamStar
   """
    
   # Unpacking the parameters
   J, Nptos, rMax = util
   
   # Unpacking background quantities
   fsN, fuN = datFunc
   
   # Computing the \lambda_*
   ln, ls = lamV
   lamStar = abs(ln + alp**2 * ls) if abs(ln + alp**2 * ls) !=0 else 1
    
   # Mapping [-1, 1] -> [0, rMax]
   utilphy = [J, Nptos, rMax]
   r_dis, D_chev, _ = chevQuant(utilphy, info=info)

   # Creating the matricial operators
   # (NOTICE that the homogeneous Dirichlet boundary conditions are defined by omitting the first and last element)
   r_dis_inner = r_dis[1:Nptos]  # radii values excluding boundaries
    
   # vector of Matrix Sigma0
   Sigma0 = [np.diag(fs(r_dis_inner)) for fs in fsN]  # [σ_x^0, σ_y^0, ...]
   
   # temporal matrices
   Rmatriz = np.diag(1 / r_dis_inner**2)  # Rm = 1/r²
   Jmatriz = J * (J + 1) * Rmatriz  # Jm = J(J+1)/r²

   # vector of Matrix Ueff
   Ueff = []
   for fu in fuN:
      U0 = np.diag(fu(r_dis_inner))  # potential u0 = E - Δ⁻¹(|σ₁⁰|²)
      Ueff.append(U0 - Jmatriz)  # Ueff = u0 - J(J+1)/r²
    
   # Derivative operators
   D2_chev = np.dot(D_chev, D_chev) / ((rMax / 2) ** 2)  # Computing D²
    
   # Extracting the interior matrix (ignoring boundaries)
   D2i_chev = D2_chev[1:Nptos, 1:Nptos]

   # Operator TrianJInv = [4*D²/r_star² - J(J+1)/r²]⁻¹
   temp = D2i_chev - Jmatriz
   TrianJInv = np.linalg.inv(temp)  # (D² - J(J+1)/r²)⁻¹

   if info:
      print('Dimensions of the matrices Sigma0 and Ueff:', Sigma0[0].shape, Ueff[0].shape)
      print('\nChecking the inverse of D²...')
      check = np.allclose(np.dot(temp, TrianJInv), np.eye(Nptos - 1))  # Check if inverse is correct
      print('Inverse check:', check)

   return Sigma0, Ueff, TrianJInv, Rmatriz, D2i_chev, r_dis_inner, lamStar

####### Block Polarization matrices
#####################################
def linBlock(Nptos, Sigma0, Ueff, TrianJInv, D2i_chev, lamV, lamStar, extras=None):
   """
   Constructs the matrix blocks M11, M12, M21, M22, and M33 using the given background operators.
   These matrices appear in the linearized equations (Eq. A2).

   Parameters:
   - Nptos: Number of discretization points.
   - Sigma0: Background sigma matrix.
   - Ueff: Effective potential matrix.
   - TrianJInv: Inverted triangular operator.
   - D2i_chev: Second derivative operator matrix.
   - lamV: Tuple (ln, ls) representing the lambda values.

   Returns:
   - M11, M12, M21, M22, M33: Block matrices.
   """
   ln, ls = np.array(lamV) / lamStar
   size = 2 * (Nptos - 1)

   # Initialize complex matrices
   M11 = np.zeros((size, size), dtype=complex)
   M12 = np.zeros((size, size), dtype=complex)
   M21 = np.zeros((size, size), dtype=complex)
   M22 = np.zeros((size, size), dtype=complex)

   Sigma0, Ueff = Sigma0[0], Ueff[0]
   sigSq = Sigma0 @ Sigma0  # Compute Sigma0^2
   sigma_Trian_sigma = Sigma0 @ TrianJInv @ Sigma0  # Precompute term to avoid redundancy

   # Define block positions
   row, col = 0, 1
   start_r, end_r = row * (Nptos - 1), (row + 1) * (Nptos - 1)
   start_c, end_c = col * (Nptos - 1), (col + 1) * (Nptos - 1)
   M11[start_r:end_r, start_c:end_c] = D2i_chev + Ueff - ln * sigSq
   M22[start_r:end_r, start_c:end_c] = D2i_chev + Ueff - (ln + 2 * ls) * sigSq

   row, col = 1, 0
   start_r, end_r = row * (Nptos - 1), (row + 1) * (Nptos - 1)
   start_c, end_c = col * (Nptos - 1), (col + 1) * (Nptos - 1)
   M11[start_r:end_r, start_c:end_c] = D2i_chev + Ueff - 3 * ln * sigSq - 2 * sigma_Trian_sigma
   M22[start_r:end_r, start_c:end_c] = D2i_chev + Ueff - ln * sigSq

   #print(np.sum(sigSq), np.sum(D2i_chev + Ueff))

   # M33 is identical to M22
   M33 = np.copy(M22)

   return M11, M12, M21, M22, M33

def circBlock(Nptos, Sigma0, Ueff, TrianJInv, D2i_chev, lamV, lamStar, extras):
   """
   Constructs the matrix blocks M11, M12, M21, M22, and M33 using the given background operators.
   These matrices appear in the linearized equations (Eq. A3).

   Parameters:
   - Nptos: Number of discretization points.
   - Sigma0: Background sigma matrix.
   - Ueff: Effective potential matrix.
   - TrianJInv: Inverted triangular operator.
   - D2i_chev: Second derivative operator matrix.
   - lamV: Tuple (ln, ls) representing the lambda values.
   - alph: Scaling factor.
   - lamStar: Scaling parameter for lambda values.

   Returns:
   - M11, M12, M21, M22, M33: Block matrices.
   """
   ln, ls = np.array(lamV) / lamStar

   alph = extras
   size = 2 * (Nptos - 1)
   # Initialize complex matrices
   M11 = np.zeros((size, size), dtype=complex)  # dtype = 'complex_'
   M12 = np.zeros((size, size), dtype=complex)
   M21 = np.zeros((size, size), dtype=complex)
   M22 = np.zeros((size, size), dtype=complex)
   M33 = np.zeros((size, size), dtype=complex)

   Sigma0, Ueff = Sigma0[0], Ueff[0]
   sigSq = Sigma0 @ Sigma0  # Compute Sigma0^2
   sigma_Trian_sigma = Sigma0 @ TrianJInv @ Sigma0  # Precompute term to avoid redundancy
   
   row, col = 0, 0
   start_r, end_r = row * (Nptos - 1), (row + 1) * (Nptos - 1)
   start_c, end_c = col * (Nptos - 1), (col + 1) * (Nptos - 1)
   M12[start_r:end_r, start_c:end_c] = 1j * alph * ls * sigSq
   M21[start_r:end_r, start_c:end_c] = -1j * alph * ((ln + 2 * ls) * sigSq + sigma_Trian_sigma)
   
   row, col = 0, 1
   start_r, end_r = row * (Nptos - 1), (row + 1) * (Nptos - 1)
   start_c, end_c = col * (Nptos - 1), (col + 1) * (Nptos - 1)
   M11[start_r:end_r, start_c:end_c] = D2i_chev + Ueff - ln * sigSq
   M22[start_r:end_r, start_c:end_c] = D2i_chev + Ueff - (2 * ln + ls) * sigSq - sigma_Trian_sigma
   M33[start_r:end_r, start_c:end_c] = D2i_chev + Ueff - (ln + ls) * sigSq

   row, col = 1, 0
   start_r, end_r = row * (Nptos - 1), (row + 1) * (Nptos - 1)
   start_c, end_c = col * (Nptos - 1), (col + 1) * (Nptos - 1)
   M11[start_r:end_r, start_c:end_c] = D2i_chev + Ueff - (2 * ln + ls) * sigSq - sigma_Trian_sigma
   M22[start_r:end_r, start_c:end_c] = D2i_chev + Ueff - ln * sigSq
   M33[start_r:end_r, start_c:end_c] = D2i_chev + Ueff -(ln + ls) * sigSq

   row, col = 1, 1
   start_r, end_r = row * (Nptos - 1), (row + 1) * (Nptos - 1)
   start_c, end_c = col * (Nptos - 1), (col + 1) * (Nptos - 1)
   M12[start_r:end_r, start_c:end_c] = 1j * alph * ((ln + 2 * ls) * sigSq + sigma_Trian_sigma)
   M21[start_r:end_r, start_c:end_c] = -1j * alph * ls * sigSq

   return M11, M12, M21, M22, M33

def radBlock(Nptos, Sigma0, Ueff, TrianJInv, D2i_chev, lamV, lamStar, extras):
   """
   Constructs the matrix blocks M11, M12, M21, M22, and M33 using the given background operators.
   These matrices appear in the linearized equations (Eq. A4).

   Parameters:
   - Nptos: Number of discretization points.
   - Sigma0: Background sigma matrix.
   - Ueff: Effective potential matrix.
   - TrianJInv: Inverted triangular operator.
   - D2i_chev: Second derivative operator matrix.
   - lamV: Tuple (ln, ls) representing the lambda values.
   - alph: Scaling factor.
   - lamStar: Scaling parameter for lambda values.

   Returns:
   - M11, M12, M21, M22, M33: Block matrices.
   """
   ln, ls = np.array(lamV) / lamStar
   Rmatriz, J = extras
   
   size = 2 * (Nptos - 1)
   # Initialize complex matrices
   M11 = np.zeros((size, size), dtype=complex)  # dtype = 'complex_'
   M12 = np.zeros((size, size), dtype=complex)
   M22 = np.zeros((size, size), dtype=complex)
   
   Sigma0, Ueff = Sigma0[0], Ueff[0]
   sigSq = Sigma0 @ Sigma0
   sigma_Trian_sigma = Sigma0 @ TrianJInv @ Sigma0  # Precompute term to avoid redundancy
   
   row, col = 0, 1
   start_r, end_r = row * (Nptos - 1), (row + 1) * (Nptos - 1)
   start_c, end_c = col * (Nptos - 1), (col + 1) * (Nptos - 1)
   M11[start_r: end_r, start_c: end_c] = D2i_chev + Ueff - 2 * Rmatriz - ln*sigSq
   M12[start_r: end_r, start_c: end_c] = 2 * np.sqrt(J * (J + 1)) * Rmatriz
   M22[start_r: end_r, start_c: end_c] = D2i_chev + Ueff - (ln + 2 * ls) * sigSq

   row, col = 1, 0
   start_r, end_r = row * (Nptos - 1), (row + 1) * (Nptos - 1)
   start_c, end_c = col * (Nptos - 1), (col + 1) * (Nptos - 1)
   M11[start_r: end_r, start_c: end_c] = D2i_chev + Ueff - 2 * Rmatriz - 3 * ln * sigSq - 2 * sigma_Trian_sigma
   M12[start_r: end_r, start_c: end_c] = 2 * np.sqrt(J * (J + 1)) * Rmatriz
   M22[start_r: end_r, start_c: end_c] = D2i_chev + Ueff - ln * sigSq

   M21 = np.copy(M12)
   M33 = np.copy(M22)
   
   return M11, M12, M21, M22, M33

def MultMii(Nptos, Sigma0i, TSigma0_Sq, Ueffi, TrianJInv, D2i_chev, ln):
   """
   Mii -> multifrequency
   """
   size = 2 * (Nptos - 1)
   Mii = np.zeros((size, size), dtype=complex)  # Initialize complex matrices
   
   sigSq = Sigma0i@Sigma0i
   sigma_Trian_sigma = Sigma0i @ TrianJInv @ Sigma0i
   
   # Block (0, 1)
   Mii[0*(Nptos - 1): 1*(Nptos - 1), 1*(Nptos - 1): 2*(Nptos - 1)] = D2i_chev + Ueffi - ln * TSigma0_Sq
   
   # Block (1,0)
   Mii[1*(Nptos - 1): 2*(Nptos - 1), 0*(Nptos - 1): 1*(Nptos - 1)] = (
      D2i_chev + Ueffi - ln * (2 * sigSq + TSigma0_Sq) - 2 * sigma_Trian_sigma
   )
   
   return Mii

def MultMij(Nptos, Sigma0i, Sigma0j, TrianJInv, ln):
   """
   Mij -> multifrequency
   """
   size = 2 * (Nptos - 1)
   Mij = np.zeros((size, size), dtype=complex)  # Initialize complex matrices
   
   sigSqij = Sigma0i@Sigma0j
   sigmai_Trian_sigmaj = Sigma0i @ TrianJInv @ Sigma0j
   
   # Block (1, 0)
   Mij[1*(Nptos - 1): 2*(Nptos - 1), 0*(Nptos - 1): 1*(Nptos - 1)] = -2 * (ln * sigSqij + sigmai_Trian_sigmaj)

   return Mij

def multBlock(Nptos, Sigma0, Ueff, TrianJInv, D2i_chev, lamV, lamStar, extras=None):
   """
   Construct a block matrix using `MultMii` and `MultMij` functions.
   """
   ln, _ = np.array(lamV) / lamStar

   TSigma0_Sq = sum(Sigma0i @ Sigma0i for Sigma0i in Sigma0)  # summation

   Mij = []
   for i in range(3):
      Sigma0i, Ueffi = Sigma0[i], Ueff[i]
      for j in range(3):
         Sigma0j = Sigma0[j]
         if j != i:
            Mij.append(MultMij(Nptos, Sigma0i, Sigma0j, TrianJInv, ln))
         else:
            Mij.append(MultMii(Nptos, Sigma0i, TSigma0_Sq, Ueffi, TrianJInv, D2i_chev, ln))

   return Mij  # M11, M12, M13, M21, M22, M23, M31, M32, M33

####### Spectral Implementation 
######################################
def espectro(datFunc, util, lamV, alp, polariz, info=False, fplot=False):
   r"""
   Computes the spectrum of a system.
    
   Type of polarizations:
   i. $\gamma=0$, $\alpha=0$ if the polarization is linear, 
   ii. $\gamma=0$, $\alpha=1$ if the polarization is circular, and 
   iii. $\gamma=1$, $\alpha=0$ if the polarization is radial.
   """
   ############################################################################
   # Cheking the in put parameters:
   # Check if the polarization is valid
   if polariz not in ['linear', 'circular', 'radial', 'multifrequency']:
      raise ValueError("Choose a valid polarization: radial, circular, linear, or multifrequency")
   # Check if the parameters are valid
   if polariz == 'linear' and alp != 0:
      raise ValueError("Alpha should be zero for linear polarization.")
   if polariz == 'circular' and alp == 0:
      raise ValueError("Alpha should be non-zero for circular polarization.")
   if polariz == 'radial' and alp != 0:
      raise ValueError("Alpha should be zero for radial polarization.")
   if polariz == 'multifrequency' and alp != 0:
      raise ValueError("Alpha should be zero for multifrequency polarization.")
   # Check if the data function is valid
   if np.any([not callable(i[0]) for i in datFunc]):
      raise ValueError("datFunc should be a list of callable functions.")
   # Check if the utility parameters are valid
   if not isinstance(util, (list, tuple)) or len(util) != 3:
      raise ValueError("util should be a list or tuple of length 3.")
   if not isinstance(lamV, (list, tuple)) or len(lamV) != 2:
      raise ValueError("lamV should be a list or tuple of length 2.")
   if not isinstance(alp, (int, float)):
      raise ValueError("alp should be an integer or float.")
   if not isinstance(info, bool):
      raise ValueError("info should be a boolean value.")
   if not isinstance(fplot, bool):
      raise ValueError("fplot should be a boolean value.")
   ############################################################################
   
   # Global variables
   J, Nptos, _ = util

   # Background operators
   Sigma0, Ueff, TrianJInv, Rmatriz, D2i_chev, r_dis2, lamStar = backgroundOper(datFunc, util, lamV, alp, info=info)

   # Define the (M_J)_ij matrix
   num = 3  # Block number: Dim(OM_chev) = num*(Nptos-1) x num*(Nptos-1)
   OM_chev = np.zeros((num*2*(Nptos-1), num*2*(Nptos-1)), dtype = 'complex_')
   
   # Block matrices functions
   blockPola = {
        'linear': linBlock,
        'circular': circBlock,
        'radial': radBlock,
        'multifrequency': multBlock
   }
   
   # Index patterns for different polarizations
   index_patterns = {
        "linear": zip([0, 1, 2], [0, 1, 2]),
        "circular": zip([0, 0, 1, 1, 2], [0, 1, 0, 1, 2]),
        "radial": zip([0, 0, 1, 1, 2], [0, 1, 0, 1, 2]),
        "multifrequency": zip([0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2])
   }
   
   # Validate polarization
   if polariz not in blockPola:
      raise ValueError("Choose a valid polarization: radial, circular, linear, or multifrequency")
   
   # Compute block data
   extras = None if polariz == "linear" else alp if polariz == "circular" else [Rmatriz, J]
   data = blockPola[polariz](Nptos, Sigma0, Ueff, TrianJInv, D2i_chev, lamV, lamStar, extras)
   
   # Fill OM_chev matrix
   for ind, (row, col) in enumerate(index_patterns[polariz]):
      OM_chev[2 * row * (Nptos - 1): 2 * (row + 1) * (Nptos - 1),
              2 * col * (Nptos - 1): 2 * (col + 1) * (Nptos - 1)] = data[ind]

   # Plot if required
   if fplot:
      for Sigma0i, Ueffi in zip(Sigma0, Ueff):
         plotImag(r_dis2, Sigma0i, Ueffi)  # + J * (J + 1) * Rmatriz

   # Compute eigenvalues and eigenvectors using scipy.linalg.eig
   # Mij@X = lambda X
   # print("===> ", np.sum(OM_chev), "size ", OM_chev.shape)
   eig_values, eig_vectors = eig(OM_chev)

   # Verify eigenvalues and eigenvectors if info is True
   # Valify that the system Ax=Lx
   if info:
      test = [np.allclose(OM_chev @ eig_vectors[:, i] - (eig_values[i] * eig_vectors[:, i]),
                          np.zeros((2 * num * (Nptos - 1)), dtype=complex))
              for i in range(len(eig_values))]
      print("Verifying that Ax = λx holds ->\n", np.array(test))
      
   # Compute the real Lambda values
   Lambda = 1j * eig_values  # Lambda = -i LambdaReal -> LambdaReal = i Lambda
    
   # Sort eigenvalues from smallest to largest
   sorted_indices = np.argsort(np.abs(Lambda))
   LambdaSorted = Lambda[sorted_indices]
   eig_vectors_sorted = eig_vectors[:, sorted_indices]
    
   return LambdaSorted, Lambda, eig_vectors_sorted, r_dis2

####### Convergence Study 
######################################
def Organize(data, Rtol=1e-02, Atol=1e-03):
   """
   Organizes the data based on a reference row.
   """
   R_row, i = Reference_row(data)  # Pivot row
   data_T = data.copy()

   Nrow = len(data_T.index)
   for ind in range(Nrow):
      if ind == i:  # Skip the pivot row
         continue
      data_T = Organize_row(ind, R_row, data_T, Rtol, Atol)

   return data_T

def Reference_row(data):
   """
   Gets the reference row (first one without NaN values).
   """
   Nrow = len(data.index)
   for i in range(Nrow):
      R_row = data.iloc[[i]]
      if not R_row.isnull().values.any():
         print(f'The reference row is {i}')
         return R_row, i
   return None, None  # In case all rows contain NaN values

def Organize_row(ind, R_row, data_T, Rtol, Atol):
   """
   Organizes row `ind` based on the reference row `R_row`.
   """
   for col in R_row.columns[1:]:  # Skip the first column if it is an index
      # Get real and imaginary parts from the reference row
      valSupI = np.imag(R_row[col].values[0])
      valSupR = np.real(R_row[col].values[0])

      # Get values from the row being sorted
      tempFrame = data_T.iloc[[ind]]
      # tempArray = tempFrame[col].values  # Extract values as an array
      tempArrayI = np.imag(tempFrame)[0] # np.imag(tempArray)
      tempArrayR = np.real(tempFrame)[0] # np.real(tempArray)

      # Comparison with tolerance
      compI = np.isclose(tempArrayI, valSupI, rtol=Rtol, atol=Atol)
      compR = np.isclose(tempArrayR, valSupR, rtol=Rtol, atol=Atol)
      filt = compI & compR  # Element-wise AND

      if filt.any():
         # Get the matching index
         j = np.where(filt)[0]  # np.where returns a tuple, extract the first element
         # val1 = tempArray[filt]  # Matching values
         val1 = (np.array(tempFrame)[0])[filt]  # Matching values
         if len(val1) > 1:
            val1 = val1[0]  # Take only the first if there are multiple
         val2 = data_T.at[ind, col]  # Original value

         # Swap values
         data_T.at[ind, col] = val1
         #data_T.at[ind, data_T.columns[j[0]]] = val2  # Ensure correct access
         data_T.loc[ind, j] = val2

   return data_T

####### Tools 
######################################
def LamJval(datFunc, rMax, lamV, alp, polariz, 
            Nptos=500, Jval=[0, 1, 2, 3], real=True, 
            info=False, fplot=False):
  """
  Computing the spectro for a set of J-values
  """
  if info:
    print('Computing the spectro for the polarizacion ', polariz)

  # Spectro
  lambdaVal, dataExtra = [], []
  for ind, J in enumerate(Jval):
     util = [J, Nptos, rMax]
     LambdaSorted, _, eig_vectors_sorted, r_dis2 = espectro(datFunc, util, lamV, alp, polariz, info=info, fplot=fplot)
     
     if real:
         jj = np.real(LambdaSorted) != 0  # reales
         lambReal = LambdaSorted[jj]
         vectReal = eig_vectors_sorted[:, jj]     
         lambdaVal.append([J, lambReal])
         dataExtra.append([J, [r_dis2, lambReal, vectReal]])
     else:
         lambdaVal.append([J, LambdaSorted])
         dataExtra.append([J, [r_dis2, LambdaSorted, eig_vectors_sorted]])
   
     progressbar(ind, len(Jval)-1)
  return lambdaVal, dataExtra


def sep(Auto_Valores, Auto_Funciones, Imag=False):
   """
   Separando los autovalores y autovectores 
   """
   ndatos = len(Auto_Valores)
   dataR, dataI = [], []
   dataRV, dataIV = [], []
   for i in range(ndatos):
        jj = np.real(Auto_Valores[i])!=0  # reales
        dataR.append(Auto_Valores[i][jj])
        dataRV.append(Auto_Funciones[i][:, jj])
   
        if Imag:
            gg = np.array([not(k) for k in jj])  # imag
            dataI.append(Auto_Valores[i][gg])
            dataIV.append(Auto_Funciones[i][:, gg])

   if Imag:
      return dataR, dataRV, dataI, dataIV
   else:
      return dataR, dataRV

def progressbar(current_value, total_value, bar_length=20, progress_char='#'): 
    """
    Display a progress bar in the console.
    :param current_value: Current progress value.
    :param total_value: Total value for completion.
    :param bar_length: Length of the progress bar.
    :param progress_char: Character used to fill the progress bar.
    """
    if total_value == 0:
        print("Error: total_value cannot be 0")
        return
    
    # Calculate the percentage and progress
    percentage = int((current_value / total_value) * 100)
    progress = int((bar_length * current_value) / total_value)
    
    # Build the progress bar string
    loadbar = f"Progress: [{progress_char * progress}{'.' * (bar_length - progress)}] {percentage}%"
    
    # Print the progress bar
    print(loadbar, end='\r')
    

def plotImag(r_dis2, Sigma0, Ueff):
   fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4),
                       sharex=False, sharey=False,
                       gridspec_kw=dict(hspace=0.0, wspace=.28))
   
   ax[0].plot(r_dis2, np.diagonal(Sigma0), 'o', markersize=2, mfc='white')
   ax[1].plot(r_dis2, np.diagonal(Ueff), 'o', markersize=2, mfc='white')
   #ax[0].set_xlim(0, 30)
   plt.show()
   return