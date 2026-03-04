# 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from matplotlib.colors import LinearSegmentedColormap


#############################################################################
def colorBar_and_normaliz(maximo, minimo, cmapStr=False):
    """    
    Creates a color map and normalization scale for visualizations.
    """

    # Ensure valid input ranges
    if minimo >= maximo:
        raise ValueError("minimo must be less than maximo")

    # Creating normalized color bar
    if cmapStr:
        cmap = LinearSegmentedColormap.from_list('', 
                                             [(0, '#c23838'),
                                              (0.25, '#f0784d'),
                                              (0.5, '#337512'),
                                              (0.75, '#1dc4ab'),
                                              (1, '#2681ab')], N=2000)
    else:
        cmap = LinearSegmentedColormap.from_list('', ['#f0784d', '#2681ab'])

    norm = mcolors.Normalize(vmin=minimo, vmax=maximo)
    
    return cmap, norm


def ShowPlaneProf(profData, X=None, Y=None, Z=None, indX=None, indY=None, indZ=None, save=False):
    """ 
    Function to validate and process plane profiles.
    """
    
    ####### Validate numerical parameters
    # Ensure the correct structure of the profile data
    if len(profData.shape) != 3:
        raise ValueError("The profile needs to have the structure: (resol, resol, resol)")

    # Ensure exactly two coordinate components are provided
    if sum(v is not None for v in [X, Y, Z]) != 2:
        raise ValueError("It is necessary to provide data for exactly two coordinate components.")

    # Ensure indices are either int or None, and only one index is provided
    if not all(isinstance(temp, (int, type(None))) for temp in [indX, indY, indZ]):
        raise ValueError("The indices must be integers or None.")
    if sum(v is not None for v in [indX, indY, indZ]) not in [0, 1]:
        raise ValueError("Only one index must be provided.")
    

    ####### Preparing data
    coord = np.array(["x", "y", "z"], dtype=object)
    xd, yd = [temp for temp in [X, Y, Z] if temp is not None]
    cxd, cyd = [coord[k] for k, temp in enumerate([X, Y, Z]) if temp is not None]

    # Extract profile based on provided index
    if indX is not None:
        prof = profData[indX, :, :]
    elif indY is not None:
        prof = profData[:, indY, :]
    elif indZ is not None:
        prof = profData[:, :, indZ]
    else:
        print("Showing the Z=0, X-Y plane")
        resol = profData.shape[0]
        prof = profData[:, :, resol // 2]  # Ensure integer division
        
    PlaneProf(prof, xd, yd, cxd, cyd, save=save)
    
    return None

def PlaneProf(prof, xd, yd, cxd, cyd, save=False):
    """ 
    Plot a 2D proyection
    """
    
    # Generate mesh grid
    xdG, ydG = np.meshgrid(xd, yd, indexing='ij')

    # Get min/max values for normalization
    maximo = np.max(prof)
    minimo = np.min(prof)
    
    cmap, norm = colorBar_and_normaliz(maximo, minimo)  # Ensure function name matches
    
    ###### Plot 3D profile
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Normalize color values
    color_values = norm(prof.flatten())  # Flatten data for color mapping
    
    # Scatter plot
    ax.scatter(xdG.flatten(), ydG.flatten(), prof.flatten(), c=color_values, cmap=cmap, norm=norm, s=0.5)

    ax.set_xlabel(r'%s' % cxd)
    ax.set_ylabel(r'%s' % cyd)
    ax.set_zlabel(r'Profile')

    ax.set_xlim(np.min(xdG), np.max(xdG))
    ax.set_ylim(np.min(ydG), np.max(ydG))
    ax.set_zlim(np.min(prof), np.max(prof))
    
    # Line plots for 2D reference
    ax.plot(xd, prof[:, prof.shape[1] // 2], zs=0, zdir='y', color='k', alpha=0.8)
    ax.plot(yd, prof[prof.shape[0] // 2, :], zs=0, zdir='x', color='r', alpha=0.8)

    # Save or show plot
    if save:
        fig.savefig('P0plot.pdf', format='pdf', pad_inches=0.1, dpi=1000, bbox_inches='tight')
    else:
        #ax.view_init(elev=90., azim=-90)
        plt.show()
    
    return None

def Plot3DCorrProf(X, Y, Z, profile, minpsi=1e-8, save=False):
    """ 
    Function to plot a 3D correlation profile.
    """
    
    # Generate mesh grid (Ensure correct shape)
    X, Y, Z = np.meshgrid(X, Y, Z, indexing='ij')  # Ensure correct indexing
    
    # Define max/min for normalization
    maximo = np.max(profile)
    minimo = np.min(profile)

    # Get colormap and normalization function
    cmap, norm = colorBar_and_normaliz(maximo, minimo)
    
    # Masking data based on `minpsi` threshold
    ind = profile > minpsi

    # Create figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Normalize color values
    color_values = norm(profile[ind])  # Normalize only valid data

    # Scatter plot with proper colormap
    sc = ax.scatter(X[ind], Y[ind], Z[ind], c=color_values, cmap=cmap, norm=norm, s=0.01)

    # Labels
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_zlabel(r'$z$')

    # Set limits
    ax.set_xlim(np.min(X), np.max(X))
    ax.set_ylim(np.min(Y), np.max(Y))
    ax.set_zlim(np.min(Z), np.max(Z))

    # Adjust view
    ax.view_init(elev=40., azim=45)

    # Save or show plot
    if save:
        fig.savefig('P0plot.pdf', format='pdf', pad_inches=0.1, dpi=1000, bbox_inches='tight')
    else:
        plt.show()

    return None