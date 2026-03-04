# Proyecto proca v.1.0
# Plot configurations

import os
import sys
import warnings
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from scipy.integrate import solve_ivp
from cycler import cycler
from matplotlib.collections import LineCollection

##############
# Get the parent directory
parent_dir =  os.path.abspath(os.path.join(os.path.dirname("Modules/Background/background.py"), ""))

# Add it to sys.path
sys.path.append(parent_dir)

# Now you can import the module
import background as bg  # Import from the parent folder (modules)

################
## SET OF COLORS
################

def get_colors():
    """
    Returns a predefined tuple of hex color codes for consistent plotting.
    
    Returns:
        tuple: A tuple of hex color strings.
    """
    return (
        '#1a1919', '#f0784d', '#2681ab', '#ab262f', '#bc92e0', 
        '#486318', '#ed5b0c', '#f0a092', '#484f07', '#694d0c'
    )

######
# MODULO PARA CONFIGURACION GENERAL DE GRAFICOS
# For more rcParams check
# https://matplotlib.org/stable/users/explain/customizing.html#matplotlibrc-sample
####

def general():
    """
    CONFIGURACIÓN GENERAL
    """ 
    for i in [FigParam(), LineParam(), axesParam(), labelParam(), fontParam()]:
        i

    return

def FigParam():
    """
    configuracion de los marcos, resolucion, etc.
    see: https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.savefig.html
    """
    # resolution
    mpl.rcParams['figure.figsize'] = [6.4, 4.8]  # (width, height) in inches
    mpl.rcParams['figure.facecolor'] = 'white'     # figure facecolor
    mpl.rcParams['figure.edgecolor'] = 'white'     # figure edgecolor
    mpl.rcParams['figure.dpi'] = 100 # resolution in dots per inch.

    # to save
    mpl.rcParams['savefig.dpi'] = 1000
    #mpl.rcParams['savefig.metadata'] = None
    mpl.rcParams['savefig.format'] = 'pdf'
    mpl.rcParams['savefig.bbox'] = 'tight' # Plot will be occupy a maximum of available space
    mpl.rcParams['savefig.pad_inches'] = 0.1
    mpl.rcParams['savefig.facecolor'] = 'auto'
    mpl.rcParams['savefig.edgecolor'] = 'auto'
    
        
def LineParam():
    # estilo de linea y grosor
    mpl.rcParams['lines.linestyle'] = '-'
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['lines.markersize'] = 10  # weight of the marker

    # orden de los colores que usará
    mpl.rcParams['axes.prop_cycle'] = cycler('color',
                                             ['#1f77b4', '#ff7f0e', '#2ca02c',
                                              '#d62728', '#9467bd', '#8c564b',
                                              '#e377c2', '#7f7f7f', '#bcbd22',
                                              '#17becf'])

def axesParam():
    mpl.rcParams['axes.grid'] = False
    mpl.rcParams['axes.formatter.limits'] = -4, 6  # use scientific notation if log10
                                                   # of the axis range is smaller than the
                                                   # first or larger than the second

    mpl.rcParams['axes.formatter.use_mathtext'] = True # When True, use mathtext for scientific notation.

    # Display axis spines, (muestra la linea de los marcos)
    mpl.rcParams['axes.spines.left'] = True
    mpl.rcParams['axes.spines.bottom'] = True
    mpl.rcParams['axes.spines.top'] = True
    mpl.rcParams['axes.spines.right'] = True

    # draw ticks on the top side
    mpl.rcParams['xtick.bottom'] = True
    mpl.rcParams['xtick.top'] = False
    mpl.rcParams['ytick.left'] = True
    mpl.rcParams['ytick.right'] = False

    # draw x axis bottom/top major ticks
    mpl.rcParams['xtick.major.top'] = False
    mpl.rcParams['xtick.major.bottom'] = True
    mpl.rcParams['ytick.major.right'] = False
    mpl.rcParams['ytick.major.left'] = True
    
    # draw x axis bottom/top minor ticks
    mpl.rcParams['xtick.minor.top'] = False
    mpl.rcParams['xtick.minor.bottom'] = False 
    mpl.rcParams['ytick.minor.right'] = False
    mpl.rcParams['ytick.minor.left'] = False 

    # direction: {in, out, inout} señalamiento de los ejes
    mpl.rcParams['xtick.direction'] = 'out'
    mpl.rcParams['ytick.direction'] = 'out'
    
    mpl.rcParams['xtick.minor.visible'] = True # visibility of minor ticks on x-axis


def labelParam():
    mpl.rcParams['xaxis.labellocation'] = 'center'  # alignment of the xaxis label: {left, right, center}
    mpl.rcParams['yaxis.labellocation'] = 'center'  # alignment of the yaxis label: {bottom, top, center}

    # axes numbers, etc.
    # 'large' tamaño de los números de las x, y
    mpl.rcParams['xtick.labelsize'] = 13
    mpl.rcParams['ytick.labelsize'] = 13
    
    # draw label on the top/bottom
    mpl.rcParams['xtick.labeltop'] = False
    mpl.rcParams['xtick.labelbottom'] = True
    mpl.rcParams['ytick.labelright'] = False
    mpl.rcParams['ytick.labelleft'] = True
    
    # labels and title
    mpl.rcParams['axes.titlepad'] = 6.0  # pad between axes and title in points
    mpl.rcParams['axes.labelpad'] = 3.0  # 10.0     # space between label and axis
    mpl.rcParams['axes.labelweight'] = 'normal'  # weight (grosor) of the x and y labels
    mpl.rcParams['axes.labelcolor'] = 'black'
    mpl.rcParams['axes.unicode_minus'] = False  # use Unicode for the minus symbol
    mpl.rcParams['axes.linewidth'] = 1  # edge linewidth, grosor del marco

    mpl.rcParams['axes.titlesize'] = 24  # title size
    mpl.rcParams['axes.labelsize'] = 15  # label size


def legendParam():
    # Legend
    mpl.rcParams['legend.loc'] = 'best'
    mpl.rcParams['legend.frameon'] = True  # if True, draw the legend on a background patch
    mpl.rcParams['legend.framealpha'] = 0.19  # 0.8 legend patch transparency
    mpl.rcParams['legend.facecolor'] = 'inherit'  # inherit from axes.facecolor; or color spec
    mpl.rcParams['legend.edgecolor'] = 'inherit' # background patch boundary color
    mpl.rcParams['legend.fancybox'] = True  # if True, use a rounded box for the

    # mpl.rcParams['legend.numpoints'] = 1 # the number of marker points in the legend line
    # mpl.rcParams['legend.scatterpoints'] = 1 # number of scatter points
    # mpl.rcParams['legend.markerscale'] = 1.0 # the relative size of legend markers vs. original
    mpl.rcParams['legend.fontsize'] = 15  # 'medium' 'large'
    mpl.rcParams['legend.title_fontsize'] = 13  # 'xx-small'

    # Dimensions as fraction of fontsize:
    mpl.rcParams['legend.borderpad'] = 0.4  # border whitespace espacio de los bordes con respecto al texto
    mpl.rcParams['legend.labelspacing'] = 0.5  # the vertical space between the legend entries
    mpl.rcParams['legend.handlelength'] = 1.5  # the length of the legend lines defauld 2
    # mpl.rcParams['legend.handleheight'] = 0.7  # the height of the legend handle
    mpl.rcParams['legend.handletextpad'] = 0.8  # the space between the legend line and legend text
    # mpl.rcParams['legend.borderaxespad'] = 0.5  # the border between the axes and legend edge
    # mpl.rcParams['legend.columnspacing'] = 8.0  # column separation


def fontParam():
    # CONFIGURACIÓN GENERAL
    # latex modo math
    # Should be: 'dejavusans' (default), 'dejavuserif', 'cm' (Computer Modern),
    #             'stix', 'stixsans' or 'custom'
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['mathtext.fallback'] = 'cm'

    # latex modo text
    # Should be: serif, sans-serif, cursive, fantasy, monospace
    mpl.rcParams['font.family'] = 'serif'
    cmfont = font_manager.FontProperties(fname=mpl.get_data_path()
                                         + '/fonts/ttf/cmr10.ttf')
    mpl.rcParams['font.serif'] = cmfont.get_name()
    mpl.rcParams['font.size'] = 16  # size of the text

    # latex
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['pgf.rcfonts'] = False
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{amssymb}'  # , \usepackage{txfonts}
    mpl.rcParams['pgf.preamble'] = r'\usepackage{amssymb}'


###################################
#### Illustrative plot structure 
###################################
def plotUsingPerf(ax, perf, ls='-', 
                  xscale='linear', yscale='linear',
                  color=get_colors()):
    """
    Plot performance profiles on given axes.

    Parameters:
    ax : matplotlib.axes.Axes or array-like of Axes
        Axes object(s) to plot on.
    perf : list or 2D array
        First element (perf[0]) is assumed to be the radial values.
        Remaining elements (perf[1:], indexed by k) are plotted against perf[0].
    """
    
    rad = perf[0]  # Assuming perf[0] contains the x-axis (radius or time values)

    # Ensure ax is iterable
    if not isinstance(ax, (list, np.ndarray)):
        ax = [ax]  # Convert single AxesSubplot into a list
    
    # Check if perf has enough data
    if len(perf) - 1 != len(ax):
        raise ValueError(f"Mismatch: {len(perf)-1} data series but {len(ax)} axes.")

    # Plot each performance metric
    for k, axi in enumerate(ax, start=1):
        axi.plot(rad, perf[k], ls=ls, c=color, label=f'Profile {k}')
        
        axi.set_xlabel(r"$r$")  # Adjust label as needed
        #axi.set_ylabel(f'Profile {k}')
        
        axi.legend(frameon=False)
        axi.set_xscale(xscale)
        axi.set_yscale(yscale)

    return ax

def plotUsingDiscSol(perf1, perf2=None, perf3=None, xscale='linear', yscale='linear'):
    """   
    Plot the solution using discrete solutions.
    
    Note: The first position of perf1 (perf1[0]) is assumed to contain the radii data.
    
    Note: scale: 'linear', 'log', 'symlog', 'logit'
    """
    col = get_colors()
    
    ncol = len(perf1) - 1  # Adjusting for radii data

    if ncol <= 0:  # Safety check
        print("Error: Solution data is empty or invalid.")
        return
    
    _, ax = plt.subplots(ncols=ncol, figsize=(4.5 * ncol, 4.5), gridspec_kw=dict(hspace=0.0, wspace=.28))

    # Ensure ax is a list if there's only one subplot
    if ncol == 1:
        ax = [ax]

    plotUsingPerf(ax, perf1, ls='-', color=col[0], xscale=xscale, yscale=yscale)
    if perf2 is not None:
        plotUsingPerf(ax, perf2, ls='--', color=col[1], xscale=xscale, yscale=yscale)
    if perf3 is not None:
        plotUsingPerf(ax, perf3, ls=':', color=col[2], xscale=xscale, yscale=yscale)

    plt.show()
    return None
    
def plotUsingSol(sol):
    """   
    Plot the solution using a solve_ivp object.
    """
    numFields = len(sol.y)//4  # Integer division
    
    if numFields == 0 or sol.t.size == 0:  # Safety check
        print("Error: Solution data is empty or invalid.")
        return

    _, ax = plt.subplots(figsize=(6, 4.5))

    # Loop through each field
    y0val = []
    for k in range(0, 2 * numFields, 2):
        ax.plot(sol.t, sol.y[k], label=fr'$\sigma_{{{k//2 + 1}}}$')
        y0val.append(sol.y[k][0])

    # ymax and ymin calculations
    ymax = np.max(y0val) if y0val else 0
    ymin = np.min([np.min(sol.y[k]) for k in range(0, 2 * numFields, 2)]) if numFields > 0 else 0

    ax.set_xlim(0, sol.t[-1])
    ax.set_ylim(ymin - 0.02, ymax + 0.02)
    ax.legend(frameon=False)
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$\sigma$")
    
    plt.show()
    return None

def plotPerf(U0, rTmax, arg, px=True, py=True,
             pz=True, lim=True, Rtol=1e-9, Atol=1e-10, met='RK45'):
    """
    Plot performance metrics based on numerical integration results.

    Parameters
    ----------
    U0 : array-like
        Initial conditions for the system.
    rTmax : float
        Maximum value of the independent variable `r`.
    LambT : float
        Parameter for the system dynamics.
    px, py, pz : bool, optional
        Flags to plot results for x, y, z dimensions. Defaults are True.
    lim : bool, optional
        Whether to set dynamic axis limits. Default is True.

    Returns
    -------
    None
    """
    # Numerical integration setup
    rmin, rmax = 0, rTmax
    Nptos = 500
    rspan = np.linspace(rmin, rmax, Nptos)
    
    try:
        solution = solve_ivp(bg.systemMultifrequency, [rmin, rmax], U0, t_eval=rspan,
                             args=(arg,), method=met, rtol=Rtol, atol=Atol)
    except Exception as e:
        print(f"Error during integration: {e}")
        return
    
    # Choosing profiles
    numFields = len(solution.y)//4  # Integer division
    if numFields == 1:
        warnings.warn("The py, pz ccomponents was off because the number of fields is %d"%numFields)
        py, pz = False, False
    elif numFields == 2:
        warnings.warn("The pz component was off because the number of fields is %d"%numFields)
        pz = False
        
    # Extract solution components
    t = solution.t
    y = solution.y

    # Plot setup
    _, ax = plt.subplots(figsize=(6, 4.5))
    
    # Plot dimensions based on flags
    if px:
        ax.plot(t, y[0], color='#1f77b4', label=fr'$\sigma_x^{(0)} = {y[0][0]:.2f}$')
    if py:
        ax.plot(t, y[2], color='#ff7f0e', label=fr'$\sigma_y^{(0)} = {y[2][0]:.2f}$')
    if pz:
        ax.plot(t, y[4], color='k', label=fr'$\sigma_z^{(0)} = {y[4][0]:.2f}$')

    ax.legend(frameon=False)

    # Dynamic axis limits
    if lim:
        ymin = np.min([y[0], y[2]]) - 0.1 * abs(np.min([y[0], y[2]]))
        ymax = np.max([y[0][0], y[2][0]]) + 0.1 * abs(np.max([y[0][0], y[2][0]]))
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(0, t[-1])

    # Additional plot settings
    ax.hlines(y=0, xmin=0, xmax=t[-1], linestyle='--', linewidth=0.5, color='k')
    ax.set_ylabel(r'$\sigma_i^{(0)}$')
    ax.set_xlabel(r'$r$')
    # ax.set_yscale('log')

    # Display the plot
    plt.show()


# https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    # Validate inputs
    x = np.asarray(x)
    y = np.asarray(y)
    c = np.asarray(c)
    
    if len(x) != len(y) or len(x) != len(c):
        raise ValueError("x, y, and c must all have the same length.")
    
    # Check for overridden 'array' in kwargs
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    # Create and add LineCollection
    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)
