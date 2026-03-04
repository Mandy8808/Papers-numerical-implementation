# modulo/__init__.py

from .Background.background import system, systemMultifrequency, systemMultFreqTot, MatrizDXDu
from .Background.background import mainExt, extend, sParam, Asy_uProf, Asy_sProf, Asy_sProf_v2, extend_Met1, extend_Met2
from .Background.background import shoot, freq_shoot, identify, MultFreq_solveG2, fitting, algebSyst
from .Background.background import profilesFromSolut

from .Plot.plot_conf import general, FigParam, LineParam, axesParam, labelParam, legendParam, fontParam
from .Plot.plot_conf import  get_colors, plotUsingDiscSol, plotUsingPerf, plotPerf, colored_line, plotUsingSol

from .EnergyMass.energy_mass import energEng, massVal

from .SpectralMeth.spectralMet import cheb, chevQuant, backgroundOper, linBlock, circBlock, radBlock, MultMii, MultMij
from .SpectralMeth.spectralMet import  Organize, Reference_row, Organize_row, LamJval
from .SpectralMeth.spectralMet import  multBlock, espectro, LamJval, sep, progressbar, plotImag

#from .SpectralMeth import check1, errorDic, SigInt, Findinterv

__all__ = [
    'system', 'systemMultifrequency', 'systemMultFreqTot', 'MatrizDXDu',
    'mainExt', 'extend', 'sParam', 'Asy_uProf', 'Asy_sProf', 'Asy_sProf_v2', 'extend_Met1', 'extend_Met2',
    'shoot', 'freq_shoot', 'identify', 'MultFreq_solveG2', 'fitting', 'algebSyst', 'profilesFromSolut',
    #
    'general', 'FigParam', 'LineParam', 'axesParam', 'labelParam', 'legendParam', 'fontParam',
    'get_colors', 'plotUsingDiscSol', 'plotUsingPerf', 'plotPerf', 'colored_line', 'plotUsingSol',
    #
    'energEng', 'massVal',
    #
    'cheb', 'chevQuant', 'backgroundOper', 'linBlock', 'circBlock', 'radBlock', 'MultMii', 'MultMij',
    'Organize', 'Reference_row', 'Organize_row', 'LamJval',
    'multBlock', 'espectro', 'LamJval', 'sep', 'progressbar', 'plotImag']