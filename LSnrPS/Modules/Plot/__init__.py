# modulo/Plot/__init__.py

from .plot_conf import general, FigParam, LineParam, axesParam, labelParam, legendParam, fontParam
from .plot_conf import  get_colors, plotUsingDiscSol, plotUsingPerf, plotPerf, colored_line, plotUsingSol

__all__ = ['general', 'FigParam', 'LineParam', 'axesParam', 'labelParam', 'legendParam', 'fontParam',
           'get_colors', 'plotUsingDiscSol', 'plotUsingPerf', 'plotPerf', 'colored_line', 'plotUsingSol']