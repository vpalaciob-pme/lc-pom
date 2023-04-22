"""
lcpom : a Python package to simulate polarized optical microscopy images for liquid crystals
Copyright (C) 2023, de Pablo Lab, Prtizker School of Molecular Engineering, University of Chicago.

"""

__title__ = "lcpom"
__name__ = "lcpom"
__author__='Chuqiao Chen (Elise), Viviana Palacio-Betancur, Pablo Zubieta, Prof. Juan de Pablo'
__license__ = "MIT"
__copyright__ = "Copyright (C) 2023, de Pablo Lab, Prtizker School of Molecular Engineering, University of Chicago."
__version__ = '0.0.1'


from lcpom.utils import *

from lcpom.orderfield import *

from lcpom.pom import *

def _isnotebook():
	try:
		shell = get_ipython().__class__.__name__
		# print(shell)
		if shell == 'ZMQInteractiveShell':
			return True   # Jupyter notebook or qtconsole
		elif shell == 'TerminalInteractiveShell':
			return False  # Terminal running IPython
		else:
			return False  # Other type (?)
	except NameError:
		return False      # Probably standard Python interpreter

if not _isnotebook():
	pass
	#import matplotlib
	#matplotlib.use('Qt5Agg')
	#matplotlib.use('TkAgg')
	#matplotlib.rcParams['font.size'] = 18
	#matplotlib.style.use('seaborn')