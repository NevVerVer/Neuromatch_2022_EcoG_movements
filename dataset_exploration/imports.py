print("Importing modules...")

# essentials
import os
import sys
import time
import warnings
from tqdm import tqdm

# science
import numpy as np
import pandas as pd
import scipy as sp


# for neural data
import mne
import nilearn
from scipy import signal, stats

# for nwb files and AJILE dataset
from brunton_lab_to_nwb.brunton_widgets import BruntonDashboard
from dandi.dandiapi import DandiAPIClient
from pynwb import NWBHDF5IO

import dandi

# plotting
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import rc
#rc('font',**{'family':'sans-serif'})
#rc('text', usetex=True)
#rc('text.latex',preamble=r'\usepackage[utf8]{inputenc}')
#rc('text.latex',preamble=r'\usepackage[russian]{babel}')
#rc('axes', **{'titlesize': '16', 'labelsize': '16'})
#rc('legend', **{'fontsize': '16'})
#rc('figure', **{'figsize': (12, 8)})

warnings.filterwarnings("ignore")
print("Module import successful.")