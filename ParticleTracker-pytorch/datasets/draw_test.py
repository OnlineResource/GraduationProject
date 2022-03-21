import numpy as np
import pandas as pd
from scipy import stats, integrate
import seaborn as sns
import matplotlib.pyplot as plt
from utils.tif_process import tif_particle

tif_path = 'data/C2-!220118 cos7 wt er endo int2s 015.tif'
x_all = tif_particle(
    tif_path, standardize=True, uint16_min=None, uint16_max=None)
x_all = x_all.reshape(-1)
sns.histplot(x_all,bins=10)
plt.show()
pass