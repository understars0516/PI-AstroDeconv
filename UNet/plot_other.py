import numpy as np
import os, sys, cv2
import matplotlib.pyplot as plt
from astropy.convolution import convolve_fft
from scipy.signal import fftconvolve
from astropy.wcs import WCS
from astropy.io import fits
import pandas as pd

import pspy, pixell
from pspy.so_config import DEFAULT_DATA_DIR
pixell.colorize.mpl_setdefault("planck")


header = fits.getheader("data/WLM_galex_FUV_int_crop.fits")
header['NAXIS1'] = 2048
header['NAXIS2'] = 2048
wcs_helix = WCS(header)

names = os.listdir("/home/nisl/npys")

label_fs = 10
title_fs = 12

for name in names:
    data = np.load("/home/nisl/npys/" + name).reshape(2048, 2048)
    print(data.shape)
    plt.subplot(111, projection=wcs_helix)
    plt.imshow(data, norm='log')#, origin='lower', cmap='cividis', aspect='equal')#, vmin=vmin_dirty, vmax=vmax_dirty)
    plt.colorbar(fraction=0.045, pad=0.05)
    plt.xticks([]); plt.yticks([]); plt.xlabel("RA", fontsize=label_fs); plt.ylabel("Dec", fontsize=label_fs)
    plt.title("Ground Truth", fontsize=title_fs)
    plt.savefig("/home/nisl/figs/%s"%name[0:-4])
    plt.close()
