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


gpu = int(sys.argv[1]); epochs = int(sys.argv[2]);
bound = str(sys.argv[3]); loss = str(sys.argv[4]);
os.environ['CUDA_VISIBLE_DEVICES'] = '%d'%gpu
img_h = 2048; dims = 1; rate = 255; batch_size = 2
psf = np.load("data/psf.npy").astype("float32"); psf = psf/np.sum(psf)


header = fits.getheader("data/WLM_galex_FUV_int_crop.fits")
header['NAXIS1'] = 2048
header['NAXIS2'] = 2048
wcs_helix = WCS(header)

print(100*"*")
print("gpu: ", gpu)
print("epochs: ", epochs)
print("learning rate bound: ", bound)
print("loss: ", loss)
print(100*"*")

nums = [int(sys.argv[5])]
norm = str(sys.argv[6])

label_fs = 10
title_fs = 12
for i in range(3):
    for num in nums:
        dirty_nosm = np.load("../Algorithm_Gen_Data/results/image_%d_%d.npy"%(num, i)) 
        if norm:
            dirty_nosm = 255*((dirty_nosm - np.min(dirty_nosm)) / (np.max(dirty_nosm) - np.min(dirty_nosm)))

        dirty_nosm = np.load("results_%s/dirty_nosm_num%d.%d_epochs-%d_bound-%s_loss-%s_norm%s.npy"%(loss, num, i, epochs, bound, loss, norm))
        dirty_sm = np.load("results_%s/dirty_sm_num%d.%d_epochs-%d_bound-%s_loss-%s_norm%s.npy"%(loss, num, i, epochs, bound, loss, norm))
        data_pred2 = np.load("results_%s/data_pred2_num%d.%d_epochs-%d_bound-%s_loss-%s_norm%s.npy"%(loss, num, i, epochs, bound, loss, norm))

        if True:
            vmin_dirty = 0; vmax_dirty = 1
            fig = plt.figure(figsize=(12, 4))
            fig.subplots_adjust(hspace=0.2, wspace=0.5)
            plt.subplot(131, projection=wcs_helix)
            plt.imshow(dirty_sm[0, :, :, 0], norm='log')#, origin='lower', cmap='cividis', aspect='equal')#, vmin=vmin_dirty, vmax=vmax_dirty)
            plt.colorbar(fraction=0.045, pad=0.05)
            plt.xticks([]); plt.yticks([]); plt.xlabel("RA", fontsize=label_fs); plt.ylabel("Dec", fontsize=label_fs)
            plt.title("Blurring Image", fontsize=title_fs)

            plt.subplot(132, projection=wcs_helix)
            plt.imshow(data_pred2[0, :, :, 0], norm='log')
            plt.colorbar(fraction=0.045, pad=0.05)
            plt.xticks([]); plt.yticks([]); plt.xlabel("RA", fontsize=label_fs); plt.ylabel("Dec", fontsize=label_fs)
            plt.title("Deblurring Image", fontsize=title_fs)

            plt.subplot(133, projection=wcs_helix)
            plt.imshow(dirty_nosm, norm='log')#, origin='lower', cmap='cividis', aspect='equal')#, vmin=vmin_dirty, vmax=vmax_dirty)
            plt.colorbar(fraction=0.045, pad=0.05)
            plt.xticks([]); plt.yticks([]); plt.xlabel("RA", fontsize=label_fs); plt.ylabel("Dec", fontsize=label_fs)
            plt.title("Ground Truth", fontsize=title_fs)

            plt.show()
