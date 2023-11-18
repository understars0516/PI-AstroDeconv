import numpy as np
import tensorflow as tf
from unet import unet_2d_fft_best
import os, sys, cv2
import matplotlib.pyplot as plt
from astropy.convolution import convolve_fft
from scipy.signal import fftconvolve
from tensorflow.image import ssim as ssim_cal
from tensorflow.image import psnr as psnr_cal
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
psf = np.load("../data/psf.npy").astype("float32"); psf = psf/np.sum(psf)


header = fits.getheader("../data/WLM_galex_FUV_int_crop.fits")
header['NAXIS1'] = 2048
header['NAXIS2'] = 2048
wcs_helix = WCS(header)

print(100*"*")
print("gpu: ", gpu)
print("epochs: ", epochs)
print("learning rate bound: ", bound)
print("loss: ", loss)
print(100*"*")

def norm(img, norm=True):
    img = img.reshape(2048, 2048, 1).astype('float32')
    if norm:
        img = 255*(img - np.min(img)) / (np.max(img) - np.min(img))
    return img

def ssim(y_true, y_pred):
    y_true_float = tf.image.convert_image_dtype(y_true, tf.float32)
    y_pred_float = tf.image.convert_image_dtype(y_pred, tf.float32)

    ssim_val = tf.image.ssim(y_true_float, y_pred_float, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)

    loss = 1 - tf.reduce_mean(ssim_val)

    return loss


if bound == 'True':
    boundaries = [1000, 2000, 4000, 8000]
    values = [0.01, 0.001, 0.0001, 0.00005, 0.00001]
    lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
else:
    lr = 0.001

nums = [int(sys.argv[5])]
norm = str(sys.argv[6])
dirtys = []
for i in range(3):
    for num in nums:
        dirty_temp = np.load("../data/image_rgb.npy"[i])
        print(dirty_temp.shape)
        if norm:
            dirty_temp = 255*((dirty_temp - np.min(dirty_temp)) / (np.max(dirty_temp) - np.min(dirty_temp)))
        dirtys.append(fftconvolve(dirty_temp, psf, 'same'))
#dirty_temp = cv2.imread("../Algorithm_Gen_Data/results/image_18.png")
#print(dirty_temp.shape)
#dirtys = np.transpose(dirty_temp, [2, 0, 1])
#dirtys = []
#for i in range(24):
#    dirty_temp = np.load("../Algorithm_Gen_Data/results/image_%d.npy"%i)
#    dirty_ = 255*((dirty_temp - np.min(dirty_temp)) / (np.max(dirty_temp) - np.min(dirty_temp)))
#    dirtys.append(fftconvolve(dirty_, psf, 'same'))

dirtys = np.array(dirtys).astype("float32"); dirtys = np.expand_dims(dirtys, -1)

psf = psf.reshape(1, img_h, img_h, 1).astype('float32')

print(10*"*")
print(dirtys.shape, psf.shape)
print(10*"*")


net = unet_2d_fft_best.unet2D(n_filters=32, conv_width=3, network_depth=3, n_channels=1, x_dim=img_h, dropout=0.2, growth_factor=2, batchnorm=True, momentum=0.9, epsilon=0.001, activation='relu', maxpool=True, psf=psf)
net = net.build_model()

if loss == 'ssim':
    net.compile(optimizer=tf.optimizers.Adam(learning_rate = lr, beta_1=0.9, beta_2=0.999), loss=ssim, metrics=['mae', 'mse'])
else:
    net.compile(optimizer=tf.optimizers.Adam(learning_rate = lr, beta_1=0.9, beta_2=0.999), loss='logcosh', metrics=['mae', 'mse'])

history = net.fit(dirtys, dirtys, epochs=epochs, batch_size=batch_size)
rgb = 'b'
if not os.path.exists("results_%s"%loss):
    os.makedirs("results_%s"%loss)
np.save("results_%s/train_loss_%s_epochs-%d_bound-%s_loss-%s.npy"%(loss, rgb, epochs, bound, loss), history.history['loss'])

layer_outputs = [layer.output for layer in net.layers]
_model = tf.keras.models.Model(inputs=net.input, outputs=layer_outputs)

label_fs = 10
title_fs = 12
ssim_sm = []; ssim_pred = []
psnr_sm = []; psnr_pred = []
labels = []
for i in range(3):
    for num in nums:
        labels.append('Image No.%d %d'%(num, i))
        dirty_nosm = np.load("../data/image_rgb.npy"[i])
        if norm:
            dirty_nosm = 255*((dirty_nosm - np.min(dirty_nosm)) / (np.max(dirty_nosm) - np.min(dirty_nosm)))
        psf = np.load("../data/psf.npy").astype("float32")
        dirty_sm = fftconvolve(dirty_nosm, psf, 'same').reshape(1, img_h, img_h, 1)
        pred_test = net.predict(dirty_sm)
        pred = _model.predict(dirty_sm)
        data_pred1 = pred[-1]
        data_pred2 = pred[-19]

        np.save("results_%s/dirty_nosm_num%d.%d_epochs-%d_bound-%s_loss-%s_norm%s.npy"%(loss, num, i, epochs, bound, loss, norm), dirty_nosm)
        np.save("results_%s/dirty_sm_num%d.%d_epochs-%d_bound-%s_loss-%s_norm%s.npy"%(loss, num, i, epochs, bound, loss, norm), dirty_sm)
        np.save("results_%s/pred_test_num%d.%d_epochs-%d_bound-%s_loss-%s_norm%s.npy"%(loss, num, i, epochs, bound, loss, norm), pred_test)
        np.save("results_%s/data_pred1_num%d.%d_epochs-%d_bound-%s_loss-%s_norm%s.npy"%(loss, num, i, epochs, bound, loss, norm), data_pred1)
        np.save("results_%s/data_pred2_num%d.%d_epochs-%d_bound-%s_loss-%s_norm%s.npy"%(loss, num, i, epochs, bound, loss, norm), data_pred2)

        #print(100*"*")
        #print("Smoothed %d Image(SSIM): "%num, ssim_cal(norm(dirty_nosm), norm(dirty_sm), max_val = 255).numpy())
        #print("Predicted %d Image(SSIM): "%num, ssim_cal(norm(dirty_nosm), norm(data_pred2), max_val = 255).numpy())
        #print(100*"*")
        #print("Smoothed %d Image(PSNR): "%num, psnr_cal(norm(dirty_nosm), norm(dirty_sm), max_val = 255).numpy())
        #print("Predicted %d Image(PSNR): "%num, psnr_cal(norm(dirty_nosm), norm(data_pred2), max_val = 255).numpy())
        #print(100*"*")
        #print()
        #ssim_sm.append(ssim_cal(norm(dirty_nosm), norm(dirty_sm), max_val = 255).numpy())
        #ssim_pred.append(ssim_cal(norm(dirty_nosm), norm(data_pred2), max_val = 255).numpy())
        #psnr_sm.append(psnr_cal(norm(dirty_nosm), norm(dirty_sm), max_val = 255).numpy())
        #psnr_pred.append(psnr_cal(norm(dirty_nosm), norm(data_pred2), max_val = 255).numpy())
        if False:
            vmin_dirty = 0; vmax_dirty = 1
            fig = plt.figure(figsize=(8, 8))
            fig.subplots_adjust(hspace=0.2, wspace=0.75)
            plt.subplot(221)
            plt.imshow(dirty_sm[0, :, :, 0])#, vmin=vmin_dirty, vmax=vmax_dirty)
            plt.colorbar()
            plt.xticks([]); plt.yticks([]); plt.xlabel("RA"); plt.ylabel("Dec")
            plt.title("Input")

            plt.subplot(222)
            plt.imshow(pred_test[0, :, :, 0])#, vmin=vmin_dirty, vmax=vmax_dirty)
            plt.colorbar()
            plt.xticks([]); plt.yticks([]); plt.xlabel("RA"); plt.ylabel("Dec")
            plt.title("Output")

            plt.subplot(223)
            plt.imshow(dirty_nosm)#, vmin=vmin_dirty, vmax=vmax_dirty)
            plt.colorbar()
            plt.xticks([]); plt.yticks([]); plt.xlabel("RA"); plt.ylabel("Dec")
            plt.title("Ground Truth")

            plt.subplot(224)
            plt.imshow(data_pred2[0, :, :, 0])#, vmin=vmin_dirty, vmax=vmax_dirty)
            plt.colorbar()
            plt.xticks([]); plt.yticks([]); plt.xlabel("RA"); plt.ylabel("Dec")
            plt.title("Predict")

            plt.savefig("results_%s/data_pred2_num%d.%d_epochs-%d_bound-%s_loss-%s.png"%(loss, num, i, epochs, bound, loss))
    #        plt.show()
        if True:
            vmin_dirty = 0; vmax_dirty = 1
            fig = plt.figure(figsize=(12, 4))
            fig.subplots_adjust(hspace=0.2, wspace=0.5)
            plt.subplot(131, projection=wcs_helix)
            plt.imshow(dirty_sm[0, :, :, 0])#, origin='lower', cmap='cividis', aspect='equal')#, vmin=vmin_dirty, vmax=vmax_dirty)
            plt.colorbar(fraction=0.045, pad=0.05)
            plt.xticks([]); plt.yticks([]); plt.xlabel("RA", fontsize=label_fs); plt.ylabel("Dec", fontsize=label_fs)
            plt.title("Blurring Image", fontsize=title_fs)

            plt.subplot(132, projection=wcs_helix)
            plt.imshow(data_pred2[0, :, :, 0])
            plt.colorbar(fraction=0.045, pad=0.05)
            plt.xticks([]); plt.yticks([]); plt.xlabel("RA", fontsize=label_fs); plt.ylabel("Dec", fontsize=label_fs)
            plt.title("Deblurring Image", fontsize=title_fs)

            plt.subplot(133, projection=wcs_helix)
            plt.imshow(dirty_nosm)#, norm='log')#, origin='lower', cmap='cividis', aspect='equal')#, vmin=vmin_dirty, vmax=vmax_dirty)
            plt.colorbar(fraction=0.045, pad=0.05)
            plt.xticks([]); plt.yticks([]); plt.xlabel("RA", fontsize=label_fs); plt.ylabel("Dec", fontsize=label_fs)
            plt.title("Ground Truth", fontsize=title_fs)

            plt.savefig("results_%s/data_pred2_gpu%d_num%d.%d_epochs-%d_bound-%s_loss-%s_norm%s.png"%(loss, gpu, num, i, epochs, bound, loss, norm), bbox_inches='tight', pad_inches=0, dpi=300)
            #plt.show()

#df = pd.DataFrame({"SSIM Smoothed Image": ssim_sm, "SSIM Pred Image": ssim_pred, "PSNR Smooth Image": psnr_sm, "PSNR Pred Image": psnr_pred})
#df.insert(0, 'Image', labels)
#df.to_csv("output_gpu%d_num%d_epochs-%d_bound-%s_loss-%s.csv"%(gpu, num, epochs, bound, loss), index=False)
