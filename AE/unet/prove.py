import os
import numpy as np
import tensorflow as tf
from scipy.signal import convolve2d
from tensorflow.signal import rfft2d, fftshift, ifftshift, irfft2d, fft2d, ifft2d



os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def conv_tf(last_layer, psf):
        last_layer = tf.cast(last_layer, dtype=tf.float64)
        psf = tf.cast(psf, dtype=tf.float64)

        last_layer = tf.transpose(last_layer, perm=[0, 3, 1, 2])
        psf = tf.transpose(psf, perm=[0, 3, 1, 2])

        s1 = tf.convert_to_tensor(tf.shape(last_layer)[-2:])
        s2 = tf.convert_to_tensor(tf.shape(psf)[-2:])
        shape1 = s1 + s1 - 1
        shape2 = s2 + s2 - 1
        sp1 = tf.signal.rfft2d(last_layer, shape1)
        sp2 = tf.signal.rfft2d(psf, shape2)
        sp1xsp2 = sp1 * sp2
        ret = irfft2d(sp1 * sp2, shape2)
        currshape = tf.shape(ret)[-2:]
        startind = (currshape - s1) // 2
        endind = startind + s1
        output_temp = ret[..., startind[0]:endind[0], startind[1]:endind[1]]
        result = tf.transpose(output_temp, perm=[0, 2, 3, 1])

        
        return result

last_layer = np.random.random((1, 64, 64, 1))#.astype("float128")
psf = np.random.random((1, 64, 64, 1))#.astype("float128")

data1 = conv_tf(last_layer, psf)[0, :, :, 0]
data2 = convolve2d(last_layer[0, :, :, 0], psf[0, :, :, 0], 'same')
data1 = np.round(data1, 14)
data2 = np.round(data2, 14)
print(data1[0] == data2[0])
