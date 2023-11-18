import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.signal import rfft2d, fftshift, ifftshift, irfft2d, fft2d, ifft2d
from tensorflow.keras.layers import Conv2D, BatchNormalization, Conv2DTranspose, concatenate, MaxPool2D, Activation
from tensorflow.keras.layers import Layer


class FFT2DLayer(Layer):
    def __init__(self, shape):
        super(FFT2DLayer, self).__init__()
        self.shape = shape

    def call(self, inputs):
        return tf.signal.rfft2d(inputs, self.shape)

class IFFT2DLayer(Layer):
    def __init__(self, shape):
        super(IFFT2DLayer, self).__init__()
        self.shape = shape

    def call(self, inputs):
        return tf.signal.irfft2d(inputs, self.shape)

class unet2D(): 
    def __init__(self,n_filters = 16, conv_width=1, 
                 network_depth = 4,
                 n_channels=32, x_dim=32, dropout = 0.0, 
                 growth_factor=2, batchnorm = True, 
                 momentum=0.9, epsilon=0.001,
                 activation='relu',
                 maxpool=True,
                 psf = False
                 ):
        
        self.n_filters = n_filters
        self.n_channels = n_channels
        self.conv_width = conv_width
        self.network_depth = network_depth
        self.x_dim = x_dim
        self.dropout = dropout
        self.growth_factor = growth_factor
        self.batchnorm = batchnorm
        self.momentum = momentum
        self.epsilon = epsilon
        self.activation = activation
        self.maxpool = maxpool
        self.psf = psf
        
        # define all layers
        
    def conv_block(self, input_tensor, n_filters, n_layers=1, strides=1, kernel_size=3, \
                           momentum=0.9, maxpool=False, batchnorm=True, layer_num=None):
        if layer_num is not None:
            if strides > 1:
                name = 'downsample_{}'.format(layer_num)
        else:
            name = None
        
        x = input_tensor       
        
        for _ in range(n_layers):        
            identity = x
            x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
                      padding = 'same', strides=strides, name=name)(x)

            if batchnorm:
                x = BatchNormalization(momentum=momentum)(x)   
            x = Activation(self.activation)(x)
        #    if l > 0:
        #        x = Add()([x, identity])
        #    x = Activation(self.activation)(x)    
        return x           
    def _centered(arr, newshape):
        # Return the center newshape portion of the array.
        currshape = tf.shape(arr)[-2:]
        startind = (currshape - newshape) // 2
        endind = startind + newshape
        return arr[..., startind[0]:endind[0], startind[1]:endind[1]]
    
    def fftconv(in1, in2, mode="same"):
        # Reorder channels to come second (needed for fft)
        complex_result = (in1.dtype.is_complex or in2.dtype.is_complex)


        in1 = tf.transpose(in1, perm=[0, 3, 1, 2])
        in2 = tf.transpose(in2, perm=[0, 3, 1, 2])

        # Extract shapes
        s1 = tf.convert_to_tensor(tf.shape(in1)[-2:])
        s2 = tf.convert_to_tensor(tf.shape(in2)[-2:])
        shape = s1 + s2 - 1

        if not complex_result:
            fft, ifft = rfft2d, irfft2d
        else:
            fft, ifft = fft2d, ifft2d
        print(100*"tt")
        print(s1, s2, shape)
        print(100*"tt")
        # Compute convolution in fourier space
        sp1 = fft(in1, shape)
        sp2 = fft(in2, shape)
        ret = ifft(sp1 * sp2, shape)

        # Crop according to mode
        if mode == "full":
            cropped = ret
        elif mode == "same":
            cropped = _centered(ret, s1)
        elif mode == "valid":
            cropped = _centered(ret, s1 - s2 + 1)
        else:
            raise ValueError("Acceptable mode flags are 'valid',"
                         " 'same', or 'full'.")

        # Reorder channels to last
        result = tf.transpose(cropped, perm=[0, 2, 3, 1])
        return result

    def build_model(self):
        """
        Function to build network with specified architecture parameters
        """
        network_depth = self.network_depth
        n_filters = self.n_filters
        growth_factor = self.growth_factor
        momentum = self.momentum

        ## Start with inputs
        inputs = keras.layers.Input(shape=(self.x_dim, self.x_dim, self.n_channels),name="image_input")
        x = inputs
        concat_down = []
        
        for h in range(network_depth):
            x = self.conv_block(x, n_filters, n_layers=self.conv_width,strides=1) 
            concat_down.append(x)
            n_filters *= growth_factor
            x = self.conv_block(x, n_filters, n_layers=1, batchnorm=True, strides=2, 
                                    maxpool=self.maxpool, layer_num=h+1)
        
        concat_down = concat_down[::-1]  
        x = self.conv_block(x, n_filters, n_layers=self.conv_width, strides=1)
        
        n_filters //= growth_factor
        for h in range(network_depth):
            n_filters //= growth_factor
            x = Conv2DTranspose(n_filters, kernel_size=3, strides=2, padding='same')(x)
            x = BatchNormalization(momentum=momentum, epsilon=self.epsilon)(x)
            x = Activation(self.activation)(x)
            x = concatenate([x, concat_down[h]])
            x = self.conv_block(x, n_filters, n_layers=self.conv_width, kernel_size=3, 
                                        strides=1, momentum=self.momentum)   
            
        last_layer = Conv2DTranspose(self.n_channels,1,padding="same",name="last_layer")(x)
        last_layer = tf.transpose(last_layer, perm=[0, 3, 1, 2])
        psf = tf.transpose(self.psf, perm=[0, 3, 1, 2])

        s1 = tf.convert_to_tensor(tf.shape(last_layer)[-2:])
        s2 = tf.convert_to_tensor(tf.shape(psf)[-2:])
        shape1 = s1 + s1 - 1
        shape2 = s2 + s2 - 1
        sp1 = FFT2DLayer(shape2)(last_layer)  # 从模型出来的late_layer输出的是封装好的, 此处要用rfft2d需要封装
        sp2 = tf.signal.rfft2d(psf, shape2)
        sp1xsp2 = sp1 * sp2
        ret = IFFT2DLayer(shape2)(sp1xsp2)  # 同上述原因
        currshape = tf.shape(ret)[-2:]
        startind = (currshape - s1) // 2
        endind = startind + s1
        output_temp = ret[..., startind[0]:endind[0], startind[1]:endind[1]]
        result = tf.transpose(output_temp, perm=[0, 2, 3, 1])

        model = keras.models.Model(inputs=inputs,outputs=result)
        return model
