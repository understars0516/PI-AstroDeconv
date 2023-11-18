# PI-AstroDeconv(Updating)

We present a straightforward and efficient approach for blind image deblurring by integrating the principles of telescope imaging and deep learning. According to telescope imaging principles, when a signal is observed through a telescope, it undergoes smoothing due to the telescope's beam or PSF, resulting in signal blurring. To tackle this issue, we incorporate a convolution kernel resembling a telescope's beam or PSF into a conventional network. We then perform a deconvolution process on the intermediate outputs to restore image clarity. The proposed model is trained end-to-end and comprehensively evaluated using both simulated and real-world image blur scenarios. Our extensive experimental results demonstrate a remarkable enhancement in deblurring performance achieved by the proposed De_Conv deconvolution network.

![De_Beam](https://github.com/understars0516/PI-AstroDeconv/assets/32385394/fa328a9d-a1ac-4cea-a297-9e1df04bace7)



### Data Generator:
image: download from [WEBB](https://webbtelescope.org/images), reshape the image to (2048, 2048).

![image](https://github.com/understars0516/PI-AstroDeconv/assets/32385394/be88243d-a4d2-43bc-9d28-da3b86629e2c)



psf: simulation by [webbpsf](https://webbpsf.readthedocs.io/en/latest/usage.html)

![image](https://github.com/understars0516/PI-AstroDeconv/assets/32385394/ede23454-ce15-45e6-a2b8-ec248ff1f74a)


simulation data $image \otimes psf$ : 

![image](https://github.com/understars0516/PI-AstroDeconv/assets/32385394/21734dd9-26e1-4cc7-83eb-4b9bce708131)


### ae: 
based AutoEncoder model
### U-Net:
based U-Net model


# Requirements
Main requirements: 
tensorflow=2.12.1, astropy=5.3.3, pspy=1.6.4, pixell=0.19.3
```bash
conda create -n env_name python=3.11
conda install tensorflow-gpu=2.12.1
```
