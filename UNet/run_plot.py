import os

gpu = 3
epochs = 16000
bound = False
loss = 'ssim'
num = 15
norm = True
os.system("python plot.py %d %d %s %s %d %s"%(gpu, epochs, bound, loss, num, norm))
#os.system("python run_imgs.py %d %d %s %s"%(gpu, epochs, bound, loss))
#os.system("python run_imgs_new.py %d %d %s %s %d -> .temp_gpu_%d_epoch_%d_bound_%s_%s_num_%d"%(gpu, epochs, bound, loss, num, gpu, epochs, bound, loss, num))