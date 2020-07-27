import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

from scipy.signal import get_window

def windowFFT(image,filename):
    #img = cv2.imread(filename,0)

    bw2d = np.outer(get_window('hanning',image.shape[0]), np.ones(image.shape[1]))
    bw2d_1 = np.transpose(np.outer(get_window('hanning',image.shape[1]), np.ones(image.shape[0])))
    w = np.sqrt(bw2d * bw2d_1)

    image = image * w

    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    output = 10*np.log(np.abs(fshift))

    output2=np.zeros((64,64))
    for i in range(0,64):
        for j in range(0,64):
            output2[i,j] = output[int(float(output.shape[0])/float(2.0))-32+i,int(float(output.shape[1])/float(2.0))-32+j]

    if not os.path.exists('Windowed'):
        os.makedirs('Windowed')

    plt.figure()
    plt.imshow(image,cmap='gray')
    #plt.colorbar()
    #plt.draw()
    plt.axis('off')
    plt.savefig('Windowed/' + filename + '_window.png' ,bbox_inches='tight',pad_inches=0)#, dpi=17.5)

    if not os.path.exists('FFT'):
        os.makedirs('FFT')

    plt.figure()
    plt.imshow(output2,cmap='gray')
    #plt.colorbar()
    #plt.draw()
    plt.axis('off')
    plt.savefig('FFT/' + filename + '_fft.png',bbox_inches='tight',pad_inches=0)#, dpi=17.5)
    plt.close()
    
    return output2
