#!/bin/python

import sys
import os.path

import numpy as np
import math

from matplotlib import pyplot as plt

import makeScalebar
import makeWindowingFFT


def localwindow(image_in,stride_size,pixel_max):
    x_max = image_in.shape[0]
    y_max = image_in.shape[1]

    i = 0
    ni = 0
    while i < x_max-pixel_max:
        j = 0
        nj = 0
        ni = ni + 1

        while j < y_max-pixel_max:
            nj = nj + 1
            image = np.zeros((pixel_max,pixel_max))
            for x in range(0,pixel_max):
                for y in range(0,pixel_max):
                    image[x,y] = image_in[x+i,y+j] 

            filename = "local_images_" + str(ni) + "_" + str(nj)
            plt.figure()
            plt.imshow(image,cmap='gray')
            #plt.colorbar()
            #plt.draw()
            plt.axis('off')
            plt.savefig(filename + '.png',bbox_inches='tight',pad_inches=0)

            makeWindowingFFT.windowFFT(image,filename)

            j += stride_size[1]


        i += stride_size[0]

