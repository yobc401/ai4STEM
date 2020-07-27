#!/bin/python

import sys
import os.path

import numpy as np
import math

from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture


def seperate_background(image,pixel_max):

        hist, bin_edges = np.histogram(image, bins=30)
        bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])

        classif = GaussianMixture(n_components=2)
        classif.fit(image.reshape((image.size, 1)))

        threshold = np.mean(classif.means_)

        new_image = np.zeros((pixel_max,pixel_max))
        for x in range(0,pixel_max):
            for y in range(0,pixel_max):
                if image[x,y] > threshold:
                    new_image[x,y] = image[x,y]

        image = new_image

        return image

