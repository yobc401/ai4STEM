import cv2
import makeScalebar
import makeWindowingFFT
import saveLocalwindow

import numpy as np
from PIL import Image
import math
from skimage.feature import peak_local_max
from matplotlib import pyplot as plt
import separateBackGMM, hierCluster, learnDescriptor

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

def segment(filename,hyperparameters):

    dx_origin = 0.004 # length value of input image per pixel (Unit: Angstrom)
    dx_new = 0.02 # length value of new image per pixel (Unit: Angstrom)
    stride_size = [64,64]
    pixel_max = 64

    image = cv2.imread(filename)
    image = image[:, :, 0]
    (Nx, Ny) = image.shape

    new_Nx = int(float(Nx)*float(dx_origin)/float(dx_new))
    new_Ny = int(float(Ny)*float(dx_origin)/float(dx_new))

    image = cv2.resize(image, dsize=(new_Nx, new_Ny), interpolation=cv2.INTER_AREA)
    image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    makeScalebar.scalebar(image, dx_new, filename)

    saveLocalwindow.localwindow(image, stride_size, pixel_max)

    num_X = int(new_Nx / stride_size[0])
    num_Y = int(new_Ny / stride_size[1])

    return num_X, num_Y

def learn(filename):

    image = Image.open(filename)
    size = (64,64)
    image.thumbnail(size)
    image = np.array(image,dtype=float)
    image = np.mean(image,axis=2)
    
    image = separateBackGMM.seperate_background(image,64)

    atom_columns = peak_local_max(image, min_distance=1)
    atom_columns = hierCluster.hier_clustering(atom_columns)

    angle, dist = learnDescriptor.calculate_angle_and_dist(atom_columns)

    return angle, dist

def map_from_dist_to_latConst(lattice_plane, dist):

    filename = 'data_points_' + lattice_plane + '.dat'
    with open(filename, 'r') as f:
        matrix = [[float(num) for num in line.split(' ')] for line in f]

    matrix = np.array(matrix)
    X = np.array(matrix[:,0])
    X = np.reshape(X,(X.shape[0],1))
    y = np.array(matrix[:,1])
    y = y.transpose()

    regressor = SVR(kernel='linear')#, C=100, gamma='auto')
    regressor.fit(X,y)

    y_pred = regressor.predict([[dist]])

    return y_pred[0]
