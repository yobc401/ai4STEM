#!/bin/python

import sys
import os.path

import numpy as np
import math

import scipy.cluster.hierarchy as hcluster

def hier_clustering(coordinates):
    # clustering
    thresh = 3
    clusters = hcluster.fclusterdata(coordinates, thresh, criterion="distance")
    #print(len(clusters))
    #print(clusters)

    clusters_list = []
    for i in range(max(clusters)):
        neighbors = []
        for j in range(len(clusters)):
            if i + 1 == clusters[j]:
                neighbors.append(coordinates[j])

        clusters_list.append(neighbors)

    #print(clusters_list)
    new_coordinates = []
    for i in range(len(clusters_list)):
        #print(np.mean(clusters_list[i], axis=0))

        new_coordinates.append(np.mean(clusters_list[i], axis=0))

    coordinates = new_coordinates
    
    return coordinates

