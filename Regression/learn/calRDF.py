import sys
import os.path

import numpy as np
from math import sqrt, ceil, floor, pow, pi

def makeRDF(atoms, side, numbins):

    maxbin = numbins # number of bins
    sideh = side/2.0
    dr = float(sideh/maxbin) # bin width
    hist = [0]*(maxbin+1) # initialize list
    rdf = {}
    count = 0


    # loop over particle pairs
    npart = len(atoms)

    for i in range(npart):

        xi = (atoms[i])[0]
        yi = (atoms[i])[1]
        #zi = (atoms[i])[2]

        for j in range(i+1, npart):
            xx = xi - (atoms[j])[0]
            yy = yi - (atoms[j])[1]
            #zz = zi - (atoms[j])[2]

            # minimum image
            if (xx < -sideh):   xx = xx + side
            if (xx > sideh):    xx = xx - side
            if (yy < -sideh):   yy = yy + side
            if (yy > sideh):    yy = yy - side
            #if (zz < -sideh):   zz = zz + side
            #if (zz > sideh):    zz = zz - side

            # distance between i and j
            rd  = xx * xx + yy * yy #+ zz * zz
            rij = sqrt(rd)

            bin = int(ceil(rij/dr)) # determine in which bin the distance falls
            if (bin <= maxbin):
                hist[bin] += 1

    # normalize
    #print("Normalizing ... ")
    phi = npart/pow(side, 2.0) # number area (N*A)
    norm = 2.0 * pi * dr * phi * npart

    for i in range(1, maxbin+1):
        rrr = (i - 0.5) * dr
        val = hist[i]/ norm / ((rrr * rrr) + (dr * dr) / 12.0)
        rdf.update({rrr:val})

    return rdf


