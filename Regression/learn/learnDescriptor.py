
import numpy as np
import math
import calRDF

def calculate_angle_and_dist(atom_columns):
    # write RDF into file
    boxsize = 64.0
    numbins = 20 # number of bins

    rdf = calRDF.makeRDF(atom_columns,boxsize,numbins)
    #print(rdf.keys())
    X = []
    Y = []
    for r in sorted(rdf.keys()): # sort before writing into file
        #print("%15.8g %15.8g\n"%(float(r)*0.1, rdf[r]))
        X.append(float(r)*0.1)
        Y.append(rdf[r])

    #print(X[Y.index(max(Y))])
    min_dist = X[Y.index(max(Y))]

    angle_list = []
    for i in range(len(atom_columns)):

        for j in range(len(atom_columns)):
            a1 = atom_columns[i]
            a2 = atom_columns[j]

            x1 = a1[0]
            y1 = a1[1]

            x2 = a2[0]
            y2 = a2[1]

            if x2 > x1:
                if y2 > y1:
                    dist = math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))*0.1

                    if dist < min_dist*1.8:
                        vec1 = [(x2-x1),(y2-y1)]
                        if vec1[0] > 0:
                            atan_value = math.atan(vec1[1]/vec1[0])
                            angle_list.append((atan_value)*180.0/math.pi)

                        else:
                            angle_list.append(90.0)

    angle_list = sorted(angle_list, key = lambda x:float(x))
    #for x in range(len(angle_list)):
    #    print(angle_list[x])
 
    min_angle = angle_list[0]

    return min_angle, min_dist
