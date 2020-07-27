import numpy as np

def mark_figure(image,atom_columns):
    new_image = np.zeros((64,64,3),'uint8')

    for x in range(0,64):
        for y in range(0,64):
            for ch in range(0,3):
                new_image[x,y,ch] = image[x,y] #* 255

    for i in range(len(atom_columns)):         
        x = float(atom_columns[i][0])
        y = float(atom_columns[i][1])

        new_image[int(x),int(y),0] = 255.0
        new_image[int(x),int(y),1] = 0.0
        new_image[int(x),int(y),2] = 0.0

    return new_image

