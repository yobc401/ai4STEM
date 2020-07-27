#!/usr/bin/python3.7

"""
This function draws a scale bar in an image based on the pixel size, typicall in pm and chosen scale bar width
typically in nm.
Input: a un- or processed image
Output: the image with a white scale bar, the image is also saved as high quality .png
"""

# Libraries
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib_scalebar.scalebar import ScaleBar


def scalebar(image, pixsize, file_path):
   # Display the image (using matplotlib) and draw scalebar
    plt.figure(1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
    sb = ScaleBar(pixsize, units='nm', length_fraction=0.2, height_fraction=0.02, location='lower left', pad=0.2, \
        color='w', box_alpha=0, scale_loc='top', label_loc='top', font_properties={'family': 'sans-serif', \
        'weight': 'normal', 'size': 16})
    plt.gca().add_artist(sb)
    plt.savefig(file_path[0:-4] + '_sb.png', format='png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()

    plt.close()
    return image


