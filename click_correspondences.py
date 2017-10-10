'''
  File name: click_correspondences.py
  Author: 
  Date created: 
'''

'''
  File clarification:
    Click correspondences between two images
    - Input im1: target image
    - Input im2: source image
    - Output im1_pts: correspondences coordiantes in the target image
    - Output im2_pts: correspondences coordiantes in the source image
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import scipy.ndimage as nd
from PIL import Image
from cpselect import cpselect
import pickle
import sys
import os
import scipy.misc


def click_correspondences(im1, im2):
    '''
    Tips:
      - use 'matplotlib.pyplot.subplot' to create a figure that shows the source and target image together
      - add arguments in the 'imshow' function for better image view
      - use function 'ginput' and click correspondences in two images in turn
      - please check the 'ginput' function documentation carefully
        + determine the number of correspondences by yourself which is the argument of 'ginput' function
        + when using ginput, left click represents selection, right click represents removing the last click
        + click points in two images in turn and once you finish it, the function is supposed to 
          return a NumPy array contains correspondences position in two images
	'''
    # Read Monica and Rachel images
    monica = np.array(Image.open("Monica.jpg").convert('RGB'))
    rachel = np.array(Image.open("Rachel.jpg").convert('RGB'))

    # Increase recursion limit to increase number of correspondence points
    sys.setrecursionlimit(3000)

    # Implementing cpselect on monica and rachel
    point_monica, point_rachel = cpselect(monica, rachel)

    # Saving the correspondence points as pickle files
    pickle.dump(point_monica, open("save1.p", "w"))
    print point_monica
    pickle.dump(point_rachel, open("save2.p", "w"))
    print point_rachel

    # Return correspondence points of the 2 images
    return point_monica, point_rachel