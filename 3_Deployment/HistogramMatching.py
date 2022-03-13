# Name: HistogramMatching.py
# Description: Adjusting colors based on histogram matching

"""Installing libraries"""
!pip install rio-hist
!pip install imagecodecs

"""Installing packages"""
from rio_hist.match import histogram_match
import tifffile
import os
import numpy as np


path_ref = "D:/Model/all_labeled_uint16/2020_2661711.0196_1231867.9929_FID_5521_20cm.tif"
reference = tifffile.imread(path_ref).astype("uint8") # reference image

copy_to_path = "D:/Model/all_labeled_uint8_hm/" # new destination path

for image in os.listdir("D:/Model/all_labeled_uint16"):
        img = tifffile.imread('D:/Model/all_labeled_uint16/'+image).astype("uint8") # source image we want to alter
        img_new = histogram_match(source=img,
                                  reference=reference,
                                  match_proportion=0.7).astype(np.uint8)
        tifffile.imsave(copy_to_path+image, img_new)