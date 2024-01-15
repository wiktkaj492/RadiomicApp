import numpy as np
import SimpleITK as sitk
import pandas as pd
import os
from Code.show_data import showData
import matplotlib.pyplot as plt

class Segmentation(object):

    def __init__(self):
        pass
    def area1Mask(self, masks):
        new_mask_sitk = []
        for segmentation_image in masks:
            unique_values = np.unique(segmentation_image)
            print("Unique values in the mask:", unique_values)

            if not np.any(segmentation_image):
                mask_array = np.ones_like(segmentation_image, dtype=np.uint8)
                mask_array[::2, ::2] = 1
            else:
                mask_array = np.where(segmentation_image == 1, 1, 0).astype(np.uint8)

            print("New Unique values in the mask:", np.unique(mask_array))

            mask_image_sitk = sitk.GetImageFromArray(mask_array)
            new_mask_sitk.append(mask_image_sitk)
        return new_mask_sitk


    def area2Mask(self, masks):
        new_mask_sitk = []
        for segmentation_image in masks:
            unique_values = np.unique(segmentation_image)
            print("Unique values in the mask:", unique_values)

            if not np.any(segmentation_image):
                mask_array = np.ones_like(segmentation_image, dtype=np.uint8)
                mask_array[1::2, 1::2] = 2
            else:
                mask_array = np.where(segmentation_image == 2, 2, 0).astype(np.uint8)

            print("New Unique values in the mask:", np.unique(mask_array))

            mask_image_sitk = sitk.GetImageFromArray(mask_array)
            new_mask_sitk.append(mask_image_sitk)
        return new_mask_sitk

    def bothMask(self, masks):
        new_mask_sitk = []
        for segmentation_image in masks:
            unique_values = np.unique(segmentation_image)
            print("Unique values in the mask:", unique_values)

            if not np.any(segmentation_image):
                mask_array = np.ones_like(segmentation_image, dtype=np.uint8)
                mask_array[::2, ::2] = 1
                mask_array[1::2, 1::2] = 2
            else:
                mask_array = np.where(segmentation_image > 0, segmentation_image, 0).astype(np.uint8)
            print("New Unique values in the mask:", np.unique(mask_array))

            mask_image_sitk = sitk.GetImageFromArray(mask_array)
            new_mask_sitk.append(mask_image_sitk)

        return new_mask_sitk
