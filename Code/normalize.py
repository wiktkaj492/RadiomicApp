import os
import sys
import numpy as np
import SimpleITK as sitk
import pandas as pd

class Normalize(object):

    def __init__(self):
        pass

    def minMaxNormalization(self, roiImage, roiMask):
        # Convert the mask to boolean for indexing
        roiImage = np.array(roiImage)
        roiMask = np.array(roiMask)

        mask_bool = roiMask.astype(bool)

        # Extract the region of interest based on the mask
        roi_values = roiImage[mask_bool]

        # Find the unique gray levels in the region of interest
        unique_levels = np.unique(roi_values)
        n = len(unique_levels)

        # Calculate the minimum and maximum values within the masked region
        min_val = roi_values.min()
        max_val = roi_values.max()

        # Check if there are at least two unique levels to avoid division by zero
        if min_val == max_val:
            raise ValueError("The region of interest has only one unique gray level.")

        # Apply the normalization formula to the region of interest
        roiImage[mask_bool] = (2 ** n) * ((roiImage[mask_bool] - min_val) / (max_val - min_val) + 1)
        return roiImage

    def meanStdNormalization(self):
        return

    def percentileNormalization(self):
        return