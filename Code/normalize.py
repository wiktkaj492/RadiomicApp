import os
import sys
import numpy as np
import SimpleITK as sitk
import pandas as pd
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod


# Strategy Interface
class NormalizationStrategy(ABC):

    @abstractmethod
    def normalize(self, roiImage, roiMask):
        pass

    def save_debug_images(self, masks, filenames, folder="debug_images"):
        # Create the folder if it doesn't exist
        if not os.path.exists(folder):
            os.makedirs(folder)

        for mask, filename in zip(masks, filenames):
            # Normalize and map the mask to a colormap for better visibility
            plt.imshow(mask, cmap='viridis', interpolation='nearest')
            plt.colorbar()
            plt.savefig(os.path.join(folder, filename))
            plt.close()

# Concrete Strategy 1
class MinMaxNormalization(NormalizationStrategy):

    def normalize(self, roiImage, roiMask):
        # Convert the mask to boolean for indexing
        roiImage = np.array(roiImage, dtype=np.float64)
        roiMask = np.array(roiMask)

        mask_bool = roiMask.astype(bool)

        # Extract the region of interest based on the mask
        roi_values = roiImage[mask_bool]

        # Find the unique gray levels in the region of interest
        # unique_levels = np.unique(roi_values)
        # n = len(unique_levels)

        # n_values = unique_levels[np.logical_and(unique_levels > 0, unique_levels < 8)]

        # Calculate the minimum and maximum values within the masked region
        min_val = roi_values.min()
        max_val = roi_values.max()

        # Check if there are at least two unique levels to avoid division by zero
        if min_val == max_val:
            raise ValueError("The region of interest has only one unique gray level.")

        # Apply the normalization formula to the region of interest
        normalized_roi = ((roiImage[mask_bool] - min_val) / (max_val - min_val + 1))

        # Find unique levels in the ROI values to determine n_bits
        unique_levels = np.unique(roi_values)
        n_bits = len(unique_levels).bit_length()

        # Ensure that n_bits is within the correct range

        # Quantization
        if n_bits != 8:
            # Calculate the scale factor based on the number of bits
            scale_factor = (2 ** n_bits) - 1
            # Apply quantization
            normalized_roi = (normalized_roi * scale_factor).round() / scale_factor

        # Map the normalized values to the range [0, 2^n_bits]
        roiImage[mask_bool] = (2 ** n_bits) * normalized_roi
        self.save_debug_images([roiImage], ["norm_minmax"])
        return roiImage

# Concrete Strategy 2
class MeanStdNormalization(NormalizationStrategy):

    def normalize(self, roiImage, roiMask):
        # Convert the mask to boolean for indexing
        roiImage = np.array(roiImage, dtype=np.float64)
        roiMask = np.array(roiMask)
        mask_bool = roiMask.astype(bool)

        # Extract the region of interest based on the mask
        roi_values = roiImage[mask_bool]

        # Calculate mean and standard deviation within the masked region
        mean_val = np.mean(roi_values)
        std_val = np.std(roi_values)

        min_val = mean_val - std_val
        max_val = mean_val + std_val

        # Check if there are at least two unique levels to avoid division by zero
        if min_val == max_val:
            raise ValueError("The region of interest has only one unique gray level.")

        # Apply the normalization formula to the region of interest
        normalized_roi = ((roiImage[mask_bool] - min_val) / (max_val - min_val + 1))

        # Find unique levels in the ROI values to determine n_bits
        unique_levels = np.unique(roi_values)
        # n_bits = len(unique_levels).bit_length()

        # Ensure that n_bits is within the correct range

        # Quantization
        if unique_levels != 8:
            # Calculate the scale factor based on the number of bits
            scale_factor = (2 ** unique_levels) - 1
            # Apply quantization
            normalized_roi = (normalized_roi * scale_factor).round() / scale_factor

        # Map the normalized values to the range [0, 2^n_bits]
        roiImage[mask_bool] = (2 ** unique_levels) * normalized_roi
        self.save_debug_images([roiImage], ["norm_meanstd"])
        return roiImage

# Concrete Strategy 3
class PercentileNormalization(NormalizationStrategy):

    def normalize(self, roiImage, roiMask):
        # Convert the mask to boolean for indexing
        roiImage = np.array(roiImage, dtype=np.float64)
        roiMask = np.array(roiMask)
        mask_bool = roiMask.astype(bool)

        # Extract the region of interest based on the mask
        roi_values = roiImage[mask_bool]

        # Calculate the 1st and 99th percentiles
        min_val = np.percentile(roi_values, 1)
        max_val = np.percentile(roi_values, 99)

        # Ensure that the calculated percentiles are not equal
        if min_val == max_val:
            raise ValueError("The calculated percentiles are equal; check the input data.")

        # Normalize the ROI values
        normalized_roi = ((roiImage[mask_bool] - min_val) / (max_val - min_val + 1))

        # Find unique levels in the ROI values to determine n_bits
        unique_levels = np.unique(roi_values)
        # n_bits = len(unique_levels).bit_length()

        # Ensure that n_bits is within the correct range

        # Quantization
        if unique_levels != 8:
            scale_factor = 2 ** unique_levels - 1
            normalized_roi = (normalized_roi * scale_factor).round() / scale_factor

        # Map the normalized values to the range [0, 2^n_bits)
        roiImage[mask_bool] = normalized_roi * (2 ** unique_levels)
        self.save_debug_images([roiImage], ["norm_per"])

        return roiImage



