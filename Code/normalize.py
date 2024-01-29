import os
import sys
import numpy as np
import SimpleITK as sitk
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod




class NormalizationStrategy(ABC):

    @abstractmethod
    def normalize(self, images):
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

    def getNormImage(self, image, min, max, n_bits):
        return (((2 ** n_bits) - 1) * (image - min) / (max - min + 1)).astype(np.uint8)

class NoNormalization(NormalizationStrategy):
    def normalize(self, images):
        # Simply return the image as is, without any normalization
        return images

class MinMaxNormalization(NormalizationStrategy):
    def normalize(self, images, n_bits, mask):
        # Convert the mask to boolean for indexing
        roiImage = np.array(images, dtype=np.float64)
        roiMask = np.array(mask, dtype=bool)
        masked_pixels = roiImage[roiMask]

        # Calculate the minimum and maximum values within the masked region
        min_val = masked_pixels.min()
        max_val = masked_pixels.max()

        # Apply the normalization formula to the region of interest
        normalized_image = self.getNormImage(roiImage, min_val, max_val, n_bits)

        self.save_debug_images([normalized_image], ["norm_minmax"])
        return normalized_image

class MeanStdNormalization(NormalizationStrategy):

    def normalize(self, image, n_bits, mask):
        roiImage = np.array(image, dtype=np.float64)
        mask = np.array(mask, dtype=bool)

        # Use the mask to select the relevant pixels
        masked_pixels = roiImage[mask]

        mean_val = np.mean(masked_pixels)
        std_val = np.std(masked_pixels)

        min_val = mean_val - std_val
        max_val = mean_val + std_val

        normalized_image = self.getNormImage(roiImage, min_val, max_val, n_bits)
        self.save_debug_images([normalized_image], ["norm_meanstd"])
        return normalized_image

class PercentileNormalization(NormalizationStrategy):

    def normalize(self, image, n_bits, mask):
        roiImage = np.array(image, dtype=np.float64)
        mask = np.array(mask, dtype=bool)

        # Use the mask to select the relevant pixels
        masked_pixels = roiImage[mask]

        min_val = np.percentile(masked_pixels, 1)
        max_val = np.percentile(masked_pixels, 99)

        normalized_image = self.getNormImage(roiImage, min_val, max_val, n_bits)
        self.save_debug_images([normalized_image], ["norm_per"])
        return normalized_image



