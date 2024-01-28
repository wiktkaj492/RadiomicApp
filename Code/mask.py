import numpy as np
import SimpleITK as sitk
import pandas as pd
import os
import matplotlib.pyplot as plt

class Segmentation(object):

    def __init__(self):
        pass

    def segmenMask(self, mask, values):
        # Create a copy of the mask to avoid modifying the original
        binary_mask = np.copy(mask)

        # Check if the mask contains only zeros
        if np.all(binary_mask == 0):
            # If the user chooses both 1 and 2, fill with 1s (since it's a binary mask)
            #fill_value = 1 if len(values) > 1 else values[0]
            #binary_mask.fill(fill_value)
            return binary_mask
        else:
            # Set specified values to 1 and others to 0
            binary_mask = np.isin(binary_mask, values).astype(int)

        return binary_mask

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


    def segmentationMask(self, masks, values):
        new_mask_sitk = []
        for segmentation_image in masks:
            mask_array = self.segmenMask(segmentation_image, values)
            self.save_debug_images([mask_array], ["mask"])
            new_mask_sitk.append(mask_array)
        return new_mask_sitk

