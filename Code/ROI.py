import numpy as np
import SimpleITK as sitk
import pandas as pd
import os
import matplotlib.pyplot as plt

class ROI(object):

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

    def roiImage(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:

        # Find the coordinates of non-zero values in the mask
        print(image.shape)
        coords = np.argwhere(mask)
        print(coords.shape)
        if coords.size == 0:
            # Handle the empty case (e.g., return the original image and mask or raise an error)
            return image, mask
        # Find the bounding box of the non-zero values in the mask
        min_y, min_x = coords.min(axis=0)
        max_y, max_x = coords.max(axis=0)

        # Crop the image and mask using the bounding box
        cropped_image = image[min_y:max_y+1, min_x:max_x+1]
        self.save_debug_images([cropped_image], ["roi"])
        cropped_mask = mask[min_y:max_y + 1, min_x:max_x + 1]
        return cropped_image, cropped_mask

