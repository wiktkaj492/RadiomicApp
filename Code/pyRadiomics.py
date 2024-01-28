import SimpleITK as sitk
import numpy as np
from radiomics import featureextractor, imageoperations
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import skimage
import trimesh
import pywt
import os
import csv


class Radiomics(object):

    def extractRadiomics(self, images, param_file, results_folder_path):

        # Initialize the feature extractor
        extractor = featureextractor.RadiomicsFeatureExtractor(param_file)
        #extractor.enableAllFeatures()

        filenames = os.path.join(results_folder_path, 'results.csv')
        with open(filenames, mode='w', newline='') as file:
            writer = None

        # Iterate over each image/mask pair
            for (image, image_path, mask) in images:

                # Convert the numpy arrays to SimpleITK images
                mask_image_sitk = sitk.GetImageFromArray(mask)
                prostate_image_sitk = sitk.GetImageFromArray(image)

                # Execute feature extraction
                results = extractor.execute(prostate_image_sitk, mask_image_sitk)
                print(results)
                # Write headers if it's the first row
                results['image_path'] = image_path
                if writer is None:
                    writer = csv.DictWriter(file, fieldnames=['image_path'] + list(results.keys()))
                    writer.writeheader()

                # Write the results
                writer.writerow(results)

