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


class Radiomics(object):

    def extractRadiomics(self, images, results_folder_path):

        # Initialize the feature extractor
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.enableAllFeatures()

        params = {
            'Square': {},  # Square image filter
            'Exponential': {},  # Exponential image filter
            'Logarithm': {},  # Logarithm image filter
        }

        for image_type, kwargs in params.items():
            extractor.enableImageTypeByName(image_type, customArgs=kwargs)

        filenames = {
            'original_': os.path.join(results_folder_path, 'original.csv'),
            'square_': os.path.join(results_folder_path, 'square.csv'),
            'logarithm_': os.path.join(results_folder_path, 'logarithm.csv'),
            'exponential_': os.path.join(results_folder_path, 'expo.csv')
        }

        dfs = {key: pd.DataFrame() for key in filenames}

        # Iterate over each image/mask pair
        for (image, image_path, mask) in images:

            # Convert the numpy arrays to SimpleITK images
            mask_image_sitk = sitk.GetImageFromArray(mask)
            prostate_image_sitk = sitk.GetImageFromArray(image)

            # Execute feature extraction
            results = extractor.execute(prostate_image_sitk, mask_image_sitk)
            for image_type, filename in filenames.items():
                features = {k: results[k] for k in results if k.startswith(image_type)}
                if features:
                    features_with_image = {'Image': image_path, **features}
                    df = pd.DataFrame(features_with_image, index=[0])

                    # Append to the DataFrame for the current image type
                    dfs[image_type] = dfs[image_type]._append(df, ignore_index=True)

        for image_type, df in dfs.items():
                df.to_csv(filenames[image_type], mode='w', index=False)

