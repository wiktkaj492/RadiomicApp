import SimpleITK as sitk
import numpy as np
from radiomics import featureextractor, imageoperations
import matplotlib.pyplot as plt
import six
import pandas as pd
import scipy
import skimage
import trimesh
import pywt
import os


class Radiomics(object):

    def __init__(self):
        self.output_path = "..\\Result"
        os.makedirs(self.output_path, exist_ok=True)


    def extractRadiomics(self, images):

        # Initialize the feature extractor
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.enableAllFeatures()

        params = {
            'LoG': {},  # Laplacian of Gaussian uses default settings
            'Wavelet': {},  # Wavelet uses default settings
            'Square': {},  # Square image filter
            'SquareRoot': {},  # Square root image filter
            'Exponential': {},  # Exponential image filter
            'Logarithm': {},  # Logarithm image filter
            'LBP2D': {},  # Local Binary Pattern 2D
            'LBP3D': {},  # Local Binary Pattern 3D
        }
        #localBinaryPatern, Gradient,Histogram, walvet, log, squareRoot

        for image_type, kwargs in params.items():
            extractor.enableImageTypeByName(image_type, customArgs=kwargs)

        filenames = {
            'original_': os.path.join(self.output_path, 'original.csv'),
            'log_': os.path.join(self.output_path, 'log.csv'),
            'wavelet_': os.path.join(self.output_path, 'walwet.csv'),
            'square_': os.path.join(self.output_path, 'square.csv'),
            'squareRoot_': os.path.join(self.output_path, 'squareRoot.csv'),
            'logarithm_': os.path.join(self.output_path, 'logarithm.csv'),
            'exponential_': os.path.join(self.output_path, 'expo.csv'),
            'lbp2d_': os.path.join(self.output_path, 'lbp2d.csv'),
            'lbp3d_': os.path.join(self.output_path, 'lbp3d.csv'),
        }

        dfs = {key: pd.DataFrame() for key in filenames}

        # Iterate over each image/mask pair
        for (image, image_path, mask) in images:

            # Convert the numpy arrays to SimpleITK images
            mask_image_sitk = sitk.GetImageFromArray(mask)
            prostate_image_sitk = sitk.GetImageFromArray(image)

            """
            gradient_image = imageoperations.getGradientImage(prostate_image_sitk, mask_image_sitk)
            lbp2d_image = imageoperations.getLBP2DImage(prostate_image_sitk, mask_image_sitk)
            lbp3d_image = imageoperations.getLBP3DImage(prostate_image_sitk, mask_image_sitk)

            gradient_results = extractor.execute(gradient_image, mask_image_sitk)
            lbp2d_results = extractor.execute(lbp2d_image, mask_image_sitk)
            lbp3d_results = extractor.execute(lbp3d_image, mask_image_sitk)
            """
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

