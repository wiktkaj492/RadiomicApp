import SimpleITK as sitk
import numpy as np
from radiomics import featureextractor
import matplotlib.pyplot as plt
import six
import pandas as pd
import os


class Radiomics:


    def __init__(self):
        self.output_path = "D:\Repo\RadiomicApp\Code\outputs3"

    def extractRadiomics(self, images, masks, image_names, img_paths):

        # Initialize the feature extractor
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.enableAllFeatures()

        extractor.enableImageTypeByName('LoG')
        extractor.enableImageTypeByName('Wavelet')
        extractor.enableImageTypeByName('Square')
        extractor.enableImageTypeByName('SquareRoot')
        extractor.enableImageTypeByName('Logarithm')
        extractor.enableImageTypeByName('Exponential')
        #localBinaryPatern, Gradient,Histogram, walvet, log, squareRoot

        filenames = {
            'original_': os.path.join(self.output_path, 'original.csv'),
            'log_': os.path.join(self.output_path, 'log.csv'),
            'wavelet_': os.path.join(self.output_path, 'walwet.csv'),
            'square_': os.path.join(self.output_path, 'square.csv'),
            'squareRoot_': os.path.join(self.output_path, 'squareRoot.csv'),
            'logarithm_': os.path.join(self.output_path, 'logarithm.csv'),
            'exponential_': os.path.join(self.output_path, 'expo.csv')
        }

        dfs = {key: pd.DataFrame() for key in filenames}

        # Iterate over each image/mask pair
        for image, mask, img_path, image_name in zip(images, masks, img_paths, image_names):
            print(f"extractRadiomics mask shape: {mask.shape}")
            print(f"extractRadiomics image shape: {image.shape}")

            # Convert the numpy arrays to SimpleITK images
            mask_image_sitk = sitk.GetImageFromArray(mask)
            prostate_image_sitk = sitk.GetImageFromArray(image)

            # Copy meta information from the loaded image
            #prostate_image_sitk.CopyInformation(sitk.ReadImage(img_path, sitk.sitkUInt8))



            # Execute feature extraction
            results = extractor.execute(prostate_image_sitk, mask_image_sitk)
            for image_type, filename in filenames.items():
                features = {k: results[k] for k in results if k.startswith(image_type)}
                if features:
                    features_with_image = {'Image': image_name, **features}
                    df = pd.DataFrame(features_with_image, index=[0])

                    # Append to the DataFrame for the current image type
                    dfs[image_type] = dfs[image_type]._append(df, ignore_index=True)

        for i, (image_type, df) in enumerate(dfs.items()):
            if i == 0:
                df.to_csv(filenames[image_type], mode='w', index=False)
            else:
                df.to_csv(filenames[image_type], mode='a', header=False, index=False)



