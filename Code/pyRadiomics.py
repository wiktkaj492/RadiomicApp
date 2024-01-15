import SimpleITK as sitk
import numpy as np
from radiomics import featureextractor
import matplotlib.pyplot as plt
import six
import pandas as pd
import os


class Radiomics:


    def __init__(self):
        self.output_path = "D:\Repo\RadiomicApp\Code\outputs3\cechy.csv"

    def extractRadiomics(self, image, mask, image_name, img_path):

        output_dir = "D:\Repo\RadiomicApp\Code\outputs3"

        mask_image_sitk = sitk.GetImageFromArray(mask)
        prostate_image_sitk = sitk.GetImageFromArray(image)

        prostate_image_sitk.CopyInformation(sitk.ReadImage(img_path, sitk.sitkUInt8))

        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.enableAllFeatures()

        names = []
        features = []


        for feature_name, feature_value in extractor.execute(prostate_image_sitk, mask_image_sitk).items():
            if feature_name.startswith("original_"):
                names.append(feature_name)
                features.append(feature_value)

        names_l = list(names)
        fig, ax = plt.subplots(figsize=[24, 5])
        plt.plot(features, color='red')
        plt.yscale('log')
        plt.xticks(range(len(names_l)), names_l, rotation=90, ha="right")
        plt.subplots_adjust(wspace=10)
        plt.title("Radiomic Feature Values");
        output_path = os.path.join(output_dir, 'example_radiomics.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=100)

        names.insert(0, "Image")
        features.insert(0, image_name)
        df = pd.DataFrame([features], columns=names)
        df = pd.DataFrame({'Feature Name': names, 'Feature Value': features})

        def make_pretty(styler):
            styler.background_gradient(axis=None, cmap="viridis")
            return styler

        styled_df = df.style.pipe(make_pretty)

        df.to_csv(self.output_path, index=False)


        return styled_df
