import SimpleITK as sitk
import numpy as np
from radiomics import featureextractor
import matplotlib.pyplot as plt
import six
import pandas as pd


class Radiomics:


    def __dir__(self):
        self.output_path = "..\Code\outputs3\cechy.csv"

    def extractRadiomics(self, image, mask, path, image_name):

        image_sitk = sitk.GetImageFromArray(image)
        image_sitk.CopyInformation(sitk.ReadImage(path, sitk.sitkUInt8))

        extractor = featureextractor.RadiomicsFeatureExtractor(force2D=True)
        extractor.enableAllFeatures()

        names = []
        features = []

        for label in [1, 2]:
            for feature_name, feature_value in extractor.execute(image_sitk, mask, label).items():
                if feature_name.startswith("original_"):
                    names.append(feature_name)
                    features.append(feature_value)

        names_l = list(names)
        fig, ax = plt.subplots(figsize=[24, 5])
        plt.plot(features, color='red')
        plt.yscale('log')
        plt.xticks(range(len(names_l)), names_l, rotation=45, ha="right")
        plt.subplots_adjust(wspace=10)
        plt.title("Radiomic Feature Values");
        plt.savefig('..\Code\outputs3\example_radiomics.png', bbox_inches='tight', dpi=100)

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
