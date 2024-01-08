import SimpleITK
import radiomics
import numpy as np
import nibabel as nib
from radiomics import featureextractor
import matplotlib.pyplot as plt
import six
import pandas as pd
from PIL import Image

def extractRadiomics(image, tempGrey):
    n = len(image)

    for i in range(n):
        img = image[i]

        if isinstance(img, str):
            img = Image.open(img)

        mask = np.ones(img.size[::-1]) * 255
        mask[:1, :1] = 0
        mask = mask.astype(np.uint8)

        mask_filename = "..\\outputs3\\mask.nii"
        print(np.unique(np.asarray(mask)))
        empty_header = nib.Nifti1Header()
        affine = np.eye(4)
        another_img = nib.Nifti1Image(mask, affine, empty_header)
        print(another_img.header.get_data_shape())
        nib.save(another_img, mask_filename)


        extractor = featureextractor.RadiomicsFeatureExtractor(force2D=True)
        output = extractor.execute(img, mask_filename, label = 255)
        # Make an array of the values
        names =  np.array([])
        features = np.array([])
        for key, value in six.iteritems(output):
            if key.startswith("original_"):
                names = np.append (names, key)
                features = np.append ( features, output[key])

        names_l = list(names)
        fig, ax = plt.subplots(figsize=[18,5])
        plt.plot(features, color='red')
        plt.yscale('log')
        plt.xticks(range(len(names_l)), names_l, rotation=90)
        plt.title ( "Radiomic Feature Values");
        plt.savefig('..\\outputs3\\example_radiomics.png', bbox_inches='tight', dpi=100)

        # Declare csv filename from Pyradiomics (zscore scaled and merged)
        fname = f'../outputs3/pyradi_features.csv'

        # Load data
        pyradi_data = pd.read_csv(fname)
        pyradi_data.shape

        def make_pretty(styler):
            styler.background_gradient(axis=None, cmap="viridis")
            return styler

        # Show the radiomic feature columns
        pyradi_original = pyradi_data.iloc[:,25:]
        pyradi_original.head().style.pipe(make_pretty)

        # Create a DataFrame from the names and features arrays
        df = pd.DataFrame({'Feature Name': names_l, 'Feature Value': features})
        # Write the DataFrame to a CSV file
        df.to_csv('..\\outputs3\\radiomics_features.csv', index=False)
