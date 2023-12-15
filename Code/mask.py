import numpy as np
from PIL import Image
import nibabel as nib
import csv
import pandas as pd
from Code.show_data import batch
from Code.data import data_dir, nii_dir, dataset

# Save a default mask file in nifti format
mask = np.ones(img.shape) *255
mask[:1, :1, :] = 0
mask = mask.astype(np.uint8)

mask_filename = "..\\outputs2\\mask.nii"
print(np.unique(np.asarray(mask)))

empty_header = nib.Nifti1Header()
affine = np.eye(4)
another_img = nib.Nifti1Image(mask, affine, empty_header)
print(another_img.header.get_data_shape())
nib.save(another_img, mask_filename)

# Write a csv file with location and label of each image in the train set
pyradiomics_header = ('Image','Mask', 'Label')
m_arr = [mask_filename] * len(dataset.imgs)
#rows = [(il[0].replace(data_dir, nii_dir).replace('.JPG', '.nii'), m, 255) for m, il in zip(m_arr, dataset.imgs)]
rows = [(il[0].replace(data_dir, nii_dir).replace('.png', '.nii'), m, 255) for m, il in zip(m_arr, dataset.imgs)]
rows.insert(0, pyradiomics_header)
arr = np.asarray(rows)
#np.savetxt('..\\outputs\\pyradiomics_samples.csv', arr, fmt="%s", delimiter=",")
#ds = pd.read_csv('..\\outputs\\pyradiomics_samples.csv')
np.savetxt('..\\outputs2\\pyradiomics_samples.csv', arr, fmt="%s", delimiter=",")
ds = pd.read_csv('..\\outputs2\\pyradiomics_samples.csv')
ds.head()