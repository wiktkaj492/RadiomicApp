import numpy as np
from PIL import Image
import csv
import pandas as pd
import os


seg_filename = "D:\\PNG_001_0000.png"
output_directory = r"..\outputs3"
mask_filename = os.path.join(output_directory, "mask.png")

# Read the image using PIL
mask_img = Image.open(seg_filename)

# Convert the PIL image to a numpy array
img_array = np.array(mask_img)

# Change the value 0 to -1
img_array[img_array == 0] = -1

# Convert the numpy array back to a PIL image
mask_img = Image.fromarray(img_array)

# Save the PIL image
mask_img.save(mask_filename)

