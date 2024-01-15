import random
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageOps
from pathlib import Path
import os
import numpy as np

def showData(image, name ):

    n = len(image)
    output_dir = os.path.abspath('..\\outputs3')
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=[70, 5])
    for i in range(n):

        if isinstance(image[i], str):
            img = cv2.imread(image[i], cv2.IMREAD_UNCHANGED)
        else:
            img = image[i]

        trans = transforms.ToTensor()
        img_tensor = trans(img)
        img_tensor = img_tensor.permute(1, 2, 0).numpy()

        plt.subplot(1, n, i + 1)
        plt.imshow(img_tensor, cmap='gray')
        plt.title(Path(name[i]).name)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'image_grayscale.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=100)
    plt.show()
