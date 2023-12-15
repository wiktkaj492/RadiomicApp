import random
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageOps
from pathlib import Path
import os
import numpy as np

def transform_image_with_mask(image, mask):
    # Pobierz minimalną i maksymalną wartość poziomu szarości w regionie oznaczonym maską
    min_intensity = np.min(image[mask > 0])
    max_intensity = np.max(image[mask > 0])

    # Przelicz wszystkie wartości w regionie zgodnie ze wzorem
    n = 8  # Możesz dostosować wartość n według potrzeb
    transformed_image = 2**n * ((image - min_intensity) / (max_intensity - min_intensity + 1))

    return transformed_image

# Przykładowe użycie
image = np.array([[100, 150, 200], [50, 75, 100], [25, 30, 35]])
mask = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

result = transform_image_with_mask(image, mask)
print(result)

def percentile_normalize(image):
    # Spłaszcz obraz do jednego wymiaru (wektora)
    flattened_image = image.flatten()

    # Sortuj wartości pikseli
    sorted_values = np.sort(flattened_image)

    # Wylicz indeksy pikseli odpowiadające 1 i 99 percentylowi
    min_index = int(0.01 * len(sorted_values))
    max_index = int(0.99 * len(sorted_values))

    # Wylicz minimalną i maksymalną wartość na podstawie percentyli
    min_intensity = sorted_values[min_index]
    max_intensity = sorted_values[max_index]

    # Przeprowadź normalizację percentylową
    normalized_image = (image - min_intensity) / (max_intensity - min_intensity + 1)

    return normalized_image

# Przykładowe użycie
image = np.array([[100, 150, 200], [50, 75, 100], [25, 30, 35]])
result = percentile_normalize(image)
print(result)

def mean_std_normalize(image, mask, pmstd):
    # Pobierz wartości pikseli w regionie oznaczonym maską
    values_in_mask = image[mask > 0]

    # Sprawdź, czy region oznaczony maską nie jest pusty
    if len(values_in_mask) == 0:
        return image  # Brak normalizacji dla pustego obszaru

    # Oblicz wartość średnią i odchylenie standardowe
    mean_intensity = np.mean(values_in_mask)
    std_intensity = np.std(values_in_mask)

    # Wylicz minimalną i maksymalną wartość na podstawie mean ± pmstd
    min_intensity = mean_intensity - pmstd * std_intensity
    max_intensity = mean_intensity + pmstd * std_intensity

    # Przeprowadź normalizację mean-std
    normalized_image = (image - min_intensity) / (max_intensity - min_intensity)

    return normalized_image

# Przykładowe użycie
image = np.array([[100, 150, 200], [50, 75, 100], [25, 30, 35]])
mask = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
pmstd = 2

result = mean_std_normalize(image, mask, pmstd)
print(result)
