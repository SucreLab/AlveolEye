import random

import cv2
import numpy as np
from scipy import ndimage


def calculate_airspace_volume_density(labelmap, labels):
    alveoli_pixels = np.count_nonzero(labelmap == labels["ALVEOLI"])
    parenchyma_pixels = np.count_nonzero(labelmap == labels["PARENCHYMA"])
    alveoli_and_parenchyma_pixels = alveoli_pixels + parenchyma_pixels

    if alveoli_and_parenchyma_pixels == 0:
        alveolar_density = 0.0
    else:
        alveolar_density = (alveoli_pixels / alveoli_and_parenchyma_pixels) * 100.0

    return alveolar_density, alveoli_pixels, alveoli_and_parenchyma_pixels


def calculate_mean_linear_intercept(labelmap, num_lines, min_length, scale, labels, randomized_distribution=False):
    labelmap = np.squeeze(labelmap)
    labelmap_shape = labelmap.shape

    if randomized_distribution:
        line_y_coordinates = np.array([random.sample(range(1, labelmap.shape[0] - 1), num_lines)])
    else:
        line_y_coordinates = np.linspace(0, labelmap_shape[0] - 1, num_lines + 2, dtype=int)[1:-1]

    test_lines_labelmap = np.zeros(labelmap_shape, dtype=np.uint8)
    test_lines_labelmap[line_y_coordinates, :] = labels["MLI_LINES_OUTSIDE"]

    chords_labelmap = np.where(labelmap != labels["ALVEOLI"], 0, test_lines_labelmap)

    counter = 0
    total_area = 0
    chord_lengths = []

    for i in range(chords_labelmap.shape[0]):
        labeled, num_components = ndimage.label(chords_labelmap[i])

        for component_label in range(1, num_components + 1):
            component_mask = (labeled == component_label)
            component_area = np.sum(component_mask)

            if component_area >= min_length:
                counter += 1
                total_area += component_area
                chord_lengths.append(component_area * scale)
            else:
                chords_labelmap[i][component_mask] = 0

    average_length = total_area * scale / counter if counter > 0 else 0

    chord_lengths = np.array(chord_lengths)
    stdev_chord_lengths = "NA" if len(chord_lengths) == 0 else np.std(chord_lengths)

    kernel = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], np.uint8)
    chords_highlighted_labelmap = cv2.dilate(chords_labelmap, kernel, iterations=5)
    chords_highlighted_labelmap = np.where(chords_labelmap, labels["MLI_LINES_INSIDE"], chords_highlighted_labelmap)

    return average_length, chords_highlighted_labelmap.astype(int), counter, stdev_chord_lengths