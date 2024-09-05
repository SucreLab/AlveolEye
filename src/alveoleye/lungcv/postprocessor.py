import cv2
import numpy as np


def greyscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def dynamic_threshold(greyscale_image):
    threshold_value = cv2.threshold(greyscale_image, 0, 255, cv2.THRESH_OTSU)[0] + 20
    return cv2.threshold(greyscale_image, threshold_value, 255, cv2.THRESH_BINARY)[1]


def manual_threshold(greyscale_image, threshold_value):
    return cv2.threshold(greyscale_image, threshold_value, 255, cv2.THRESH_BINARY)[1]


def invert_binary(binary_image):
    return cv2.bitwise_not(binary_image)


def clean(binary_image, minimum_size):
    connected_components = cv2.connectedComponentsWithStats(binary_image)
    quantity, image, stats = connected_components[:3]
    sizes = stats[:, -1][1:]

    small_blobs = np.where(sizes <= minimum_size)[0] + 1
    binary_image[np.isin(image, small_blobs)] = 0

    return binary_image


def create_processing_labelmap(model_output, shape, confidence_threshold, labels):
    confidence_threshold = confidence_threshold / 100

    if len(shape) == 3:
        shape = shape[:2]

    airway_epithelium_labelmap = create_class_labelmap_from_model(model_output, 1, confidence_threshold)
    vessel_epithelium_labelmap = create_class_labelmap_from_model(model_output, 2, confidence_threshold)

    final_labelmap = np.zeros(shape, dtype="uint8")
    final_labelmap = np.where(airway_epithelium_labelmap, labels["AIRWAY_EPITHELIUM"], final_labelmap)
    final_labelmap = np.where(vessel_epithelium_labelmap, labels["VESSEL_ENDOTHELIUM"], final_labelmap)

    return final_labelmap


def create_class_labelmap_from_model(model_output, class_id, confidence_threshold):
    model_output_mask = np.array([mask.cpu().numpy() for idx, mask in enumerate(model_output["masks"]) if
                                  (model_output["labels"][idx].cpu().numpy() == class_id and
                                   model_output["scores"][idx] > confidence_threshold)])
    mask_with_confidence = (model_output_mask > confidence_threshold).any(axis=0)

    return mask_with_confidence


def create_postprocessing_labelmap(masks_labelmap, thresholded_labelmap, labels):
    parenchyma_labelmap = np.where(thresholded_labelmap, labels["ALVEOLI"], labels["PARENCHYMA"])
    airway_epithelium_labelmap = np.where(masks_labelmap == labels["AIRWAY_EPITHELIUM"],
                                          labels["AIRWAY_EPITHELIUM"], 0)
    vessel_epithelium_labelmap = np.where(masks_labelmap == labels["VESSEL_ENDOTHELIUM"],
                                          labels["VESSEL_ENDOTHELIUM"], 0)

    airway_complete_labelmap = create_complete_class_labelmap(airway_epithelium_labelmap, thresholded_labelmap,
                                                              labels["AIRWAY_EPITHELIUM"], labels["AIRWAY_LUMEN"])
    vessel_complete_labelmap = create_complete_class_labelmap(vessel_epithelium_labelmap, thresholded_labelmap,
                                                              labels["VESSEL_ENDOTHELIUM"], labels["VESSEL_LUMEN"])

    final_labelmap = np.zeros(masks_labelmap.shape, dtype="uint8")
    final_labelmap = np.where(parenchyma_labelmap, parenchyma_labelmap, final_labelmap)
    final_labelmap = np.where(airway_complete_labelmap, airway_complete_labelmap, final_labelmap)
    final_labelmap = np.where(vessel_complete_labelmap, vessel_complete_labelmap, final_labelmap)

    return final_labelmap


def create_complete_class_labelmap(class_epithelium_labelmap, thresholded_image, epithelium_label, lumen_label):
    all_lumens = cv2.connectedComponents(thresholded_image)[1]

    class_epithelium_labelmap = class_epithelium_labelmap.astype(np.uint8).squeeze()
    outline_labelmap = class_epithelium_labelmap.copy()
    class_epithelium_labelmap_parenchyma_overlap = np.where(thresholded_image, 0, class_epithelium_labelmap)

    contours = cv2.findContours(class_epithelium_labelmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)[0]

    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Fill edges if the drawn / predicted labels touch the edge
    for contour in contours:
        contour_mask = cv2.drawContours(np.zeros_like(class_epithelium_labelmap, dtype=np.uint8),
                                        [contour], -1, 255, thickness=cv2.FILLED)

        edge_offset = 10

        touching_left = np.any(contour_mask[:, :edge_offset] != 0)
        touching_right = np.any(contour_mask[:, -1 * edge_offset:] != 0)
        touching_top = np.any(contour_mask[:edge_offset, :] != 0)
        touching_bottom = np.any(contour_mask[-1 * edge_offset:, :] != 0)

        if touching_right:
            class_epithelium_labelmap[:, -1 * edge_offset:] = epithelium_label
        if touching_left:
            class_epithelium_labelmap[:, :edge_offset] = epithelium_label
        if touching_top:
            class_epithelium_labelmap[:edge_offset, edge_offset:-1 * edge_offset] = epithelium_label
        if touching_bottom:
            class_epithelium_labelmap[-1 * edge_offset:, edge_offset:-1 * edge_offset] = epithelium_label

    contours = cv2.findContours(class_epithelium_labelmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)[0]

    # Run on image considering parenchyma
    for contour in contours:
        contour_mask = cv2.drawContours(np.zeros_like(class_epithelium_labelmap, dtype=np.uint8),
                                        [contour], -1, 255, thickness=cv2.FILLED)

        eroded_contour_mask = cv2.erode(contour_mask, kernel, iterations=1)

        rough_empty_spaces = cv2.bitwise_and(eroded_contour_mask, cv2.bitwise_not(class_epithelium_labelmap))
        rough_empty_spaces = np.where(rough_empty_spaces == 255, rough_empty_spaces, 0)

        centroids = cv2.connectedComponentsWithStats(rough_empty_spaces)[3]

        # Flood fill spaces
        for centroid in centroids[1:]:
            centroid_x, centroid_y = map(int, centroid)
            component = all_lumens[centroid_y, centroid_x]

            if component:
                class_epithelium_labelmap[(all_lumens == component) & (eroded_contour_mask == 255)] = lumen_label
                class_epithelium_labelmap[eroded_contour_mask == 255] = lumen_label
                # add back in outline
                class_epithelium_labelmap[outline_labelmap == epithelium_label] = epithelium_label

    return class_epithelium_labelmap
