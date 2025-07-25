import cv2
import numpy as np


def convert_to_grayscale(image, callback=None):
    grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if callback:
        callback(grayscaled, "CONVERT_TO_GRAYSCALE")

    return grayscaled


def apply_dynamic_threshold(grayscale_image, callback=None):
    threshold_value = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_OTSU)[0] + 20
    thresholded = cv2.threshold(grayscale_image, threshold_value, 255, cv2.THRESH_BINARY)[1]

    if callback:
        callback(thresholded, "APPLY_DYNAMIC_THRESHOLD")

    return thresholded


def apply_manual_threshold(grayscale_image, threshold_value, callback=None):
    thresholded = cv2.threshold(grayscale_image, threshold_value, 255, cv2.THRESH_BINARY)[1]

    if callback:
        callback(thresholded, "APPLY_MANUAL_THRESHOLD")

    return thresholded


def invert_image_binary(binary_image, callback=None):
    inverted = cv2.bitwise_not(binary_image)

    if callback:
        callback(inverted, "INVERT_IMAGE_BINARY")

    return inverted


def remove_small_components(binary_image, minimum_size, callback=None):
    connected_components = cv2.connectedComponentsWithStats(binary_image)
    quantity, image, stats = connected_components[:3]
    sizes = stats[:, -1][1:]

    small_blobs = np.where(sizes <= minimum_size)[0] + 1
    binary_image[np.isin(image, small_blobs)] = 0

    if callback:
        callback(binary_image, "REMOVE_SMALL_COMPONENTS")

    return binary_image


def generate_processing_labelmap(model_output, shape, confidence_threshold, labels, callback=None):
    confidence_threshold = confidence_threshold / 100

    if len(shape) == 3:
        shape = shape[:2]

    airway_epithelium_labelmap = extract_class_labelmap_from_model(model_output, shape, 1, confidence_threshold)
    vessel_epithelium_labelmap = extract_class_labelmap_from_model(model_output, shape, 2, confidence_threshold)

    final_labelmap = np.zeros(shape, dtype="uint8")
    final_labelmap = np.where(airway_epithelium_labelmap, labels["AIRWAY_EPITHELIUM"], final_labelmap)
    final_labelmap = np.where(vessel_epithelium_labelmap, labels["VESSEL_ENDOTHELIUM"], final_labelmap)

    if callback:
        callback(airway_epithelium_labelmap, "GENERATE_PROCESSING_LABELMAP_AIRWAY")
        callback(vessel_epithelium_labelmap, "GENERATE_PROCESSING_LABELMAP_VESSEL")
        callback(final_labelmap, "GENERATE_PROCESSING_LABELMAP_COMBINED")

    return final_labelmap


def extract_class_labelmap_from_model(model_output, shape, class_id, confidence_threshold):
    model_output_mask = np.array([mask.cpu().numpy() for idx, mask in enumerate(model_output["masks"]) if (model_output["labels"][idx].cpu().numpy() == class_id and model_output["scores"][idx] > confidence_threshold)])
    mask_with_confidence = (model_output_mask > confidence_threshold).any(axis=0)

    if mask_with_confidence is False:
        mask_with_confidence = np.full(shape, False, dtype=bool)

    return mask_with_confidence


def generate_postprocessing_labelmap(masks_labelmap, thresholded_labelmap, labels, callback=None):
    intermediate_label = max(labels.values()) + 1

    parenchyma_labelmap = np.where(thresholded_labelmap, labels["ALVEOLI"], labels["PARENCHYMA"])
    airway_epithelium_labelmap = np.where(masks_labelmap == labels["AIRWAY_EPITHELIUM"], labels["AIRWAY_EPITHELIUM"], 0)
    vessel_epithelium_labelmap = np.where(masks_labelmap == labels["VESSEL_ENDOTHELIUM"], labels["VESSEL_ENDOTHELIUM"], 0)
    blocking_labelmap = np.where(masks_labelmap == labels["BLOCKER"], labels["BLOCKER"], 0)

    airway_complete_labelmap = generate_complete_class_labelmap(airway_epithelium_labelmap, thresholded_labelmap, labels["AIRWAY_EPITHELIUM"], labels["AIRWAY_LUMEN"])
    vessel_complete_labelmap = generate_complete_class_labelmap(vessel_epithelium_labelmap, thresholded_labelmap, labels["VESSEL_ENDOTHELIUM"], labels["VESSEL_LUMEN"])
    blocking_complete_labelmap = generate_complete_class_labelmap(blocking_labelmap, thresholded_labelmap, labels["BLOCKER"], intermediate_label, True)

    final_labelmap = np.zeros(masks_labelmap.shape, dtype="uint8")
    final_labelmap = np.where(parenchyma_labelmap, parenchyma_labelmap, final_labelmap)
    final_labelmap = np.where(airway_complete_labelmap, airway_complete_labelmap, final_labelmap)
    final_labelmap = np.where(vessel_complete_labelmap, vessel_complete_labelmap, final_labelmap)

    final_labelmap = np.where(blocking_complete_labelmap, blocking_complete_labelmap, final_labelmap)
    final_labelmap[final_labelmap == intermediate_label] = 0

    if callback:
        callback(airway_complete_labelmap, "GENERATE_POSTPROCESSING_LABELMAP_AIRWAY")
        callback(vessel_complete_labelmap, "GENERATE_POSTPROCESSING_LABELMAP_VESSEL")
        callback(final_labelmap, "GENERATE_POSTPROCESSING_LABELMAP_COMBINED")

    return final_labelmap


def generate_complete_class_labelmap(class_epithelium_labelmap, thresholded_image, epithelium_label, lumen_label, blocking=False, edge_distance=10):
    class_epithelium_labelmap = class_epithelium_labelmap.astype(np.uint8).squeeze()

    if class_epithelium_labelmap.ndim == 3 and class_epithelium_labelmap.shape[2] == 3:
        class_epithelium_labelmap = class_epithelium_labelmap[:, :, 0]

    non_tissue_spaces = cv2.connectedComponents(thresholded_image)[1]

    labelmap_without_overlap = class_epithelium_labelmap.copy()
    labelmap_with_overlap = np.where(thresholded_image, 0, class_epithelium_labelmap)

    contours = cv2.findContours(class_epithelium_labelmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    height, width = class_epithelium_labelmap.shape

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width - 1, x + w), min(height - 1, y + h)

        if x1 <= edge_distance:
            cv2.line(class_epithelium_labelmap, (x1, y1), (x1, y2), epithelium_label, thickness=1)
        if x2 >= width - edge_distance:
            cv2.line(class_epithelium_labelmap, (x2, y1), (x2, y2), epithelium_label, thickness=1)
        if y1 <= edge_distance:
            cv2.line(class_epithelium_labelmap, (x1, y1), (x2, y1), epithelium_label, thickness=1)
        if y2 >= height - edge_distance:
            cv2.line(class_epithelium_labelmap, (x1, y2), (x2, y2), epithelium_label, thickness=1)

    contours = cv2.findContours(class_epithelium_labelmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    kernel = np.ones((5, 5), np.uint8)

    for contour in contours:
        mask = np.zeros_like(class_epithelium_labelmap, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        eroded_mask = cv2.erode(mask, kernel)
        empty_spaces = cv2.bitwise_and(eroded_mask, cv2.bitwise_not(class_epithelium_labelmap))

        empty_spaces = np.where(empty_spaces == 255, empty_spaces, 0)

        centroids = cv2.connectedComponentsWithStats(empty_spaces, connectivity=8)[3]

        if blocking:
            labelmap_without_overlap[eroded_mask == 255] = lumen_label
        else:
            for centroid in centroids[1:]:
                cx, cy = map(int, centroid)
                component = non_tissue_spaces[cy, cx]
                if component:
                    labelmap_with_overlap[(non_tissue_spaces == component)] = lumen_label

    return labelmap_without_overlap if blocking else labelmap_with_overlap
