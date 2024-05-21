<!--
[![License MIT](https://img.shields.io/pypi/l/AlveolEye.svg?color=green)](https://github.com/SucreLab/AlveolEye/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/AlveolEye.svg?color=green)](https://pypi.org/project/AlveolEye)
[![Python Version](https://img.shields.io/pypi/pyversions/AlveolEye.svg?color=green)](https://python.org)
[![tests](https://github.com/SucreLab/AlveolEye/workflows/tests/badge.svg)](https://github.com/SucreLab/AlveolEye/actions)
[![codecov](https://codecov.io/gh/SucreLab/AlveolEye/branch/main/graph/badge.svg)](https://codecov.io/gh/SucreLab/AlveolEye)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/AlveolEye)](https://napari-hub.org/plugins/AlveolEye)
-->

# AlveolEye Usage Tutorial

## Annotated Diagram
![annotated diagram](./Colored%20AlveolEye.png)

## Processing: Identify and segment vessel and airway epithelium with an AI computer vision model.
1. ![#FF3333](https://placehold.co/15x15/FF3333/FF3333.png) **Select an image**: The remaining steps will concern this image.

   a. Click the “Import Image” button.

   b. Use operating system default file dialogue to select an image (*.jpg, *.png, or *.tiff).
   
   c. Check the image in the “Image” layer (of the Napari Viewer) and the file name (displayed to the right of the “Import Image” button) to confirm that the image loaded correctly.
2. ![#FF9933](https://placehold.co/15x15/FF9933/FF9933.png) **Select a model**</span>: The selected model will run on the image and predict (segment) vessel and airway epithelium. 

   a. To use the default model, skip to step 3; otherwise, proceed to step 2b. Use the provided default model unless you have a specific reason not to.
   
   b. Click the “Import Weights” button.
   
   c. Use operating system file dialogue to select a model (*.pth).
   
   d. Check the file name (displayed to the right of the “Import Weights” button) to confirm that the model loaded correctly.
3. ![#FFFF33](https://placehold.co/15x15/FFFF33/FFFF33.png) **Select a confidence level**: Type a percentage and/or click the “-” and “+” buttons in the “Minimum confidence” input box to set the confidence level. Predictions with lower confidence then the set confidence level will not appear.
4. ![#f03c15](https://placehold.co/15x15/33FF33/33FF33.png) **Run processing**: Click the “Run Processing” button to run the model and segment vessel and airway epithelium filtered by confidence level. Once completed, manually edit the prediction as necessary with the built-in napari tools to the left of the displayed image layer. See Napari documentation for more information about how to use these tools. 

---

## Postprocessing: Identify alveolar tissue, and airwary and vessel lumens with “classical” (non-AI) methods; remove small particles and holes to prepare for assessments.
1. ![#FF3333](https://placehold.co/15x15/FF3333/FF3333.png) **Toggle manual thresholding**: To manually set a threshold value, toggle manual threshold; otherwise, a threshold value will be determined with Otsu's method.

   a. To use manual thresholding, check the “Manual thresholding” box and proceed; to use automatic thresholding, leave the box unchecked and skip to step 2.

   b. Type a percentage and/or click the “-” and “+” buttons in the “Manual thresholding” input box to set the threshold level.
2. ![#FF9933](https://placehold.co/15x15/FF9933/FF9933.png) **Remove small particles**: Type a percentage and/or click the “-” and “+” buttons in the “Remove small particles” input box to set the maximum size cutoff for particles to remove. Particles with fewer pixels than the set number will be removed. 
3. ![#FFFF33](https://placehold.co/15x15/FFFF33/FFFF33.png) **Remove small holes**: Type a percentage and/or click the “-” and “+” buttons in the “Remove small holes” input box to set the maximum size cutoff for holes to remove. Holes with fewer pixels than the set number will be removed.
4. ![#f03c15](https://placehold.co/15x15/33FF33/33FF33.png) **Run postprocessing**: Click “Run Postprocessing” button to identify alveolar tissue, airwary lumens, and vessel lumens, and to remove small particles and holes. Once completed, manually edit the post-processing layer as necessary with the built-in napari tools to the left of the displayed image layer. See Napari documentation for more information about how to use these tools.

---

## Assessments: Calculate morphometry assessments—mean linear intercept (MLI) and airspace volume density (ASVD) on the fully classified image. 
1. ![#FF3333](https://placehold.co/15x15/FF3333/FF3333.png) **Select ASVD**: To include ASVD calculations in results, check the “ASVD” checkbox; otherwise, leave the box unchecked. Leave the box unchecked to increase the speed of the assessments calculation or to exclude unnecessary data from the final export file.
2. ![#FF9933](https://placehold.co/15x15/FF9933/FF9933.png) **Select MLI**: To include MLI calculations in results, check the “MLI” checkbox; otherwise, leave the box unchecked. Leave the box unchecked to increase the speed of the assessments calculation or to exclude unnecessary data from the final export file.
3. ![#FFFF33](https://placehold.co/15x15/FFFF33/FFFF33.png) **Set number of lines**: Type a number and/or click the “-” and “+” buttons in the “number of lines” input box to set the number of MLI lines.
4. ![#f03c15](https://placehold.co/15x15/33FF33/33FF33.png) **Set minimum length**: Type a number and/or click the “-” and “+” buttons in the “minimum length” input box to set the minimum length required for a chord to be included in the mean calculation.
5. ![#f03c15](https://placehold.co/15x15/3359FF/3359FF.png) **Set scale**: Type a number and/or click the “-” and “+” buttons in the “scale” input box to set the scale factor (i.e. a pixel to physical space multiplier). 
6. ![#f03c15](https://placehold.co/15x15/C433FF/C433FF.png) **Run Assessments**: Click the “Run Assessments” button to calculate the selected assessments. The ASVD and MLI calculation results will display to the right of the assessment checkboxes.

---

## Export Results: Collect assessment results for each image and export all the data into a file when done (*.csv or *.json).
- **Interpreting Results**
    - **MLI:** Mean Linear Intercept for the given image
    - **Standard deviation:** The standard deviation of the lengths of the chord used to calculate MLI. 
    - **Number of chords:** The number of chords used to calculate MLI.
    - **ASVD:** Airspace Volume Density calculation for the given image.
    - **Airspace pixels:** The total number of airspace pixels
    - **Non airspace pixels:** The total number non-airspace pixels.
1. ![#FF3333](https://placehold.co/15x15/FF3333/FF3333.png) **Add last result**: Click the “Add” button to add the assessment data to the final export file. Once the results are added, you can return to the "Processing" step and do another image.
2. ![#FF9933](https://placehold.co/15x15/FF9933/FF9933.png) **Remove last result**: Click the “Remove” button to remove the last results added to the export file.
3. ![#FFFF33](https://placehold.co/15x15/FFFF33/FFFF33.png) **Clear export data**: Click the “Clear” button to clear the export data file.
4. ![#f03c15](https://placehold.co/15x15/33FF33/33FF33.png) **Export Results**: Click the “Export Results” button to open a file dialogue for saving the assessments results. Note that the plugin supports two export result file types, *.csv and *.json that you can choose between.
