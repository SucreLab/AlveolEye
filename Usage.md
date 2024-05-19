<!--
[![License MIT](https://img.shields.io/pypi/l/automated-lung-morphometry.svg?color=green)](https://github.com/Quooooooookka/automated-lung-morphometry/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/automated-lung-morphometry.svg?color=green)](https://pypi.org/project/automated-lung-morphometry)
[![Python Version](https://img.shields.io/pypi/pyversions/automated-lung-morphometry.svg?color=green)](https://python.org)
[![tests](https://github.com/Quooooooookka/automated-lung-morphometry/workflows/tests/badge.svg)](https://github.com/Quooooooookka/automated-lung-morphometry/actions)
[![codecov](https://codecov.io/gh/Quooooooookka/automated-lung-morphometry/branch/main/graph/badge.svg)](https://codecov.io/gh/Quooooooookka/automated-lung-morphometry)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/automated-lung-morphometry)](https://napari-hub.org/plugins/automated-lung-morphometry)
-->
## Processing: Identify and segment vessel and airway epithelium with an AI driven computer vision model.
1. **Select an image**
- Click the “Import Image” button.
- Use operating system defualt file dialogue to select an image (*.jpg, *.png, or *.tiff).
- Check the image in the “Image” layer (of the Napari Viewer) and the file name (displayed to the right of the “Import Image” button) to confirm that the image loaded correctly.
2. **Select a model**
- To use the default model, skip to step 3; otherwise, proceed within step 2.
- Click the “Import Weights” button.
- Use operating system file dialogue to select a model (*.pth).
- Check the file name (displayed to the right of the “Import Weights” button) to confirm that the model loaded correctly.
3. **Select a confidence level** 
- Type a percentage and/or click the “-” and “+” buttons in the “Minimum confidence” input box to set the confidence level.
4. **Run processing** 
- Click the “Run Processing” button to have the model run identify and segment vessel and airway epithelium, filtered by confidence level.  

## Postprocessing: Identify alveolar tissue, and airwary and vessel lumens with “classical” (non-AI) methods; remove small particles and holes to prepare for assessments.
1. **Toggle manual thresholding**
- To use manual thresholding, check the “Manual thresholding” box and proceed; to use automatic thresholding, leave the box unchecked and skip to step 2.
- Type a percentage and/or click the “-” and “+” buttons in the “Manual thresholding” input box to set the threshold level.
2. **Remove small particles** 
- Type a percentage and/or click the “-” and “+” buttons in the “Remove small particles” input box to set the maximum size cutoff for particles to remove.
3. **Remove small holes** 
- Type a percentage and/or click the “-” and “+” buttons in the “Remove small holes” input box to set the maximum size cutoff for holes to remove.
4. **Run postprocessing**
- Click “Run Postprocessing” button to identify alveolar tissue, airwary lumens, and vessel lumens, and to remove small particles and holes.

## Assessments: Calculate morphometry assessments—mean linear intercept (MLI) and airspace volume density (ASVD) on the fully classified image. 
1. **Select ASVD** 
- To include ASVD calculations in results, check the “ASVD” checkbox; otherwise, leave the box unchecked. 
2. **Select MLI** 
- To include MLI calculations in results, check the “MLI” checkbox; otherwise, leave the box unchecked. 
3. **Set number of lines** 
- Type a number and/or click the “-” and “+” buttons in the “number of lines” input box to set the number of MLI test lines.
4. **Set minimum length** 
- Type a number and/or click the “-” and “+” buttons in the “minimum length” input box to set the minimum length required for a chord to be included in the mean calculation.
5. **Set scale** 
- Type a number and/or click the “-” and “+” buttons in the “scale” input box to set the scale factor (i.e. pixels to physical space multiplier). 
6. **Run Assessments** 
- Click the “Run Assessments” button to calculate the selected assessments. 

## Export Results: Collect assessment results for each image and export all the data into a file when done (*.csv or *.json).
- **Interpreting Results**
    - **MLI:** Mean Linear Intercept for the given image
    - **Standard deviation:** The standard deviation of the chord lengths used to calculate MLI. 
    - **Number of chords:** The number of chords used to calculate MLI.
    - **ASVD:** Airspace Volume Density calculation for the given image.
    - **Airspace pixels:** The total number of airspace pixels
    - **Non airspace pixels:** The total number non-airspace pixels.
1. **Add last result** 
- Click the “Add” button to add the assessment data to the final results file
2. **Clear export data** 
- Click the “Clear” button to clear the export data file
3. **Export Results** 
- Click the “Export Results” button to open a file dialogue for saving the assessments results. Note that the plugin supports two export result file types, *.csv and *.json that you can choose between.  
