# AlveolEye: Automated Lung Morphometry Made Easy

[![Napari Plugin](https://img.shields.io/badge/Napari-Plugin-1157c4?logo=napari)](https://www.napari-hub.org/plugins/AlveolEye)
![Python Version](https://img.shields.io/badge/Python-3.9%20|%203.10%20|%203.11-blue)
![OS Support](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-blue)
![GitHub Release](https://img.shields.io/github/v/release/SucreLab/AlveolEye?display_name=tag)
![License](https://img.shields.io/github/license/SucreLab/AlveolEye)
[![PyPI Downloads](https://img.shields.io/pypi/dm/AlveolEye)](https://pypi.org/project/AlveolEye/)
![Maintenance](https://img.shields.io/maintenance/yes/2025)
![Last Commit](https://img.shields.io/github/last-commit/SucreLab/AlveolEye)
![Issues](https://img.shields.io/github/issues/SucreLab/AlveolEye)

This repository contains the beta version of AlveolEye, created by the [Sucre lab](https://www.sucrelab.org/).  
The code is authored by Samuel Hirsh, Joseph Hirsh, Nick Negretti, and Shawyon Shirazi.

AlveolEye is a Napari plugin that uses computer vision and classical image processing  
to calculate mean linear intercept (MLI) and airspace volume density (ASVD) of histologic images.

The goal of this tool is to aid researchers, not provide a complete automated annotation solution.

We welcome Git issues and feedback!

## Installation

The goal of this process is to create a conda environment containing Napari and all AlveolEye requirements.

*If you already have conda set up, you can skip step 1.*

1. **Install Miniconda** by downloading the appropriate version from [here](https://www.anaconda.com/docs/getting-started/miniconda/install):  
   - Choose the version that matches your processor.  
   - Download the `.pkg` version for easy installation.

2. **Clone the repository** (by opening a terminal or Miniconda prompt and running the following)
   ```
   git clone https://github.com/SucreLab/AlveolEye
   ```

3. **Navigate to the directory**:
   ```
   cd AlveolEye
   ```

4. **Create the conda environment**:
   ```
   conda env create -f ./environment.yml
   ```

5. **Activate the environment**:
   ```
   conda activate AlveolEye
   ```

6. **Install the plugin**:
   ```
   pip install .
   ```

7. **Launch Napari** and locate the plugin in the plugin menu:
   ```
   napari
   ```

## Running Post-Installation

1. **Activate the environment** in the terminal or Miniconda prompt:
   ```
   conda activate AlveolEye
   ```

2. **Run Napari** in the terminal:
   ```
   napari
   ```

<div align="right">
  <a href="#alveoleye-automated-lung-morphometry-made-easy">Back to Top</a>
</div>

## Usage

### Processing: Identify and Segment Vessel Endothelium and Airway Epithelium with Computer Vision

![processing diagram](https://raw.githubusercontent.com/SucreLab/AlveolEye/main/docs/PROCESSING_FINAL.svg)

1. **Import image**  
   - Click the "Import Image" button.  
   - Use the file dialog to select an image (`.jpg`, `.png`, or `.tiff`).  
   - Verify that the image correctly loaded. The file name should appear to the right of the button.

2. **Toggle processing with computer vision**  
   - Keep the checkbox selected to process the image with computer vision (continue to step 3).  
   - Deselect to skip computer vision processing (skip to step 5).

3. **Import weights**  
   - To use the default model, proceed to step 4.  
   - To use a custom model:  
     - Click the "Import Weights" button.  
     - Select a model file (`.pth`).  
     - Verify that the weights correctly loaded. The file name should appear to the right of the button.

4. **Set minimum confidence**  
   - Adjust the minimum confidence using the input box or the "-/+" buttons.  
   - Predictions from the computer vision model with lower confidence than this threshold will not appear.

5. **Run processing**  
   - Click the "Run Processing" button.  
   - Once completed, manually edit the prediction as needed using Napari's built-in tools.

<div align="right">
  <a href="#alveoleye-automated-lung-morphometry-made-easy">Back to Top</a>
</div>

### Postprocessing: Segment Alveolar Tissue and Find Vessel and Aireway Lumens

![postprocessing diagram](https://raw.githubusercontent.com/SucreLab/AlveolEye/main/docs/POSTPROCESSING_FINAL.svg)

1. **Configure thresholding**  
   - For manual thresholding: Select the "Manual threshold" checkbox and use the spinbox to set the threshold level.  
   - For automatic thresholding ([Otsu's method](https://learnopencv.com/otsu-thresholding-with-opencv/)): Leave the box unchecked.

2. **Remove small particles**  
   - Set the minimum size cutoff.
   - Particles with fewer pixels than this value will be removed.

3. **Remove small holes**  
   - Set the minimum size cutoff.  
   - Holes with fewer pixels than this value will be removed.

4. **Run postprocessing**  
   - Click the "Run Postprocessing" button.  
   - Once completed, manually edit the results as needed using Napari's built-in tools.

<div align="right">
  <a href="#alveoleye-automated-lung-morphometry-made-easy">Back to Top</a>
</div>

### Assessments: Calculate Morphometry Measurements

![assessments diagram](https://raw.githubusercontent.com/SucreLab/AlveolEye/main/docs/ASSESSMENTS_FINAL.svg)

1. **Airspace Volume Density (ASVD)**
   - Select the checkbox to run ASVD calculation.
   - Deselect the checkbox to exclude data from export and increase processing speed.

2. **Mean Linear Intercept (MLI)**
   - Select the checkbox to run MLI calculation.
   - Deselect the checkbox to exclude data from export and increase processing speed.

3. **Number of lines**
   - Set the number of lines used for MLI calculation.

5. **Minimum length**
   - Set the minimum chord length for inclusion in MLI calculations.
   - Note: Chords are the line segments that span across an airspace between two alveolar tissue boundaries during MLI calculation.

7. **Scale**
   - Set the pixel-to-physical space multiplier.

9. **Run assessments**
   - Click the "Run Assessments" button.
   - View results displayed beside assessment checkboxes and in the export box.

<div align="right">
  <a href="#alveoleye-automated-lung-morphometry-made-easy">Back to Top</a>
</div>

### Export Results: Save Assessment Results as a CSV or JSON File

![export diagram](https://raw.githubusercontent.com/SucreLab/AlveolEye/main/docs/EXPORT_FINAL.svg)

1. **Add results**
   - Click "Add" to include current assessment data in the export file.

3. **Remove last result**
   - Click "Remove" to delete the last added results from the export file.

5. **Clear export data**
   - Click "Clear" to empty the export file.

7. **Export results**
   - Click "Export Results" to save the data (`.csv` or `.json` format).

**Results Key**

- **MLI**: Mean Linear Intercept for the tissue image
 
- **Standard deviation**: Standard deviation of chord lengths used in MLI calculation
  
- **Number of chords**: Number of chords used in MLI calculation

- **ASVD**: Airspace Volume Density for the image
 
- **Airspace pixels**: Total number of airspace pixels
   
- **Non-airspace pixels**: Total number of non-airspace pixels

<div align="right">
  <a href="#alveoleye-automated-lung-morphometry-made-easy">Back to Top</a>
</div>

## Manual Annotation Guide

### Label Reference

| Structure          | Label Number |
|--------------------|--------------|
| Blocker            | 1            |
| Airway Epithelium  | 2            |
| Vessel Endothelium | 3            |
| Airway Lumen       | 4            |
| Vessel Lumen       | 5            |
| Parenchyma         | 6            |
| Alveoli            | 7            |

### Annotation Tips

- **Eyedropper tool**: Click the eyedropper tool, then click a pixel in the image to set your active label (for drawing and editing) to that pixel's label.  
- **Layer selection**: Ensure you're working on the correct layer before annotating.  
- **Visibility control**: Hide unnecessary layers using the eye icon on the layer boxes (to the left of the image viewer) for clearer viewing.
- **Blocking**: Encircle airways and vessels in the blocking label, and everything within that closed shape will be discounted from assessments calculation. 

<div align="right">
  <a href="#alveoleye-automated-lung-morphometry-made-easy">Back to Top</a>
</div>

## Additional Information

### Theme Settings

Toggle between dark and light mode using:

- **Windows/Linux**: `Ctrl + Shift + T`  
- **macOS**: `Cmd + Shift + T`

Or through Napari preferences:

1. Select "napari" in the menu bar.
   
2. Choose "Preferences."
   
3. Click "Appearance" in the left menu.
     
4. Select "dark," "light," or "system" in the theme dropdown.

<div align="right">
  <a href="#alveoleye-automated-lung-morphometry-made-easy">Back to Top</a>
</div>
