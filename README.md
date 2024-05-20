<!--
[![License MIT](https://img.shields.io/pypi/l/automated-lung-morphometry.svg?color=green)](https://github.com/Quooooooookka/automated-lung-morphometry/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/automated-lung-morphometry.svg?color=green)](https://pypi.org/project/automated-lung-morphometry)
[![Python Version](https://img.shields.io/pypi/pyversions/automated-lung-morphometry.svg?color=green)](https://python.org)
[![tests](https://github.com/Quooooooookka/automated-lung-morphometry/workflows/tests/badge.svg)](https://github.com/Quooooooookka/automated-lung-morphometry/actions)
[![codecov](https://codecov.io/gh/Quooooooookka/automated-lung-morphometry/branch/main/graph/badge.svg)](https://codecov.io/gh/Quooooooookka/automated-lung-morphometry)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/automated-lung-morphometry)](https://napari-hub.org/plugins/automated-lung-morphometry)
-->

# AlveolEye: Automated lung morphometry made easy

---

This repository contains the beta version of AlveolEye, which is created by the Sucre lab.
This code is authored by Joseph Hirsh, Samuel Hirsh, Nick Negretti, and Shawyon Shirazi.

This project is a Napari plugin that uses computer vision tools and classical image processing
to calculate mean linear intercept (MLI) and airspace volume density (ASVD) from histological images.

A primary goal of this tool is to be an aid to the researcher, and not be a complete automated annotation solution.

## Installation

---

The target of this process is to create a conda environment that has both napari, and all of the AlveolEye requirements.

If you already have conda setup, you can skip step 1

1. Install miniconda by downlading the appropriate version from (here)[https://docs.anaconda.com/free/miniconda/]

   a. Choose the version that matches your processor

   b. Download the "pkg" version for easy install
3. Open a terminal, or miniconda prompt, and clone this git repository by running:

    ```git clone https://github.com/SucreLab/AlveolEye```
4. Go to the AlveolEye directory

    ```cd AlveolEye```
5. Create the conda environment

    ```conda env create -f ./environment.yml```
6. Activate the new environment

    ```conda activate AlveolEye```
5. Install the plugin

    ```pip install .```
6. Launch napari, followed by locating the plugin in the plugin menu

    ```napari```

## Running post-installation

---

1. Open a terminal, or miniconda prompt, activate the environment and run napari

```
conda activate AlveolEye
napari
```

Usage instructions can be found in (Usage.md)[Usage.md]
