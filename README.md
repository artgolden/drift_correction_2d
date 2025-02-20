EXPERIMENTAL! Most of the code and documentation generated with LLMs.

# Drift Correction for 2D z-Projections

This repository contains a Python script that applies drift correction to a folder of 2D z-projections (TIFF files) generated from microscope imaging. The script groups TIFF files by common base name (ignoring the timepoint indicated by `TP-XXXX` in the filename) and processes each group separately. Within each group, the projection with the lowest timepoint is used as the reference image for drift correction.

## Features

- **Grouping by File Name:**  
  Files are expected to be named in the following format:
  ```
  timelapseID-20250124-114341_SPC-0001_TP-0000_ILL-0_CAM-1_CH-01_PL-0000-outOf-0090.tif
  ```
  The script parses the filename to extract the timepoint (`TP`) and illumination (`ILL`) values.

- **Illumination Merging (Default Behavior):**  
  When multiple files share the same base name and timepoint but differ in the illumination value (e.g., `ILL-0` and `ILL-1`), the script will average these images to produce a single merged projection. The merged file will have `ILL-merged` in its name. This allows for correction on a combined image when two illuminations per timepoint are present.

- **Optional Split by Illumination:**  
  Using the `--split_ill` flag, you can instruct the script to treat files with different illumination values as separate groups, bypassing the merging process.

- **Drift Correction:**  
  Within each group (spanning multiple timepoints), the image corresponding to the lowest timepoint is used as the reference image for drift correction. Drift is computed using `dexp.processing.registration.translation_nd` and applied with `scipy.ndimage.shift`, followed by cropping to remove border artifacts.

---

## Installation

This repository uses [mamba](https://github.com/mamba-org/mamba) for package management. Follow the steps below to create a new environment and install the required dependencies.

1. **Install Mamba (if not already installed):**

   ...

2. **Create a New Environment and Install Dependencies:**

   ```bash
   # Create a new environment (here named "drift_correction")
   mamba create -n drift_correction python=3.9 -y
   mamba activate drift_correction

   # Install required libraries from conda-forge
   mamba install -c conda-forge tifffile scipy numpy

   # Install dexp via pip
   pip install dexp
   ```

---

## Usage

1. **Prepare Your TIFF Files:**

   Ensure that your TIFF files follow the naming convention:
   ```
   timelapseID-20250124-114341_SPC-0001_TP-0000_ILL-0_CAM-1_CH-01_PL-0000-outOf-0090.tif
   ```
   Files with multiple illuminations for the same timepoint (e.g., one with `ILL-0` and one with `ILL-1`) will be merged by default. If you wish to process each illumination separately, use the `--split_ill` flag.

2. **Run the Script:**

   ```bash
   # To merge illuminations (default behavior):
   python drift_correction_for_z_proj.py /path/to/input_folder /path/to/output_folder --crop_margin 10

   # To treat illuminations separately (no merging):
   python drift_correction_for_z_proj.py /path/to/input_folder /path/to/output_folder --crop_margin 10 --split_ill
   ```

   - `/path/to/input_folder`: Directory containing your TIFF images.
   - `/path/to/output_folder`: Directory where the corrected images will be saved.
   - `--crop_margin`: Optional; number of pixels to crop from each border after drift correction (default: 10).
   - `--split_ill`: Optional flag; if provided, images with different illumination values are processed as separate groups without merging.

---

## Script Overview

The `drift_correction_for_z_proj.py` script performs the following steps:

1. **Grouping Files:**  
   Uses a regular expression to group TIFF files by their base name (ignoring the timepoint).

2. **Reference Selection:**  
   For each group, the file with the lowest timepoint is chosen as the reference image.

3. **Drift Correction:**  
   Each image in the group is corrected relative to the reference image using:
   - `register_translation_nd` from the dexp package to compute the drift.
   - `scipy.ndimage.shift` to apply the computed translation.
   - Cropping with the specified margin to remove edge artifacts.

4. **Saving Results:**  
   Corrected images are saved to the output folder with the same filename as the input.


