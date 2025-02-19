#!/usr/bin/env python3
"""
Drift Correction Script for 2D z-Projections

This script reads TIFF images from an input folder, groups them based on a common base name
(i.e. files with identical names except for the timepoint part, e.g. TP-0000, TP-0001, etc.),
and then for each group uses the image with the lowest timepoint as the reference.
Drift is computed with `register_translation_nd` and applied using `scipy.ndimage.shift`.
After correction a crop margin is applied to remove edge artifacts.

Expected file naming convention:
  timelapseID-20250124-114341_SPC-0001_TP-0000_ILL-0_CAM-1_CH-01_PL-0000-outOf-0090.tif

Usage:
    python drift_correction.py /path/to/input_folder /path/to/output_folder --crop_margin 10
"""

import os
import re
import argparse
import logging

import numpy as np
import tifffile
from scipy.ndimage import shift as scipy_shift

# Import the drift registration function.
from dexp.processing.registration.translation_nd import register_translation_nd


def logging_broadcast(message):
    """Log and print a message."""
    logging.info(message)
    print(message)


def group_files(input_dir):
    """
    Group files in the input directory based on the common base name (excluding timepoint).

    The expected file naming format is:
        <base>_TP-<timepoint><suffix>
    For example:
        timelapseID-20250124-114341_SPC-0001_TP-0000_ILL-0_CAM-1_CH-01_PL-0000-outOf-0090.tif

    Returns
    -------
    groups : dict
        Dictionary mapping a group key (base + suffix) to a list of (timepoint, filename) tuples.
    """
    groups = {}
    # Pattern to capture the base, the timepoint, and the suffix (including file extension).
    pattern = re.compile(r"^(.*)_TP-(\d+)(_.+\.tiff?)$", re.IGNORECASE)
    for f in os.listdir(input_dir):
        if not f.lower().endswith(('.tif', '.tiff')):
            continue
        m = pattern.match(f)
        if not m:
            logging_broadcast(f"File {f} does not match the expected naming pattern. Skipping.")
            continue
        base = m.group(1)
        timepoint_str = m.group(2)
        suffix = m.group(3)
        try:
            tp = int(timepoint_str)
        except ValueError:
            logging_broadcast(f"Timepoint value {timepoint_str} in file {f} is not an integer. Skipping.")
            continue
        # The group key is the base plus suffix (thus ignoring the timepoint).
        group_key = f"{base}{suffix}"
        groups.setdefault(group_key, []).append((tp, f))
    return groups


def process_group(input_dir, output_dir, group_key, file_list, crop_margin=10):
    """
    Process one group of images: the one with the lowest timepoint is used as the reference
    and all other images in the group are drift corrected relative to that reference.

    Parameters
    ----------
    input_dir : str
        Folder containing the input TIFF images.
    output_dir : str
        Folder where corrected images will be saved.
    group_key : str
        Group identifier (base name and suffix).
    file_list : list
        List of (timepoint, filename) tuples.
    crop_margin : int, optional
        Number of pixels to crop from each border after drift correction (default: 10).
    """
    # Sort the files by timepoint (ascending)
    file_list_sorted = sorted(file_list, key=lambda x: x[0])
    ref_tp, ref_fname = file_list_sorted[0]
    ref_path = os.path.join(input_dir, ref_fname)
    logging_broadcast(f"Group '{group_key}': Using {ref_fname} (TP-{ref_tp:04d}) as reference.")
    
    ref_img = tifffile.imread(ref_path)
    # Check if the reference image is large enough for the crop margin.
    if (ref_img.shape[0] <= 2 * crop_margin) or (ref_img.shape[1] <= 2 * crop_margin):
        logging_broadcast(
            f"Reference image {ref_fname} dimensions {ref_img.shape} are too small for crop margin {crop_margin}. "
            "No cropping will be applied."
        )
        use_crop = False
    else:
        use_crop = True

    if use_crop:
        ref_img_corrected = ref_img[crop_margin:-crop_margin, crop_margin:-crop_margin]
    else:
        ref_img_corrected = ref_img

    # Save the (optionally cropped) reference image.
    ref_out_path = os.path.join(output_dir, ref_fname)
    tifffile.imwrite(ref_out_path, ref_img_corrected)
    logging_broadcast(f"Group '{group_key}': Saved reference image to {ref_out_path}")

    # Process each subsequent image in the group.
    for tp, fname in file_list_sorted[1:]:
        file_path = os.path.join(input_dir, fname)
        logging_broadcast(f"Group '{group_key}': Processing image {fname} (TP-{tp:04d}).")
        img = tifffile.imread(file_path)

        if (img.shape[0] <= 2 * crop_margin) or (img.shape[1] <= 2 * crop_margin):
            logging_broadcast(
                f"Image {fname} dimensions {img.shape} are too small for crop margin {crop_margin}. "
                "Skipping drift correction and cropping for this image."
            )
            corrected_img = img
        else:
            # Keep a copy in case drift correction fails.
            img_copy = img.copy()
            try:
                # Compute the drift relative to the reference image.
                translation_model = register_translation_nd(ref_img, img)
                logging_broadcast(f"Group '{group_key}': Computed shift for {fname}: {translation_model.shift_vector}")
                shifted_img = scipy_shift(img, shift=translation_model.shift_vector)
                corrected_img = shifted_img[crop_margin:-crop_margin, crop_margin:-crop_margin]
            except Exception as err:
                logging_broadcast(f"Group '{group_key}': Drift correction failed for {fname}: {err}")
                corrected_img = img_copy

        # Save the corrected image.
        out_path = os.path.join(output_dir, fname)
        tifffile.imwrite(out_path, corrected_img)
        logging_broadcast(f"Group '{group_key}': Saved corrected image to {out_path}")


def process_images(input_dir, output_dir, crop_margin=10):
    """
    Process TIFF images in the input directory, grouping by common base name (excluding TP)
    and applying drift correction to each group separately.
    
    Parameters
    ----------
    input_dir : str
        Folder containing input TIFF images.
    output_dir : str
        Folder where corrected images will be saved.
    crop_margin : int, optional
        Number of pixels to crop from each border after drift correction (default: 10).
    """
    groups = group_files(input_dir)
    if not groups:
        logging_broadcast("No valid TIFF files found in input directory.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for group_key, file_list in groups.items():
        logging_broadcast(f"Processing group: {group_key} with {len(file_list)} file(s).")
        process_group(input_dir, output_dir, group_key, file_list, crop_margin=crop_margin)


def main():
    parser = argparse.ArgumentParser(
        description="Apply drift correction to groups of 2D z-projections (TIFF files) using the reference image (lowest TP)."
    )
    parser.add_argument("input_folder", type=str, help="Path to folder containing input TIFF images")
    parser.add_argument("output_folder", type=str, help="Path to folder where corrected images will be saved")
    parser.add_argument(
        "--crop_margin",
        type=int,
        default=10,
        help="Number of pixels to crop from each border after drift correction (default: 10)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    process_images(args.input_folder, args.output_folder, args.crop_margin)


if __name__ == "__main__":
    main()
