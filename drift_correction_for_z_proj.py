#!/usr/bin/env python3
"""
Drift Correction Script for 2D z-Projections with Optional Illumination Merging

This script reads TIFF images from an input folder and groups them based on their naming convention.
Filenames are expected to have the following format:
    {prefix}_TP-{tp}_ILL-{ill}{suffix}
For example:
    timelapseID-20250124-114341_SPC-0001_TP-0000_ILL-0_CAM-1_CH-01_PL-0000-outOf-0090.tif

Two processing modes are available:
  1. Merge illuminations (default):  
     Files with the same prefix and suffix (i.e. the same overall sample and channel)
     but with different illumination values (ILL) and same timepoint (TP) are merged (averaged)
     before drift correction. The merged image is saved with "ILL-merged" in the name.
  2. Treat illuminations separately (--split_ill):  
     Files with different ILL values are processed as separate groups.

Within each group (which spans multiple timepoints), the projection with the lowest TP is used
as the reference image for drift correction. Drift is computed with dexp’s
`register_translation_nd` and applied using `scipy.ndimage.shift`.

Usage:
    python drift_correction.py /path/to/input_folder /path/to/output_folder --crop_margin 10 [--split_ill]

Requirements:
    - tifffile
    - dexp (install via pip: pip install dexp)
    - scipy
    - numpy
    - Python 3.x
"""

import os
import re
import argparse
import logging

import numpy as np
import tifffile
from scipy.ndimage import shift as scipy_shift
import cupy as cp

from dexp.utils import xpArray
from dexp.utils.backends import Backend, BestBackend, CupyBackend

# Import the drift registration function.
from dexp.processing.registration.translation_nd import register_translation_nd


def logging_broadcast(message):
    """Log and print a message."""
    logging.info(message)
    print(message)


# Regular expression to parse filenames.
# Expected format: {prefix}_TP-{tp}_ILL-{ill}{suffix}
filename_pattern = re.compile(
    r"^(?P<prefix>.*)_TP-(?P<tp>\d+)_ILL-(?P<ill>\d+)(?P<suffix>.*\.tiff?)$",
    re.IGNORECASE,
)


def group_files_merge_ill(input_dir):
    """
    Group files for merging illuminations.
    Group key is (prefix, suffix) so that files with different ILL values but same prefix/suffix
    and same TP will be merged (averaged) later.
    
    Returns
    -------
    groups : dict
        Mapping (prefix, suffix) -> list of tuples (tp, ill, filename)
    """
    groups = {}
    for f in os.listdir(input_dir):
        if not f.lower().endswith(('.tif', '.tiff')):
            continue
        m = filename_pattern.match(f)
        if not m:
            logging_broadcast(f"File {f} does not match the expected naming pattern. Skipping.")
            continue
        prefix = m.group("prefix")
        tp = int(m.group("tp"))
        ill = int(m.group("ill"))
        suffix = m.group("suffix")
        key = (prefix, suffix)
        groups.setdefault(key, []).append((tp, ill, f))
    return groups


def group_files_split_ill(input_dir):
    """
    Group files treating illuminations separately.
    Group key is (prefix, ill, suffix) so that each illumination is processed as a separate group.
    
    Returns
    -------
    groups : dict
        Mapping (prefix, ill, suffix) -> list of tuples (tp, filename)
    """
    groups = {}
    for f in os.listdir(input_dir):
        if not f.lower().endswith(('.tif', '.tiff')):
            continue
        m = filename_pattern.match(f)
        if not m:
            logging_broadcast(f"File {f} does not match the expected naming pattern. Skipping.")
            continue
        prefix = m.group("prefix")
        tp = int(m.group("tp"))
        ill = m.group("ill")  # Keep as string for filename reconstruction
        suffix = m.group("suffix")
        key = (prefix, ill, suffix)
        groups.setdefault(key, []).append((tp, f))
    return groups


def process_group_merge_ill(input_dir, output_dir, group_key, file_list, crop_margin=10):
    """
    Process one group of images when merging illuminations.
    Files in file_list are tuples (tp, ill, filename).
    For each unique timepoint, if there are multiple files (with different ILL values), average them.
    The merged image is saved with "ILL-merged" in the filename.
    Drift correction is performed using the image for the lowest TP as reference.
    """
    prefix, suffix = group_key
    # Group files by timepoint.
    tp_dict = {}
    for tp, ill, fname in file_list:
        tp_dict.setdefault(tp, []).append((ill, fname))
    merged_images = []  # List of tuples: (tp, image, out_fname)
    for tp in sorted(tp_dict.keys()):
        files = tp_dict[tp]
        # Load all images for this timepoint.
        imgs = []
        for ill, fname in files:
            img = tifffile.imread(os.path.join(input_dir, fname))
            imgs.append(img)
        if len(imgs) > 1:
            # Average the images pixel-wise.
            avg_img = np.mean(np.stack(imgs, axis=0), axis=0)
            avg_img = avg_img.astype(imgs[0].dtype)
            out_fname = f"{prefix}_TP-{tp:04d}_ILL-merged{suffix}"
            logging_broadcast(f"Timepoint {tp:04d}: Merged {len(imgs)} illuminations into one image.")
        else:
            avg_img = imgs[0]
            # Even if not merged, name the file with ILL-merged for consistency.
            out_fname = f"{prefix}_TP-{tp:04d}_ILL-merged{suffix}"
        merged_images.append((tp, avg_img, out_fname))

    if not merged_images:
        logging_broadcast("No images found in group for merging.")
        return

    # Use the image with the lowest TP as reference.
    ref_tp, ref_img, ref_out_fname = merged_images[0]
    logging_broadcast(f"Group '{prefix}{suffix}': Using {ref_out_fname} (TP-{ref_tp:04d}) as reference.")

    # Check if the reference image is large enough for cropping.
    if (ref_img.shape[0] <= 2 * crop_margin) or (ref_img.shape[1] <= 2 * crop_margin):
        logging_broadcast(
            f"Reference image dimensions {ref_img.shape} are too small for crop margin {crop_margin}. "
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
    ref_out_path = os.path.join(output_dir, ref_out_fname)
    tifffile.imwrite(ref_out_path, ref_img_corrected)
    logging_broadcast(f"Group '{prefix}{suffix}': Saved reference image to {ref_out_path}")

    
    with BestBackend() as bkg:
        # Process each subsequent timepoint.
        for tp, img, out_fname in merged_images[1:]:
            logging_broadcast(f"Group '{prefix}{suffix}': Processing TP-{tp:04d} ({out_fname}).")
            # Check image dimensions for cropping.
            if (img.shape[0] <= 2 * crop_margin) or (img.shape[1] <= 2 * crop_margin):
                logging_broadcast(
                    f"Image for TP-{tp:04d} dimensions {img.shape} are too small for crop margin {crop_margin}. "
                    "Skipping drift correction and cropping."
                )
                corrected_img = img
            else:
                img_copy = img.copy()
                try:
                    ref_img_gpu = Backend.to_backend(ref_img)
                    img_gpu = Backend.to_backend(img)
                    if isinstance(bkg, CupyBackend):
                        ref_img_gpu =  ref_img_gpu.astype(cp.float32)
                        img_gpu = img_gpu.astype(cp.float32)
                    translation_model = register_translation_nd(ref_img_gpu, img_gpu)
                    logging_broadcast(f"TP-{tp:04d}: Computed shift vector: {translation_model.shift_vector}")
                    if isinstance(bkg, CupyBackend):
                        shift_vec = cp.asnumpy(translation_model.shift_vector)
                    else:
                        shift_vec = translation_model.shift_vector
                    shifted_img = scipy_shift(img, shift=shift_vec)
                    corrected_img = shifted_img[crop_margin:-crop_margin, crop_margin:-crop_margin]
                except Exception as err:
                    logging_broadcast(f"Drift correction failed for TP-{tp:04d}: {err}")
                    corrected_img = img_copy

        out_path = os.path.join(output_dir, out_fname)
        tifffile.imwrite(out_path, corrected_img)
        logging_broadcast(f"Group '{prefix}{suffix}': Saved corrected image to {out_path}")


def process_group_split_ill(input_dir, output_dir, group_key, file_list, crop_margin=10):
    """
    Process one group of images when treating illuminations separately.
    Files in file_list are tuples (tp, filename).
    If multiple files exist for the same timepoint within this group, they are averaged.
    The group key is (prefix, ill, suffix) and output filenames are left unchanged.
    Drift correction is performed using the image for the lowest TP as reference.
    """
    prefix, ill, suffix = group_key
    # Group files by timepoint.
    tp_dict = {}
    for tp, fname in file_list:
        tp_dict.setdefault(tp, []).append(fname)
    processed_images = []  # List of (tp, image, out_fname)
    for tp in sorted(tp_dict.keys()):
        files = tp_dict[tp]
        imgs = [tifffile.imread(os.path.join(input_dir, f)) for f in files]
        if len(imgs) > 1:
            avg_img = np.mean(np.stack(imgs, axis=0), axis=0)
            avg_img = avg_img.astype(imgs[0].dtype)
            out_fname = f"{prefix}_TP-{tp:04d}_ILL-{ill}{suffix}"
            logging_broadcast(f"TP-{tp:04d}: Averaged {len(imgs)} duplicate files for illumination {ill}.")
        else:
            avg_img = imgs[0]
            out_fname = f"{prefix}_TP-{tp:04d}_ILL-{ill}{suffix}"
        processed_images.append((tp, avg_img, out_fname))

    if not processed_images:
        logging_broadcast("No images found in group.")
        return

    # Use the image with the lowest TP as reference.
    ref_tp, ref_img, ref_out_fname = processed_images[0]
    logging_broadcast(f"Group '{prefix}_ILL-{ill}{suffix}': Using {ref_out_fname} (TP-{ref_tp:04d}) as reference.")

    # Check cropping possibility.
    if (ref_img.shape[0] <= 2 * crop_margin) or (ref_img.shape[1] <= 2 * crop_margin):
        logging_broadcast(
            f"Reference image dimensions {ref_img.shape} are too small for crop margin {crop_margin}. "
            "No cropping will be applied."
        )
        use_crop = False
    else:
        use_crop = True

    if use_crop:
        ref_img_corrected = ref_img[crop_margin:-crop_margin, crop_margin:-crop_margin]
    else:
        ref_img_corrected = ref_img

    # Save the reference image.
    ref_out_path = os.path.join(output_dir, ref_out_fname)
    tifffile.imwrite(ref_out_path, ref_img_corrected)
    logging_broadcast(f"Group '{prefix}_ILL-{ill}{suffix}': Saved reference image to {ref_out_path}")

    # Process remaining timepoints.
    for tp, img, out_fname in processed_images[1:]:
        logging_broadcast(f"Group '{prefix}_ILL-{ill}{suffix}': Processing TP-{tp:04d} ({out_fname}).")
        if (img.shape[0] <= 2 * crop_margin) or (img.shape[1] <= 2 * crop_margin):
            logging_broadcast(
                f"Image for TP-{tp:04d} dimensions {img.shape} are too small for crop margin {crop_margin}. "
                "Skipping drift correction and cropping."
            )
            corrected_img = img
        else:
            img_copy = img.copy()
            try:
                translation_model = register_translation_nd(ref_img, img)
                logging_broadcast(f"TP-{tp:04d}: Computed shift vector: {translation_model.shift_vector}")
                shifted_img = scipy_shift(img, shift=translation_model.shift_vector)
                corrected_img = shifted_img[crop_margin:-crop_margin, crop_margin:-crop_margin]
            except Exception as err:
                logging_broadcast(f"Drift correction failed for TP-{tp:04d}: {err}")
                corrected_img = img_copy

        out_path = os.path.join(output_dir, out_fname)
        tifffile.imwrite(out_path, corrected_img)
        logging_broadcast(f"Group '{prefix}_ILL-{ill}{suffix}': Saved corrected image to {out_path}")


def process_images_merge_ill(input_dir, output_dir, crop_margin=10):
    """
    Process TIFF images from input_dir in merge-illumination mode.
    Files are grouped by (prefix, suffix) so that different ILL values at the same TP are merged.
    """
    groups = group_files_merge_ill(input_dir)
    if not groups:
        logging_broadcast("No valid TIFF files found in input directory for merge mode.")
        return

    os.makedirs(output_dir, exist_ok=True)
    for group_key, file_list in groups.items():
        logging_broadcast(f"Processing group (merged illuminations): {group_key} with {len(file_list)} file(s).")
        process_group_merge_ill(input_dir, output_dir, group_key, file_list, crop_margin=crop_margin)


def process_images_split_ill(input_dir, output_dir, crop_margin=10):
    """
    Process TIFF images from input_dir in split-illumination mode.
    Files are grouped by (prefix, ill, suffix) so that each illumination is processed separately.
    """
    groups = group_files_split_ill(input_dir)
    if not groups:
        logging_broadcast("No valid TIFF files found in input directory for split mode.")
        return

    os.makedirs(output_dir, exist_ok=True)
    for group_key, file_list in groups.items():
        logging_broadcast(f"Processing group (separate illuminations): {group_key} with {len(file_list)} file(s).")
        process_group_split_ill(input_dir, output_dir, group_key, file_list, crop_margin=crop_margin)


def main():
    parser = argparse.ArgumentParser(
        description="Apply drift correction to groups of 2D z-projections (TIFF files). "
                    "By default, images from different illuminations (ILL) for the same TP are merged "
                    "by averaging before correction. Use --split_ill to treat different ILL values separately."
    )
    parser.add_argument("input_folder", type=str, help="Path to folder containing input TIFF images")
    parser.add_argument("output_folder", type=str, help="Path to folder where corrected images will be saved")
    parser.add_argument(
        "--crop_margin",
        type=int,
        default=10,
        help="Number of pixels to crop from each border after drift correction (default: 10)",
    )
    parser.add_argument(
        "--split_ill",
        action="store_true",
        help="Treat files with different illuminations (different value after 'ILL-') as separate groups (no merging)."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.split_ill:
        process_images_split_ill(args.input_folder, args.output_folder, args.crop_margin)
    else:
        process_images_merge_ill(args.input_folder, args.output_folder, args.crop_margin)


if __name__ == "__main__":
    main()
