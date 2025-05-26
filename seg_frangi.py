#!/usr/bin/env python3
import argparse
import os
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
from skimage.filters import frangi, threshold_otsu
from skimage.morphology import remove_small_objects, ball, binary_dilation, binary_opening

# --- Configuration ---

# Vessel and Lumen Dimensions (in pixels)
FRANGI_SCALE_RANGE_LUMEN = (1, 3)
FRANGI_SCALE_STEP_LUMEN = 1
FRANGI_BETA1 = 0.5
FRANGI_BETA2 = 15

# Pre-processing
PRE_SMOOTHING_SIGMA_PIXELS = 0.5

# **** New: Background Removal Parameters ****
# Percentile to define "very dark" background in the original image.
# Pixels below this intensity percentile might be pushed towards a common low value.
BACKGROUND_PERCENTILE_THRESHOLD = 5.0 # e.g., darkest 5% of pixels are considered deep background
# Value to set these background pixels to before inversion.
# Setting to 0 ensures they become the max value after a `effective_max - val` inversion.
# Or, if using `(val - eff_min) / (eff_max - eff_min)` then `1 - ...` for inversion,
# setting background to `eff_min` makes them 0 after normalization, then 1 after `1 - ...`.
# Let's go with a simpler approach: identify background and ensure it maps to highest value post-inversion.

# **** New: Robust Inversion Parameters ****
# Percentiles for robust min/max scaling before inversion
# This helps if some "background" pixels are still brighter than true signal min after initial cleanup
INVERSION_ROBUST_MIN_PERCENTILE = 0.5  # Pixels below this are clipped to this value before inversion scaling
INVERSION_ROBUST_MAX_PERCENTILE = 99.5 # Pixels above this are clipped to this value

# Segmentation
THRESHOLD_METHOD = 'otsu'

# Post-processing on Lumen Segmentation
MIN_LUMEN_OBJECT_SIZE_VOXELS = 50
OPENING_RADIUS_LUMEN_PIXELS = 0

# Optional: Reconstruct Full Vessel
RECONSTRUCT_FULL_VESSEL = True
WALL_THICKNESS_FOR_DILATION_PIXELS = 2


def segment_capillaries(input_volume_path: str, output_volume_path: str):
    print(f"Starting capillary segmentation for: {input_volume_path}")
    print(f"Output will be saved to: {output_volume_path}")

    # 1. Load Input Volume
    print("Loading input volume...")
    try:
        original_volume_zyx = tifffile.imread(input_volume_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_volume_path}")
        return
    except Exception as e:
        print(f"Error loading input TIFF: {e}")
        return
    print(f"Input volume shape (ZYX): {original_volume_zyx.shape}, dtype: {original_volume_zyx.dtype}")
    
    processed_volume = original_volume_zyx.astype(np.float32)

    # 2. Pre-processing: Smoothing
    if PRE_SMOOTHING_SIGMA_PIXELS is not None and PRE_SMOOTHING_SIGMA_PIXELS > 0:
        print(f"Applying Gaussian pre-smoothing with sigma: {PRE_SMOOTHING_SIGMA_PIXELS} pixels...")
        processed_volume = gaussian_filter(processed_volume, sigma=PRE_SMOOTHING_SIGMA_PIXELS, mode='reflect')

    # 3. **** New: Initial Background Pixel Handling ****
    # Identify very dark pixels in the (optionally smoothed) original image
    # These are assumed to be strong background.
    if BACKGROUND_PERCENTILE_THRESHOLD > 0:
        print(f"Identifying and adjusting deep background pixels (below {BACKGROUND_PERCENTILE_THRESHOLD}th percentile)...")
        background_thresh_value = np.percentile(processed_volume, BACKGROUND_PERCENTILE_THRESHOLD)
        print(f"  Background intensity threshold: {background_thresh_value:.2f}")
        # Option 1: Set these background pixels to a common low value (e.g., the threshold itself, or 0)
        # This makes them uniformly "dark" before inversion.
        # processed_volume[processed_volume < background_thresh_value] = background_thresh_value
        # Option 2 (Potentially better for `max - val` inversion):
        # Mark them so they become max after inversion.
        # For `max_intensity_for_inversion - pixel_value` inversion, we want these background pixels
        # to effectively become 0 AFTER inversion. So, they should be set to `max_intensity_for_inversion` BEFORE it.
        # This is tricky before knowing the `max_intensity_for_inversion`.
        # A simpler approach: just ensure these pixels are truly at the bottom of the intensity range.
        # Let's try setting them to the minimum found in the image, to not artificially create a new minimum.
        current_min_val = np.min(processed_volume)
        processed_volume[processed_volume < background_thresh_value] = current_min_val
        print(f"  Deep background pixels set to: {current_min_val:.2f}")


    # 4. Image Inversion (Lumen becomes bright) using Robust Min/Max
    print("Inverting image (making lumen bright) using robust percentiles...")
    # Determine robust min and max for scaling before inversion
    # This happens on the `processed_volume` which has had deep background handled
    robust_min = np.percentile(processed_volume, INVERSION_ROBUST_MIN_PERCENTILE)
    robust_max = np.percentile(processed_volume, INVERSION_ROBUST_MAX_PERCENTILE)
    print(f"  Robust min for inversion scaling ({INVERSION_ROBUST_MIN_PERCENTILE}th percentile): {robust_min:.2f}")
    print(f"  Robust max for inversion scaling ({INVERSION_ROBUST_MAX_PERCENTILE}th percentile): {robust_max:.2f}")

    if robust_max <= robust_min: # Handle flat or nearly flat images
        print("Warning: Robust max <= robust min. Image may be flat. Inverted image will be zero.")
        inverted_volume_scaled = np.zeros_like(processed_volume)
    else:
        # Clip values to the robust range
        clipped_volume = np.clip(processed_volume, robust_min, robust_max)
        # Scale to 0-1 based on this robust range
        normalized_volume_robust = (clipped_volume - robust_min) / (robust_max - robust_min)
        # Invert: 1 - normalized_value
        inverted_volume_scaled = 1.0 - normalized_volume_robust
        
    # `inverted_volume_scaled` is now in the range [0, 1], with lumens bright.

    # 5. Vessel Enhancement Filter (Frangi) on INVERTED and SCALED image
    print(f"Applying Frangi filter for bright lumen detection (scales: {FRANGI_SCALE_RANGE_LUMEN}, step: {FRANGI_SCALE_STEP_LUMEN})...")
    # `inverted_volume_scaled` is already normalized, suitable for Frangi.
    enhanced_lumen_volume = frangi(
        inverted_volume_scaled, # Use the robustly inverted and scaled image
        sigmas=range(FRANGI_SCALE_RANGE_LUMEN[0], FRANGI_SCALE_RANGE_LUMEN[1] + 1, FRANGI_SCALE_STEP_LUMEN),
        alpha=0.5, beta=FRANGI_BETA1, gamma=FRANGI_BETA2,
        black_ridges=False # Looking for BRIGHT lumens
    )
    print("Frangi filter applied.")

    # 6. Thresholding (on the enhanced lumen image)
    # (No changes here, same as before)
    print(f"Thresholding enhanced lumen volume using '{THRESHOLD_METHOD}' method...")
    binary_lumen_volume = np.zeros_like(enhanced_lumen_volume, dtype=np.uint8)
    if isinstance(THRESHOLD_METHOD, (float, int)):
        binary_lumen_volume = (enhanced_lumen_volume > float(THRESHOLD_METHOD)).astype(np.uint8)
    elif THRESHOLD_METHOD == 'otsu':
        mask_for_otsu = enhanced_lumen_volume > 1e-5
        if np.any(mask_for_otsu):
            try:
                threshold_val = threshold_otsu(enhanced_lumen_volume[mask_for_otsu])
                binary_lumen_volume = (enhanced_lumen_volume > threshold_val).astype(np.uint8)
                print(f"Otsu threshold applied: {threshold_val:.4f}")
            except ValueError as e:
                print(f"Warning: Otsu thresholding failed ({e}). Defaulting to empty segmentation.")
        else:
            print("No significant signal found for Otsu thresholding. Resulting lumen segmentation will be empty.")
    else:
        print(f"Error: Unknown threshold method: {THRESHOLD_METHOD}")
        return

    # 7. Post-processing on Lumen Segmentation
    # (No changes here, same as before)
    if OPENING_RADIUS_LUMEN_PIXELS > 0:
        print(f"Applying binary opening to lumen segmentation (radius: {OPENING_RADIUS_LUMEN_PIXELS} pixels)...")
        opening_selem = ball(OPENING_RADIUS_LUMEN_PIXELS)
        binary_lumen_volume = binary_opening(binary_lumen_volume.astype(bool), footprint=opening_selem).astype(np.uint8)

    if MIN_LUMEN_OBJECT_SIZE_VOXELS > 0:
        print(f"Removing small lumen objects (less than {MIN_LUMEN_OBJECT_SIZE_VOXELS} voxels)...")
        binary_lumen_volume = remove_small_objects(binary_lumen_volume.astype(bool),
                                                   min_size=MIN_LUMEN_OBJECT_SIZE_VOXELS).astype(np.uint8)
    print("Lumen segmentation post-processing complete.")


    # 8. Optional: Reconstruct Full Vessel by Dilating Lumen
    # (No changes here, same as before)
    final_output_volume = binary_lumen_volume
    if RECONSTRUCT_FULL_VESSEL:
        if WALL_THICKNESS_FOR_DILATION_PIXELS > 0:
            print(f"Dilating segmented lumen by {WALL_THICKNESS_FOR_DILATION_PIXELS} pixels to reconstruct full vessel...")
            dilation_selem = ball(WALL_THICKNESS_FOR_DILATION_PIXELS)
            final_output_volume = binary_dilation(binary_lumen_volume.astype(bool), footprint=dilation_selem).astype(np.uint8)
            print("Full vessel reconstruction complete.")
        else:
            print("Skipping full vessel reconstruction as WALL_THICKNESS_FOR_DILATION_PIXELS is 0.")
    else:
        print("Skipping full vessel reconstruction (RECONSTRUCT_FULL_VESSEL is False). Outputting lumen segmentation.")

    # 9. Save Output Volume
    # (No changes here, same as before)
    print(f"Saving final segmented volume to: {output_volume_path}")
    try:
        os.makedirs(os.path.dirname(output_volume_path), exist_ok=True)
        tifffile.imwrite(output_volume_path, final_output_volume, imagej=True)
        print("Segmentation saved successfully.")
    except Exception as e:
        print(f"Error saving output TIFF: {e}")

def main():
    parser = argparse.ArgumentParser(description="Capillary vessel segmentation pipeline. Inverts image, handles background.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input 3D TIFF volume.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output segmented 3D TIFF volume.")
    
    args = parser.parse_args()
    
    segment_capillaries(args.input, args.output)

if __name__ == "__main__":
    main()
