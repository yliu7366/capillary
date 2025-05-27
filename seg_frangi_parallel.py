#!/usr/bin/env python3
import argparse
import os
import time
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
from skimage.filters import frangi, threshold_otsu
from skimage.morphology import remove_small_objects, ball, binary_dilation, binary_opening
import multiprocessing

# --- Configuration (Defaults, can be overridden by CLI) ---
DEFAULT_BLOCK_SIZE = 128
DEFAULT_OVERLAP_PIXELS = 16
DEFAULT_NUM_CORES = 24
DEFAULT_SAVE_INTERMEDIATES = False
DEFAULT_INVERT_BEFORE_FRANGI = False
DEFAULT_GLOBAL_INTENSITY_THRESH = 100

# Vessel and Lumen Dimensions (in pixels)
FRANGI_SCALE_RANGE_LUMEN_OR_WALL = (1, 3) # Unified, context depends on inversion
FRANGI_SCALE_STEP = 1
FRANGI_BETA1 = 0.5
FRANGI_BETA2 = 15

# Pre-processing
PRE_SMOOTHING_SIGMA_PIXELS = 0.5

# Background Removal Parameters (Percentiles for robust scaling if inversion happens)
BACKGROUND_PERCENTILE_THRESHOLD_GLOBAL = 5.0 # Applied if NOT using fixed global_intensity_thresh or after it
INVERSION_ROBUST_MIN_PERCENTILE_GLOBAL = 0.5
INVERSION_ROBUST_MAX_PERCENTILE_GLOBAL = 99.5

# Segmentation
THRESHOLD_METHOD = 'otsu'

# Post-processing on Lumen Segmentation
MIN_LUMEN_OBJECT_SIZE_VOXELS = 50
OPENING_RADIUS_LUMEN_PIXELS = 0

# Optional: Reconstruct Full Vessel (Meaning changes if not inverting)
RECONSTRUCT_FULL_VESSEL = True # If inverting, this dilates lumen. If not, this is the direct vessel.
WALL_THICKNESS_FOR_DILATION_PIXELS = 2 # Only used if inverting AND reconstructing

# ==============================================================================
# TOP-LEVEL FUNCTIONS FOR MULTIPROCESSING
# ==============================================================================

def process_single_3d_block_mp(input_block_data_with_overlap, params_dict):
    """
    Processes a single 3D block.
    Inversion logic is now conditional based on params_dict['perform_inversion'].
    """
    # Unpack parameters
    perform_inversion = params_dict['perform_inversion']
    # Background/robust thresholds are now specific to inversion if it happens
    # If not inverting, these might not be used for the inversion step,
    # but global_background_thresh_val might still be used for initial cleanup.
    global_background_thresh_val_for_inversion = params_dict.get('global_background_thresh_val_for_inversion') # May be None
    global_robust_min_val_for_inversion = params_dict.get('global_robust_min_val_for_inversion') # May be None
    global_robust_max_val_for_inversion = params_dict.get('global_robust_max_val_for_inversion') # May be None

    frangi_scales = params_dict['frangi_scales']
    frangi_beta1 = params_dict['frangi_beta1']
    frangi_beta2 = params_dict['frangi_beta2']
    frangi_black_ridges = params_dict['frangi_black_ridges'] # New: depends on inversion
    
    threshold_method_val = params_dict['threshold_method_val']
    opening_radius = params_dict['opening_radius'] # Generic name now
    min_obj_size = params_dict['min_obj_size']
    
    reconstruct_if_inverted = params_dict['reconstruct_if_inverted'] # Specific to inversion case
    wall_thickness_dilation = params_dict['wall_thickness_dilation']


    # Start with input block (already a copy from the main process's working_volume)
    # which has had global fixed thresholding and global smoothing applied.
    processed_block = input_block_data_with_overlap.astype(np.float32, copy=False)

    frangi_input = processed_block # Default input to Frangi

    if perform_inversion:
        # Apply background handling specifically for inversion, using global inversion-specific thresholds
        if global_background_thresh_val_for_inversion is not None:
            # This step assumes processed_block's values are still in original-like scale before inversion
            processed_block_for_inv = processed_block.copy() # Modify a copy for inversion
            processed_block_for_inv[processed_block_for_inv < global_background_thresh_val_for_inversion] = global_background_thresh_val_for_inversion
        else:
            processed_block_for_inv = processed_block # Use as-is if no specific bg thresh for inversion

        # Image Inversion using GLOBAL Robust Min/Max for inversion
        if global_robust_max_val_for_inversion is None or global_robust_min_val_for_inversion is None or \
           global_robust_max_val_for_inversion <= global_robust_min_val_for_inversion:
            inverted_volume_scaled = np.zeros_like(processed_block_for_inv, dtype=np.float32)
        else:
            clipped_volume = np.clip(processed_block_for_inv, global_robust_min_val_for_inversion, global_robust_max_val_for_inversion)
            normalized_volume_robust = (clipped_volume - global_robust_min_val_for_inversion) / \
                                       (global_robust_max_val_for_inversion - global_robust_min_val_for_inversion)
            inverted_volume_scaled = (1.0 - normalized_volume_robust).astype(np.float32)
        frangi_input = inverted_volume_scaled # Frangi operates on the inverted image
    
    # Ensure frangi_input is float32
    if frangi_input.dtype != np.float32:
        frangi_input = frangi_input.astype(np.float32)

    # Vessel Enhancement Filter (Frangi)
    enhanced_block = frangi(
        frangi_input,
        sigmas=frangi_scales,
        alpha=0.5, beta=frangi_beta1, gamma=frangi_beta2,
        black_ridges=frangi_black_ridges # Use appropriate setting
    )
    if enhanced_block.dtype != np.float32: # Ensure float32 for consistency
        enhanced_block = enhanced_block.astype(np.float32)

    # Thresholding
    binary_block = np.zeros_like(enhanced_block, dtype=np.uint8)
    # ... (Otsu/fixed thresholding logic as before, on `enhanced_block`)
    if isinstance(threshold_method_val, (float, int)):
        binary_block = (enhanced_block > float(threshold_method_val)).astype(np.uint8)
    elif threshold_method_val == 'otsu':
        mask_for_otsu = enhanced_block > 1e-5
        if np.any(mask_for_otsu):
            try:
                threshold_val = threshold_otsu(enhanced_block[mask_for_otsu])
                binary_block = (enhanced_block > threshold_val).astype(np.uint8)
            except ValueError:
                 pass
    else:
        raise ValueError(f"Unknown threshold method in block processing: {threshold_method_val}")


    # Post-processing (e.g., remove small objects, opening)
    if opening_radius > 0:
        opening_selem = ball(opening_radius)
        binary_block = binary_opening(binary_block.astype(bool), footprint=opening_selem).astype(np.uint8)

    if min_obj_size > 0:
        binary_block = remove_small_objects(binary_block.astype(bool),
                                             min_size=min_obj_size).astype(np.uint8)

    # Optional: Reconstruct Full Vessel (only if inversion was done and reconstruction is flagged)
    final_block_segmentation = binary_block
    if perform_inversion and reconstruct_if_inverted and wall_thickness_dilation > 0:
        dilation_selem = ball(wall_thickness_dilation)
        final_block_segmentation = binary_dilation(binary_block.astype(bool), footprint=dilation_selem).astype(np.uint8)
    
    return final_block_segmentation

# ==============================================================================

# --- Main Segmentation Function ---
def segment_capillaries_3d_blocks(input_volume_path: str, output_volume_path: str,
                                  block_size_val: int, overlap_val: int, num_cores_val: int,
                                  save_intermediates_flag: bool,
                                  perform_inversion_flag: bool, # New argument
                                  global_intensity_threshold_val: int): # New argument
    
    print(f"Starting capillary segmentation for: {input_volume_path}")
    print(f"  Output will be saved to: {output_volume_path}")
    print(f"  Perform inversion before Frangi: {perform_inversion_flag}")
    print(f"  Global intensity threshold for initial background: {global_intensity_threshold_val if global_intensity_threshold_val >=0 else 'Disabled'}")
    # ... (other printouts) ...
    start_time_total = time.time()

    input_basename = os.path.splitext(os.path.basename(input_volume_path))[0]
    output_dir = os.path.dirname(output_volume_path)
    if not output_dir: output_dir = "."
    os.makedirs(output_dir, exist_ok=True)

    print("Loading input volume...")
    try:
        original_volume_zyx = tifffile.imread(input_volume_path)
    except FileNotFoundError: # ... (error handling) ...
        return
    except Exception as e: # ... (error handling) ...
        return
    load_time = time.time()
    print(f"Input volume shape (ZYX): {original_volume_zyx.shape}, dtype: {original_volume_zyx.dtype}. Loaded in {load_time - start_time_total:.2f}s.")

    working_volume = original_volume_zyx.astype(np.float32, copy=True)

    # 1. **** NEW: Apply Global Fixed Intensity Threshold ****
    if global_intensity_threshold_val >= 0: # Apply if not negative
        print(f"Applying global fixed intensity threshold: values < {global_intensity_threshold_val} set to {global_intensity_threshold_val}...")
        # Option 1: set to the threshold value
        working_volume[working_volume < global_intensity_threshold_val] = global_intensity_threshold_val
        # Option 2: set to 0 (might be better if not inverting, and Frangi looks for bright structures)
        # working_volume[working_volume < global_intensity_threshold_val] = 0
        print("Global fixed intensity threshold applied.")
        if save_intermediates_flag:
            inter_fixed_thresh_path = os.path.join(output_dir, f"{input_basename}_fixedthreshed.tif")
            print(f"  Saving fixed-thresholded volume to: {inter_fixed_thresh_path}")
            tifffile.imwrite(inter_fixed_thresh_path, working_volume.astype(original_volume_zyx.dtype), imagej=True) # Save in original-like type


    # 2. GLOBAL Pre-processing: Smoothing (on potentially thresholded volume)
    if PRE_SMOOTHING_SIGMA_PIXELS is not None and PRE_SMOOTHING_SIGMA_PIXELS > 0:
        print(f"Applying GLOBAL Gaussian pre-smoothing with sigma: {PRE_SMOOTHING_SIGMA_PIXELS} pixels...")
        working_volume = gaussian_filter(working_volume, sigma=PRE_SMOOTHING_SIGMA_PIXELS, mode='reflect')
        print("Global smoothing complete.")
        if save_intermediates_flag:
            inter_smooth_path = os.path.join(output_dir, f"{input_basename}_smoothed.tif")
            print(f"  Saving smoothed volume to: {inter_smooth_path}")
            tifffile.imwrite(inter_smooth_path, working_volume.astype(np.float32), imagej=True)

    # 3. GLOBAL Calculation of Percentile-Based Thresholds (primarily for inversion if used)
    glob_bg_thresh_val_for_inv = None
    glob_robust_min_val_for_inv = None
    glob_robust_max_val_for_inv = None

    if perform_inversion_flag:
        print("Calculating GLOBAL thresholds for robust inversion (from current working_volume)...")
        if BACKGROUND_PERCENTILE_THRESHOLD_GLOBAL > 0: # This percentile is for further bg cleanup before inversion
            glob_bg_thresh_val_for_inv = np.percentile(working_volume, BACKGROUND_PERCENTILE_THRESHOLD_GLOBAL)
            print(f"  Global background intensity threshold for inversion ({BACKGROUND_PERCENTILE_THRESHOLD_GLOBAL}th perc.): {glob_bg_thresh_val_for_inv:.2f}")

        glob_robust_min_val_for_inv = np.percentile(working_volume, INVERSION_ROBUST_MIN_PERCENTILE_GLOBAL)
        glob_robust_max_val_for_inv = np.percentile(working_volume, INVERSION_ROBUST_MAX_PERCENTILE_GLOBAL)
        print(f"  Global robust min for inversion scaling ({INVERSION_ROBUST_MIN_PERCENTILE_GLOBAL}th perc.): {glob_robust_min_val_for_inv:.2f}")
        print(f"  Global robust max for inversion scaling ({INVERSION_ROBUST_MAX_PERCENTILE_GLOBAL}th perc.): {glob_robust_max_val_for_inv:.2f}")

        if glob_robust_max_val_for_inv <= glob_robust_min_val_for_inv:
            print("Warning: Global robust max <= global robust min for inversion. Inverted image in blocks will be zero.")
        
        if save_intermediates_flag:
            # Create a temporary globally inverted volume just for saving this intermediate step
            temp_inverted_for_save = np.zeros_like(working_volume, dtype=np.float32)
            if glob_robust_max_val_for_inv > glob_robust_min_val_for_inv:
                temp_working_for_inv = working_volume.copy()
                if glob_bg_thresh_val_for_inv is not None:
                    temp_working_for_inv[temp_working_for_inv < glob_bg_thresh_val_for_inv] = glob_bg_thresh_val_for_inv
                
                temp_clipped = np.clip(temp_working_for_inv, glob_robust_min_val_for_inv, glob_robust_max_val_for_inv)
                temp_normalized = (temp_clipped - glob_robust_min_val_for_inv) / \
                                  (glob_robust_max_val_for_inv - glob_robust_min_val_for_inv)
                temp_inverted_for_save = (1.0 - temp_normalized).astype(np.float32)
            
            inter_inverted_path = os.path.join(output_dir, f"{input_basename}_globally_inverted_preview.tif")
            print(f"  Saving preview of globally inverted_scaled volume to: {inter_inverted_path}")
            tifffile.imwrite(inter_inverted_path, temp_inverted_for_save, imagej=True)
            del temp_inverted_for_save # Free memory

    # Prepare output array and block generation
    output_volume_zyx = np.zeros_like(original_volume_zyx, dtype=np.uint8)
    dim_z, dim_y, dim_x = working_volume.shape
    block_info_list_for_workers = [] 
    # ... (block generation logic as before, extracting blocks from `working_volume`) ...
    print("Generating 3D block coordinates for processing...")
    for z_start_orig in range(0, dim_z, block_size_val):
        z_end_orig = min(z_start_orig + block_size_val, dim_z)
        read_z_start = max(0, z_start_orig - overlap_val)
        read_z_end = min(dim_z, z_end_orig + overlap_val)
        for y_start_orig in range(0, dim_y, block_size_val):
            y_end_orig = min(y_start_orig + block_size_val, dim_y)
            read_y_start = max(0, y_start_orig - overlap_val)
            read_y_end = min(dim_y, y_end_orig + overlap_val)
            for x_start_orig in range(0, dim_x, block_size_val):
                x_end_orig = min(x_start_orig + block_size_val, dim_x)
                read_x_start = max(0, x_start_orig - overlap_val)
                read_x_end = min(dim_x, x_end_orig + overlap_val)
                input_block_with_overlap = working_volume[
                    read_z_start:read_z_end,
                    read_y_start:read_y_end,
                    read_x_start:read_x_end
                ].copy()
                block_info = {
                    'input_data': input_block_with_overlap,
                    'orig_slices_for_stitching': (slice(z_start_orig, z_end_orig),
                                    slice(y_start_orig, y_end_orig),
                                    slice(x_start_orig, x_end_orig)),
                    'crop_in_block_slices_for_stitching': (
                        slice(z_start_orig - read_z_start, (z_start_orig - read_z_start) + (z_end_orig - z_start_orig)),
                        slice(y_start_orig - read_y_start, (y_start_orig - read_y_start) + (y_end_orig - y_start_orig)),
                        slice(x_start_orig - read_x_start, (x_start_orig - read_x_start) + (x_end_orig - x_start_orig))
                    )
                }
                block_info_list_for_workers.append(block_info)

    num_blocks = len(block_info_list_for_workers)
    print(f"Total number of 3D blocks to process: {num_blocks}")

    # Determine Frangi's black_ridges based on inversion flag
    frangi_black_ridges_setting = True if perform_inversion_flag else False
    # If NOT inverting, we assume original image has bright vessels (dark lumen)
    # So Frangi should look for bright ridges (black_ridges=False).
    # If we ARE inverting, lumen becomes bright, so Frangi looks for bright ridges (black_ridges=False).
    # Wait, this logic is simpler:
    # If inverting (lumen becomes bright), black_ridges=False.
    # If NOT inverting (lumen is dark, walls are bright), black_ridges=False (to find bright walls).
    # So, black_ridges is generally False for these scenarios.
    # If vessels were dark tubes in bright background (and not inverting), then black_ridges=True.
    # Let's assume current setup is always for bright structures of interest after any inversion.
    frangi_black_ridges_setting = False


    shared_processing_params = {
        'perform_inversion': perform_inversion_flag,
        'global_background_thresh_val_for_inversion': glob_bg_thresh_val_for_inv, # For inversion step if active
        'global_robust_min_val_for_inversion': glob_robust_min_val_for_inv,       # For inversion step if active
        'global_robust_max_val_for_inversion': glob_robust_max_val_for_inv,       # For inversion step if active
        'frangi_scales': range(FRANGI_SCALE_RANGE_LUMEN_OR_WALL[0], FRANGI_SCALE_RANGE_LUMEN_OR_WALL[1] + 1, FRANGI_SCALE_STEP),
        'frangi_beta1': FRANGI_BETA1, 'frangi_beta2': FRANGI_BETA2,
        'frangi_black_ridges': frangi_black_ridges_setting, # Pass this to the block processor
        'threshold_method_val': THRESHOLD_METHOD,
        'opening_radius': OPENING_RADIUS_LUMEN_PIXELS, # Renamed to be more generic
        'min_obj_size': MIN_LUMEN_OBJECT_SIZE_VOXELS, # Renamed
        'reconstruct_if_inverted': RECONSTRUCT_FULL_VESSEL, # Specific flag
        'wall_thickness_dilation': WALL_THICKNESS_FOR_DILATION_PIXELS
    }
    
    starmap_args = [(bi['input_data'], shared_processing_params) for bi in block_info_list_for_workers]

    processing_start_time = time.time()
    # ... (multiprocessing pool logic as before) ...
    if num_cores_val > 1 and num_blocks > 1:
        print(f"Processing {num_blocks} blocks in parallel using {num_cores_val} cores...")
        with multiprocessing.Pool(processes=num_cores_val) as pool:
            processed_blocks_with_overlap = pool.starmap(process_single_3d_block_mp, starmap_args)
    else:
        print(f"Processing {num_blocks} blocks sequentially...")
        processed_blocks_with_overlap = []
        for i, (block_data, params_dict) in enumerate(starmap_args):
            if (i + 1) % (max(1, num_blocks // 10 if num_blocks >=10 else 1)) == 0 or i == num_blocks -1 :
                 print(f"  Processing block {i+1}/{num_blocks}...")
            processed_blocks_with_overlap.append(process_single_3d_block_mp(block_data, params_dict))

    processing_time = time.time() - processing_start_time
    print(f"All blocks processed in {processing_time:.2f}s.")

    # Stitch results
    # ... (stitching logic as before) ...
    print("Stitching processed blocks...")
    for i, processed_block_ov in enumerate(processed_blocks_with_overlap):
        block_info_for_stitching = block_info_list_for_workers[i]
        orig_slices = block_info_for_stitching['orig_slices_for_stitching']
        crop_slices = block_info_for_stitching['crop_in_block_slices_for_stitching']
        valid_part = processed_block_ov[crop_slices]
        output_volume_zyx[orig_slices] = valid_part
    print(f"Stitching complete.")


    # Save Output Volume
    # ... (saving logic as before) ...
    print(f"Saving final segmented volume to: {output_volume_path}")
    try:
        os.makedirs(os.path.dirname(output_volume_path), exist_ok=True)
        tifffile.imwrite(output_volume_path, output_volume_zyx, imagej=True)
        print("Segmentation saved successfully.")
    except Exception as e:
        print(f"Error saving output TIFF: {e}")
    
    total_duration = time.time() - start_time_total
    print(f"Total pipeline execution time: {total_duration:.2f}s")

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Capillary vessel segmentation with 3D block processing.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input 3D TIFF volume.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output segmented 3D TIFF volume.")
    parser.add_argument("--block_size", type=int, default=DEFAULT_BLOCK_SIZE,
                        help=f"Size of the 3D processing blocks. Default: {DEFAULT_BLOCK_SIZE}")
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP_PIXELS,
                        help=f"Overlap in pixels between blocks. Default: {DEFAULT_OVERLAP_PIXELS}")
    parser.add_argument("--cores", type=int, default=DEFAULT_NUM_CORES,
                        help=f"Number of CPU cores to use. Default: {DEFAULT_NUM_CORES}")
    parser.add_argument("--save_intermediates", action='store_true', default=DEFAULT_SAVE_INTERMEDIATES,
                        help="Save intermediate smoothed and (if applicable) inverted volumes. Default: False")
    parser.add_argument("--invert_before_frangi", action='store_true', default=DEFAULT_INVERT_BEFORE_FRANGI,
                        help="Invert the volume (lumen becomes bright) before Frangi filtering. Default: False")
    parser.add_argument("--global_intensity_thresh", type=int, default=DEFAULT_GLOBAL_INTENSITY_THRESH,
                        help="Global intensity threshold to apply to input volume (values below are set to this value). Applied before smoothing. Set to -1 to disable. Default: 100")

    args = parser.parse_args()

    total_cpus = multiprocessing.cpu_count()
    if total_cpus <= 1:
        num_cores_to_use = 1
        if args.cores > 1: print(f"Warning: Only {total_cpus} core(s) available. Running sequentially.")
    else:
        max_usable_cores = total_cpus - 1
        num_cores_to_use = min(max(1, args.cores), max_usable_cores)
        if args.cores > max_usable_cores and args.cores > 1 : # only warn if user asked for more than 1 and more than usable
            print(f"Warning: Requested {args.cores} cores. Capping at {num_cores_to_use} to reserve 1 core for the system (total available: {total_cpus}).")
        elif args.cores < 1:
            print(f"Warning: Requested {args.cores} cores. Using 1 core.")
            num_cores_to_use = 1
        
    segment_capillaries_3d_blocks(args.input, args.output,
                                  args.block_size, args.overlap, num_cores_to_use,
                                  args.save_intermediates,
                                  args.invert_before_frangi,
                                  args.global_intensity_thresh) # Pass new args

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
