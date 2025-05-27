#!/usr/bin/env python3
import argparse
import os
import time
import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter
from skimage.filters import frangi, threshold_otsu, apply_hysteresis_threshold
from skimage.morphology import remove_small_objects, ball, binary_dilation, binary_opening, skeletonize
import multiprocessing

# --- Configuration (Defaults, can be overridden by CLI) ---
DEFAULT_BLOCK_SIZE = 128
DEFAULT_OVERLAP_PIXELS = 16
DEFAULT_NUM_CORES = 24
DEFAULT_SAVE_INTERMEDIATES = False
DEFAULT_GLOBAL_INTENSITY_THRESH = 100

# Frangi parameters (Targeting bright structures, e.g., walls or vessels)
FRANGI_SCALE_RANGE = (1, 2) 
FRANGI_SCALE_STEP = 1
FRANGI_BETA1 = 0.5
FRANGI_BETA2 = 15

# Pre-processing
PRE_SMOOTHING_SIGMA_PIXELS = 0.5

# Segmentation
DEFAULT_THRESHOLD_METHOD = 'hysteresis'
DEFAULT_HYSTERESIS_LOW = 0.05
DEFAULT_HYSTERESIS_HIGH = 0.15
DEFAULT_FIXED_THRESHOLD_VALUE = 0.1

# Post-processing
MIN_OBJECT_SIZE_VOXELS = 50
OPENING_RADIUS_PIXELS = 0

# ==============================================================================
# TOP-LEVEL FUNCTIONS FOR MULTIPROCESSING
# ==============================================================================

def process_single_3d_block_mp(input_block_data_with_overlap, params_dict):
    """
    Processes a single 3D block: Frangi, thresholding, post-processing.
    No inversion is performed. Input block is assumed to be globally preprocessed.
    Returns a tuple: (final_binary_segmentation, frangi_enhanced_block_calibrated)
    """
    # Unpack parameters
    frangi_scales = params_dict['frangi_scales']
    frangi_beta1 = params_dict['frangi_beta1']
    frangi_beta2 = params_dict['frangi_beta2']
    
    threshold_method_val = params_dict['threshold_method_val']
    hysteresis_low_val = params_dict['hysteresis_low']
    hysteresis_high_val = params_dict['hysteresis_high']
    fixed_threshold_val = params_dict['fixed_threshold']

    opening_radius = params_dict['opening_radius']
    min_obj_size = params_dict['min_obj_size']
    
    processed_block = input_block_data_with_overlap.astype(np.float32, copy=False)
    
    if processed_block.dtype != np.float32:
        frangi_input = processed_block.astype(np.float32)
    else:
        frangi_input = processed_block

    # Vessel Enhancement Filter (Frangi)
    enhanced_block_raw_frangi = frangi(
        frangi_input,
        sigmas=frangi_scales,
        alpha=0.5, beta=frangi_beta1, gamma=frangi_beta2,
        black_ridges=False # Target bright structures
    )
    if enhanced_block_raw_frangi.dtype != np.float32:
        enhanced_block_raw_frangi = enhanced_block_raw_frangi.astype(np.float32)

    enhanced_block_for_thresh = np.nan_to_num(enhanced_block_raw_frangi, nan=0.0, posinf=1.0, neginf=0.0)
    min_fr, max_fr = np.min(enhanced_block_for_thresh), np.max(enhanced_block_for_thresh)
    if max_fr > min_fr:
        enhanced_block_calibrated = (enhanced_block_for_thresh - min_fr) / (max_fr - min_fr)
    else:
        enhanced_block_calibrated = np.zeros_like(enhanced_block_for_thresh, dtype=np.float32)

    binary_block = np.zeros_like(enhanced_block_calibrated, dtype=np.uint8)
    if threshold_method_val == 'otsu':
        mask_for_otsu = enhanced_block_calibrated > 1e-5
        if np.any(mask_for_otsu):
            try:
                threshold_val = threshold_otsu(enhanced_block_calibrated[mask_for_otsu])
                binary_block = (enhanced_block_calibrated > threshold_val).astype(np.uint8)
            except ValueError: pass
    elif threshold_method_val == 'hysteresis':
        binary_block = apply_hysteresis_threshold(enhanced_block_calibrated, hysteresis_low_val, hysteresis_high_val).astype(np.uint8)
    elif threshold_method_val == 'fixed':
        binary_block = (enhanced_block_calibrated > fixed_threshold_val).astype(np.uint8)
    else:
        raise ValueError(f"Unknown threshold method: {threshold_method_val}")

    if opening_radius > 0:
        opening_selem = ball(opening_radius)
        binary_block = binary_opening(binary_block.astype(bool), footprint=opening_selem).astype(np.uint8)
    if min_obj_size > 0:
        binary_block = remove_small_objects(binary_block.astype(bool), min_size=min_obj_size).astype(np.uint8)
    
    final_block_segmentation = binary_block
    
    return final_block_segmentation, enhanced_block_calibrated

# ==============================================================================

# --- Main Segmentation Function ---
def segment_capillaries_3d_blocks(input_volume_path: str, output_volume_path: str,
                                  block_size_val: int, overlap_val: int, num_cores_val: int,
                                  save_intermediates_flag: bool,
                                  global_intensity_threshold_val: int,
                                  arg_threshold_method: str,
                                  arg_hysteresis_low: float,
                                  arg_hysteresis_high: float,
                                  arg_fixed_threshold: float,
                                  skeleton_output_path: str = None):
    
    print(f"Starting capillary segmentation for: {input_volume_path}")
    print(f"  Output will be saved to: {output_volume_path}")
    if skeleton_output_path:
        print(f"  Skeleton output will be saved to: {skeleton_output_path}")
    print(f"  Global intensity threshold (pixels < thresh set to 0): {global_intensity_threshold_val if global_intensity_threshold_val >=0 else 'Disabled'}")
    print(f"  Segmentation threshold method: {arg_threshold_method}")
    if arg_threshold_method == 'hysteresis':
        print(f"    Hysteresis Low: {arg_hysteresis_low}, High: {arg_hysteresis_high}")
    elif arg_threshold_method == 'fixed':
        print(f"    Fixed Threshold: {arg_fixed_threshold}")
    print(f"  Processing with 3D blocks: {block_size_val}^3 pixels")
    print(f"  Overlap: {overlap_val} pixels")
    print(f"  Number of CPU cores to use: {num_cores_val}")
    print(f"  Save intermediate steps: {save_intermediates_flag}")

    start_time_total = time.time()
    input_basename = os.path.splitext(os.path.basename(input_volume_path))[0]
    output_dir_main = os.path.dirname(output_volume_path)
    if not output_dir_main: output_dir_main = "."
    os.makedirs(output_dir_main, exist_ok=True)
    
    output_dir_inter_skel = output_dir_main
    if skeleton_output_path:
        skel_dir = os.path.dirname(skeleton_output_path)
        if skel_dir : os.makedirs(skel_dir, exist_ok=True)

    print("Loading input volume...")
    try:
        original_volume_zyx = tifffile.imread(input_volume_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_volume_path}")
        return
    except Exception as e:
        print(f"Error loading input TIFF: {e}")
        return
    load_time = time.time()
    print(f"Input volume shape (ZYX): {original_volume_zyx.shape}, dtype: {original_volume_zyx.dtype}. Loaded in {load_time - start_time_total:.2f}s.")

    working_volume = original_volume_zyx.astype(np.float32, copy=True)

    if global_intensity_threshold_val >= 0:
        print(f"Applying global fixed intensity threshold: values < {global_intensity_threshold_val} set to 0...")
        working_volume[working_volume < global_intensity_threshold_val] = 0
        print("Global fixed intensity threshold applied.")
        if save_intermediates_flag:
            inter_path = os.path.join(output_dir_inter_skel, f"{input_basename}_01_fixedthreshed.tif")
            print(f"  Saving fixed-thresholded volume to: {inter_path}")
            tifffile.imwrite(inter_path, working_volume.astype(original_volume_zyx.dtype), imagej=True)

    if PRE_SMOOTHING_SIGMA_PIXELS is not None and PRE_SMOOTHING_SIGMA_PIXELS > 0:
        print(f"Applying GLOBAL Gaussian pre-smoothing with sigma: {PRE_SMOOTHING_SIGMA_PIXELS} pixels...")
        working_volume = gaussian_filter(working_volume, sigma=PRE_SMOOTHING_SIGMA_PIXELS, mode='reflect')
        print("Global smoothing complete.")
        if save_intermediates_flag:
            inter_path = os.path.join(output_dir_inter_skel, f"{input_basename}_02_smoothed.tif")
            print(f"  Saving smoothed volume to: {inter_path}")
            tifffile.imwrite(inter_path, working_volume.astype(np.float32), imagej=True)

    output_volume_zyx = np.zeros_like(original_volume_zyx, dtype=np.uint8)
    if save_intermediates_flag:
        full_frangi_response_volume = np.zeros_like(working_volume, dtype=np.float32)
    else:
        full_frangi_response_volume = None

    dim_z, dim_y, dim_x = working_volume.shape
    block_info_list_for_workers = []
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

    shared_processing_params = {
        # 'perform_inversion': False, # Removed as it's always False
        'frangi_scales': range(FRANGI_SCALE_RANGE[0], FRANGI_SCALE_RANGE[1] + 1, FRANGI_SCALE_STEP),
        'frangi_beta1': FRANGI_BETA1, 'frangi_beta2': FRANGI_BETA2,
        # 'frangi_black_ridges' is hardcoded to False in process_single_3d_block_mp
        'threshold_method_val': arg_threshold_method,
        'hysteresis_low': arg_hysteresis_low,
        'hysteresis_high': arg_hysteresis_high,
        'fixed_threshold': arg_fixed_threshold,
        'opening_radius': OPENING_RADIUS_PIXELS,
        'min_obj_size': MIN_OBJECT_SIZE_VOXELS,
    }
    
    starmap_args = [(bi['input_data'], shared_processing_params) for bi in block_info_list_for_workers]

    processing_start_time = time.time()
    if num_cores_val > 1 and num_blocks > 1:
        print(f"Processing {num_blocks} blocks in parallel using {num_cores_val} cores...")
        with multiprocessing.Pool(processes=num_cores_val) as pool:
            results_from_workers = pool.starmap(process_single_3d_block_mp, starmap_args)
    else:
        print(f"Processing {num_blocks} blocks sequentially...")
        results_from_workers = []
        for i, (block_data, params_dict) in enumerate(starmap_args):
            if (i + 1) % (max(1, num_blocks // 10 if num_blocks >=10 else 1)) == 0 or i == num_blocks -1 :
                 print(f"  Processing block {i+1}/{num_blocks}...")
            results_from_workers.append(process_single_3d_block_mp(block_data, params_dict))

    processing_time = time.time() - processing_start_time
    print(f"All blocks processed in {processing_time:.2f}s.")

    print("Stitching processed blocks...")
    for i, (processed_binary_block_ov, processed_frangi_block_ov) in enumerate(results_from_workers):
        block_info_for_stitching = block_info_list_for_workers[i]
        orig_slices = block_info_for_stitching['orig_slices_for_stitching']
        crop_slices = block_info_for_stitching['crop_in_block_slices_for_stitching']
        valid_binary_part = processed_binary_block_ov[crop_slices]
        output_volume_zyx[orig_slices] = valid_binary_part
        if save_intermediates_flag and full_frangi_response_volume is not None:
            valid_frangi_part = processed_frangi_block_ov[crop_slices]
            full_frangi_response_volume[orig_slices] = valid_frangi_part
    print(f"Stitching complete.")

    if save_intermediates_flag and full_frangi_response_volume is not None:
        inter_path = os.path.join(output_dir_inter_skel, f"{input_basename}_03_frangi_response.tif")
        print(f"  Saving full Frangi response volume to: {inter_path}")
        tifffile.imwrite(inter_path, full_frangi_response_volume.astype(np.float32), imagej=True)

    if skeleton_output_path:
        print("Performing 3D skeletonization on the final segmented volume...")
        try:
            skeleton_start_time = time.time()
            skeleton_volume = skeletonize(output_volume_zyx.astype(bool))
            skeleton_volume_uint8 = skeleton_volume.astype(np.uint8) * 255
            skeleton_time = time.time() - skeleton_start_time
            print(f"Skeletonization complete in {skeleton_time:.2f}s.")
            skel_out_dir_actual = os.path.dirname(skeleton_output_path)
            if skel_out_dir_actual : os.makedirs(skel_out_dir_actual, exist_ok=True)
            print(f"Saving skeletonized volume to: {skeleton_output_path}")
            tifffile.imwrite(skeleton_output_path, skeleton_volume_uint8, imagej=True)
            print("Skeleton saved successfully.")
        except MemoryError:
            print("MemoryError during global skeletonization. The segmented volume might be too large.")
        except Exception as e:
            print(f"Error during skeletonization or saving skeleton: {e}")

    print(f"Saving final segmented volume to: {output_volume_path}")
    try:
        tifffile.imwrite(output_volume_path, output_volume_zyx, imagej=True)
        print("Segmentation saved successfully.")
    except Exception as e:
        print(f"Error saving output TIFF: {e}")
    
    total_duration = time.time() - start_time_total
    print(f"Total pipeline execution time: {total_duration:.2f}s")

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description="Capillary vessel segmentation with 3D block processing.")
    parser.add_argument("--input", type=str, required=True, help="Path to input 3D TIFF.")
    parser.add_argument("--output", type=str, required=True, help="Path to save output segmentation TIFF.")
    parser.add_argument("--skeleton_output", type=str, default=None, help="Optional: Path to save skeleton TIFF.")
    parser.add_argument("--block_size", type=int, default=DEFAULT_BLOCK_SIZE, help=f"Block size. Default: {DEFAULT_BLOCK_SIZE}")
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP_PIXELS, help=f"Overlap pixels. Default: {DEFAULT_OVERLAP_PIXELS}")
    parser.add_argument("--cores", type=int, default=DEFAULT_NUM_CORES, help=f"CPU cores. Default: {DEFAULT_NUM_CORES}")
    parser.add_argument("--save_intermediates", action='store_true', default=DEFAULT_SAVE_INTERMEDIATES, help="Save intermediates. Default: False")
    parser.add_argument("--global_intensity_thresh", type=int, default=DEFAULT_GLOBAL_INTENSITY_THRESH,
                        help=f"Global intensity threshold (pixels < thresh set to 0). -1 to disable. Default: {DEFAULT_GLOBAL_INTENSITY_THRESH}")
    parser.add_argument("--threshold_method", type=str, default=DEFAULT_THRESHOLD_METHOD,
                        choices=['otsu', 'hysteresis', 'fixed'], help=f"Threshold method. Default: {DEFAULT_THRESHOLD_METHOD}")
    parser.add_argument("--hysteresis_low", type=float, default=DEFAULT_HYSTERESIS_LOW, help=f"Low for hysteresis. Default: {DEFAULT_HYSTERESIS_LOW}")
    parser.add_argument("--hysteresis_high", type=float, default=DEFAULT_HYSTERESIS_HIGH, help=f"High for hysteresis. Default: {DEFAULT_HYSTERESIS_HIGH}")
    parser.add_argument("--fixed_threshold_value", type=float, default=DEFAULT_FIXED_THRESHOLD_VALUE, help=f"Value for fixed threshold. Default: {DEFAULT_FIXED_THRESHOLD_VALUE}")

    args = parser.parse_args()

    total_cpus = multiprocessing.cpu_count()
    if total_cpus <= 1:
        num_cores_to_use = 1
        if args.cores > 1: print(f"Warning: Only {total_cpus} core(s) available. Running sequentially.")
    else:
        max_usable_cores = total_cpus - 1
        num_cores_to_use = min(max(1, args.cores), max_usable_cores)
        if args.cores > max_usable_cores and args.cores > 1 :
            print(f"Warning: Requested {args.cores} cores. Capping at {num_cores_to_use} (total available: {total_cpus}).")
        elif args.cores < 1:
            print(f"Warning: Requested {args.cores} cores. Using 1 core.")
            num_cores_to_use = 1
        
    segment_capillaries_3d_blocks(args.input, args.output,
                                  args.block_size, args.overlap, num_cores_to_use,
                                  args.save_intermediates,
                                  args.global_intensity_thresh,
                                  args.threshold_method,
                                  args.hysteresis_low,
                                  args.hysteresis_high,
                                  args.fixed_threshold_value,
                                  args.skeleton_output)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
