#!/usr/bin/env python3
import argparse
import os
import glob
import time
import numpy as np
import tifffile
import traceback

try:
    import vtk
except ImportError:
    print("Error: vtk library not found. Please install it: pip install vtk")
    vtk = None

try:
    from scipy.ndimage import zoom
except ImportError:
    print("Warning: scipy library not found. Downsampling feature will be unavailable.")
    print("         Install it for downsampling: pip install scipy")
    zoom = None

# Default values
DEFAULT_MARCHING_CUBES_LEVEL = 0.5
DEFAULT_SMOOTHING_ITERATIONS = 2
DEFAULT_SMOOTHING_PASSBAND = 0.1
DEFAULT_DOWNSAMPLE_FACTOR = (1.0, 1.0, 1.0)
DEFAULT_DECIMATION_REDUCTION = 0.0

def clean_polydata(polydata: vtk.vtkPolyData) -> vtk.vtkPolyData:
    """
    Cleans a vtkPolyData object using the high-performance vtkStaticCleanPolyData.
    This filter is optimized for removing degenerate cells and unused points without
    modifying point coordinates.
    """
    original_cell_count = polydata.GetNumberOfCells()
    if original_cell_count == 0:
        return polydata

    print(f"  Performing high-performance mesh cleanup with vtkStaticCleanPolyData...")
    cleaner = vtk.vtkStaticCleanPolyData()
    cleaner.SetInputData(polydata)
    
    # This is a valid method and good practice to keep.
    cleaner.SetRemoveUnusedPoints(True)
    
    # Degenerate cell removal is an implicit, default behavior of this filter.
    # The erroneous SetRemoveDegenerateCells(True) call has been removed.
    cleaner.Update()

    cleaned_polydata = cleaner.GetOutput()
    new_cell_count = cleaned_polydata.GetNumberOfCells()
    
    removed_count = original_cell_count - new_cell_count
    if removed_count > 0:
        print(f"  Mesh Cleanup Report: Removed {removed_count} degenerate faces.")
    else:
        print(f"  Mesh Cleanup Report: No degenerate faces found. Mesh is clean.")
        
    return cleaned_polydata

def convert_mask_to_surface_obj(mask_filepath: str, 
                                obj_filepath: str, 
                                level: float,
                                smoothing_iterations: int,
                                smoothing_passband: float,
                                voxel_spacing_original_zyx: tuple,
                                downsample_factor_zyx: tuple,
                                decimation_reduction: float,
                                ):
    """
    Converts a 3D binary mask TIFF file to a surface mesh OBJ file using VTK.
    Includes downsampling, decimation, and a final mesh cleanup step to ensure valid output.
    """
    if vtk is None:
        print("VTK library not available. Cannot perform conversion.")
        return False
    if not os.path.exists(mask_filepath):
        print(f"Error: Input mask file not found: {mask_filepath}")
        return False

    print(f"Processing {os.path.basename(mask_filepath)} -> {os.path.basename(obj_filepath)} using VTK...")
    start_time = time.time()

    try:
        mask_volume_numpy = tifffile.imread(mask_filepath)
        if mask_volume_numpy.ndim != 3:
            print(f"Error: Input is not a 3D volume. Dimensions: {mask_volume_numpy.ndim}")
            return False
        print(f"  Original mask shape: {mask_volume_numpy.shape}, dtype: {mask_volume_numpy.dtype}")
        
        if any(f != 1.0 for f in downsample_factor_zyx):
            if zoom is None:
                print("  Error: Scipy is not installed, but downsampling was requested. Skipping file.")
                return False
            print(f"  Downsampling volume with factors (Z,Y,X): {downsample_factor_zyx}...")
            downsampled_mask = zoom(mask_volume_numpy.astype(np.float32), downsample_factor_zyx, order=1)
            mask_volume_numpy = (downsampled_mask > level).astype(np.float32)
            adjusted_spacing_zyx = tuple(o / f for o, f in zip(voxel_spacing_original_zyx, downsample_factor_zyx))
            print(f"  New downsampled shape: {mask_volume_numpy.shape}")
            print(f"  Adjusted voxel spacing (Z,Y,X): {adjusted_spacing_zyx}")
        else:
            adjusted_spacing_zyx = voxel_spacing_original_zyx
            if mask_volume_numpy.max() > 1 and np.issubdtype(mask_volume_numpy.dtype, np.integer):
                mask_volume_numpy = (mask_volume_numpy > 0).astype(np.float32)
            elif mask_volume_numpy.dtype != np.float32:
                mask_volume_numpy = mask_volume_numpy.astype(np.float32)
        
        if not np.any(mask_volume_numpy > 0):
            print(f"  Warning: Mask volume in {mask_filepath} is empty. Skipping OBJ creation.")
            with open(obj_filepath, 'w') as f: f.write("# Empty OBJ file from empty mask\n")
            return True

        data_importer = vtk.vtkImageImport()
        data_string = mask_volume_numpy.tobytes('C')
        data_importer.CopyImportVoidPointer(data_string, len(data_string))
        data_importer.SetDataScalarTypeToFloat()
        data_importer.SetNumberOfScalarComponents(1)
        dim_z, dim_y, dim_x = mask_volume_numpy.shape
        data_importer.SetDataExtent(0, dim_x - 1, 0, dim_y - 1, 0, dim_z - 1)
        data_importer.SetWholeExtent(0, dim_x - 1, 0, dim_y - 1, 0, dim_z - 1)
        data_importer.SetDataSpacing(adjusted_spacing_zyx[2], adjusted_spacing_zyx[1], adjusted_spacing_zyx[0])
        data_importer.Update()
        vtk_image_data = data_importer.GetOutput()
        
        mc = vtk.vtkMarchingCubes()
        mc.SetInputData(vtk_image_data)
        mc.SetValue(0, level)
        mc.ComputeNormalsOn()
        print(f"  Applying vtkMarchingCubes (level={level})...")
        mc.Update()
        current_polydata = mc.GetOutput()

        if current_polydata.GetNumberOfPoints() == 0 or current_polydata.GetNumberOfCells() == 0:
            print(f"  Warning: Marching Cubes produced no surface. Skipping OBJ creation.")
            with open(obj_filepath, 'w') as f: f.write("# Empty OBJ file - VTK Marching Cubes found no surface\n")
            return True
        print(f"  Marching Cubes generated {current_polydata.GetNumberOfPoints()} vertices and {current_polydata.GetNumberOfCells()} faces.")
        
        if decimation_reduction > 0.0:
            print(f"  Applying mesh decimation (target reduction: {decimation_reduction * 100:.1f}%)...")
            decimate = vtk.vtkDecimatePro()
            decimate.SetInputData(current_polydata)
            decimate.SetTargetReduction(decimation_reduction)
            decimate.PreserveTopologyOn()
            decimate.Update()
            current_polydata = decimate.GetOutput()
            print(f"  Decimation reduced mesh to {current_polydata.GetNumberOfPoints()} vertices and {current_polydata.GetNumberOfCells()} faces.")

        polydata_to_write = current_polydata
        if smoothing_iterations > 0:
            print(f"  Applying Laplacian smoothing ({smoothing_iterations} iterations, passband={smoothing_passband})...")
            smoother = vtk.vtkSmoothPolyDataFilter()
            smoother.SetInputData(current_polydata)
            smoother.SetNumberOfIterations(smoothing_iterations)
            smoother.SetRelaxationFactor(smoothing_passband)
            smoother.Update()
            polydata_to_write = smoother.GetOutput()
        
        cleaned_polydata = clean_polydata(polydata_to_write)
        
        print(f"  Writing final valid mesh to {obj_filepath}...")
        writer = vtk.vtkOBJWriter()
        writer.SetFileName(obj_filepath)
        writer.SetInputData(cleaned_polydata)
        writer.Write()

        conversion_time = time.time() - start_time
        print(f"  Successfully converted. Output: {obj_filepath} ({conversion_time:.2f}s)")
        return True

    except Exception as e:
        print(f"Error during conversion of {mask_filepath}: {e}")
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Convert 3D binary mask TIFF files to smaller, valid OBJ surface meshes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the directory containing input mask TIFF files.")
    parser.add_argument("--output_dir", type=str, default=None, help="Path to the directory where .obj files will be saved. Defaults to input dir.")
    parser.add_argument("--level", type=float, default=DEFAULT_MARCHING_CUBES_LEVEL, help="Iso-value for Marching Cubes.")
    parser.add_argument("--smoothing_iter", type=int, default=DEFAULT_SMOOTHING_ITERATIONS, help="Laplacian smoothing iterations. 0 for no smoothing.")
    parser.add_argument("--smoothing_passband", type=float, default=DEFAULT_SMOOTHING_PASSBAND, help="Passband for Laplacian smoothing.")
    parser.add_argument("--voxel_size", type=float, nargs=3, default=[1.0, 1.0, 1.0], help="Original voxel size in Z Y X order (e.g., '2.0 1.0 1.0').")
    parser.add_argument("--file_suffix", type=str, default="_frangi_seg.tif", help="Suffix of input TIFF files to identify them.")
    parser.add_argument("--output_suffix", type=str, default="_mesh", help="Suffix for output OBJ file base name.")
    parser.add_argument("--downsample_factor", type=float, nargs=3, default=list(DEFAULT_DOWNSAMPLE_FACTOR), help="Downsample the volume by this factor in Z Y X before meshing. E.g., '0.5 0.5 0.5' for 2x downsampling. Requires 'scipy'.")
    parser.add_argument("--decimation_reduction", type=float, default=DEFAULT_DECIMATION_REDUCTION, help="Target reduction of triangles after meshing (0.0 to 1.0). E.g., 0.9 means reduce triangle count by 90%%. 0.0 means no decimation.")

    args = parser.parse_args()
    
    if vtk is None: return
    if not (0.0 <= args.decimation_reduction < 1.0):
        print(f"Error: --decimation_reduction must be between 0.0 and 1.0. Got {args.decimation_reduction}")
        return
    if any(f != 1.0 for f in args.downsample_factor) and zoom is None:
        print("Error: --downsample_factor was specified, but 'scipy' is not installed.")
        return
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        return

    output_directory = args.output_dir if args.output_dir else args.input_dir
    os.makedirs(output_directory, exist_ok=True)
    
    search_pattern = os.path.join(args.input_dir, f"*{args.file_suffix}")
    mask_files = glob.glob(search_pattern)

    if not mask_files:
        print(f"No files found matching pattern '{search_pattern}' in {args.input_dir}")
        return

    print(f"Found {len(mask_files)} mask files to process.")
    print("-" * 50)
    
    success_count = 0
    failure_count = 0
    for mask_file_path in mask_files:
        base_name = os.path.basename(mask_file_path).replace(args.file_suffix, '')
        obj_filename = f"{base_name}{args.output_suffix}.obj"
        obj_file_path = os.path.join(output_directory, obj_filename)
        
        if convert_mask_to_surface_obj(
            mask_file_path, 
            obj_file_path,
            level=args.level,
            smoothing_iterations=args.smoothing_iter,
            smoothing_passband=args.smoothing_passband,
            voxel_spacing_original_zyx=tuple(args.voxel_size),
            downsample_factor_zyx=tuple(args.downsample_factor),
            decimation_reduction=args.decimation_reduction
            ):
            success_count += 1
        else:
            failure_count += 1
        print("-" * 30)

    print("\nConversion Summary:")
    print(f"  Successfully converted: {success_count}")
    print(f"  Failed conversions: {failure_count}")

if __name__ == "__main__":
    main()
