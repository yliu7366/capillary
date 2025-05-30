#!/usr/bin/env python3
import argparse
import os
import time
import numpy as np
import tifffile
import pandas as pd
import traceback # Import the traceback module

try:
    import skan
except ImportError:
    print("Error: 'skan' library not found. Please install it: pip install skan")
    skan = None

try:
    import vtk
except ImportError:
    print("Error: 'vtk' library not found. Please install it: pip install vtk")
    vtk = None


def extract_polylines_from_skeleton(skeleton_volume_path: str,
                                    csv_output_path: str,
                                    voxel_size_zyx: tuple = (1.0, 1.0, 1.0)):
    """
    Extracts polylines (centerlines) from a 3D binary skeleton volume using skan.
    Saves the path coordinates to a CSV file.

    Args:
        skeleton_volume_path (str): Path to the input binary skeleton TIFF volume.
        csv_output_path (str): Path to save the output CSV file with polyline coordinates.
        voxel_size_zyx (tuple): Voxel dimensions (Z, Y, X) for skan's metric calculations.
    Returns:
        bool: True if successful, False otherwise.
    """
    if skan is None:
        print("skan library not available. Cannot extract polylines.")
        return False
    if pd is None:
        print("pandas library not available. Cannot save polylines to CSV.")
        return False

    print(f"Loading skeleton volume from: {skeleton_volume_path}...")
    try:
        skeleton_image_uint8 = tifffile.imread(skeleton_volume_path)
    except FileNotFoundError:
        print(f"Error: Skeleton file not found at {skeleton_volume_path}")
        return False
    except Exception as e:
        print(f"Error loading skeleton TIFF: {e}")
        traceback.print_exc() 
        return False

    if skeleton_image_uint8.ndim != 3:
        print(f"Error: Input skeleton is not a 3D volume. Dimensions: {skeleton_image_uint8.ndim}")
        return False
    
    print(f"Skeleton volume shape: {skeleton_image_uint8.shape}, dtype: {skeleton_image_uint8.dtype}")

    skeleton_bool = (skeleton_image_uint8 > 0).astype(bool, order='C')

    if not np.any(skeleton_bool):
        print("Warning: Skeleton volume is empty (all zero values). No polylines to extract.")
        empty_df = pd.DataFrame(columns=['path_id', 'point_order', 'z', 'y', 'x'])
        try:
            csv_dir = os.path.dirname(csv_output_path)
            if csv_dir: os.makedirs(csv_dir, exist_ok=True)
            empty_df.to_csv(csv_output_path, index=False)
            print(f"Empty CSV saved to {csv_output_path}")
        except Exception as e_csv:
            print(f"Error saving empty CSV: {e_csv}")
            traceback.print_exc()
        return True

    print(f"Converting skeleton to polylines/graph representation using skan with voxel_size (Z,Y,X): {voxel_size_zyx}...")
    print(f"  Input skeleton_bool shape: {skeleton_bool.shape}, dtype: {skeleton_bool.dtype}, C-contiguous: {skeleton_bool.flags['C_CONTIGUOUS']}")
    polyline_conv_start_time = time.time()
    
    try:
        skan_skeleton_obj = skan.Skeleton(skeleton_bool, spacing=voxel_size_zyx,
                                          source_image=skeleton_image_uint8) 

        polylines_list = []
        num_paths = skan_skeleton_obj.n_paths
        
        if num_paths > 0:
            print(f"  Extracting polylines from {num_paths} paths found by skan...")
            
            for i in range(num_paths): # i is the path index, can also serve as a simple path_id
                path_coords_indices = skan_skeleton_obj.path_coordinates(i)
                
                # Use the path index 'i' as the path_id for this CSV output
                # If you need more specific IDs from skan (like branch IDs from summarize),
                # you would need to link this back to the skan.summarize() DataFrame.
                # For now, using 'i' is simple and guarantees a unique ID for each path here.
                current_path_id = i 

                if path_coords_indices is None or path_coords_indices.shape[0] < 2:
                    # print(f"    Skipping path index {i} (effective ID: {current_path_id}) as it has < 2 points or is None.")
                    continue 

                for point_idx in range(path_coords_indices.shape[0]):
                    polylines_list.append({
                        'path_id': current_path_id, # Using the loop index as the path ID
                        'point_order': point_idx,
                        'z': path_coords_indices[point_idx, 0],
                        'y': path_coords_indices[point_idx, 1],
                        'x': path_coords_indices[point_idx, 2]
                    })
            
            if polylines_list:
                polylines_df = pd.DataFrame(polylines_list)
                csv_dir = os.path.dirname(csv_output_path)
                if csv_dir: os.makedirs(csv_dir, exist_ok=True)
                polylines_df.to_csv(csv_output_path, index=False)
                print(f"  Polyline coordinates (pixel indices) saved to: {csv_output_path}")
            else:
                print("  No valid polylines (>=2 points) extracted by skan, though paths might have been reported by n_paths.")
        
        else: 
             print("  Skan did not find any paths in the skeleton.")

        if not polylines_list:
            empty_df = pd.DataFrame(columns=['path_id', 'point_order', 'z', 'y', 'x'])
            try:
                csv_dir = os.path.dirname(csv_output_path)
                if csv_dir: os.makedirs(csv_dir, exist_ok=True)
                if num_paths > 0 and not polylines_list: 
                    print(f"Empty CSV saved to {csv_output_path} as no valid polylines were generated from detected paths.")
                elif num_paths == 0:
                    print(f"Empty CSV saved to {csv_output_path} as no paths were found.")
                empty_df.to_csv(csv_output_path, index=False)
            except Exception as e_csv_empty:
                print(f"Error saving empty CSV: {e_csv_empty}")
                traceback.print_exc()

        polyline_conv_time = time.time() - polyline_conv_start_time
        print(f"Polyline extraction complete in {polyline_conv_time:.2f}s.")
        
        # Optionally, if you want the branch summary data from skan:
        # print("Generating branch summary data from skan...")
        # branch_data_summary = skan.summarize(skan_skeleton_obj)
        # if not branch_data_summary.empty:
        #     summary_filename = csv_output_path.replace(".csv", "_branch_summary.csv")
        #     branch_data_summary.to_csv(summary_filename, index=False)
        #     print(f"  Skan branch summary saved to: {summary_filename}")
        # else:
        #     print("  Skan did not produce a branch summary (skeleton might be too simple or empty).")

        return True

    except Exception as e:
        print(f"Error during skan processing or saving polyline CSV: {e}")
        print("-" * 60)
        print("Full traceback for the error:")
        traceback.print_exc()
        print("-" * 60)
        return False

# ... (save_polylines_csv_to_vtk function remains the same) ...
def save_polylines_csv_to_vtk(csv_filepath: str, vtk_filepath: str, voxel_size_zyx: tuple = (1.0, 1.0, 1.0)):
    if vtk is None:
        print("VTK library not available. Cannot save VTK file.")
        return False
    if not os.path.exists(csv_filepath):
        print(f"Error: CSV file not found at {csv_filepath}")
        return False
    print(f"Converting polylines from {csv_filepath} to VTK format at {vtk_filepath}...")
    start_time = time.time()
    try:
        df = pd.read_csv(csv_filepath)
    except Exception as e:
        print(f"Error reading CSV file {csv_filepath}: {e}")
        traceback.print_exc()
        return False

    if not all(col in df.columns for col in ['path_id', 'point_order', 'z', 'y', 'x']):
        print("Error: CSV file does not contain the expected columns: 'path_id', 'point_order', 'z', 'y', 'x'")
        return False
    
    poly_data = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    path_id_cell_data = vtk.vtkIntArray()
    path_id_cell_data.SetName("PathID")

    if df.empty:
        print("Warning: CSV file is empty. No polylines to convert to VTK.")
    else:
        grouped = df.groupby('path_id')
        for path_id_val, group in grouped:
            path_points = group.sort_values('point_order')
            if len(path_points) < 2: continue
            line = vtk.vtkPolyLine()
            num_points_in_line = len(path_points)
            line.GetPointIds().SetNumberOfIds(num_points_in_line)
            for i, (_, row) in enumerate(path_points.iterrows()):
                world_x = row['x'] * voxel_size_zyx[2]
                world_y = row['y'] * voxel_size_zyx[1]
                world_z = row['z'] * voxel_size_zyx[0]
                point_id = points.InsertNextPoint(world_x, world_y, world_z)
                line.GetPointIds().SetId(i, point_id)
            lines.InsertNextCell(line)
            path_id_cell_data.InsertNextValue(int(path_id_val))

    poly_data.SetPoints(points)
    poly_data.SetLines(lines)
    if lines.GetNumberOfCells() > 0 and lines.GetNumberOfCells() == path_id_cell_data.GetNumberOfTuples():
        poly_data.GetCellData().AddArray(path_id_cell_data)
    elif lines.GetNumberOfCells() > 0:
        print(f"Warning: Mismatch lines ({lines.GetNumberOfCells()}) vs path_id data ({path_id_cell_data.GetNumberOfTuples()}).")

    writer = None
    if vtk_filepath.lower().endswith('.vtp'):
        writer = vtk.vtkXMLPolyDataWriter()
    elif vtk_filepath.lower().endswith('.vtk'):
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileTypeToASCII()
    else:
        print(f"Error: Unknown VTK file extension for {vtk_filepath}. Use .vtp or .vtk.")
        return False
    writer.SetFileName(vtk_filepath)
    writer.SetInputData(poly_data)
    writer.Write()
    end_time = time.time()
    print(f"Polylines saved to {vtk_filepath} in {end_time - start_time:.2f}s")
    return True

# ... (main function remains the same) ...
def main():
    parser = argparse.ArgumentParser(description="Convert a 3D binary skeleton volume to polylines (CSV and VTK).")
    parser.add_argument("--skeleton_input", type=str, required=True,
                        help="Path to the input 3D binary skeleton TIFF volume.")
    parser.add_argument("--csv_output", type=str, required=True,
                        help="Path to save the extracted polylines (path coordinates) as CSV.")
    parser.add_argument("--vtk_output", type=str, required=True,
                        help="Path to save the extracted polylines as a VTK file (.vtp or .vtk).")
    parser.add_argument("--voxel_size", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help="Voxel size in Z Y X order (e.g., '2.0 1.0 1.0'). Used for skan metrics and VTK scaling.")
    args = parser.parse_args()

    # Ensure output directories exist (create only the directory part)
    csv_dir = os.path.dirname(args.csv_output)
    vtk_dir = os.path.dirname(args.vtk_output)
    if csv_dir: os.makedirs(csv_dir, exist_ok=True)
    if vtk_dir: os.makedirs(vtk_dir, exist_ok=True)
    
    csv_success = extract_polylines_from_skeleton(
        args.skeleton_input,
        args.csv_output,
        tuple(args.voxel_size)
    )

    if csv_success and os.path.exists(args.csv_output):
        save_polylines_csv_to_vtk(
            args.csv_output,
            args.vtk_output,
            tuple(args.voxel_size)
        )
    elif csv_success and not os.path.exists(args.csv_output): # Handle case where skeleton was empty
        print(f"CSV file {args.csv_output} was not created (skeleton might have been empty or no paths found). Skipping VTK conversion.")
    else: # csv_success is False
        print("Skipping VTK conversion due to errors in polyline extraction from skeleton.")


if __name__ == "__main__":
    main()
