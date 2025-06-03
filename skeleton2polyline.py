#!/usr/bin/env python3
import argparse
import os
import time
import numpy as np
import tifffile
import pandas as pd
import traceback 

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

# Default voxel size (can be overridden by CLI)
DEFAULT_VOXEL_SIZE_ZYX = (1.0, 1.0, 1.0)

def extract_polylines_and_metrics(skeleton_volume_path: str,
                                  voxel_size_zyx: tuple = (1.0, 1.0, 1.0),
                                  save_points_csv_path: str = None, # Optional path
                                  save_summary_csv_path: str = None # Optional path
                                  ):
    """
    Extracts polylines, calculates branch metrics using skan.
    Optionally saves all path point coordinates and branch summary with metrics to CSV files.
    
    Args:
        skeleton_volume_path (str): Path to the input binary skeleton TIFF volume.
        voxel_size_zyx (tuple): Voxel dimensions (Z, Y, X) for skan's metric calculations.
        save_points_csv_path (str, optional): Path to save CSV of all polyline point coordinates.
        save_summary_csv_path (str, optional): Path to save CSV of branch summary with metrics.
    Returns:
        pd.DataFrame or None: DataFrame containing branch summary with metrics if successful, else None.
        pd.DataFrame or None: DataFrame containing all path point coordinates if successful, else None.
    """
    if skan is None or pd is None:
        print("skan or pandas library not available.")
        return None, None

    print(f"Loading skeleton volume from: {skeleton_volume_path}...")
    try:
        skeleton_image_uint8 = tifffile.imread(skeleton_volume_path)
        if skeleton_image_uint8.ndim != 3:
            print(f"Error: Input skeleton is not 3D. Dimensions: {skeleton_image_uint8.ndim}")
            return None, None
        skeleton_bool = (skeleton_image_uint8 > 0).astype(bool, order='C')
        print(f"Skeleton volume shape: {skeleton_bool.shape}, dtype: {skeleton_bool.dtype}")
    except Exception as e:
        print(f"Error loading skeleton TIFF: {e}"); traceback.print_exc(); return None, None

    empty_points_df = pd.DataFrame(columns=['path_id', 'point_order', 'z', 'y', 'x'])
    empty_summary_df = pd.DataFrame(columns=['branch-id', 'branch-distance', 'euclidean-distance', 
                                             'tortuosity', 'straightness', 'node-id-src', 'node-id-dst', 
                                             'coord-src-z','coord-src-y','coord-src-x',
                                             'coord-dst-z','coord-dst-y','coord-dst-x'])

    if not np.any(skeleton_bool):
        print("Warning: Skeleton volume is empty. No polylines or metrics to extract.")
        if save_points_csv_path:
            try:
                empty_points_df.to_csv(save_points_csv_path, index=False)
                print(f"Empty points CSV saved to {save_points_csv_path}")
            except Exception as e_csv: print(f"Error saving empty points CSV: {e_csv}"); traceback.print_exc()
        if save_summary_csv_path:
            try:
                empty_summary_df.to_csv(save_summary_csv_path, index=False)
                print(f"Empty summary CSV saved to {save_summary_csv_path}")
            except Exception as e_csv: print(f"Error saving empty summary CSV: {e_csv}"); traceback.print_exc()
        return empty_summary_df, empty_points_df 

    print(f"Analyzing skeleton with skan, voxel_size (Z,Y,X): {voxel_size_zyx}...")
    analysis_start_time = time.time()
    
    branch_summary_df = None
    points_df = None

    try:
        skan_skeleton_obj = skan.Skeleton(skeleton_bool, spacing=voxel_size_zyx,
                                          source_image=skeleton_image_uint8)
        
        print("  Summarizing branches with skan...")
        branch_summary_df = skan.summarize(skan_skeleton_obj)

        if branch_summary_df.empty:
            print("  Skan did not find any branches to summarize.")
            branch_summary_df = empty_summary_df # Ensure it's an empty df, not None
        else:
            print(f"  Found {len(branch_summary_df)} branches.")
            branch_summary_df['tortuosity'] = np.where(
                branch_summary_df['euclidean-distance'] > 1e-6,
                branch_summary_df['branch-distance'] / branch_summary_df['euclidean-distance'], 1.0)
            branch_summary_df['straightness'] = np.where(
                branch_summary_df['branch-distance'] > 1e-6, 
                branch_summary_df['euclidean-distance'] / branch_summary_df['branch-distance'], 1.0)
            branch_summary_df = branch_summary_df.replace([np.inf, -np.inf], np.nan)
            branch_summary_df['tortuosity'] = branch_summary_df['tortuosity'].fillna(1.0) 
            branch_summary_df['straightness'] = branch_summary_df['straightness'].fillna(1.0)
            branch_summary_df.rename(columns={
                'coord-src-0': 'coord-src-z', 'coord-src-1': 'coord-src-y', 'coord-src-2': 'coord-src-x',
                'coord-dst-0': 'coord-dst-z', 'coord-dst-1': 'coord-dst-y', 'coord-dst-2': 'coord-dst-x'
            }, inplace=True)

        if save_summary_csv_path:
            branch_summary_df.to_csv(save_summary_csv_path, index=False)
            print(f"  Branch summary with metrics saved to: {save_summary_csv_path}")

        polylines_list_of_points = []
        num_paths = skan_skeleton_obj.n_paths
        
        if num_paths > 0:
            print(f"  Extracting polyline coordinates from {num_paths} paths...")
            for i in range(num_paths):
                path_coords_indices = skan_skeleton_obj.path_coordinates(i)
                current_path_id = i 
                if path_coords_indices is None or path_coords_indices.shape[0] < 1: continue 
                for point_idx in range(path_coords_indices.shape[0]):
                    polylines_list_of_points.append({
                        'path_id': current_path_id, 
                        'point_order': point_idx,
                        'z': path_coords_indices[point_idx, 0],
                        'y': path_coords_indices[point_idx, 1],
                        'x': path_coords_indices[point_idx, 2]
                    })
            if polylines_list_of_points:
                points_df = pd.DataFrame(polylines_list_of_points)
                if save_points_csv_path:
                    points_df.to_csv(save_points_csv_path, index=False)
                    print(f"  All path point coordinates saved to: {save_points_csv_path}")
            else:
                print("  No path coordinates extracted (all paths might be single points).")
                points_df = empty_points_df # Ensure it's an empty df
        else:
             print("  Skan did not find any paths in the skeleton.")
             points_df = empty_points_df # Ensure it's an empty df
        
        if save_points_csv_path and (points_df is None or points_df.empty) and not polylines_list_of_points:
             empty_points_df.to_csv(save_points_csv_path, index=False)
             print(f"Empty points CSV saved to {save_points_csv_path}")


        analysis_time = time.time() - analysis_start_time
        print(f"Skeleton analysis complete in {analysis_time:.2f}s.")
        return branch_summary_df, points_df

    except Exception as e:
        print(f"Error during skan processing: {e}")
        traceback.print_exc(); return None, None


def save_polylines_to_vtk(points_df, # Expects a DataFrame
                          branch_summary_df, # DataFrame from skan.summarize()
                          vtk_filepath: str,
                          voxel_size_zyx: tuple = (1.0, 1.0, 1.0)):
    if vtk is None or pd is None:
        print("VTK or pandas library not available.")
        return False
    
    print(f"Saving polylines to VTK format at {vtk_filepath}...")
    start_time = time.time()

    if points_df is None:
        print("Error: Points DataFrame is None. Cannot proceed with VTK conversion.")
        return False

    if not all(col in points_df.columns for col in ['path_id', 'point_order', 'z', 'y', 'x']):
        if not points_df.empty:
             print("Error: Points data does not contain expected columns: 'path_id', 'point_order', 'z', 'y', 'x'"); return False
    
    is_summary_valid = branch_summary_df is not None and not branch_summary_df.empty
    if not is_summary_valid:
        print("Warning: Branch summary data is empty or None. VTK lines will be generated without metrics.");

    poly_data = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints() 
    lines = vtk.vtkCellArray()
    
    vtk_branch_length = vtk.vtkDoubleArray(); vtk_branch_length.SetName("BranchLength")
    vtk_tortuosity = vtk.vtkDoubleArray(); vtk_tortuosity.SetName("Tortuosity")
    vtk_straightness = vtk.vtkDoubleArray(); vtk_straightness.SetName("Straightness")
    vtk_path_id_cell = vtk.vtkIntArray(); vtk_path_id_cell.SetName("PathID_SkanBranchIndex") # This is the 0-indexed path/branch

    if points_df.empty:
        print("Warning: Points data is empty. No polylines to convert to VTK.")
    else:
        grouped = points_df.groupby('path_id')
        for path_id_val, group in grouped: # path_id_val here is our 0-indexed path identifier
            path_points_coords = group.sort_values('point_order')
            if len(path_points_coords) < 2: continue

            line = vtk.vtkPolyLine()
            num_points_in_line = len(path_points_coords)
            line.GetPointIds().SetNumberOfIds(num_points_in_line)
            
            for i, (_, row) in enumerate(path_points_coords.iterrows()):
                world_x = row['x'] * voxel_size_zyx[2]
                world_y = row['y'] * voxel_size_zyx[1]
                world_z = row['z'] * voxel_size_zyx[0]
                point_id_vtk = vtk_points.InsertNextPoint(world_x, world_y, world_z)
                line.GetPointIds().SetId(i, point_id_vtk)
            
            lines.InsertNextCell(line)
            vtk_path_id_cell.InsertNextValue(int(path_id_val)) # Store the path_id_val (0-indexed branch index)

            # Add metrics from branch_summary_df
            if is_summary_valid:
                try:
                    # path_id_val (our 0-indexed path identifier) corresponds to the
                    # integer index of the branch_summary_df from skan.summarize()
                    branch_metrics = branch_summary_df.iloc[path_id_val] 
                    
                    vtk_branch_length.InsertNextValue(branch_metrics.get('branch-distance', 0.0))
                    vtk_tortuosity.InsertNextValue(branch_metrics.get('tortuosity', 1.0))
                    vtk_straightness.InsertNextValue(branch_metrics.get('straightness', 1.0))
                except IndexError: 
                    # This means path_id_val is out of bounds for branch_summary_df
                    print(f"Warning: Metrics for path_id {path_id_val} out of bounds in summary. Using default values.")
                    vtk_branch_length.InsertNextValue(0.0)
                    vtk_tortuosity.InsertNextValue(1.0)
                    vtk_straightness.InsertNextValue(1.0)
                except Exception as e_metric:
                    print(f"Warning: Error accessing metrics for path_id {path_id_val}: {e_metric}. Using default values.")
                    vtk_branch_length.InsertNextValue(0.0)
                    vtk_tortuosity.InsertNextValue(1.0)
                    vtk_straightness.InsertNextValue(1.0)
            else: # No summary data to look up
                vtk_branch_length.InsertNextValue(0.0)
                vtk_tortuosity.InsertNextValue(1.0)
                vtk_straightness.InsertNextValue(1.0)

    poly_data.SetPoints(vtk_points)
    poly_data.SetLines(lines)
    
    if lines.GetNumberOfCells() > 0: 
        if lines.GetNumberOfCells() == vtk_path_id_cell.GetNumberOfTuples():
            poly_data.GetCellData().AddArray(vtk_path_id_cell)
        else: print(f"Warning: Mismatch path_id cell data tuples: {vtk_path_id_cell.GetNumberOfTuples()} vs lines: {lines.GetNumberOfCells()}")
        if lines.GetNumberOfCells() == vtk_branch_length.GetNumberOfTuples():
            poly_data.GetCellData().AddArray(vtk_branch_length)
        else: print(f"Warning: Mismatch branch_length cell data tuples: {vtk_branch_length.GetNumberOfTuples()} vs lines: {lines.GetNumberOfCells()}")
        if lines.GetNumberOfCells() == vtk_tortuosity.GetNumberOfTuples():
            poly_data.GetCellData().AddArray(vtk_tortuosity)
        else: print(f"Warning: Mismatch tortuosity cell data tuples: {vtk_tortuosity.GetNumberOfTuples()} vs lines: {lines.GetNumberOfCells()}")
        if lines.GetNumberOfCells() == vtk_straightness.GetNumberOfTuples():
            poly_data.GetCellData().AddArray(vtk_straightness)
        else: print(f"Warning: Mismatch straightness cell data tuples: {vtk_straightness.GetNumberOfTuples()} vs lines: {lines.GetNumberOfCells()}")

    writer = None
    if vtk_filepath.lower().endswith('.vtp'):
        writer = vtk.vtkXMLPolyDataWriter()
    elif vtk_filepath.lower().endswith('.vtk'):
        writer = vtk.vtkPolyDataWriter(); writer.SetFileTypeToASCII()
    else:
        print(f"Error: Unknown VTK file extension for {vtk_filepath}. Use .vtp or .vtk."); return False
    writer.SetFileName(vtk_filepath)
    writer.SetInputData(poly_data)
    writer.Write()
    end_time = time.time()
    print(f"Polylines with metrics saved to {vtk_filepath} in {end_time - start_time:.2f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert 3D skeleton TIFF to polylines (optional CSVs, mandatory VTK).")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to the input 3D binary skeleton TIFF volume.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to the output directory where all files will be saved.")
    parser.add_argument("--voxel_size", type=float, nargs=3, default=DEFAULT_VOXEL_SIZE_ZYX,
                        help=f"Voxel size in Z Y X order (e.g., '2.0 1.0 1.0'). Default: {DEFAULT_VOXEL_SIZE_ZYX}")
    parser.add_argument("--save_points_csv", action='store_true', default=False,
                        help="If present, save all path point coordinates to a CSV file.")
    parser.add_argument("--save_summary_csv", action='store_true', default=False,
                        help="If present, save branch summary with metrics to a CSV file.")
    
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Derive output filenames
    input_basename = os.path.splitext(os.path.basename(args.input))[0]
    
    vtk_output_filepath = os.path.join(args.output_dir, f"{input_basename}.vtp")
    
    points_csv_path = None
    if args.save_points_csv:
        points_csv_path = os.path.join(args.output_dir, f"{input_basename}_points.csv")

    summary_csv_path = None
    if args.save_summary_csv:
        summary_csv_path = os.path.join(args.output_dir, f"{input_basename}_summary.csv")
    
    branch_summary_df, points_df = extract_polylines_and_metrics(
        args.input,
        tuple(args.voxel_size), # Pass voxel_size here
        save_points_csv_path=points_csv_path,   # Pass optional path
        save_summary_csv_path=summary_csv_path # Pass optional path
    )

    if points_df is not None and branch_summary_df is not None: # Check if extraction was successful
        save_polylines_to_vtk(
            points_df, # Pass the DataFrame of all points
            branch_summary_df, # Pass the summary with metrics
            vtk_output_filepath,
            tuple(args.voxel_size)
        )
    else:
        print("Skipping VTK conversion due to errors or no data from polyline/metrics extraction.")

if __name__ == "__main__":
    main()
