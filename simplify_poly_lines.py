import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import numpy as np
from collections import defaultdict
import time
import os
import argparse
import random

def rdp(points, epsilon):
    """
    The Ramer-Douglas-Peucker algorithm for polyline geometric simplification.
    `points` is a NumPy array of shape (N, 3).
    Returns a list of indices of the points to keep.
    """
    if len(points) < 2:
        return list(range(len(points)))

    dmax = 0.0
    index = 0
    end = len(points) - 1
    
    # Using vector math for efficiency
    p1, p2 = points[0], points[end]
    line_vec = p2 - p1
    line_len_sq = np.sum(line_vec**2)

    for i in range(1, end):
        # Find distance from point i to the line segment p1-p2
        p = points[i]
        if line_len_sq == 0: # p1 and p2 are the same point
            d = np.linalg.norm(p - p1)
        else:
            # Project p onto the line, but clamp between p1 and p2
            t = max(0, min(1, np.dot(p - p1, line_vec) / line_len_sq))
            projection = p1 + t * line_vec
            d = np.linalg.norm(p - projection)
        
        if d > dmax:
            index = i
            dmax = d

    # If max distance is greater than epsilon, recursively simplify
    if dmax > epsilon:
        rec_results1 = rdp(points[:index + 1], epsilon)
        # Offset the results from the second recursive call
        rec_results2 = [i + index for i in rdp(points[index:], epsilon)]
        # Combine results, removing the duplicated middle point
        return rec_results1[:-1] + rec_results2
    else:
        # All points are within tolerance, so we only keep the start and end
        return [0, end]

def simplify_polylines_with_rdp(input_path, output_path, num_samples=0, epsilon=0.0):
    start_time = time.time()
    print(f"Reading VTP file: {input_path}", flush=True)
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(input_path)
    reader.Update()
    source_polydata = reader.GetOutput()
    
    if num_samples > 0:
        source_polydata = create_polydata_from_sample(source_polydata, num_samples)

    print("Applying vtkStaticCleanPolyData...", flush=True)
    cleaner = vtk.vtkStaticCleanPolyData()
    cleaner.SetInputData(source_polydata)
    cleaner.SetTolerance(1e-6)
    cleaner.Update()
    cleaned_polydata = cleaner.GetOutput()
    
    print("Building graph and analyzing topology...", flush=True)
    points_np = vtk_to_numpy(cleaned_polydata.GetPoints().GetData())
    lines_vtk = cleaned_polydata.GetLines()
    id_list = vtk.vtkIdList()
    adjacency_list = defaultdict(list)
    lines_vtk.InitTraversal()
    while lines_vtk.GetNextCell(id_list):
        point_ids = [id_list.GetId(i) for i in range(id_list.GetNumberOfIds())]
        for i in range(len(point_ids) - 1):
            p1, p2 = point_ids[i], point_ids[i+1]
            adjacency_list[p1].append(p2)
            adjacency_list[p2].append(p1)
    point_degrees = {pid: len(neighbors) for pid, neighbors in adjacency_list.items()}
    special_points = {pid for pid, degree in point_degrees.items() if degree != 2}
    print(f"Identified {len(special_points)} special points (endpoints and branches).", flush=True)

    print("Tracing topologically simplified paths...", flush=True)
    topologically_simplified_lines = []
    visited_points = set()
    for point_id in range(cleaned_polydata.GetNumberOfPoints()):
        if point_id not in visited_points and (point_id in special_points or point_degrees.get(point_id, 0) == 2):
            for neighbor_id in adjacency_list[point_id]:
                if neighbor_id in visited_points: continue
                current_path = [point_id, neighbor_id]
                visited_points.add(point_id)
                visited_points.add(neighbor_id)
                prev_point, current_point = point_id, neighbor_id
                while current_point not in special_points:
                    neighbors = adjacency_list[current_point]
                    next_point = -1
                    for n_id in neighbors:
                        if n_id != prev_point:
                            next_point = n_id
                            break
                    if next_point == -1 or next_point in visited_points: break
                    current_path.append(next_point)
                    visited_points.add(next_point)
                    prev_point, current_point = current_point, next_point
                topologically_simplified_lines.append(current_path)

    print("Applying geometric simplification (RDP)...", flush=True)
    geometrically_simplified_lines = []
    if epsilon > 0:
        for path in topologically_simplified_lines:
            path_coords = points_np[path]
            kept_indices = rdp(path_coords, epsilon)
            # Map local indices back to original point IDs
            simplified_path_ids = [path[i] for i in kept_indices]
            geometrically_simplified_lines.append(simplified_path_ids)
    else:
        # If no epsilon, just use the topologically simplified lines
        geometrically_simplified_lines = topologically_simplified_lines

    print("Rebuilding final geometry and associated data...", flush=True)
    # The rest of the script now uses the *final* geometrically simplified lines
    final_points_original_ids = sorted(list(set(p for line in geometrically_simplified_lines for p in line)))
    original_point_count = cleaned_polydata.GetNumberOfPoints()
    final_point_count = len(final_points_original_ids)
    point_reduction = 100 * (1 - final_point_count / original_point_count) if original_point_count > 0 else 0
    print(f"Point count reduced from {original_point_count} to {final_point_count} ({point_reduction:.2f}% reduction).", flush=True)
    
    # ... (The data rebuilding part is identical to the previous script) ...
    old_to_new_id_map = {old_id: new_id for new_id, old_id in enumerate(final_points_original_ids)}
    new_points_array = points_np[final_points_original_ids]
    new_points = vtk.vtkPoints()
    new_points.SetData(numpy_to_vtk(new_points_array))
    new_lines = vtk.vtkCellArray()
    new_length_data = vtk.vtkDoubleArray()
    new_length_data.SetName("Length")
    new_tortuosity_data = vtk.vtkDoubleArray()
    new_tortuosity_data.SetName("Tortuosity")
    for original_path in geometrically_simplified_lines:
        num_points = len(original_path)
        if num_points < 2: continue
        path_coords = points_np[original_path]
        path_length = np.sum(np.linalg.norm(np.diff(path_coords, axis=0), axis=1))
        new_length_data.InsertNextValue(path_length)
        endpoint_dist = np.linalg.norm(path_coords[0] - path_coords[-1])
        tortuosity = path_length / endpoint_dist if endpoint_dist > 1e-6 else 1.0
        new_tortuosity_data.InsertNextValue(tortuosity)
        vtk_line = vtk.vtkPolyLine()
        vtk_line.GetPointIds().SetNumberOfIds(num_points)
        for i, old_pid in enumerate(original_path):
            vtk_line.GetPointIds().SetId(i, old_to_new_id_map[old_pid])
        new_lines.InsertNextCell(vtk_line)
    new_polydata = vtk.vtkPolyData()
    new_polydata.SetPoints(new_points)
    new_polydata.SetLines(new_lines)
    new_polydata.GetCellData().AddArray(new_length_data)
    new_polydata.GetCellData().AddArray(new_tortuosity_data)
    final_point_data_container = new_polydata.GetPointData()
    original_point_data = cleaned_polydata.GetPointData()
    for i in range(original_point_data.GetNumberOfArrays()):
        original_array = original_point_data.GetArray(i)
        array_name = original_array.GetName()
        new_array = vtk.vtkDataArray.CreateDataArray(original_array.GetDataType())
        new_array.SetName(array_name)
        new_array.SetNumberOfComponents(original_array.GetNumberOfComponents())
        new_array.SetNumberOfTuples(final_point_count)
        for new_id, old_id in enumerate(final_points_original_ids):
            new_array.SetTuple(new_id, original_point_data.GetTuple(old_id))
        final_point_data_container.AddArray(new_array)
    
    print("Writing new VTP file...", flush=True)
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_path)
    writer.SetInputData(new_polydata)
    writer.Write()
    print("\n--- SCRIPT FINISHED ---", flush=True)
    print(f"Total time: {time.time() - start_time:.2f} seconds", flush=True)
    print(f"Original line count: {source_polydata.GetNumberOfLines()}", flush=True)
    print(f"Simplified line count: {new_polydata.GetNumberOfLines()}", flush=True)
    print(f"Simplified file saved to: {output_path}", flush=True)

# The create_polydata_from_sample function is needed if you use the --sample flag
# This function does not need to be changed.
def create_polydata_from_sample(polydata, num_samples):
    # (code from previous answer is correct)
    total_lines = polydata.GetNumberOfLines()
    if num_samples > total_lines:
        num_samples = total_lines
    print(f"Creating an in-memory sample of {num_samples} out of {total_lines} polylines...", flush=True)
    sampled_original_line_indices = random.sample(range(total_lines), num_samples)
    original_points_np = vtk_to_numpy(polydata.GetPoints().GetData())
    sampled_line_definitions = {}
    master_point_ids = set()
    id_list = vtk.vtkIdList()
    for i in sampled_original_line_indices:
        polydata.GetCellPoints(i, id_list)
        point_ids_for_line = [id_list.GetId(j) for j in range(id_list.GetNumberOfIds())]
        sampled_line_definitions[i] = point_ids_for_line
        master_point_ids.update(point_ids_for_line)
    final_points_original_ids = sorted(list(master_point_ids))
    old_to_new_id_map = {old_id: new_id for new_id, old_id in enumerate(final_points_original_ids)}
    new_points = vtk.vtkPoints()
    new_points.SetData(numpy_to_vtk(original_points_np[final_points_original_ids]))
    new_point_data_container = vtk.vtkPointData()
    original_point_data = polydata.GetPointData()
    for i in range(original_point_data.GetNumberOfArrays()):
        original_array = original_point_data.GetArray(i)
        new_array = vtk.vtkDataArray.CreateDataArray(original_array.GetDataType())
        new_array.SetName(original_array.GetName())
        new_array.SetNumberOfComponents(original_array.GetNumberOfComponents())
        new_array.SetNumberOfTuples(len(final_points_original_ids))
        for new_id, old_id in enumerate(final_points_original_ids):
            new_array.SetTuple(new_id, original_point_data.GetTuple(old_id))
        new_point_data_container.AddArray(new_array)
    new_lines = vtk.vtkCellArray()
    new_cell_data_container = vtk.vtkCellData()
    original_cell_data = polydata.GetCellData()
    new_cell_arrays = []
    for i in range(original_cell_data.GetNumberOfArrays()):
        original_array = original_cell_data.GetArray(i)
        new_array = vtk.vtkDataArray.CreateDataArray(original_array.GetDataType())
        new_array.SetName(original_array.GetName())
        new_array.SetNumberOfComponents(original_array.GetNumberOfComponents())
        new_cell_arrays.append(new_array)
    for original_line_index in sampled_original_line_indices:
        line_point_ids = sampled_line_definitions[original_line_index]
        vtk_line = vtk.vtkPolyLine()
        vtk_line.GetPointIds().SetNumberOfIds(len(line_point_ids))
        for i, old_pid in enumerate(line_point_ids):
            vtk_line.GetPointIds().SetId(i, old_to_new_id_map[old_pid])
        new_lines.InsertNextCell(vtk_line)
        for i, new_array in enumerate(new_cell_arrays):
            new_array.InsertNextTuple(original_cell_data.GetArray(i).GetTuple(original_line_index))
    for arr in new_cell_arrays:
        new_cell_data_container.AddArray(arr)
    sampled_polydata = vtk.vtkPolyData()
    sampled_polydata.SetPoints(new_points)
    sampled_polydata.SetLines(new_lines)
    sampled_polydata.GetPointData().ShallowCopy(_new_point_data_container)
    sampled_polydata.GetCellData().ShallowCopy(new_cell_data_container)
    return sampled_polydata

def main():
    parser = argparse.ArgumentParser(description="Clean, topologically simplify, and geometrically simplify a VTP polyline network.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input .vtp file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the simplified output .vtp file.")
    parser.add_argument("--sample", type=int, default=0, help="If > 0, randomly sample this many lines for a debug run. Default is 0 (process the whole file).")
    parser.add_argument("--epsilon", type=float, default=0.0,
        help="Tolerance for RDP geometric simplification. Larger values mean more aggressive simplification. Default is 0.0 (no geometric simplification)."
    )
    args = parser.parse_args()
    simplify_polylines_with_rdp(args.input, args.output, args.sample, args.epsilon)

if __name__ == "__main__":
    main()
