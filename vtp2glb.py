#!/usr/bin/env python3
import argparse
import os
import glob
import time 
import traceback # For detailed error messages in convert function
import shutil # For cleaning up intermediate files

try:
    import vtk
except ImportError:
    print("Error: 'vtk' library not found. Please install it: pip install vtk")
    vtk = None

try:
    from pygltflib import GLTF2 # BufferFormat might not be needed directly
except ImportError:
    print("Error: 'pygltflib' library not found. Please install it: pip install pygltflib")
    GLTF2 = None


def convert_vtp_to_final_glb(vtp_filepath, final_glb_filepath, temp_dir_for_gltf):
    if vtk is None or GLTF2 is None:
        print("VTK or pygltflib library not available. Cannot perform conversion.")
        return False

    if not os.path.exists(vtp_filepath):
        print(f"Error: Input VTP file not found: {vtp_filepath}")
        return False

    os.makedirs(temp_dir_for_gltf, exist_ok=True) 

    base_filename = os.path.splitext(os.path.basename(vtp_filepath))[0]
    intermediate_gltf_path = os.path.join(temp_dir_for_gltf, f"{base_filename}.gltf")

    print(f"Converting {os.path.basename(vtp_filepath)} to intermediate .gltf at {intermediate_gltf_path}...")
    start_time_vtk_export = time.time()

    try:
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(vtp_filepath)
        reader.Update()
        polydata = reader.GetOutput()

        if not polydata or polydata.GetNumberOfPoints() == 0:
            print(f"Warning: No data or no points found in {vtp_filepath}. Skipping GLB creation.")
            return True 

        render_window = vtk.vtkRenderWindow()
        render_window.SetOffScreenRendering(1)
        renderer = vtk.vtkRenderer()
        render_window.AddRenderer(renderer)
        
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        renderer.AddActor(actor)

        vtk_exporter = vtk.vtkGLTFExporter()
        vtk_exporter.SetFileName(intermediate_gltf_path) 
        vtk_exporter.SetInput(render_window)
        vtk_exporter.Write() # This should create .gltf and associated .bin files in temp_dir_for_gltf
        
        vtk_export_time = time.time() - start_time_vtk_export
        print(f"  Intermediate .gltf (and .bin files) saved in {vtk_export_time:.2f}s by VTK.")

    except Exception as e_vtk:
        print(f"Error during VTK export to intermediate .gltf: {e_vtk}")
        traceback.print_exc()
        return False

    if not os.path.exists(intermediate_gltf_path):
        print(f"  Error: Intermediate .gltf file not found at {intermediate_gltf_path} after VTK export.")
        return False

    print(f"  Packaging {intermediate_gltf_path} into {final_glb_filepath} using pygltflib...")
    start_time_pygltf = time.time()
    
    gltf_obj = None
    try:
        # When loading, pygltflib will parse URIs.
        # For external buffers, buffer.data will likely be None at this stage.
        gltf_obj = GLTF2().load(intermediate_gltf_path) 
    except Exception as e_load:
        print(f"    Error loading intermediate .gltf file '{intermediate_gltf_path}' with pygltflib: {e_load}")
        traceback.print_exc()
        return False
    
    # --- Debugging: Print buffer states AFTER load, BEFORE save_binary ---
    print("  --- Buffer State AFTER pygltflib.load() ---")
    if gltf_obj and gltf_obj.buffers:
        for i, b_check in enumerate(gltf_obj.buffers):
            # Check if 'data' attribute exists before trying to access it for length
            has_data_attr = hasattr(b_check, 'data')
            data_is_none = b_check.data is None if has_data_attr else "N/A (no data attr)"
            data_len_str = f"len={len(b_check.data)}" if has_data_attr and b_check.data is not None else "N/A"
            print(f"    Buffer {i}: URI: {b_check.uri}, byteLength: {b_check.byteLength}, Has 'data' attr: {has_data_attr}, Data is None: {data_is_none}, Data type: {type(b_check.data) if has_data_attr else 'N/A'}, {data_len_str}")
            if b_check.uri and (not has_data_attr or (has_data_attr and b_check.data is None)):
                # This is expected for external buffers after load, before save_binary resolves them.
                print(f"      Buffer {i} has URI '{b_check.uri}', data not yet loaded into .data by pygltflib.load().")
    else:
        print("  No buffers found in GLTF object or GLTF object is None.")


    # save_binary is expected to handle the reading of URI-specified .bin files
    # (relative to the path of intermediate_gltf_path) and embed them.
    try:
        print(f"    Calling gltf_obj.save_binary('{final_glb_filepath}')...")
        gltf_obj.save_binary(final_glb_filepath) 

    except TypeError as e_save: 
        print(f"    TypeError during pygltflib.save_binary: {e_save}")
        print(f"    This indicates pygltflib encountered a None value where it expected data during binary packaging.")
        print(f"    This might happen if a .bin file URI could not be resolved or the file was empty/corrupt,")
        print(f"    and pygltflib's internal handling led to a None value being processed.")
        traceback.print_exc()
        # It might be useful to inspect the gltf_obj.buffers again here if possible,
        # but the error occurs inside save_binary.
        return False
    except FileNotFoundError as e_fnf: 
        print(f"    FileNotFoundError during pygltflib.save_binary: {e_fnf}")
        print(f"    This means pygltflib could not find a .bin file referenced by a URI,")
        print(f"    relative to '{os.path.dirname(intermediate_gltf_path)}'.")
        traceback.print_exc()
        return False
    except Exception as e_save_other:
        print(f"    Error during pygltflib.save_binary: {e_save_other}")
        traceback.print_exc()
        return False

    pygltf_time = time.time() - start_time_pygltf
    print(f"  Successfully packaged to {os.path.basename(final_glb_filepath)} in {pygltf_time:.2f}s.")
    
    # ... (Cleanup of intermediate files as before) ...
    try:
        if os.path.exists(intermediate_gltf_path):
            os.remove(intermediate_gltf_path)
        path_to_search_bins = os.path.dirname(intermediate_gltf_path)
        if os.path.isdir(path_to_search_bins): # Ensure directory exists before listing
            for item in os.listdir(path_to_search_bins):
                if item.endswith(".bin"):
                    if item.startswith("buffer") or (base_filename and item.startswith(base_filename)):
                        item_path = os.path.join(path_to_search_bins, item)
                        if os.path.isfile(item_path):
                             # print(f"    Removing intermediate .bin: {item_path}") # Optional print
                             os.remove(item_path)
        print(f"  Attempted cleanup of intermediate files from {temp_dir_for_gltf} for {base_filename}.")
    except Exception as e_clean:
        print(f"  Warning: Could not clean up all intermediate files for {base_filename}: {e_clean}")
            
    return True
    
    
def main():
    parser = argparse.ArgumentParser(description="Convert all .vtp files in an input folder to .glb files.")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to the directory containing .vtp files.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Path to the directory where .glb files will be saved. "
                             "Defaults to the input directory if not specified.")
    parser.add_argument("--temp_subdir", type=str, default="gltf_temp_intermediates", # More descriptive name
                        help="Name of the subdirectory (within output_dir) for intermediate .gltf/.bin files during conversion. Default: gltf_temp_intermediates")

    args = parser.parse_args()

    if vtk is None or GLTF2 is None:
        print("VTK or pygltflib library not available. Exiting.")
        return

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        return

    output_directory = args.output_dir if args.output_dir else args.input_dir
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)
        print(f"Created output directory: {output_directory}")
    elif not os.path.isdir(output_directory):
        print(f"Error: Specified output path {output_directory} exists but is not a directory.")
        return

    # Common temporary directory for all conversions, created inside the final output directory
    common_temp_dir = os.path.join(output_directory, args.temp_subdir)
    if os.path.exists(common_temp_dir):
        print(f"Cleaning up existing common temporary directory: {common_temp_dir}")
        try:
            shutil.rmtree(common_temp_dir)
        except Exception as e_rm:
            print(f"  Warning: Could not remove old common temp directory: {e_rm}")
    os.makedirs(common_temp_dir, exist_ok=True)


    vtp_files = glob.glob(os.path.join(args.input_dir, "*.vtp"))

    if not vtp_files:
        print(f"No .vtp files found in {args.input_dir}")
        if os.path.isdir(common_temp_dir) and not os.listdir(common_temp_dir):
            try: os.rmdir(common_temp_dir)
            except OSError: pass # If other processes made files there
        return

    print(f"Found {len(vtp_files)} .vtp files to convert to .glb in '{output_directory}'")
    
    success_count = 0
    failure_count = 0

    for vtp_file in vtp_files:
        basename = os.path.splitext(os.path.basename(vtp_file))[0]
        final_glb_file = os.path.join(output_directory, f"{basename}.glb") 
        
        # Pass the common temporary directory for this conversion's intermediates
        if convert_vtp_to_final_glb(vtp_file, final_glb_file, common_temp_dir):
            success_count += 1
        else:
            failure_count += 1
        print("-" * 30)

    # Final cleanup of the common temporary directory
    try:
        if os.path.isdir(common_temp_dir):
            print(f"Attempting final cleanup of common temporary directory: {common_temp_dir}")
            shutil.rmtree(common_temp_dir)
            print("Common temporary directory cleaned up.")
    except Exception as e_final_clean:
        print(f"  Warning: Could not fully clean up common temporary directory {common_temp_dir}: {e_final_clean}")

    print("\nConversion Summary:")
    print(f"  Successfully converted: {success_count}")
    print(f"  Failed conversions: {failure_count}")

if __name__ == "__main__":
    main()
