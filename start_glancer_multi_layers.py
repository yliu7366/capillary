import neuroglancer
import webbrowser
import numpy as np
try:
    from cloudvolume import CloudVolume
    CLOUDVOLUME_AVAILABLE = True
except ImportError:
    CLOUDVOLUME_AVAILABLE = False
    print("Warning: cloudvolume library not found. Dataset validation and some automatic viewer property settings will be skipped or limited.")

import os
import argparse
import json
import http.server
import socketserver
import threading
import sys
import re
import gzip
import socket
import urllib.parse 
import posixpath 
import time
import functools

# Get the local machine's IP address
def get_local_ip():
    """Get the local machine's IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.1) 
        s.connect(("8.8.8.8", 80)) 
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

# Custom HTTP server
class NeuroglancerHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    # Class variable to store the root directory. Will be set by start_http_server.
    # This is a bit of a hack for SimpleHTTPRequestHandler. A more robust way for custom
    # handlers is to pass it in __init__ and store on self.server.directory or similar.
    # However, SimpleHTTPRequestHandler uses os.getcwd() directly in translate_path.
    # So, we override translate_path.

    # We need to pass the server_root to the handler.
    # The standard way to do this with TCPServer is to make the handler factory take arguments.
    
    def __init__(self, *args, server_root_directory=None, **kwargs):
        if server_root_directory is None:
            raise ValueError("server_root_directory must be provided to NeuroglancerHTTPRequestHandler")
        self.server_root_directory = server_root_directory
        super().__init__(*args, **kwargs)


    def translate_path(self, path):
        """
        Translate a /-separated PATH to the local filename syntax.
        Path components constituting the requested file path are unquoted.
        The resulting local path is normalized.
        This version uses self.server_root_directory instead of os.getcwd().
        """
        # abandon query parameters
        path = path.split('?',1)[0]
        path = path.split('#',1)[0]
        # Don't forget explicit trailing slash when normalizing. Issue17324
        trailing_slash = path.rstrip().endswith('/')
        try:
            path = urllib.parse.unquote(path, errors='surrogatepass')
        except UnicodeDecodeError:
            path = urllib.parse.unquote(path)
        path = posixpath.normpath(path)
        words = path.split('/')
        words = filter(None, words)
        
        # Use the explicit server_root_directory
        path = self.server_root_directory 
        for word in words:
            if os.path.dirname(word) or word in (os.curdir, os.pardir):
                # Ignore components that are not simple names
                continue
            path = os.path.join(path, word)
        if trailing_slash:
            path += '/'
        
        # --- DIAGNOSTIC PRINT for translate_path ---
        # print(f"[HANDLER.translate_path] Input path: {original_path_arg}, Server Root: {self.server_root_directory}, Translated: {path}")
        return path


    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        self.send_header('Access-Control-Expose-Headers', '*')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        # --- DIAGNOSTIC PRINTS ---
        # print(f"\n[HANDLER] Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        # print(f"[HANDLER] Requested raw path from client: {self.path}")
        # print(f"[HANDLER] Explicit server root for this handler: {self.server_root_directory}")
        # --- END DIAGNOSTIC PRINTS ---
        
        path_query = self.path.split('?', 1)
        request_path_component = path_query[0] 
        
        # translated_fs_path will now use self.server_root_directory via our overridden translate_path
        translated_fs_path = self.translate_path(request_path_component) 
        
        # --- DIAGNOSTIC PRINTS ---
        # print(f"[HANDLER] URL path component used for translation: {request_path_component}")
        # print(f"[HANDLER] Translated to absolute filesystem path: {translated_fs_path}")

        # if os.path.exists(translated_fs_path):
        #     print(f"[HANDLER] Filesystem check: Exact path {translated_fs_path} EXISTS.")
        # else:
        #     print(f"[HANDLER] Filesystem check: Exact path {translated_fs_path} DOES NOT exist.")
        # --- END DIAGNOSTIC PRINTS ---

        gzipped_fs_path = translated_fs_path + '.gz'
        
        # --- DIAGNOSTIC PRINTS ---
        # if os.path.exists(gzipped_fs_path):
        #     print(f"[HANDLER] Filesystem check: Gzipped path {gzipped_fs_path} EXISTS.")
        # else:
        #     print(f"[HANDLER] Filesystem check: Gzipped path {gzipped_fs_path} DOES NOT exist.")
        # --- END DIAGNOSTIC PRINTS ---

        if not os.path.exists(translated_fs_path) and os.path.exists(gzipped_fs_path):
            # print(f"[HANDLER] Attempting to serve GZIP version: {gzipped_fs_path}")
            try:
                with gzip.open(gzipped_fs_path, 'rb') as f:
                    content = f.read()
                self.send_response(200)
                mimetype = 'application/octet-stream'
                if request_path_component.endswith('info'):
                    mimetype = 'application/json'
                self.send_header('Content-Type', mimetype)
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                # print(f"[HANDLER] Successfully served GZIP file {gzipped_fs_path}")
                return
            except Exception as e:
                print(f"[HANDLER] Error serving GZIP file {gzipped_fs_path}: {e}")
                self.send_error(500, f"Error processing GZIP file: {e}")
                return
        
        # print(f"[HANDLER] Falling back to SimpleHTTPRequestHandler.do_GET for raw client path: {self.path}")
        # The base class's do_GET will call our translate_path again.
        return http.server.SimpleHTTPRequestHandler.do_GET(self)
        
def start_http_server(directory_to_serve_from, port=8000):
    # We no longer need to os.chdir here for the server itself,
    # but NeuroglancerHTTPRequestHandler.translate_path will handle paths correctly.
    # original_cwd = os.getcwd() # Not strictly needed for this specific problem anymore
    # print(f"\n[SERVER START] Original CWD (will be restored): {original_cwd}") # Informational
    
    abs_directory_to_serve_from = os.path.abspath(directory_to_serve_from)
    print(f"[SERVER START] Absolute directory HTTP server will serve from: {abs_directory_to_serve_from}")

    # Create a handler factory that passes the serving directory to our custom handler
    handler_factory = functools.partial(NeuroglancerHTTPRequestHandler, 
                                        server_root_directory=abs_directory_to_serve_from)

    httpd = None
    while httpd is None:
        try:
            socketserver.TCPServer.allow_reuse_address = True
            # Pass the factory to TCPServer
            httpd = socketserver.TCPServer(("", port), handler_factory) 
        except OSError as e:
            if e.errno == socket.errno.EADDRINUSE: 
                print(f"[SERVER START] Port {port} is in use, trying {port+1}")
                port += 1
            else:
                print(f"[SERVER START] EXCEPTION during TCPServer creation: {e}")
                raise 
    
    ip_address = get_local_ip()
    print(f"[SERVER START] HTTP server starting at http://{ip_address}:{port}/, serving files from: {abs_directory_to_serve_from}")
    server_thread = threading.Thread(target=httpd.serve_forever, name="HTTPServerThread")
    server_thread.daemon = True
    server_thread.start()
    print(f"[SERVER START] HTTP server thread '{server_thread.name}' started.")
    # No need to os.chdir back if we didn't change it for the server.
    # If other parts of your script rely on a specific CWD, manage it outside this function.
    # print(f"[SERVER START] CWD in main thread is now: {os.getcwd()}\n")
    return port, ip_address, httpd

def get_neuroglancer_source_url(dataset_full_path, use_http_server, 
                                server_ip=None, server_port=None, http_server_root_abs_path=None):
    if not os.path.isabs(dataset_full_path): # Should already be absolute from main()
        dataset_full_path = os.path.abspath(dataset_full_path)

    print(f"[URL GEN] Checking dataset path: {dataset_full_path}")
    if not os.path.isdir(dataset_full_path):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_full_path}")
    
    info_file = os.path.join(dataset_full_path, 'info')
    info_file_gz = os.path.join(dataset_full_path, 'info.gz')
    if not os.path.exists(info_file) and not os.path.exists(info_file_gz):
         raise FileNotFoundError(f"Info file (or info.gz) not found in {dataset_full_path}")
    print(f"[URL GEN] Info file confirmed at: {dataset_full_path}")

    if use_http_server:
        if server_ip is None or server_port is None or http_server_root_abs_path is None:
            raise ValueError("[URL GEN] Server IP, port, and root must be provided for HTTP source from local files.")
        if not os.path.isabs(http_server_root_abs_path):
             raise ValueError("[URL GEN] http_server_root_abs_path must be an absolute path.")

        relative_path_to_dataset = os.path.relpath(dataset_full_path, http_server_root_abs_path)
        relative_path_url = relative_path_to_dataset.replace(os.sep, '/')
        
        print(f"[URL GEN] HTTP Root: {http_server_root_abs_path}")
        print(f"[URL GEN] Dataset Full Path: {dataset_full_path}")
        print(f"[URL GEN] Relative path for URL: {relative_path_url}")
        
        # This check ensures the dataset is actually INSIDE the http_server_root
        if os.path.isabs(relative_path_url) or relative_path_url.startswith("..") or relative_path_url.startswith("/"):
            raise ValueError(
                f"[URL GEN] Dataset path {dataset_full_path} is not a valid subdirectory "
                f"of HTTP server root {http_server_root_abs_path}. "
                f"Resulting relative path for URL: {relative_path_url}"
            )
        final_url = f'precomputed://http://{server_ip}:{server_port}/{relative_path_url}'
        print(f"[URL GEN] Generated HTTP source URL: {final_url}")
        return final_url
    else: # file://
        abs_path_url = dataset_full_path.replace('\\', '/')
        if os.name == 'nt': 
            if not re.match(r"^[a-zA-Z]:/", abs_path_url):
                 abs_path_url = '/' + abs_path_url 
            if not abs_path_url.startswith('/'): # Should be like /C:/path
                 abs_path_url = '/' + abs_path_url
        else: 
            if not abs_path_url.startswith('/'):
                 abs_path_url = '/' + abs_path_url
        final_url = f'precomputed://file://{abs_path_url}'
        print(f"[URL GEN] Generated File source URL: {final_url}")
        return final_url


def main():
    parser = argparse.ArgumentParser(description='Visualize precomputed Neuroglancer datasets from a root directory.')
    parser.add_argument('--root_dir', required=True, help='Root directory containing the dataset subfolders.')
    parser.add_argument('--raw_subdir', required=True, help='Subfolder name for the raw image volume (relative to root_dir).')
    parser.add_argument('--seg_subdir', default=None, help='Optional: Subfolder name for the segmentation mask (relative to root_dir).')
    parser.add_argument('--skel_subdir', default=None, help='Optional: Subfolder name for skeletons (must be a precomputed segmentation source with skeleton data, relative to root_dir).')
    
    parser.add_argument('--no-browser', action='store_true', help='Do not open the browser automatically.')
    parser.add_argument('--http', action='store_true', help='Use HTTP server with CORS support for local files. Highly recommended.')
    parser.add_argument('--port', type=int, default=8000, help='Port for HTTP server (default: 8000)')
    parser.add_argument('--bind-address', default=None, help='Manually specify the Neuroglancer server IP address for viewer URL.')
    args = parser.parse_args()

    print("[MAIN] Script arguments:", args)

    abs_root_dir = os.path.abspath(args.root_dir)
    print(f"[MAIN] Absolute root directory resolved to: {abs_root_dir}")
    if not os.path.isdir(abs_root_dir):
        print(f"Error: Root directory not found: {abs_root_dir}")
        sys.exit(1)

    http_server_instance = None
    http_server_port_actual = args.port # Will be updated if port is in use
    http_server_ip_actual = None
    
    if args.http:
        print("[MAIN] --http flag is set. Starting HTTP server.")
        http_server_port_actual, http_server_ip_actual, http_server_instance = start_http_server(abs_root_dir, args.port)
    elif not args.http :
         print("[MAIN] --http flag is NOT set. Using file:// sources for local datasets.")
         print("Warning: Browser security may restrict access. The --http option is strongly recommended.")

    active_layers_config = {} 
    vol_for_viewer_properties = None 

    try:
        # Raw Volume
        raw_vol_path_abs = os.path.join(abs_root_dir, args.raw_subdir)
        print(f"[MAIN] Absolute path for raw volume: {raw_vol_path_abs}")
        raw_source_url = get_neuroglancer_source_url(
            raw_vol_path_abs, args.http, http_server_ip_actual, http_server_port_actual, abs_root_dir
        )
        active_layers_config['image'] = {'source_url': raw_source_url, 'path': raw_vol_path_abs}
        # print(f"Raw Volume source: {raw_source_url}") # Already printed by get_neuroglancer_source_url

        # Segmentation
        if args.seg_subdir:
            seg_path_abs = os.path.join(abs_root_dir, args.seg_subdir)
            print(f"[MAIN] Absolute path for segmentation: {seg_path_abs}")
            try:
                seg_source_url = get_neuroglancer_source_url(
                    seg_path_abs, args.http, http_server_ip_actual, http_server_port_actual, abs_root_dir
                )
                active_layers_config['segmentation'] = {'source_url': seg_source_url, 'path': seg_path_abs}
                # print(f"Segmentation source: {seg_source_url}")
            except (FileNotFoundError, ValueError) as e:
                print(f"Warning: Segmentation dataset from '{args.seg_subdir}' not loaded. Error: {e}")
        
        # Skeletons
        if args.skel_subdir:
            skel_path_abs = os.path.join(abs_root_dir, args.skel_subdir)
            print(f"[MAIN] Absolute path for skeletons: {skel_path_abs}")
            try:
                skel_source_url = get_neuroglancer_source_url(
                    skel_path_abs, args.http, http_server_ip_actual, http_server_port_actual, abs_root_dir
                )
                active_layers_config['skeletons'] = {'source_url': skel_source_url, 'path': skel_path_abs}
                # print(f"Skeletons source: {skel_source_url}")
            except (FileNotFoundError, ValueError) as e:
                print(f"Warning: Skeleton dataset from '{args.skel_subdir}' not loaded. Error: {e}")

    except (FileNotFoundError, ValueError) as e: 
        print(f"Critical Error preparing raw volume: {e}")
        if http_server_instance:
            http_server_instance.shutdown()
            http_server_instance.server_close()
        sys.exit(1)

    if CLOUDVOLUME_AVAILABLE:
        for layer_name, config in active_layers_config.items():
            print(f"[MAIN] Attempting to validate {layer_name} with CloudVolume using URL: {config['source_url']}")
            cv_path = config['source_url'].replace('precomputed://', '')
            try:
                vol = CloudVolume(cv_path, mip=0, progress=False, fill_missing=True) 
                print(f"  {layer_name.capitalize()} CV loaded: shape={vol.shape}, dtype={vol.dtype}, resolution={vol.resolution}")
                config['cloudvolume_instance'] = vol
                if layer_name == 'image' or vol_for_viewer_properties is None:
                    vol_for_viewer_properties = vol
            except Exception as e:
                print(f"  Warning: Failed to load/validate {layer_name} with CloudVolume: {e}") # This is where your error occurs
    
    if not CLOUDVOLUME_AVAILABLE and 'image' in active_layers_config: 
        print("[MAIN] Attempting to parse info file for viewer properties (CloudVolume not available)...")
        try:
            info_json_path = os.path.join(active_layers_config['image']['path'], 'info')
            info_json_gz_path = info_json_path + '.gz'
            info_data = None
            if os.path.exists(info_json_path):
                with open(info_json_path, 'r') as f: info_data = json.load(f)
            elif os.path.exists(info_json_gz_path):
                with gzip.open(info_json_gz_path, 'rt') as f: info_data = json.load(f)
            
            if info_data and info_data.get("scales"):
                highest_res_scale = info_data["scales"][0]
                shape_xyz = highest_res_scale.get("size") 
                res_xyz = highest_res_scale.get("resolution") 
                if shape_xyz and res_xyz:
                    vol_for_viewer_properties = type('MockVol', (), {'shape': np.array(shape_xyz), 'resolution': np.array(res_xyz)})()
                    print(f"  Extracted shape {shape_xyz} and resolution {res_xyz} from info file.")
        except Exception as e:
            print(f"  Warning: Could not parse info file for viewer properties: {e}")


    neuroglancer.set_server_bind_address('0.0.0.0', bind_port=0) 
    viewer = neuroglancer.Viewer()
    ng_server_ip_for_url = args.bind_address if args.bind_address else get_local_ip()

    with viewer.txn() as s:
        if 'image' in active_layers_config:
            s.layers.append(
                name='image',
                layer=neuroglancer.ImageLayer(
                    source=active_layers_config['image']['source_url'],
                    shader="""
                    #uicontrol float brightness slider(min=-1, max=1, default=0)
                    #uicontrol float contrast slider(min=-1, max=1, default=0)
                    void main() {
                      float val = toNormalized(getDataValue());
                      emitGrayscale(val * (1.0 + contrast) + brightness);
                    }
                    """
                )
            )
        
        if 'segmentation' in active_layers_config:
            s.layers.append(
                name='segmentation',
                layer=neuroglancer.SegmentationLayer(
                    source=active_layers_config['segmentation']['source_url']
                )
            )

        if 'skeletons' in active_layers_config:
            s.layers.append(
                name='skeletons',
                layer=neuroglancer.SegmentationLayer( 
                    source=active_layers_config['skeletons']['source_url']
                )
            )
        
        # default layout works without setting layout explicitly
        #s.layout = 'xy-3d'

        if vol_for_viewer_properties:
            center_coords_xyz = np.array(vol_for_viewer_properties.shape[:3]) / 2.0
            s.position = center_coords_xyz 
            
            if hasattr(vol_for_viewer_properties, 'resolution') and vol_for_viewer_properties.resolution is not None:
                # Ensure resolution is a list/array of numbers
                scales_xyz = [float(r) for r in vol_for_viewer_properties.resolution[:3]]
                s.dimensions = neuroglancer.CoordinateSpace(
                    names=['x', 'y', 'z'], # Or ['c^', 'y', 'x'] or similar if your data is ordered differently
                    units=['nm', 'nm', 'nm'], # Specify units for each dimension
                    scales=scales_xyz # Voxel size for each dimension
                )
           
        try:
            s.show_scale_bar = True
            s.cross_section_scale = 1.0 
        except AttributeError:
            print("Warning: Some viewer state properties (e.g., show_scale_bar) may not be supported in this Neuroglancer version.")


    viewer_url_str = str(viewer)
    match = re.search(r'http://[^:]+:(\d+)/v/(.*)', viewer_url_str)
    if match:
        ng_port, ng_path = match.groups()
        remote_url = f'http://{ng_server_ip_for_url}:{ng_port}/v/{ng_path}'
        print(f"\nNeuroglancer viewer URLs:\n- Local access:  {viewer_url_str}\n- Remote access: {remote_url}")
    else:
        print(f"\nNeuroglancer viewer URL: {viewer_url_str}")
        print(f"Note: For remote access, replace hostname with your IP: {ng_server_ip_for_url}")

    if not args.no_browser:
        try:
            webbrowser.open(viewer_url_str)
        except Exception as e:
            print(f"Failed to open browser: {e}. Please open the URL manually.")

    print("\nKeep this terminal window open to maintain the Neuroglancer and HTTP servers.")
    print("Press Ctrl+C to exit.")
    try:
        while True:
            import time # Ensure time is imported
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        if http_server_instance:
            http_server_instance.shutdown()
            http_server_instance.server_close()
            print("HTTP server stopped.")
        neuroglancer.stop()
        print("Neuroglancer server stopped.")
        sys.exit(0)

if __name__ == '__main__':
    # Add this import at the top of the file
    import time 
    main()
