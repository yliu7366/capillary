import neuroglancer
import webbrowser
import numpy as np
from cloudvolume import CloudVolume
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

# Get the local machine's IP address
def get_local_ip():
    """Get the local machine's IP address"""
    try:
        # Create a socket connection to an external server
        # This doesn't actually establish a connection, but gives us the IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        # Fallback to localhost if we can't determine the IP
        return "127.0.0.1"

# Custom HTTP server that handles compressed files properly for Neuroglancer
class NeuroglancerHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
  def end_headers(self):
    # Add CORS headers
    self.send_header('Access-Control-Allow-Origin', '*')
    self.send_header('Access-Control-Allow-Methods', 'GET')
    self.send_header('Access-Control-Allow-Headers', '*')
    self.send_header('Access-Control-Expose-Headers', '*')
    super().end_headers()
  
  def do_OPTIONS(self):
    self.send_response(200)
    self.end_headers()
  
  def do_GET(self):
    """Handle GET requests, checking for .gz files when needed"""
    # Clean up URL (remove query string if present)
    path = self.path.split('?')[0]
    
    # Convert URL path to file path
    file_path = self.translate_path(path)
    
    # If the exact file doesn't exist, try with .gz extension
    if not os.path.exists(file_path) and os.path.exists(file_path + '.gz'):
      try:
        # File exists with .gz extension
        with gzip.open(file_path + '.gz', 'rb') as f:
          content = f.read()
        
        self.send_response(200)
        
        # Set the content type
        if path.endswith('info'):
          self.send_header('Content-Type', 'application/json')
        else:
          # For chunk files
          self.send_header('Content-Type', 'application/octet-stream')
        
        # Content length
        self.send_header('Content-Length', str(len(content)))
        
        # Already decompressed, so no need for Content-Encoding
        self.end_headers()
        
        # Send the decompressed content
        self.wfile.write(content)
        return
      except Exception as e:
        print(f"Error serving gzip file {file_path + '.gz'}: {e}")
    
    # Fall back to normal handling if no .gz file or error occurred
    return http.server.SimpleHTTPRequestHandler.do_GET(self)

def start_http_server(directory, port=8000):
  """Start a HTTP server with CORS headers in a separate thread"""
  os.chdir(directory)
  handler = NeuroglancerHTTPRequestHandler
  
  # Try to start the server, incrementing port if needed
  while True:
    try:
      # Allow socket address reuse to prevent "Address already in use" errors
      socketserver.TCPServer.allow_reuse_address = True
      # Use "" instead of "localhost" to bind to all network interfaces
      httpd = socketserver.TCPServer(("", port), handler)
      break
    except OSError:
      print(f"Port {port} is in use, trying {port+1}")
      port += 1
  
  # Get the server's IP address for external access
  ip_address = get_local_ip()
  print(f"Starting HTTP server at http://{ip_address}:{port}/")
  server_thread = threading.Thread(target=httpd.serve_forever)
  server_thread.daemon = True  # Allow the thread to be terminated when the main program exits
  server_thread.start()
  return port, ip_address, httpd  # Return server instance for clean shutdown

def main():
  parser = argparse.ArgumentParser(description='Visualize precomputed Neuroglancer dataset.')
  parser.add_argument('--output', required=True, help='Path to precomputed dataset directory')
  parser.add_argument('--no-browser', action='store_true', help='Do not open the browser automatically')
  parser.add_argument('--http', action='store_true', help='Use HTTP server with CORS support')
  parser.add_argument('--port', type=int, default=8000, help='Port for HTTP server (default: 8000)')
  parser.add_argument('--bind-address', default=None, help='Manually specify the server IP address')
  args = parser.parse_args()
  
  # Path to the precomputed dataset
  output_path = args.output
  abs_output_path = os.path.abspath(output_path)
  
  # Validate output directory
  if not os.path.exists(output_path):
    raise FileNotFoundError(f"Precomputed dataset not found at: {output_path}")
  
  # Check for info file (both with and without .gz)
  info_path = os.path.join(output_path, 'info')
  info_path_gz = os.path.join(output_path, 'info.gz')
  
  if not os.path.exists(info_path) and not os.path.exists(info_path_gz):
    raise FileNotFoundError(f"Info file not found at: {info_path} or {info_path_gz}")
  
  # Variable to store the HTTP server instance for proper shutdown
  http_server = None
  
  # Start HTTP server if requested
  if args.http:
    port, server_ip, http_server = start_http_server(output_path, args.port)
    
    # Use the provided bind address if specified
    if args.bind_address:
      server_ip = args.bind_address
      
    cloudvolume_path = f'http://{server_ip}:{port}'
    print(f"Using HTTP source with CORS support: {cloudvolume_path}")
    print(f"For remote access, use: http://{server_ip}:{port}")
  else:
    # File approach - warn user about potential issues
    cloudvolume_path = f'file://{abs_output_path}'
    print(f"Using file source: {cloudvolume_path}")
    print("NOTE: Browser security may prevent direct file access. --http option is recommended.")
    print("NOTE: For remote access, you must use --http option.")
    
  """
  # Print info file contents for debugging
  try:
    # Try reading info file (with or without .gz)
    if os.path.exists(info_path):
      with open(info_path, 'r') as f:
        info = json.load(f)
    elif os.path.exists(info_path_gz):
      with gzip.open(info_path_gz, 'rt') as f:
        info = json.load(f)
    
    print(f"Info file contents: {json.dumps(info, indent=2)}")
  except Exception as e:
    print(f"Warning: Could not read info file: {e}")
  """
  
  # Verify the dataset with CloudVolume
  try:
    vol = CloudVolume(cloudvolume_path, mip=0)
    print(f"Dataset loaded: shape={vol.shape}, dtype={vol.dtype}")
  except Exception as e:
    print(f"Warning: Failed to load precomputed dataset with CloudVolume: {e}")
    print("Continuing with Neuroglancer setup anyway...")
    vol = None
  
  # Initialize Neuroglancer viewer
  neuroglancer.set_server_bind_address('0.0.0.0')  # Allow external access
  viewer = neuroglancer.Viewer()
  
  # Get the local machine's IP for Neuroglancer server
  local_ip = args.bind_address if args.bind_address else get_local_ip()
  
  # Configure the viewer state
  with viewer.txn() as s:
    try:
      # Add the precomputed volume as an image layer
      s.layers['volume'] = neuroglancer.ImageLayer(
        source=f'precomputed://{cloudvolume_path}',
        shader="""
          #uicontrol float min_threshold slider(min=0, max=1, default=0)
          #uicontrol float max_threshold slider(min=0, max=1, default=1)
          #uicontrol float brightness slider(min=-1, max=1, default=0)
          #uicontrol float contrast slider(min=-1, max=1, default=0)
          void main() {
            float value = toNormalized(getDataValue());
            if (value < min_threshold || value > max_threshold) {
              emitGrayscale(0.0);
            } else {
              value = value + brightness;
              value = value * (1.0 + contrast);
              emitGrayscale(clamp(value, 0.0, 1.0));
            }
          }
        """
      )
    except Exception as e:
      print(f"Failed to add ImageLayer: {e}")
      raise
    
    # Set initial view parameters
    s.layout = '3d'  # 3D view
    
    # Set position if volume size is known
    if vol is not None:
      s.position = np.array(vol.shape[:3]) / 2  # Center of the volume
    
    # Try to enable scale bar (fallback if not supported)
    try:
      s.show_scale_bar = True
    except AttributeError:
      print("Warning: 'show_scale_bar' not supported in this Neuroglancer version.")
  
  # Get the viewer URL, replacing localhost with the server IP for remote access
  viewer_url = str(viewer)
  
  # Extract the hostname and port from the Neuroglancer URL
  # It can be in format: http://localhost:xxxxx/v/yyyyy or http://hostname:xxxxx/v/yyyyy
  match = re.search(r'http://[^:]+:(\d+)/v/(.*)', viewer_url)
  if match:
    ng_port = match.group(1)
    path = match.group(2)
    # Create a URL with the machine's real IP
    remote_url = f'http://{local_ip}:{ng_port}/v/{path}'
    print(f"\nNeuroglancer viewer URLs:")
    print(f"- Local access:  {viewer_url}")
    print(f"- Remote access: {remote_url}")
    viewer_url_to_open = viewer_url  # Default to opening local URL
  else:
    print(f"\nNeuroglancer viewer URL: {viewer_url}")
    print("Note: Could not parse URL to generate remote access link")
    print("For remote access, try replacing the hostname in the URL with your IP address:", local_ip)
  
  if not args.no_browser:
    try:
      webbrowser.open(viewer_url_to_open)
    except Exception as e:
      print(f"Failed to open browser: {e}. Please open the URL manually.")
  
  print("\nKeep this terminal window open to maintain the servers.")
  print("\nPress Ctrl+C to exit.")
  
  try:
    while True:
      # Keep the script running to maintain the viewer
      # Use a short sleep to allow keyboard interrupts
      import time
      time.sleep(1)
  except KeyboardInterrupt:
    print("\nShutting down servers...")
    # Properly shut down the HTTP server if it exists
    if http_server:
      http_server.shutdown()
      http_server.server_close()
      print("HTTP server stopped.")
    # Also shut down the Neuroglancer server
    neuroglancer.stop()
    print("Neuroglancer server stopped.")
    sys.exit(0)

if __name__ == '__main__':
  main()
