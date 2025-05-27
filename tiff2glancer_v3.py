import argparse
import numpy as np
from tifffile import memmap
from cloudvolume import CloudVolume
import os
from scipy.ndimage import zoom
from tqdm import tqdm
import time

import sys

def main():
  parser = argparse.ArgumentParser(description='Convert a TIFF volume to Neuroglancer precomputed format with mipmaps.')
  parser.add_argument('--input', required=True, help='Path to input TIFF file')
  parser.add_argument('--output', required=True, help='Path to output directory for Neuroglancer dataset')
  parser.add_argument('--voxel-size', type=int, nargs=3, default=[500, 500, 500], metavar=('Z', 'Y', 'X'),
                      help='Voxel size in nanometers (default: 500 500 500)')
  parser.add_argument('--chunk-size', type=int, nargs=3, default=[128, 128, 128], metavar=('Z', 'Y', 'X'),
                      help='Chunk size (default: 128 128 128)')
  parser.add_argument('--mip-levels', type=int, default=2, help='Number of mipmap levels (default: 2)')

  args = parser.parse_args()
  print(args, flush=True)
  
  tiff_path = args.input
  output_path = os.path.abspath(args.output)
  voxel_size = args.voxel_size
  chunk_size = args.chunk_size

  # Validate inputs
  if not os.path.exists(tiff_path):
    raise FileNotFoundError(f"TIFF file not found: {tiff_path}")
  if any(v <= 0 for v in voxel_size):
    raise ValueError("Voxel sizes must be positive")
  if any(c <= 0 for c in chunk_size):
    raise ValueError("Chunk sizes must be positive")

  os.makedirs(output_path, exist_ok=True)
  cloudvolume_path = f'file://{output_path}'

  print(f"Reading TIFF from: {tiff_path}")
  tiff = memmap(tiff_path)
  shape = tiff.shape  # (Z, Y, X)
  dtype = tiff.dtype.name
  print(f"Shape: {shape}, dtype: {dtype}")

  # Validate dtype
  supported_dtypes = ['uint8', 'uint16', 'uint32', 'float32']
  if dtype not in supported_dtypes:
    print(f"Input data dtype {dtype} not supported")
    sys.exit(0)
  if not tiff.flags['C_CONTIGUOUS']:
    print("Input data is not C_CONTIGUOUS")
    sys.exit(0)
  
  # Create base info for mip level 0
  info = CloudVolume.create_new_info(
    num_channels=1,
    layer_type='image',
    data_type=dtype,
    encoding='raw',
    resolution=voxel_size[::-1],  # X, Y, Z
    voxel_offset=[0, 0, 0],
    chunk_size=chunk_size[::-1],  # X, Y, Z
    volume_size=shape[::-1],      # X, Y, Z
  )

  # Create and add mip levels with proper chunk sizes
  factors = [(2**i, 2**i, 2**i) for i in range(1, args.mip_levels+1)]
  for i, factor in enumerate(factors):
    # Calculate the downsampled volume size for this mip level
    scale = 2**(i+1)
    scaled_size = [sh // scale for sh in shape[::-1]]  # X, Y, Z
    
    # Add the scale to the info with the specified chunk size
    new_resolution = [r * scale for r in voxel_size[::-1]]  # X, Y, Z
    info['scales'].append({
      'key': f"{new_resolution[0]}_{new_resolution[1]}_{new_resolution[2]}",
      'resolution': new_resolution,
      'voxel_offset': [0, 0, 0],
      'chunk_sizes': [[chunk_size[2], chunk_size[1], chunk_size[0]]],  # X, Y, Z
      'size': scaled_size,
      'encoding': 'raw',
    })

  # Create CloudVolume with all scales defined
  vol = CloudVolume(cloudvolume_path, info=info, cache=True)
  try:
    vol.commit_info()
  except Exception as e:
    raise RuntimeError(f"Failed to commit info file: {e}")

  print("Uploading full-res volume...")
  tStart = time.time()
  # Get volume shape in ZYX and convert chunk size to match
  z_dim, y_dim, x_dim = tiff.shape
  cz, cy, cx = chunk_size  # chunk size in ZYX

  # Calculate total iterations for the progress bar
  total_z_iters = (z_dim + cz - 1) // cz
  total_y_iters = (y_dim + cy - 1) // cy
  total_x_iters = (x_dim + cx - 1) // cx
  total_chunks = total_z_iters * total_y_iters * total_x_iters

  try:
    # Create a single progress bar for all chunks
    with tqdm(total=total_chunks, desc="Uploading volume", unit="chunks") as pbar:
      for z in range(0, z_dim, cz):
        for y in range(0, y_dim, cy):
          for x in range(0, x_dim, cx):
            z1 = min(z + cz, z_dim)
            y1 = min(y + cy, y_dim)
            x1 = min(x + cx, x_dim)
            # Read chunk from TIFF (ZYX order)
            chunk = tiff[z:z1, y:y1, x:x1]
            # force little endian
            chunk = chunk.astype(chunk.dtype.newbyteorder('<'))
            # Transpose to XYZ and add channel dimension
            chunk = np.transpose(chunk, (2, 1, 0))  # to XYZ
            chunk = chunk[..., np.newaxis]          # to XYZ1
            # Write chunk to CloudVolume
            vol[x:x1, y:y1, z:z1] = chunk
            # Update the progress bar
            pbar.update(1)
  except Exception as e:
    raise RuntimeError(f"Failed to upload full-res volume: {e}")

  print(f"Full-res volume processed in {(time.time() - tStart):.2f} seconds")

  print("Generating mipmaps...")
  # Now we use the scales already defined
  for mip in range(1, len(factors)+1):
    src = CloudVolume(cloudvolume_path, mip=mip-1, cache=True)
    dst = CloudVolume(cloudvolume_path, mip=mip, cache=True)
    print(f"Generating mip {mip} from mip {mip-1}...")
    scale_factor = 2  # Each mip level is downsampled by a factor of 2
    
    # Get destination chunk size from the info
    dst_chunk_size = dst.info['scales'][0]['chunk_sizes'][0]  # [X, Y, Z]
    
    # Source chunk size to produce aligned destination chunk
    src_chunk_size = [cs * scale_factor for cs in dst_chunk_size]  # [X, Y, Z]
    
    src_x_size, src_y_size, src_z_size = src.shape[:3]  # X, Y, Z of source
    dst_x_size, dst_y_size, dst_z_size = dst.shape[:3]  # X, Y, Z of destination
    
    # Iterate over source volume in aligned chunks
    for x in tqdm(range(0, src_x_size, src_chunk_size[0]), desc=f"Mip {mip} X"):
      x_end = min(x + src_chunk_size[0], src_x_size)
      dst_x = x // scale_factor
      dst_x_end = x_end // scale_factor
      for y in range(0, src_y_size, src_chunk_size[1]):
        y_end = min(y + src_chunk_size[1], src_y_size)
        dst_y = y // scale_factor
        dst_y_end = y_end // scale_factor
        for z in range(0, src_z_size, src_chunk_size[2]):
          z_end = min(z + src_chunk_size[2], src_z_size)
          dst_z = z // scale_factor
          dst_z_end = z_end // scale_factor
          # Read source chunk
          chunk = src[x:x_end, y:y_end, z:z_end, :]
          # Downsample
          downsampled = zoom(chunk, zoom=[1/scale_factor, 1/scale_factor, 1/scale_factor, 1], order=1)

          # Calculate target shape - what we actually need
          target_x = min(dst_chunk_size[0], dst_x_size - dst_x)
          target_y = min(dst_chunk_size[1], dst_y_size - dst_y)
          target_z = min(dst_chunk_size[2], dst_z_size - dst_z)

          # Handle both padding (if too small) and cropping (if too large)
          if downsampled.shape[0] != target_x or downsampled.shape[1] != target_y or downsampled.shape[2] != target_z:
            # Create array of the correct size
            fixed_shape = np.zeros((target_x, target_y, target_z, downsampled.shape[3]), dtype=downsampled.dtype)
            
            # Copy what we can (handles both crop and pad cases)
            sx = min(downsampled.shape[0], target_x)
            sy = min(downsampled.shape[1], target_y)
            sz = min(downsampled.shape[2], target_z)
            
            fixed_shape[:sx, :sy, :sz, :] = downsampled[:sx, :sy, :sz, :]
            downsampled = fixed_shape
            
            # Update destination bounds to reflect actual shape
            dst_x_end = dst_x + downsampled.shape[0]
            dst_y_end = dst_y + downsampled.shape[1]
            dst_z_end = dst_z + downsampled.shape[2]

          # Write to destination chunk
          dst[dst_x:dst_x_end, dst_y:dst_y_end, dst_z:dst_z_end, :] = downsampled

  print(f"Conversion complete with mipmaps.")
  print(f"Dataset saved at: {output_path}")

if __name__ == '__main__':
  main()
