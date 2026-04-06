import numpy as np
import requests
import time
import os

API_URL = "https://api.opentopodata.org/v1/mapzen"

def normalize_coordinates(coords):
    """
    Core data cleaning function:
    1. Correct longitude range: convert 0~360 to -180~180
    2. Correct latitude boundaries: prevent exactly -90/90 from causing API edge case failures
    input: coords shape (N, 2), assumed format [Lon, Lat]
    """
    coords_norm = coords.copy()
    
    # Process longitude (Column 0): if > 180, subtract 360
    lon_mask = coords_norm[:, 0] > 180
    coords_norm[lon_mask, 0] -= 360
    
    # Ensure longitude is between -180 and 180
    coords_norm[:, 0] = np.clip(coords_norm[:, 0], -180.0, 180.0)
    
    # Ensure latitude is between -90 and 90
    coords_norm[:, 1] = np.clip(coords_norm[:, 1], -89.9999, 89.9999)
    
    return coords_norm

def get_elevation_batch(coords_chunk):
    locations_str = "|".join([f"{lat:.4f},{lon:.4f}" for lon, lat in coords_chunk])
    
    data = {
        'locations': locations_str,
        'interpolation': 'cubic'
    }
    
    try:
        response = requests.post(API_URL, data=data, timeout=30)
        response.raise_for_status()
        result_json = response.json()
        
        if 'results' in result_json:
            return [res.get('elevation') for res in result_json['results']]
        else:
            return [None] * len(coords_chunk)
            
    except Exception as e:
        print(f"  [Error] Request failed: {e}")
        # Print first coordinate for debugging
        print(f"  [Debug] Current batch first coordinate: {coords_chunk[0]}")
        return [None] * len(coords_chunk)

def process_file(file_key, file_path):
    print(f"\nProcessing: {file_key} ({file_path})")
    
    if not os.path.exists(file_path):
        print(f"  [Skipped] File does not exist")
        return

    # Load data
    coords = np.load(file_path)
    
    # Shape correction (ensure N rows and 2 columns)
    if coords.shape[0] < coords.shape[1] and coords.shape[0] == 2:
        coords = coords.T
        
    # Format detection and correction (Lon, Lat) vs (Lat, Lon)
    if np.max(np.abs(coords[:, 0])) > 90:
        print("  [Info] Detected (Lon, Lat) format")
        # No action needed, default is Lon, Lat
    elif np.max(np.abs(coords[:, 1])) > 90:
        print("  [Info] Detected (Lat, Lon) format, flipping...")
        coords = np.fliplr(coords)
    else:
        print("  [Info] Cannot distinguish lat/lon from value range, defaulting to (Lon, Lat)")

    # Normalize coordinates
    print("  [Info] Performing coordinate normalization (0-360 -> -180/180)...")
    coords_clean = normalize_coordinates(coords)
    
    print(f"  Data ready, total {len(coords_clean)} points")
    
    elevations = []
    batch_size = 80
    
    for i in range(0, len(coords_clean), batch_size):
        chunk = coords_clean[i : i + batch_size]
        
        print(f"  Progress: {i} / {len(coords_clean)} ...", end="\r")
        
        chunk_elevs = get_elevation_batch(chunk)
        elevations.extend(chunk_elevs)
        
        time.sleep(1.2) 

    output_path = file_path.replace("coords_data.npy", "elevation_data.npy")
    np.save(output_path, np.array(elevations))
    print(f"\n  [Done] Results saved to: {output_path}")

if __name__ == "__main__":
    # Configuration section
    FILES = {
        "cn": "./storage/era5/bin_data_cn/coords_data.npy",
        "global": "./storage/era5/bin_data_global/coords_data.npy",
        "shenzhen": "./storage/sz_weather/bin_data/coords_data.npy"
    }

    for key, path in FILES.items():
        process_file(key, path)