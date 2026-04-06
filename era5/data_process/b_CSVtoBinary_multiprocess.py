import os
import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


class CSVToBinConverter:
    def __init__(self, 
                 data_dir, 
                 target_dir, 
                 keep_columns, 
                 final_features, 
                 log_transform_columns,
                 num_grids,
                 time_steps_per_day=24):
        """
        Initialize the converter with external parameters
        """
        self.data_dir = Path(data_dir)
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        # Parameter configuration
        self.keep_columns = keep_columns
        self.final_features = final_features
        self.log_transform_columns = log_transform_columns
        self.num_grids = num_grids
        self.time_steps_per_day = time_steps_per_day
        
    def calculate_derived_features(self, df):
        """Calculate derived physical variables: RH and WS"""
        # --- RH Calculation ---
        if 't2m' in df.columns and 'd2m' in df.columns:
            t_c = df['t2m'] - 273.15
            td_c = df['d2m'] - 273.15
            
            # Magnus formula
            es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
            e = 6.112 * np.exp((17.67 * td_c) / (td_c + 243.5))
            
            df['rh'] = (e / es).clip(0, 1) * 100

        # --- WS Calculation ---
        if 'u10' in df.columns and 'v10' in df.columns:
            df['ws'] = np.sqrt(df['u10']**2 + df['v10']**2)
        
        return df

    def process_units_and_log(self, df):
        """Physical unit conversion and Log1p transformation"""
        
        # Temperature: K -> C
        for col in ['t2m', 'd2m', 'skt']:
            if col in df.columns:
                df[col] = df[col] - 273.15

        # Pressure: Pa -> bar
        for col in ['sp', 'msl']:
            if col in df.columns:
                df[col] = df[col] / 1000.0

        # Height: m / m2s2 -> km
        if 'z' in df.columns:
            df['z'] = df['z'] / 9806.65  # Geopotential -> Height (km)
        
        if 'blh' in df.columns:
            df['blh'] = df['blh']  # m -> km

        # Percentage clipping
        for col in ['lcc', 'tcc']:
            if col in df.columns:
                df[col] = df[col].clip(0, 1)

        # Precipitation: m -> mm
        if 'tp' in df.columns:
            df['tp'] = (df['tp'] * 1000).clip(lower=0)

        # Radiation: J/m2 -> W/m2
        if 'ssrd' in df.columns:
             df['ssrd'] = (df['ssrd'] / 3600.0).clip(lower=0)

        # Log1p transformation (based on configuration list)
        for col in self.log_transform_columns:
            if col in df.columns:
                df[col] = np.log1p(df[col])
                
        return df

    def _process_wrapper(self, csv_file):
        """Multiprocessing wrapper"""
        try:
            bin_file = self.target_dir / f"{csv_file.stem}.bin"
            time_strs, grid_coords = self.process_single_file(csv_file, bin_file)
            
            return {
                "status": "success",
                "stem": csv_file.stem,
                "time_strs": time_strs,
                "grid_coords": grid_coords
            }
        except Exception as e:
            return {
                "status": "error",
                "file": csv_file.name,
                "error_msg": str(e)
            }

    def process_single_file(self, csv_file, bin_file):
        """Core processing logic"""
        # For maximum speed, try engine='pyarrow'
        df = pd.read_csv(csv_file)
        
        # Filter columns
        existing_cols = [c for c in self.keep_columns if c in df.columns]
        df = df[existing_cols]
        
        # Feature engineering
        df = self.calculate_derived_features(df)
        
        # Unit conversion and log transformation
        df = self.process_units_and_log(df)
        
        # Align grid
        grid_data = df.copy()
        expected_len = self.time_steps_per_day * self.num_grids
        
        if len(grid_data) != expected_len:
             if len(grid_data) > expected_len:
                 grid_data = grid_data.iloc[-expected_len:]
             else:
                 raise ValueError(f"Length mismatch: {len(grid_data)} < {expected_len}")
        
        # Dynamically generate GRIDID
        unique_grids_coords = grid_data[['LON_CENTER', 'LAT_CENTER']].drop_duplicates()
        unique_grids_coords = unique_grids_coords.sort_values(by=['LAT_CENTER', 'LON_CENTER']).reset_index(drop=True)
        unique_grids_coords['GRIDID'] = unique_grids_coords.index

        grid_data = grid_data.merge(unique_grids_coords, on=['LON_CENTER', 'LAT_CENTER'], how='left')
        grid_data = grid_data.sort_values(['DDATETIME', 'GRIDID'])
        
        # Extract features
        for col in self.final_features:
            if col not in grid_data.columns:
                # Fill with zeros if a configured feature fails or is missing
                grid_data[col] = 0.0
                
        # Extract and convert to float32
        weather_features = grid_data[self.final_features].values.astype(np.float32)
        
        # Reshape: [Time, Grid, Feature]
        data_3d = weather_features.reshape(
            self.time_steps_per_day, self.num_grids, len(self.final_features)
        )
        
        # Extract coordinates (only take the first time step)
        coords = grid_data[['LON_CENTER', 'LAT_CENTER']].values
        grid_coords = coords[:self.num_grids].astype(np.float32)
        
        # Extract time
        unique_times = grid_data['DDATETIME'].unique()
        time_strs = unique_times.tolist()
        
        # Write to binary file
        with open(bin_file, 'wb') as f:            
            data_3d.tofile(f)
        
        return time_strs, grid_coords

    def process_all_files(self, max_workers=None):
        """Multiprocessing entry point"""
        csv_files = sorted(list(self.data_dir.glob("*.csv")))
        
        if not csv_files:
            print(f"No CSV files found in {self.data_dir}")
            return

        global_time_strs = {}
        first_grid_coords = None 
        
        if max_workers is None:
            max_workers = max(1, os.cpu_count() - 2)

        print(f"Starting processing {len(csv_files)} files with {max_workers} workers...")
        print(f"   Input: {self.data_dir}")
        print(f"   Output: {self.target_dir}")
        print(f"   Grid Size: {self.num_grids}, Features: {len(self.final_features)}")
        
        start_time = time.time()

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._process_wrapper, f) for f in csv_files]
            
            for future in tqdm(as_completed(futures), total=len(csv_files), unit="file"):
                result = future.result()
                
                if result['status'] == 'success':
                    global_time_strs[result['stem']] = result['time_strs']
                    if first_grid_coords is None:
                        first_grid_coords = result['grid_coords']
                else:
                    print(f"\n[Error] File {result['file']} failed: {result['error_msg']}")

        # Organize metadata
        sorted_keys = sorted(global_time_strs.keys())
        sorted_global_times = {k: global_time_strs[k] for k in sorted_keys}

        if first_grid_coords is not None:
            self.save_global_metadata(sorted_global_times, first_grid_coords)
            print(f"\nDone! Total time: {time.time() - start_time:.2f}s")
        else:
            print("\nFailed to process any files successfully.")
    
    def save_global_metadata(self, global_time_strs, grid_coords):
        """Save global metadata"""
        print("Saving global metadata...")
        global_metadata = {
            'num_features': len(self.final_features),
            'num_grids': self.num_grids,
            'time_steps_per_day': self.time_steps_per_day,
            'feature_names': self.final_features,
            'coord_names': ['LON_CENTER', 'LAT_CENTER'],
            'dtype': 'float32',
            'log_transformed_features': self.log_transform_columns,
            'units_note': {
                'temperature': 'Celsius',
                'pressure': '100kPa (Bar)',
                'height_z_blh': 'Kilometers (km)',
                'percentage': '0-1 decimal',
                'precip': 'mm',
                'radiation': 'W/m2'
            }
        }

        with open(self.target_dir / 'metadata.json', 'w') as f:
            json.dump(global_metadata, f, indent=2)

        with open(self.target_dir / 'date_data.json', 'w') as f:
            json.dump(global_time_strs, f, indent=2)

        np.save(self.target_dir / 'coords_data.npy', grid_coords)


# ================= Configuration and Execution Section =================
if __name__ == "__main__":
    
    # Configuration
    # For global processing.
    INPUT_DIR = './storage/era5/era5_daily_data_global'
    OUTPUT_DIR = './storage/era5/bin_data_global'
    AREA = None
    GRID = [5, 5]

    # For regional processing.
    # INPUT_DIR = './storage/era5/era5_daily_data_cn'
    # OUTPUT_DIR = './storage/era5/bin_data_cn'
    # AREA = [54, 73, 3, 135]
    # GRID = [1, 1]

    # --- Calculate NUM_GRIDS ---
    lat_step, lon_step = GRID

    if AREA is None:
        # Global
        num_lat = int(round(180 / lat_step)) + 1
        num_lon = int(round(360 / lon_step))
        print(f"Mode: Global (Default)")
    else:
        # Regional
        north, west, south, east = AREA
        num_lat = int(round((north - south) / lat_step)) + 1
        # Calculate longitude (need to handle wrap-around cases)
        if east < west:
            lon_span = (east + 360) - west
        else:
            lon_span = east - west
        # If the span is very close to 360 degrees, treat as global mode, no +1
        if abs(lon_span - 360) < 1e-6:
            num_lon = int(round(lon_span / lon_step))
            print(f"🌍 Mode: Explicit Global (360° detected)")
        else:
            # Local region, start and end not connected -> need +1
            num_lon = int(round(lon_span / lon_step)) + 1
            print(f"🗺️  Mode: Regional Crop")

    # Total number of grids
    NUM_GRIDS = num_lat * num_lon
    
    print(f"Grid Calculation Info:")
    print(f"   Latitude Points : {num_lat}")
    print(f"   Longitude Points: {num_lon}")
    print(f"   Total Grids     : {num_lat} * {num_lon} = {NUM_GRIDS}")
    # -----------------------

    TIME_STEPS = 24

    RAW_COLUMNS = [
        'DDATETIME', 'LON_CENTER', 'LAT_CENTER',
        # --- Wind Basis ---
        'u10', 'v10',       # 10m zonal/meridional wind components: used to synthesize wind speed (WS) and determine wind direction
        # --- Temperature & Humidity Basis ---
        'd2m',              # 2m dewpoint temperature: key for calculating relative humidity (RH) and air density
        't2m',              # 2m temperature: directly corresponds to AWS observations, baseline for regression analysis (converted to °C)
        # --- Pressure Basis ---
        'msl',              # Mean sea level pressure: large-scale background pressure for eliminating altitude differences (converted to Bar)
        'sp',               # Surface pressure: pressure based on grid mean altitude, requires vertical correction with z (converted to Bar)
        # --- Extremes & Convection ---
        'i10fg',            # Instantaneous 10m wind gust: captures typhoon/severe convection extremes (long-tail distribution -> Log)
        'cape',             # Convective available potential energy: predicts thunderstorm/gale potential, physically-based criterion (long-tail distribution -> Log)
        # --- Clouds & Visibility ---
        'lcc',              # Low cloud cover: associated with high humidity and fog, key feature for visibility
        'tcc',              # Total cloud cover: affects radiative cooling and daytime warming amplitude (0-1)
        # --- Boundary Layer & Water Vapor ---
        'blh',              # Boundary layer height: determines vertical diffusion of pollutants, related to visibility
        'tcwv',             # Total column water vapor (kg/m2 or mm): typical range 1-70, represents water depth if fully condensed, absolute moisture condition for heavy rain
        # --- Surface Thermal & Radiation ---
        'skt',              # Skin temperature: responds quickly to radiation, used for monitoring urban heat island effect (converted to °C)
        'ssrd',             # Surface solar radiation downwards: direct driver of temperature rise (J/m2 -> W/m2 -> Log)
        # --- Precipitation ---
        'tp',               # Total precipitation: directly corresponds to AWS rain gauge (m -> mm -> Log)
        # --- Geo-Static Information ---
        'z',                # Geopotential: terrain height (z/9.8), used for vertical lapse rate correction of temperature/pressure
        'lsm'               # Land-sea mask: distinguishes coastal characteristics (land/ocean thermal differences)
    ]
        
    # Final feature list to output to .bin file
    # Model input channel order
    FINAL_FEATURES = [
        # Group 1: Wind
        'u10', 'v10',       # Wind vector components
        'ws',               # [Derived] Wind speed
        # Group 2: Temperature & Humidity
        't2m',              # Air temperature (Celsius)
        'd2m',              # Dewpoint (Celsius)
        'rh',               # [Derived] Relative humidity (0-1, calculated via Magnus formula)
        # Group 3: Pressure
        'sp',               # Surface pressure (Bar / 100kPa)
        'msl',              # Mean sea level pressure (Bar / 100kPa)
        # Group 4: Extremes & Potential
        'i10fg',            # Wind gust (Log)
        'cape',             # Convective potential (Log)
        # Group 5: Water & Light (Drivers)
        'tp',               # Precipitation (mm + Log)
        'ssrd',             # Radiation (W/m2 + Log)
        # Group 6: Clouds
        'lcc',              # Low cloud (0-1)
        'tcc',              # Total cloud (0-1)
        # Group 7: Environmental State
        'blh',              # Boundary layer height (km) - affects diffusion
        'tcwv',             # Total column water vapor
        'skt',              # Skin temperature (Celsius)
        # Group 8: Static Geography
        'z',                # Terrain height (km) - core for vertical correction
        'lsm'               # Land-sea mask
    ]

    LOG_TRANSFORM_COLS = ['ssrd', 'tp', 'cape', 'i10fg']

    converter = CSVToBinConverter(
        data_dir=INPUT_DIR,
        target_dir=OUTPUT_DIR,
        keep_columns=RAW_COLUMNS,
        final_features=FINAL_FEATURES,
        log_transform_columns=LOG_TRANSFORM_COLS,
        num_grids=NUM_GRIDS,
        time_steps_per_day=TIME_STEPS
    )

    converter.process_all_files(max_workers=1)