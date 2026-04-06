import cdsapi
import xarray as xr
import pandas as pd
import os
import datetime
import zipfile
import glob
import shutil
from collections import defaultdict
import traceback


class ERA5Downloader:
    def __init__(self, output_dir, variables, area_grid, chunk_days=5, area=None):
        """
        Initialize the downloader
        :param output_dir: Root directory for data storage
        :param variables: List of variables to download
        :param area_grid: Resolution [lat_step, lon_step]
        :param chunk_days: Number of days to pack per request
        """
        self.output_dir = output_dir
        self.variables = variables
        self.area = area
        self.grid = area_grid
        self.chunk_days = chunk_days
        
        # Initialize CDS API client
        self.client = cdsapi.Client()
        
        # Ensure output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")

    def _get_csv_path(self, date_obj):
        """Generate the final CSV file path"""
        date_str = date_obj.strftime("%Y%m%d")
        return os.path.join(self.output_dir, f"{date_str}.csv")

    def _check_missing_dates(self, start_date, end_date):
        """Check which dates have not yet been downloaded"""
        all_dates = []
        curr = start_date
        while curr <= end_date:
            all_dates.append(curr)
            curr += datetime.timedelta(days=1)
        
        missing = [d for d in all_dates if not os.path.exists(self._get_csv_path(d))]
        return missing

    def _download_cds_chunk(self, year, month, days, zip_filename):
        """Execute CDS API request"""
        request_dict = {
                "product_type": "reanalysis",
                "variable": self.variables,
                "year": str(year),
                "month": f"{month:02d}",
                "day": days,
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": self.area,
                "grid": self.grid,
                "format": "netcdf",
            }
        
        if self.area is None:
            request_dict.pop('area')

        self.client.retrieve(
            "reanalysis-era5-single-levels",
            request_dict,
            zip_filename
        )

    def _clean_dataframe(self, df):
        """Data cleaning logic: rename, drop unnecessary columns, handle expver"""
        # Unify time column name
        if 'valid_time' in df.columns:
            df.rename(columns={'valid_time': 'DDATETIME'}, inplace=True)
        elif 'time' in df.columns:
            df.rename(columns={'time': 'DDATETIME'}, inplace=True)
        else:
            raise KeyError("Time column (time or valid_time) not found in data")

        # Unify longitude/latitude column names
        if 'longitude' in df.columns:
            df.rename(columns={'longitude': 'LON_CENTER'}, inplace=True)
        if 'latitude' in df.columns:
            df.rename(columns={'latitude': 'LAT_CENTER'}, inplace=True)

        # Select columns to keep
        cols_to_keep = ['DDATETIME', 'LON_CENTER', 'LAT_CENTER'] 
        drop_list = ['number', 'step', 'surface', 'heightAboveGround', 'entireAtmosphere', 'expver']
        
        for col in df.columns:
            if col not in cols_to_keep and col not in drop_list:
                cols_to_keep.append(col)
        
        df = df[cols_to_keep]

        # Deduplicate (handle ERA5 expver mixed data issue)
        df = df.groupby(['DDATETIME', 'LON_CENTER', 'LAT_CENTER'], as_index=False).first()
        
        # Ensure time type
        df['DDATETIME'] = pd.to_datetime(df['DDATETIME'])
        
        return df

    def _save_daily_csv(self, df):
        """Split cleaned DataFrame by day and save as CSV"""
        df['date_key'] = df['DDATETIME'].dt.date
        grouped = df.groupby('date_key')

        for date_key, group_df in grouped:
            date_str = date_key.strftime("%Y%m%d")
            final_csv_path = os.path.join(self.output_dir, f"{date_str}.csv")
            
            # Format time and save
            save_df = group_df.drop(columns=['date_key']).copy()
            save_df['DDATETIME'] = save_df['DDATETIME'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            save_df.to_csv(final_csv_path, index=False)

    def process_chunk(self, start_date, end_date):
        """Core processing logic: process one time chunk (download -> clean -> save)"""
        # Check for missing dates
        missing_dates = self._check_missing_dates(start_date, end_date)
        if not missing_dates:
            print(f"{start_date} to {end_date} all exist, skipping.")
            return

        print(f"Processing time period: {start_date} to {end_date} ({len(missing_dates)} days need to be downloaded)...")

        # Prepare temporary folder
        chunk_id = start_date.strftime("%Y%m%d")
        extract_folder = f"temp_extract_chunk_{chunk_id}"
        if not os.path.exists(extract_folder):
            os.makedirs(extract_folder)

        # Build request mapping (Year, Month) -> [Days]
        requests_map = defaultdict(list)
        for d in missing_dates:
            requests_map[(d.year, d.month)].append(d.strftime("%d"))

        try:
            # Loop through downloads
            for (year, month), days in requests_map.items():
                print(f"Requesting CDS: {year}-{month:02d}, number of days: {len(days)}")
                zip_filename = f"temp_download_{chunk_id}_{year}{month:02d}.zip"
                
                self._download_cds_chunk(year, month, days, zip_filename)

                # Extract / move files
                if zipfile.is_zipfile(zip_filename):
                    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                        zip_ref.extractall(extract_folder)
                else:
                    # Handle case where API returns a netCDF file directly
                    shutil.move(zip_filename, os.path.join(extract_folder, f"data_{year}{month:02d}.nc"))
                
                if os.path.exists(zip_filename):
                    os.remove(zip_filename)

            # Merge netCDF files
            nc_files = glob.glob(os.path.join(extract_folder, "*.nc"))
            if not nc_files:
                raise FileNotFoundError("Download successful but no .nc files found in extraction directory")

            print(f"Parsing {len(nc_files)} netCDF files...")
            
            # Read using xarray
            with xr.open_mfdataset(nc_files, engine="netcdf4", combine='by_coords', compat='override') as ds:
                df_temp = ds.to_dataframe().reset_index()

            # Clean data
            df_cleaned = self._clean_dataframe(df_temp)

            # Save results
            print(f"Splitting and saving CSV...")
            self._save_daily_csv(df_cleaned)
            
            print(f"Time period {start_date} to {end_date} completed.")

        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
        finally:
            # Clean up temporary directory
            if os.path.exists(extract_folder):
                try:
                    shutil.rmtree(extract_folder)
                except OSError:
                    pass

    def run(self, start_date_str, end_date_str):
        """Main entry point: iterate over the entire date range"""
        start_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.datetime.strptime(end_date_str, "%Y-%m-%d").date()
        
        today = datetime.date.today()
        current_date = start_date

        while current_date <= end_date:
            if current_date > today:
                print(f"Date {current_date} is in the future, stopping processing.")
                break
            
            # Calculate current chunk end date
            chunk_end = current_date + datetime.timedelta(days=self.chunk_days - 1)
            
            # Boundary correction
            if chunk_end > end_date:
                chunk_end = end_date
            if chunk_end > today:
                chunk_end = today

            self.process_chunk(current_date, chunk_end)
            
            current_date = chunk_end + datetime.timedelta(days=1)
        
        print(f"\nAll tasks completed! Data saved to: {self.output_dir}")


# ================= Configuration and Execution Section =================
if __name__ == "__main__":
    # Print process ID for monitoring
    print(f"Current process ID: {os.getpid()}")

    # Define variable list
    ERA5_VARIABLES = [
        "10m_u_component_of_wind", "10m_v_component_of_wind", "2m_dewpoint_temperature",
        "2m_temperature", "mean_sea_level_pressure", "surface_pressure",
        "total_precipitation", "instantaneous_10m_wind_gust", "surface_solar_radiation_downwards",
        "low_cloud_cover", "total_cloud_cover", "boundary_layer_height",
        "convective_available_potential_energy", "geopotential", "land_sea_mask",
        "total_column_water_vapour", "skin_temperature"
    ]

    # Instantiate downloader
    # Global
    downloader = ERA5Downloader(
        output_dir="./storage/era5/era5_daily_data_global",  # Storage path
        variables=ERA5_VARIABLES,                            # Variable list
        area_grid=[5, 5],                                    # Resolution
        chunk_days=15                                        # Number of days per request
    )

    # CN
    # downloader = ERA5Downloader(
    #     output_dir="./storage/era5/era5_daily_data_cn",     # Storage path
    #     variables=ERA5_VARIABLES,                           # Variable list
    #     area_grid=[1, 1],                                   # Resolution
    #     area=[54, 73, 3, 135],                              # China region
    #     chunk_days=15                                       # Number of days per request
    # )

    # Run download task (modify the time range here)
    downloader.run(
        start_date_str="2015-01-01",
        end_date_str="2019-12-30"
    )