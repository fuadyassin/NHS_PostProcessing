"""
## Description
Script to fetch streamflow forecasts at gauge stations from the National 
Surface and River Prediction System using GeoMet's Web Coverage Service API.

Script params can be modified below the line `if __name__ == '__main__':`

Before running this script for the first time, make sure your credentials are
specified in `config.cfg`. The `config.cfg` and `nsrps_model_stn_locs.json`
should live in the same directory as this script, or the paths to these files
should be adjusted below the `if __name__ == '__main__':` conditional block.


## Dependencies
- pandas
- xarray (w/ netCDF4 engine)
- OWSLib >= 0.31.0 (more info: https://github.com/geopython/OWSLib)


## Anticipated Changes
Currently, streamflow forecasts are available from the Deterministic Hydrologic
Prediction System (DHPS). The forecasts from DHPS are the single, "best guess"
of streamflow for the coming six days. Soon, forecasts will also be available
from the corresponding Ensemble Hydrologic Prediction System (EHPS). EHPS 
produces a collection of probable streamflow forecasts out 16 days (32 days 
on Thursdays).

To query for streamflow forecasts at gauge stations, we must know the lat / lon
coordinates of the gauge stations in the "model world". For now, we can find
the lat / lon model world coordinates from the file, 
`nsrps_model_stn_locs.json`. Soon, the model world coordinates will also be
available on GeoMet. This script will require light modification to 
accommodate this change (see the function `stns_on_grid`).


## Known Limitations
NSRPS assimilates observations from WSC and partner gauge stations. At the risk
of grossly oversimplifying the assimilation process, we can think of the 
assimilated observations as being used "nudge" or "course correct" the 
forecasts. We have "model world" coordinates for 603 WSC stations whose obs are
assimilated by the model. In theory, we can query for forecasts for gauge 
stations whose data is not assimilated (i.e. they don't have model world 
coordinates), but this is more complicated. **Note from Natalie: I have scripts
to do this, but I've ommited the code here for simplicity and because I need to
verify the results - happy to provide this code further into the pilot, if it's
needed.

Program runtime is a significant issue when querying for forecasts for more 
than a few dozen stations. I've tried several strategies for speeding things up
so I'm detailing them here for future reference. Before that, some notes about
querying the Web Coverage Service:
    - A query must be made for each forecast timestep. For a 6-day DHPS 
    forecast, there are timesteps at an hourly interval, so a single forecast 
    has 144 timesteps in total.
    - Queries to the WCS can be made for the whole grid, or for a subset of the
    grid.
    - In general, there are two strategies for querying the WCS for forecasts
    at gauge stations:
        1. Query the whole grid for each forecast timestep and then extract
        streamflow at gauge stations after.
        2. Query a small subset of the grid containing one station. Do this for
        each station for each forecast timestep.
        3. Query for larger subsets of the grid and extract streamflow at 
        several gauge stations in that subset after. Do this until forecasts
        for all stations have been extracted.
    - Unfortunately, methods 1 and 2 described above have approximately the 
    same runtime. Method 2 might be slightly faster (90 mins to 2 hours to 
    query for all 603 gauge stations). I did not try method 3.
     
Things I tried for method 2:

A. Adding threading
    This works well when a thread pool is used to asynchronously query for 
    forecast timesteps. Forecasts for each stations are queried in serial. 
    **This is the method I use in this script.
B. Adding multiprocessing with the threading from method `A`
    I tried adding a process pool to asynchronously submit queries for
    stations, rather than querying stations in serial. This was definitely 
    faster than method `A`, but the script would always fail - it seems that 
    this method causes an exceedance of the max allowable number of queries to 
    the GeoMet servers. Maybe sleeping some of the process workers might 
    help...we could probably do a little more investigating here...
C. Using asyncio and aiohttp with a GET request, rather than the OWSLib API
    This was much slower than method `A`, so I didn't pursue it further, but 
    that could also be my ignorance - I have limited experience with asyncio.


## Contact
natalie.gervasi@ec.gc.ca


## Last Modified
September 2024 
By Uchechukwu UDENZE
Uchechukwu.Udenze@ec.gc.ca

"""

# for type casting
from __future__ import annotations
from typing import Union, Dict

import re
import time
import json
import logging
import configparser

import numpy as np
import xarray as xr
import pandas as pd

from functools import partial
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta
from pathlib import Path

from collections import defaultdict
from natsort import natsorted


# for multithreading
from concurrent.futures import ThreadPoolExecutor

# for querying GeoMet for observations
from GeneralProcessing.gen_streamflow_file import GenStreamflowFile
from postprocessinglib.evaluation import data




# web services 
from owslib.wms import WebMapService
from owslib.wcs import WebCoverageService
from owslib.wcs import Authentication

# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_auth(
        config_path: str
    ) -> Union[str, str]:
    """Reads username and password from configuration file.

    Parameters
    ----------
    config_path : str
        Path to configuration file.

    Returns
    -------
    Union[str, str]
        Username and password.
    """
    # load login information to connect to hidden GeoMet layers
    config = configparser.ConfigParser()
    config.read_file(open(config_path)) 

    username = config['Login']['Username']
    password = config['Login']['Password']
    
    return username, password


def fcst_times_from_wms(
        layer_name: str, 
        username: str=None,
        password: str=None,
    ) -> Union[str, list[str]]:
    """Retrieves the forecast time metadata from the WebMapService for the 
    GeoMet layer (product) of interest.

    Parameters
    ----------
    layer_name : str
        Name of the GeoMet layer.
    username : str, optional
        Username to access secured layer from WMS API, by default None.
    password : str, optional
        Password to access secured layer from WMS API, by default None.

    Returns
    -------
    Union[str, list[str]]
        The issuse / publication timestamp for the latest forecast.
        A list of timestamps corresponding to the latest forecast's timesteps.
    """
    # create WMS object
    wms = WebMapService(
        f'https://geo.weather.gc.ca/geomet?&SERVICE=WMS&LAYERS={layer_name}',
        version='1.3.0',
        auth=Authentication(username=username, password=password),
        timeout=300
    )

    layer_metadata = wms[layer_name]
    name = layer_name.split('_')

    if name[0] == "DHPS" or name[0] == "DHPS-Analysis" or name[0] == "EHPS-Analysis":
        # oldest_fcast = the oldest forecast available from GeoMet, typically b/w 2 and 3 weeks old
        # latest_fcast = the most recently created forecast
        # issue_interval = how frequently the forecasts are published
        oldest_fcast, latest_fcast, issue_interval = layer_metadata.dimensions['reference_time']['values'][0].split('/')

    elif name[0] == "EHPS":
        # oldest_fcast = the oldest forecast available from GeoMet, typically the last couple hours
        # latest_fcast = the most recently created forecast
        # oldest_fcast= wms[layer_name].dimensions['reference_time']['values'][0]
        latest_fcast= layer_metadata.dimensions['reference_time']['values'][-1]

    # the following variables correspond to the latest_fcast issue
    # first_time = the first forecast datetime
    # last_time = the last datetime in the forecast i.e. the extent of the forecast horizon
    # time_interval = the temporal resolution of the forecast
    first_time, last_time, time_interval = layer_metadata.dimensions['time']['values'][0].split('/')

    iso_format = "%Y-%m-%dT%H:%M:%SZ"

    # convert date strings to datetime objects
    first = datetime.strptime(first_time, iso_format)
    last = datetime.strptime(last_time, iso_format)

    # remove anything that isn't a number from the datetime interval (time between forecasts)
    intvl = int(re.sub(r'\D', '', time_interval))

    # create a list of forecast datetimes
    hrs = [first]
    while first < last:
        first = first + timedelta(hours=intvl)
        hrs.append(first)

    # create a list of ISO formatted forecast datetime strings
    fcasthrs = [datetime.strftime(hr, iso_format) for hr in hrs]

    return oldest_fcast, latest_fcast, fcasthrs

def connect_to_wcs(
        layer_name: str,
        username: str=None,
        password: str=None,
    ) -> WebCoverageService:
    """Establishes a connection to GeoMet's WCS API.

    Parameters
    ----------
    layer_name : str
        Name of the GeoMet layer.
    username : str, optional
        Username to access secured layer from WCS API, by default None.
    password : str, optional
        Password to access secured layer from WCS API, by default None.

    Returns
    -------
    WebCoverageService
        A connection object for the WCS API.
    """
    # create WCS object
    wcs = WebCoverageService(
        f'https://geo.weather.gc.ca/geomet?&SERVICE=WCS&COVERAGEID={layer_name}', 
        auth=Authentication(
            username=username,  
            password=password
        ),
        version='2.0.1',
        timeout=300
    )
    return wcs


def predictions_from_wcs(
        wcs: WebCoverageService, 
        layer: str,
        ref_time: str, 
        lat: float, 
        lon: float, 
        width: int,
        time: str,
    ) -> xr.Dataset:
    """Queries the WMS for a subset of the model grid for a single timestep 
    from the forecast of interest. For example, we could query for the a grid
    subset with a bounding box of [ lat: (50, 52), lon: (-118, 116) ] at 
    timestep 2023-04-15T03:00:00 for the forecast published (issued) at 
    2023-04-15T00:00:00.

    Parameters
    ----------
    wcs : WebCoverageService
        A connection object for the WCS API.
    layer_name : str
        Name of the GeoMet layer.
    ref_time : str
        The issue / publication time for the forecast of interest.
    lat : float
        The latitude of the target gauge station.
    lon : float
        The longitude of the target gauge station. Uses [-180, 180] convention.
    width : int
        The half-width of the grid subset that will be queried.
    time : str
        The foreacst timestep.

    Returns
    -------
    xr.Dataset
        A subset of the model grid at a given forecast timestep for the 
        forecast of interest.
    """
    # make the WCS request
    response = wcs.getCoverage(
        identifier=[layer], 
        format='image/netcdf', 
        subsettingcrs='EPSG:4326', 
        subsets=[('lat', lat-width, lat+width), ('lon', lon-width, lon+width)],
        DIM_REFERENCE_TIME=ref_time, # capitalization here is important
        TIME=time # capitalization here is important
    )    
    # read into an xarray
    ds = xr.open_dataset(response.read()).load()
    # add time metadata as a new dimension and coordinate
    ds = ds.expand_dims(
        time=[datetime.strptime(time, "%Y-%m-%dT%H:%M:%SZ")]
    )
    return ds


def stns_on_grid(
        stn_locs_file: str=None,
        out_format: str='dataframe',
    ) -> Union[pd.DataFrame, list[str]]:
    """Retrieves the "model world" gauge station locations from file.
    TO-DO: when the station locations are available from GeoMet, this function
    should be modified to query the locations from the OGC Features API.

    Parameters
    ----------
    stn_locs_file : str, optional
        Path to geoJSON file containing station locations, by default None.

    Returns
    -------
    pd.DataFrame
        Tabulated gauge station coordinates. DataFrame has the format:

        ---------------------------------
        | id      |   lat   |    lon    |
        ---------------------------------
        | 05AC012 | 50.1714 | -112.7203 |
        | ...     |   ...   |    ...    |
        ---------------------------------
        
        ** DataFrame index = `id`

    list[str]
        A list of station IDs corresponding to the rows in the DataFrame.

    Raises
    ------
    RuntimeError
        While the station locations are not available from GeoMet, an error is
        raised if the user does not provide a geoJSON file with the station
        locations.
    """
    # once available, query the OGC Features API for "model world" station locations
    if stn_locs_file is None:
        raise RuntimeError(
            "Model station locations not available from GeoMet; specify `stn_locs_file`"
        )
    # temporary solution to grab "model world" station lat/lons from file
    else:
        with open(stn_locs_file, 'r') as file:
            stns = json.loads(
                file.read()
            )
        
        rows = []
        for feature in stns['features']:
            id = feature['properties']['STATION_NUMBER']
            coords = feature['geometry']['coordinates']
            # Station ID | NSRPS Lat | NSRPS Lon
            rows.append((id, coords[1], coords[0]))
        
        df = pd.DataFrame(rows, columns=['id', 'lat', 'lon'])
        df = df.set_index('id')

    # print("Stations on grid:\n", df)
    if out_format == 'list':
        # return a list of station IDs
        return df.index.tolist()
    return df


def find_stn_target_gridcell(
        ds: xr.Dataset, 
        nsrps_lat: float, 
        nsrps_lon: float,
    ) -> Union[float, float]:
    """Returns the lat / lon index labels for the grid cell containing the 
    "model world" gauge station.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing a subset of the model grid.
    nsrps_lat : float
        Latitude for the gauge station on the model grid.
    nsrps_lon : float
        Longitude for the gauge station on the model grid.

    Returns
    -------
    Union[float, float]
        Lat / lon index labels for the target grid cell.
    """
    
    # find grid lat
    closest_lat_idx = (np.abs(ds.lat.data - nsrps_lat)).argmin()
    lat = ds.lat.data[closest_lat_idx]
    
    # find grid lon
    closest_lon_idx = (np.abs(ds.lon.data - nsrps_lon)).argmin()
    lon = ds.lon.data[closest_lon_idx]
    
    # print("Closest lat and lon values", lat, lon)
    return lat, lon


def generate_nsrps(
        auth_path: str,
        stn_list: list[str],
        layer_name: str,
        stn_locs_file: str=None,
    ) -> pd.DataFrame:
    """Queries for streamflow forecasts from the WCS for a given list of gauge
    stations.

    Parameters
    ----------
    auth_path : str
        Path to configuration file containing authentication information.
    stn_list : list[str]
        The list of stations where we want to query for forecasts.
    layer_name : str
        Name of the GeoMet layer.
    stn_locs_file : str, optional
        Path to geoJSON file containing station locations, by default None.

    Returns
    -------
    pd.DataFrame
        Tabulated forecasts at gauge stations. DataFrame has the format:

        ----------------------------------------------------------------
        | time                |   00XX001   |   00XX002   |   ......   |
        ----------------------------------------------------------------
        | YYYY-MM-DD HH:mm:ss |     xxxx    |     xxxx    |    ...     |
        | ...                 |     ....    |     ....    |    ...     |
        ----------------------------------------------------------------
        
        ** DataFrame index = `time` (in UTC)
    """

    # read authentication info from config file
    username, password = get_auth(auth_path)

    # find the "model world" gauge station locations
    stns = stns_on_grid(stn_locs_file) 

    # fetch time steps from the latest forecast from GeoMet's Web Map Service
    oldest_fcast, latest_fcast, fcst_dtimes = fcst_times_from_wms(
        layer_name=layer_name, 
        username=username, 
        password=password
    )

    # connect to the Web Coverage Service
    wcs = connect_to_wcs(
        layer_name=layer_name, 
        username=username, 
        password=password
    )

    # create empty dataframe to store forecast data for stations
    data = pd.DataFrame([])

    # dictionary to store the closest lat / lon coordinates for each station
    # these are the coordinates of the grid cell that is closest to the station's
    closest_coords = {}

    for i, stn in enumerate(stn_list):

        # "model world" coordinates extracted with nsrps_stns_to_geojson.py
        lat_nsrps = stns.loc[stn]['lat']
        lon_nsrps = stns.loc[stn]['lon']

        fcsts_from_wcs = partial(
            predictions_from_wcs, 
            wcs, 
            layer_name, 
            latest_fcast, 
            lat_nsrps, 
            lon_nsrps, 
            1 # half-width in degrees of the grid subset we'll query
        )

        fcst_arrays = []
        # a query needs to be made for each forecast timestep and each stn
        # DHPS 6-day fcast has 144 timesteps
        # querying in this way is very I/O bound, so we use threading to speed up the job
        with ThreadPoolExecutor() as executor:
            # query the WCS for each forecast timestep
            iterator = executor.map(
                fcsts_from_wcs, 
                fcst_dtimes
            )
            # grab the results from the object returned by the executor
            for result in iterator:
                fcst_arrays.append(result)
        
        # concatenate all forecast timesteps into one xarray dataset
        ds = xr.concat([fcst for fcst in fcst_arrays], dim='time')

        # find the lat / lot corresponding to the target grid cell's indices
        lat, lon = find_stn_target_gridcell(ds, lat_nsrps, lon_nsrps)

        # Added here just to test the actual vs closest lat / lon values
        closest_coords[stn] = (lat, lon)

        # extract the streamflow at the station of interest
        # note: using NSRPS lat / lon to `select` streamflow from target cell
        # is not always reliable b/c NSRPS lat / lons are not always co-located
        # with the target cell's lat / lon label indexers
        strflw = ds.sel(
            lat=lat, 
            lon=lon, 
            method='nearest'
        )
        # print(strflw)
        # print("\n")

        # store streamflow in dataframe
        df = strflw.to_dataframe()
        # print(df)
        # print("\n")
        
        # dropping unneeded information
        df = df.drop(['crs', 'lat', 'lon'], axis=1) 
        df = df.rename(columns={'RiverDischarge': f"QOSIM_{stn}"})

        # append the forecasts for this station to the big DataFrame
        data = pd.concat([data, df], axis=1)
        # print(data)
        # print("\n")

        logger.info(
            f"Finished forecast queries for station: {stn}; iteration: {i}"
        )
    # return data, closest_coords, stns
    return data


def _set_multiindex_columns(df):
    """
    Change the column headers to Multi-indexed headers to match the rest of the library's Dataframes.
    """
    station_counts = defaultdict(int)
    multi_cols = []

    for col in df.columns:
        station = col.split('_')[-1]
        count = station_counts[station]

        if count == 0:
            label = 'QOMEAS'
        elif count == 1 and col.startswith('QOSIM'):
            label = 'QOSIM'
        else:
            label = f'QOSIM{count}'

        station_counts[station] += 1
        multi_cols.append((station, label))

    # Apply MultiIndex to the DataFrame columns
    df.columns = pd.MultiIndex.from_tuples(multi_cols)
    return df

def _compute_percent_error(DATA: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute percent error between simulated and observed values for all stations and CSVs.
    """
    error_data = defaultdict(dict)

    # Create a dictionary to hold the mean observed values at each index for each station
    station_mean_obs = defaultdict(lambda: defaultdict(list))

    # Loop through each CSV and gather the QOMEAS values for each station at each index (0, 12, 24, ..., 144)
    for key, df in DATA.items():
        stations = df.columns.get_level_values(0).unique()
        
        for station in stations:
            obs_col = (station, "QOMEAS")  # After renaming, 'QOMEAS' columns are just the station name
            if obs_col in df.columns:
                obs = df[obs_col]
                # For each station, store the 'QOMEAS' value for each index in all CSVs
                for idx in range(0, obs.index[-1]+1, 12):  # 0, 12, 24, 36, ..., 144
                    station_mean_obs[station][idx].append(obs.loc[idx])

    # Now calculate the mean for each station at each index (0, 12, 24, ..., 144)
    mean_station_obs = {station: {idx: np.nanmean(values) for idx, values in idx_vals.items()}
                        for station, idx_vals in station_mean_obs.items()}

    # Loop through each CSV and compute the percent error for each station at every index
    for key, df in DATA.items():
        stations = df.columns.get_level_values(0).unique()
        
        for station in stations:
            obs_col = (station, "QOMEAS")  # Again, 'QOMEAS' columns are just the station name
            sim_col = (station, "QOSIM")  # 'QOSIM' columns have the '_sim' suffix
            
            if obs_col in df.columns and sim_col in df.columns:
                obs = df[obs_col]
                sim = df[sim_col]
                
                # Loop over every index to calculate percent error
                percent_errors = []
                # print(obs.index[-1])
                for idx in range(0, obs.index[-1]+1, 12):  # 0, 12, 24, 36, ..., 144
                    # Get the mean of the observed values at this index across all CSVs for this station
                    mean_obs = mean_station_obs[station][idx]
                    
                    # Calculate percent error for this index using the mean observed value for this index
                    percent_error = ((sim.loc[idx] - obs.loc[idx]) / mean_obs) * 100
                    
                    # Append the result
                    percent_errors.append(percent_error)
                
                # Store the percent error for this station and CSV file
                error_data[station][key] = pd.Series(percent_errors, index=[i * 12 for i in range(len(percent_errors))])

    # Build DataFrame from the nested dict containing percent error
    final_error_df = pd.DataFrame({
        (station, key): series
        for station, csv_dict in error_data.items()
        for key, series in csv_dict.items()
        if key in natsorted(csv_dict.keys())  # <-- This line filters the first 16 keys
    })

    final_error_df.columns = pd.MultiIndex.from_tuples(final_error_df.columns)
    
    return final_error_df


def _aggregate_statistics(final_error_df: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """
    Aggregate percent error DataFrame using the specified statistical methods.
    Assumes a 'data.stat_aggregate' method exists; replace with your implementation.
    """
    combined_cols = []
    combined_data = []

    # Assuming 'data.stat_aggregate' is your function that accepts df and method
    for method in methods:
        temp_df = data.stat_aggregate(df=final_error_df, method=method)  # Replace with your actual function
        for station in temp_df.columns.get_level_values(0).unique():
            temp_df_cols = temp_df[station]
            if isinstance(temp_df_cols, pd.Series):
                temp_df_cols = temp_df_cols.to_frame()
            for col in temp_df_cols.columns:
                combined_cols.append((station, col))
            combined_data.append(temp_df_cols)

    final_df = pd.concat(combined_data, axis=1)
    final_df.columns = pd.MultiIndex.from_tuples(combined_cols)
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]

    sorted_stations = natsorted(final_df.T.index.get_level_values(0).unique())
    new_blocks = [
        final_df.T[final_df.T.index.get_level_values(0) == station].T
        for station in sorted_stations
    ]
    final_df = pd.concat(new_blocks, axis=1)
    final_df.columns = pd.MultiIndex.from_tuples(final_df.columns)

    return final_df


def prepare_historical_dataframes(short_term_df, long_term_df, upper_bounds_dfs, lower_bounds_dfs):
    """
    Prepare climatology DataFrames expanded to match timestamps of short_term_df,
    with separate lists of upper and lower bound DataFrames.

    Args:
      short_term_df: DataFrame with datetime index.
      long_term_df: DataFrame indexed by jday (1-366) for median or central tendency.
      upper_bounds_dfs: List of DataFrames indexed by jday (1-366) for upper bounds (e.g. q95, max, q75).
      lower_bounds_dfs: List of DataFrames indexed by jday (1-366) for lower bounds (e.g. q5, min, q25).

    Returns:
      expanded_climatology: DataFrame with same index as short_term_df (median/centerline)
      expanded_upper_bounds: List of DataFrames with same index as short_term_df (upper bounds)
      expanded_lower_bounds: List of DataFrames with same index as short_term_df (lower bounds)
    """
    
    # Extract day of year (jday) for each timestamp in short_term_df
    jdays = short_term_df.index.dayofyear

    # Map median climatology values from jday to timestamps
    expanded_climatology = long_term_df.reindex(jdays).set_index(short_term_df.index)

    # Map each upper bound climatology similarly
    expanded_upper_bounds = []
    for upper_df in upper_bounds_dfs:
        expanded_upper = upper_df.reindex(jdays).set_index(short_term_df.index)
        expanded_upper_bounds.append(expanded_upper)

    # Map each lower bound climatology similarly
    expanded_lower_bounds = []
    for lower_df in lower_bounds_dfs:
        expanded_lower = lower_df.reindex(jdays).set_index(short_term_df.index)
        expanded_lower_bounds.append(expanded_lower)

    return expanded_climatology, expanded_upper_bounds, expanded_lower_bounds



def forecast_data_extraction(obs_format: str="realtime",
                             obs_file: str=None,
                             historical:bool = True,
                             historical_start_date:str="1980-01-01",
                             historical_end_date:str="2024-12-12",
                             historical_stat_lines: Union[str, list[str]]=['median'],                             
                             historical_stat_ubounds: Union[str, list[str]]=['q75'],
                             historical_stat_lbounds: Union[str, list[str]]=['q25'],
                             stn_list: list[str]=None,
                             stn_locs_file: str=None,
                             prediction_files: Union[str, pd.DataFrame, list[str], list[pd.DataFrame]] = None,
                             start_date:str=None, end_date:str=None):
    
    
    # STATIONS
    # Ensure a list of stations is provided else a JSON file with the stations
    if stn_list is None and stn_locs_file is None:
        raise ValueError("Either `stn_list` or `stn_locs_file` must be provided.")
    elif stn_list is None and stn_locs_file is not None:
        # Generate and extract the stations list from the JSON file
        stn_list = stns_on_grid(stn_locs_file, out_format='list')

    # If the user provided a list of stations, ensure that the list is not empty
    if stn_list is not None and len(stn_list) == 0:
        raise ValueError("The provided `stn_list` is empty. Please provide valid station IDs.")

    # OBSERVED DATA
    # Generate the observed streamflow data
    # REAL TIME DATA
    if obs_format == "realtime":
        # generate the real time observed dataframe
        gen_flow = GenStreamflowFile()
        end_dt = datetime.now(timezone.utc).replace(microsecond=0) # Get the time today, right now as the end date.
        start_dt = end_dt - relativedelta(months=1) # set the start date to a be a month into the past i.e., past 1 month data
        start = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        end   = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        df_rt, meta_rt = gen_flow.fetch_hydrometric_realtime_full_range(
            station_numbers=stn_list, 
            start=start, end=end,
            window_days=1, freq_hours=12
        )
        # set the columns to be multi-indexed to match the rest of the library's DataFrames
        if df_rt.empty:
            logger.warning("No real-time data found for the specified stations.")
            return pd.DataFrame()
        df_rt = _set_multiindex_columns(df_rt)

    # ARCHIVED DATA
    # elif obs_format == "csv":
    #     if obs_file is None:
    #         raise ValueError("obs_file must be provided when obs_format is 'csv'")
    #     # generate the observed dataframe from the CSV file

    else:
        raise ValueError("obs_format must be either 'realtime' or 'csv'")

    # HISTORICAL DATA
    if historical:
        lt = {}
        # ensure the historical statistics lines and bounds are provided and formatted correctly
        if historical_stat_lines is None and historical_stat_ubounds is None or historical_stat_lbounds is None:
            raise ValueError(
                "Both `historical_stat_lines` and `historical_stat_bounds` must be provided."
            )
        if isinstance(historical_stat_lines, str):
            historical_stat_lines = [historical_stat_lines]
        if isinstance(historical_stat_ubounds, str):
            historical_stat_ubounds = [historical_stat_ubounds]
        if isinstance(historical_stat_lbounds, str):
            historical_stat_ubounds = [historical_stat_ubounds]
        # generate the historical observed dataframe
        df_hst, meta_hst = gen_flow.fetch_hydrometric_data_ca(stn_list, historical_start_date, historical_end_date)
        # Make sure each value of the dataframe is in fact numeric
        for i in stn_list:
            df_hst[i] = pd.to_numeric(df_hst[i], errors='coerce')
        for stat in historical_stat_lines + historical_stat_ubounds + historical_stat_lbounds:
            lt[f"lt_{stat}"] = data.long_term_seasonal(df=df_hst, method = stat)

    
    # PREDICTIONS DATA
    if prediction_files is None:
        prediction_files = []
    else:
        if isinstance(prediction_files, (str, pd.DataFrame)):
            prediction_files = [prediction_files]
        if not isinstance(prediction_files, list):
            raise ValueError("`prediction_files` must be a list of DataFrames or file paths.")

        for i, item in enumerate(prediction_files):
            # Load DataFrame
            if isinstance(item, str):
                df = pd.read_csv(item)
            elif isinstance(item, pd.DataFrame):
                df = item.copy()
            else:
                raise ValueError("Each item must be a file path (str) or a DataFrame.")

            # Strip leading/trailing whitespace from column names
            df.columns = df.columns.str.strip()
            # Normalize time column names for detection, without altering the original DataFrame
            time_cols = {col.lower() for col in df.columns}
            mesh_cols = {'year', 'jday', 'hour', 'mins'}

            if mesh_cols.issubset(time_cols):
                # MESH detected
                # Drop unnamed columns (likely empty)
                df = df.loc[:, ~df.columns.str.contains('Unnamed')]
                # Standardize actual column names (map lowercase to actual column names)
                col_map = {col.lower(): col for col in df.columns}

                # Access real column names safely using lowercase keys
                df['datetime_utc'] = (
                    pd.to_datetime(df[col_map['year']], format='%Y') +
                    pd.to_timedelta(df[col_map['jday']] - 1, unit='D') +
                    pd.to_timedelta(df[col_map['hour']], unit='h') +
                    pd.to_timedelta(df[col_map['mins']], unit='m')
                )
                
                df = df.set_index('datetime_utc')
                df = df.drop(columns=[col_map['year'], col_map['jday'], col_map['hour'], col_map['mins']])

                # localize the UTC index to align with the NSRPS data for the GDPS
                if not df.index.tz:
                    df.index = df.index.tz_localize('UTC')
                # resample to 12 hourly frequency to match Observed data
                df = df.resample('12h').mean()

                # Drop QOMEAS_ columns
                df = df.drop(columns=df.filter(regex='QOMEAS_').columns)

                # Rename columns to multiindex
                new_cols = df.columns.str.extract(r'(QOSIM)_(\w+)$')
                if new_cols.isnull().any().any():
                    raise ValueError("Unexpected column format in MESH data")
                df.columns = pd.MultiIndex.from_arrays([new_cols[1], [f"QOSIM{i+1}"] * len(df.columns)])

                # Align the MESH values to the observed data
                for station in stn_list:
                    meas_col = (station, 'QOMEAS')
                    sim_col = (station, f'QOSIM{i+1}')
                    if meas_col not in df_rt.columns:
                        continue
                    if sim_col not in df.columns:
                        continue
                    meas_series = df_rt[meas_col]                    
                    sim_series = df[sim_col]
                    sim_first_valid_time = sim_series.first_valid_index()

                    # check that there is a corresponding non-null value in the observed data
                    if sim_first_valid_time in meas_series.index and not pd.isnull(meas_series[sim_first_valid_time]):
                        # Align the MESH value to the observed value
                        meas_value = meas_series.loc[sim_first_valid_time]
                        sim_value = sim_series.loc[sim_first_valid_time]
                        delta = meas_value - sim_value
                        corrected_sim = sim_series + delta

                        #Apply correction to the DataFrame if no value is Negative
                        if (corrected_sim < 0).any():
                            print(f"Skipping correction for {sim_col} due to negative values.")
                        else:
                            df.loc[:, sim_col] = corrected_sim
                    else:
                        print(f"No corresponding observed value for {sim_col} at {sim_first_valid_time}. Skipping correction.")
                
                # Store the DataFrame in the prediction_files list
                prediction_files[i] = df

            else:
                # NSRPS: ensure datetime index
                if not isinstance(df.index, pd.DatetimeIndex):
                    # Try to find columns with datetime dtype (e.g., datetime64[ns])
                    datetime_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
                    
                    # If no datetime dtype columns found,
                    # but there is a column literally named 'datetime' (possibly object/string type),
                    # add it to the list to try using it as datetime index anyway
                    if not datetime_cols:
                        for fallback_col in ['datetime', 'time']:
                            if fallback_col in df.columns:
                                datetime_cols = [fallback_col]
                                break
                    
                    # If we found any datetime candidate columns,
                    # set the first one as the DataFrame index
                    # (usually there should be only one datetime column)
                    if datetime_cols:
                        df = df.set_index(datetime_cols[0])
                    else:
                        # If no datetime index or datetime column found, raise an error
                        raise ValueError("NSRPS DataFrame must have a datetime index or a datetime column")
                
                    # Ensure it is in fact a datetime index
                    # If index is not a DatetimeIndex, try to convert it
                    if not isinstance(df.index, pd.DatetimeIndex):
                        try:
                            df.index = pd.to_datetime(df.index)
                        except Exception:
                            raise ValueError("Could not convert index to datetime. Ensure your CSV has datetime-formatted index.")

                
                # Ensure the index is in UTC
                if not df.index.tz:
                    df.index = df.index.tz_localize('UTC')
                # Resample to 12 hourly frequency to match Observed data
                df = df.resample('12h').mean()

                # Rename columns to multiindex
                new_cols = df.columns.str.extract(r'(QOSIM)_(\w+)$')
                if new_cols.isnull().any().any():
                    raise ValueError("Unexpected column format in NSRPS data")
                df.columns = pd.MultiIndex.from_arrays([new_cols[1], [f"QOSIM{i+1}"] * len(df.columns)])

                prediction_files[i] = df
            
            # Now filter & reorder columns by station_list (common to both)
            df = prediction_files[i]
            filtered_cols = [col for station in stn_list for col in df.columns if col[0] == station]
            prediction_files[i] = df.loc[:, filtered_cols]


    # MERGE DATA - OBSERVED and PREDICTIONS
    merged_cols = []
    for station in df_rt.columns.get_level_values(0).unique():
        obs_col = (station, 'QOMEAS')
        if obs_col in df_rt.columns:
            merged_cols.append(df_rt[obs_col])
        for i in range(1, len(prediction_files) + 1):
            pred_col = (station, f'QOSIM{i}')
            if pred_col in prediction_files[i - 1].columns:
                merged_cols.append(prediction_files[i - 1][pred_col])
    if not merged_cols:
        raise ValueError("No matching stations found between observed and prediction data.")
    merged = pd.concat(merged_cols, axis=1)

    # Filter by start_date and end_date if provided
    if start_date:
        merged = merged[merged.index >= pd.to_datetime(start_date).tz_localize('UTC')]
    if end_date:
        merged = merged[merged.index <= pd.to_datetime(end_date).tz_localize('UTC')]


    if historical:
        return merged, lt
    else:
        return merged


def forecast_error_calculation(files: Union[str, pd.DataFrame, list[str], list[pd.DataFrame]],
                               observed: Union[str, pd.DataFrame] = 'realtime',
                               stats: Union[str, list[str]] = ['q10', 'q25', 'q50', 'q75', 'q90' ]
) -> pd.DataFrame:
    """
    Calculate forecast errors for a list of files or DataFrames.
    """

    # Ensure files was passed as a list of DataFrames or file paths
    if files is None:
        raise ValueError("`files` must be provided as a list of DataFrames or file paths.")
    if isinstance(files, (str, pd.DataFrame)):
        files = [files]
    if not isinstance(files, list):
        raise ValueError("`files` must be a list of DataFrames or file paths.")
    
    # Ensure stats is a list
    if isinstance(stats, str):
        stats = [stats]

    ## 1. PROCESS EACH FILE OR DATAFRAME
    for i, item in enumerate(files):
        # Load DataFrame
        if isinstance(item, (str, Path)):
            df = pd.read_csv(item)
        elif isinstance(item, pd.DataFrame):
            df = item.copy()
        else:
            raise ValueError("Each item must be a file path (str) or a DataFrame.")
        
        # Strip leading/trailing whitespace from column names
        df.columns = df.columns.str.strip()
        # Normalize time column names for detection, without altering the original DataFrame
        time_cols = {col.lower() for col in df.columns}
        mesh_cols = {'year', 'jday', 'hour', 'mins'}

        if mesh_cols.issubset(time_cols):
            # MESH detected
            # Drop unnamed columns (likely empty)
            df = df.loc[:, ~df.columns.str.contains('Unnamed')]
            # Standardize actual column names (map lowercase to actual column names)
            col_map = {col.lower(): col for col in df.columns}

            # Access real column names safely using lowercase keys
            df['datetime_utc'] = (
                pd.to_datetime(df[col_map['year']], format='%Y') +
                pd.to_timedelta(df[col_map['jday']] - 1, unit='D') +
                pd.to_timedelta(df[col_map['hour']], unit='h') +
                pd.to_timedelta(df[col_map['mins']], unit='m')
            )
            
            df = df.set_index('datetime_utc')
            df = df.drop(columns=[col_map['year'], col_map['jday'], col_map['hour'], col_map['mins']])

            # localize the UTC index to align with the NSRPS data for the GDPS
            if not df.index.tz:
                df.index = df.index.tz_localize('UTC')
            # resample to 12 hourly frequency to match Observed data
            df = df.resample('12h').mean()
            # Drop QOMEAS_ columns
            df = df.drop(columns=df.filter(regex='QOMEAS_').columns)
        else:
            # NSRPS: ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                # Try to find columns with datetime dtype (e.g., datetime64[ns])
                datetime_cols = df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
                
                # If no datetime dtype columns found,
                # but there is a column literally named 'datetime' (possibly object/string type),
                # add it to the list to try using it as datetime index anyway
                if not datetime_cols:
                    for fallback_col in ['datetime', 'time']:
                        if fallback_col in df.columns:
                            datetime_cols = [fallback_col]
                            break
                
                # If we found any datetime candidate columns,
                # set the first one as the DataFrame index
                # (usually there should be only one datetime column)
                if datetime_cols:
                    df = df.set_index(datetime_cols[0])
                else:
                    # If no datetime index or datetime column found, raise an error
                    raise ValueError("NSRPS DataFrame must have a datetime index or a datetime column")
            
                # Ensure it is in fact a datetime index
                # If index is not a DatetimeIndex, try to convert it
                if not isinstance(df.index, pd.DatetimeIndex):
                    try:
                        df.index = pd.to_datetime(df.index)
                    except Exception:
                        raise ValueError("Could not convert index to datetime. Ensure your CSV has datetime-formatted index.")
                
            # Ensure the index is in UTC
            if not df.index.tz:
                df.index = df.index.tz_localize('UTC')
            # Resample to 12 hourly frequency to match Observed data
            df = df.resample('12h').mean()

        files[i] = df
        
    ## 2. PROCESS OBSERVED DATA
    # Load observed DataFrame
    if isinstance(observed, str) and observed == 'realtime':
        # Generate the real time observed dataframe
        gen_flow = GenStreamflowFile()
        end_dt = datetime.now(timezone.utc).replace(microsecond=0) # Get the time today, right now as the end date.
        start_dt = end_dt - relativedelta(months=1) # set the start date to a be a month into the past i.e., past 1 month data
        start = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        end   = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        df_obs, meta_obs = gen_flow.fetch_hydrometric_realtime_full_range(
            station_numbers=[col.split('_', 1)[1] for col in files[0].columns], 
            start=start, end=end,
            window_days=1, freq_hours=12
        )
    elif isinstance(observed, pd.DataFrame):
        df_obs = observed.copy()
    else:
        raise ValueError("`observed` must be 'realtime' or a DataFrame.")
    
    if df_obs.empty:
        raise ValueError("No observed data found for the specified stations.")   

    ## 3. ADD OBSERVED COLUMNS TO EACH FORECAST DATAFRAME
    DATA={} # Dictionary to hold the dataframes
    for indx, df in enumerate(files):
        i = indx + 1  # for display/logging
        # Identify only QOSIM columns
        qosim_cols = [col for col in df.columns if col.startswith('QOSIM_')]
        if not qosim_cols:
            print(f"⚠️ Skipping file/index {i} — no QOSIM columns.")
            continue

        # Drop rows with any NaNs in QOSIM columns
        df_qosim = df[qosim_cols].dropna(how='any')

        # Extract station names
        stations = [col.split('_', 1)[1] for col in qosim_cols]

        # Add observed columns and rename columns properly
        new_df = pd.DataFrame(index=df_qosim.index)
        for station in stations:
            sim_col = f"QOSIM_{station}"
            obs_col = f"QOMEAS_{station}"

            # Add obs column if available
            if station in df_obs.columns:
                new_df[obs_col] = df_obs[station].reindex(df_qosim.index)
            else:
                new_df[obs_col] = np.nan

            # Add sim column
            new_df[sim_col] = df_qosim[sim_col]

        # Change index to multiples of 12
        new_df.index = [idx * 12 for idx in range(len(new_df))]

        # Rename columns to multi-index
        new_df = _set_multiindex_columns(new_df)

        # Store cleaned, properly ordered DataFrame
        DATA[f"csv_{i}"] = new_df  

    ## 4. COMPUTE PERCENT ERROR
    if not DATA:
        raise ValueError("No valid dataframes found in `files`.")
    final_error_df = _compute_percent_error(DATA)

    ## 5. AGGREGATE STATISTICS
    df_stats = _aggregate_statistics(final_error_df, stats)

    # Return the final DataFrame with percent errors and aggregated statistics
    return df_stats
