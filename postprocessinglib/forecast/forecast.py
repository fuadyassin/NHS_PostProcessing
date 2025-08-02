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
from typing import Union

import re
import time
import json
import logging
import configparser

import numpy as np
import xarray as xr
import pandas as pd

from functools import partial
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

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


# def fcst_times_from_wms(
#         layers: list[str], 
#         username: str=None,
#         password: str=None,
#     ) -> Union[str, list[str]]:
#     """Retrieves the forecast time metadata from the WebMapService for the 
#     GeoMet layer (product) of interest.

#     Parameters
#     ----------
#     layers : list [str]
#         Name of the GeoMet layers.
#     username : str, optional
#         Username to access secured layer from WMS API, by default None.
#     password : str, optional
#         Password to access secured layer from WMS API, by default None.

#     Returns
#     -------
#     Union[str, list[str]]
#         The issuse / publication timestamp for the latest forecast.
#         A list of timestamps corresponding to the latest forecast's timesteps.
#     """
#     # 3 layers for the 3 days we will be querying
#     first = [''] * len(layers)
#     last = [None] * len(layers)
#     intvl = [None] * len(layers)
#     end_of_day = [None] * len(layers)

#     # create a list of forecast datetimes
#     hrs = []

#     for i in range(0, len(layers)):
#         # create WMS object
#         wms = WebMapService(
#             f'https://geo.weather.gc.ca/geomet?&SERVICE=WMS&LAYERS={layers[i]}',
#             version='1.3.0',
#             auth=Authentication(username=username, password=password),
#             timeout=300
#         )

#         layer_metadata = wms[layers[i]]
#         name = layers[i].split('_')    

#         if name[0] == "DHPS" or name[0] == "DHPS-Analysis" or name[0] == "EHPS-Analysis":
#             # oldest_fcast = the oldest forecast available from GeoMet, typically b/w 2 and 3 weeks old
#             # latest_fcast = the most recently created forecast
#             # issue_interval = how frequently the forecasts are published
#             oldest_fcast, latest_fcast, issue_interval = layer_metadata.dimensions['reference_time']['values'][0].split('/')

#         elif name[0] == "EHPS":
#             # oldest_fcast = the oldest forecast available from GeoMet, typically the last couple hours
#             # latest_fcast = the most recently created forecast
#             # oldest_fcast= wms[layer_name].dimensions['reference_time']['values'][0]
#             latest_fcast= layer_metadata.dimensions['reference_time']['values'][-1]

#         # the following variables correspond to the latest_fcast issue
#         # first_time = the first forecast datetime
#         # last_time = the last datetime in the forecast i.e. the extent of the forecast horizon
#         # time_interval = the temporal resolution of the forecast
#         first_time, last_time, time_interval = layer_metadata.dimensions['time']['values'][0].split('/')

#         iso_format = "%Y-%m-%dT%H:%M:%SZ"

#         # remove anything that isn't a number from the datetime interval (time between forecasts)
#         intvl[i] = int(re.sub(r'\D', '', time_interval))

#         # convert date strings to datetime objects
#         if i == 0:
#             first[i] = datetime.strptime(first_time, iso_format)
#         else:
#             if i == len(layers)-1:
#                 first[i] = first[i-1] - timedelta(hours=intvl[i-1])
#             else:
#                 first[i] = first[i-1] - timedelta(hours=intvl[i-1]) + timedelta(hours=intvl[i])
#         last[i] = datetime.strptime(last_time, iso_format)

#         end_of_day[i] = first[i] + timedelta(hours=(intvl[i] * (24/intvl[i])))

#         if i < len(layers):
#             # update the list of forecast dates
#             while first[i] < end_of_day[i]:
#                 hrs.append(first[i])
#                 first[i] = first[i] + timedelta(hours=intvl[i])
        

#     # create a list of ISO formatted forecast datetime strings
#     fcasthrs = [datetime.strftime(hr, iso_format) for hr in hrs]
#     fcasthrs.pop()

#     return latest_fcast, fcasthrs


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
    ) -> pd.DataFrame:
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


def main(
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
        df = df.rename(columns={'RiverDischarge': f"{stn}_sim"})

        # append the forecasts for this station to the big DataFrame
        data = pd.concat([data, df], axis=1)
        # print(data)
        # print("\n")

        logger.info(
            f"Finished forecast queries for station: {stn}; iteration: {i}"
        )
    # return data, closest_coords, stns
    return data


if __name__ == '__main__':
    
    # name of forecast layer from GeoMet
    # DHPS = Deterministic Hydrologic Prediction System (a subsystem of NSRPS)
    # when EHPS is available, the `layer_name` could be changed
    layer_name = 'DHPS_1km_RiverDischarge'
    # layer_name = 'EHPS_1km_RiverDischarge-Min24h-Day17to32'

    # PILOT: gauge stations in Newfoundland whose data is assimilated by NSRPS
    nfl_stns = [
        # '02ZB001',
        # '02YJ001',
        '05GG001',
        # '02YL003',
        # '02YL001',
        # '02YK005',
        # '02YL008',
        '05CC002',
        '05AC003'
    ]
    # nfl_stns = ['05AA024', '05AC003', '05AD007', '05AG006', '05AJ001', '05BB001', '05BG010', '05BH004', '05BL024', '05BN012', '05CA009', '05CB001', '05CC002', '05CE001', '05CK004', '05DB006', '05DC001', '05DF001', '05EF001', '05FA001', '05FE004', '05GG001', '05HD039', '05HG001', '05KD003', '05KJ001']

    # data, closest_coords, stns = main(
    data = main(
        # auth_path='config.cfg',
        auth_path='../config.cfg',
        stn_list=nfl_stns,
        layer_name=layer_name, 
        stn_locs_file='available_stations.json'
    )
    
    # sanity check that the flows look reasonable
    # quick comparison with lower right inset station plots from: 
    # https://web.science.gc.ca/~vfo001/nsrps_viewer/dhps/gsl/
    logger.info(
        # f"First 5 rows of forecasted streamflow: \n {data.head()}"
        # f"\nFirst 20 rows of forecasted streamflow: \n {data.iloc[:20]}"
        f"\nForecasted streamflow: \n {data}"
        # f"\nForecasted streamflow: \n {data.tail(40)}"

    )
    # logger.info(
    #         f"\nComparing Actual Lat Lon from JSON file and closest lat lon from WCS: \n"
    # )
    # for stn in nfl_stns:
    #     clat, clon = closest_coords[stn]
    #     logger.info(
    #         f"Station {stn}:\n"
    #         f"Actual Lat: {stns.loc[stn]['lat']}, Actual Lon: {stns.loc[stn]['lon']}\n"
    #         f"Closest Lat: {clat}, Closest Lon: {clon}\n"
    # )


# def main(
#         auth_path: str,
#         stn_list: list[str],
#         layer_name: str,
#         layer_index = int,
#         stn_locs_file: str=None,
#     ) -> pd.DataFrame:
#     """Queries for streamflow forecasts from the WCS for a given list of gauge
#     stations.

#     Parameters
#     ----------
#     auth_path : str
#         Path to configuration file containing authentication information.
#     stn_list : list[str]
#         The list of stations where we want to query for forecasts.
#     layer_name : str
#         Name of the GeoMet layer.
#     layer_index: int
#         An identifier for which set of layers to query for the first 3 days.
#     stn_locs_file : str, optional
#         Path to geoJSON file containing station locations, by default None.

#     Returns
#     -------
#     pd.DataFrame
#         Tabulated forecasts at gauge stations. DataFrame has the format:

#         ----------------------------------------------------------------
#         | time                |   00XX001   |   00XX002   |   ......   |
#         ----------------------------------------------------------------
#         | YYYY-MM-DD HH:mm:ss |     xxxx    |     xxxx    |    ...     |
#         | ...                 |     ....    |     ....    |    ...     |
#         ----------------------------------------------------------------
        
#         ** DataFrame index = `time` (in UTC)
#     """

#     # read authentication info from config file
#     username, password = get_auth(auth_path)

#     # Layers
#     layers = [[""]*3, [""]*3, [""]*3]
#     layers[0][0] = "EHPS_1km_RiverDischarge-Min3h"
#     layers[0][1] = "EHPS_1km_RiverDischarge-Min6h"
#     layers[0][2] = "EHPS_1km_RiverDischarge-Min24h"

#     layers[1][0] = "EHPS_1km_RiverDischarge-Max3h"
#     layers[1][1] = "EHPS_1km_RiverDischarge-Max6h"
#     layers[1][2] = "EHPS_1km_RiverDischarge-Max24h"

#     layers[2][0] = "EHPS_1km_RiverDischarge-Avg3h"
#     layers[2][1] = "EHPS_1km_RiverDischarge-Avg6h"
#     layers[2][2] = "EHPS_1km_RiverDischarge-Avg24h" 

#     # find the "model world" gauge station locations
#     stns = stns_on_grid(stn_locs_file) 

#     # fetch time steps from the latest forecast from GeoMet's Web Map Service
#     latest_fcast, fcst_dtimes = fcst_times_from_wms(
#         layers=layers[layer_index], 
#         username=username, 
#         password=password
#     )

#     # connect to the Web Coverage Service
#     wcs = connect_to_wcs(
#         layer_name=layer_name, 
#         username=username, 
#         password=password
#     )

#     # create empty dataframe to store forecast data for stations
#     data = pd.DataFrame([])

#     for i, stn in enumerate(stn_list):

#         # "model world" coordinates extracted with nsrps_stns_to_geojson.py
#         lat_nsrps = stns.loc[stn]['lat']
#         lon_nsrps = stns.loc[stn]['lon']

#         fcsts_from_wcs = partial(
#             predictions_from_wcs, 
#             wcs, 
#             layer_name, 
#             latest_fcast, 
#             lat_nsrps, 
#             lon_nsrps, 
#             1 # half-width in degrees of the grid subset we'll query
#         )

#         fcst_arrays = []
#         # a query needs to be made for each forecast timestep and each stn
#         # DHPS 6-day fcast has 144 timesteps
#         # querying in this way is very I/O bound, so we use threading to speed up the job
#         with ThreadPoolExecutor() as executor:
#             # query the WCS for each forecast timestep
#             iterator = executor.map(
#                 fcsts_from_wcs, 
#                 fcst_dtimes
#             )
#             # grab the results from the object returned by the executor
#             for result in iterator:
#                 fcst_arrays.append(result)
        
#         # concatenate all forecast timesteps into one xarray dataset
#         ds = xr.concat([fcst for fcst in fcst_arrays], dim='time')

#         # find the lat / lot corresponding to the target grid cell's indices
#         lat, lon = find_stn_target_gridcell(ds, lat_nsrps, lon_nsrps)

#         # extract the streamflow at the station of interest
#         # note: using NSRPS lat / lon to `select` streamflow from target cell
#         # is not always reliable b/c NSRPS lat / lons are not always co-located
#         # with the target cell's lat / lon label indexers
#         strflw = ds.sel(
#             lat=lat, 
#             lon=lon, 
#             method='nearest'
#         )

#         # store streamflow in dataframe
#         df = strflw.to_dataframe()
        
#         # dropping unneeded information
#         df = df.drop(['crs', 'lat', 'lon'], axis=1) 
#         df = df.rename(columns={'Band1': stn})

#         # append the forecasts for this station to the big DataFrame
#         data = pd.concat([data, df], axis=1)

#         logger.info(
#             f"Finished forecast queries for station: {stn}; iteration: {i}: Layer = {layer_name}"
#         )
#     return data



# if __name__ == '__main__':
    
#     # name of forecast layer from GeoMet
#     # DHPS = Deterministic Hydrologic Prediction System (a subsystem of NSRPS)
#     # when EHPS is available, the `layer_name` could be changed
#     # layer_name = 'DHPS_1km_RiverDischarge'
#     # layer_name = 'EHPS_1km_RiverDischarge-Max3h'

#     layer = [""] * 3
#     layer[0] = "EHPS_1km_RiverDischarge-Min3h"
#     layer[1] = "EHPS_1km_RiverDischarge-Max3h"
#     layer[2] = "EHPS_1km_RiverDischarge-Avg3h"

#     # PILOT: gauge stations in Newfoundland whose data is assimilated by NSRPS
#     nfl_stns = [
#         # '02ZB001',
#         # '02YJ001',
#         '02YK002',
#         # '02YL003',
#         # '02YL001',
#         # '02YK005',
#         # '02YL008',
#         # '02YC001',
#     ]

#     data = [pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([])]
#     time_start = time.time()
#     for i in range(0, 3):
#         time_start_1 = time.time()
#         data[i] = main(
#             # auth_path='config.cfg',
#             auth_path='../config.cfg',
#             stn_list=nfl_stns,
#             layer_index= i,
#             # layer_name=layer_name,
#             layer_name=layer[i], 
#             stn_locs_file='nsrps_model_LOTW_stn_locs.json'
#         )
#         time_end_1 = time.time()
#         print("wms data in %s seconds" %(time_end_1 - time_start_1))
#         print('-')
#     time_end = time.time()
#     print("Prediction data in %s seconds" %(time_end - time_start))
#     print('---')

#     # data = main(
#     #     # auth_path='config.cfg',
#     #     auth_path='../config.cfg',
#     #     stationId="02YL003",
#     #     stn_locs_file='nsrps_model_stn_locs.json'
#     # )

#     # data = main(
#     #         # auth_path='config.cfg',
#     #         auth_path='../config.cfg',
#     #         stn_list=nfl_stns,
#     #         layer_index= 0,
#     #         # layer_name=layer_name,
#     #         layer_name=layer[0], 
#     #         stn_locs_file='nsrps_model_LOTW_stn_locs.json'
#     #     )
    
#     # sanity check that the flows look reasonable
#     # quick comparison with lower right inset station plots from: 
#     # https://web.science.gc.ca/~vfo001/nsrps_viewer/dhps/gsl/
#     logger.info(
#         # f"First 5 rows of forecasted streamflow: \n {data.head()}"
#         # f"\nFirst 20 rows of forecasted streamflow: \n {data.iloc[:20]}"
#         f"\nForecasted streamflow: \n {data}"
#     )


