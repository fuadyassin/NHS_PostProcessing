"""
The preprocessing module contains all of the functions used alongside the metrics to filter, limit and validate the data 
before it gets evaluated.

"""

import numpy as np
import pandas as pd

from postprocessinglib.utilities import helper_functions as hlp

def station_dataframe(observed: pd.DataFrame, simulated: pd.DataFrame,
               stations: list[int]=[]) -> pd.DataFrame:
    """ Extracts a stations data from the observed and simulated 

    Parameters
    ---------- 
    observed: pd.DataFrame
            Observed values[1: Day of the year; 2: Streamflow Values]
    simulated: pd.DataFrame
            Simulated values[1: Day of the year; 2: Streamflow Values]
    stations: List[int]
            numbers pointing to the location of the stations in the list of stations.
            (Values can be any number from 1 to number of stations in the data)

    Returns
    -------
    pd.DataFrame:
        The station(s) observed and simulated data

    Example
    -------
    Extraction of the Data from Individual Stations

    >>> from postprocessinglib.evaluation import data
    >>> DATAFRAMES = data.generate_dataframes(csv_fpath=path, warm_up=365)
    >>> observed = DATAFRAMES["DF_OBSERVED"] 
    >>> simulated = DATAFRAMES["DF_SIMULATED"]
    >>> STATIONS = data.station_dataframe(observed=observed, simulated=simulated)
    >>> for station in STATIONS:
            print(station)
                QOMEAS_05BB001  QOSIM_05BB001
    1980-12-31           10.20       2.530770
    1981-01-01            9.85       2.518999
    1981-01-02           10.20       2.507289
    1981-01-03           10.00       2.495637
    1981-01-04           10.10       2.484073
    ...                    ...            ...
    2017-12-27           -1.00       4.418050
    2017-12-28           -1.00       4.393084
    2017-12-29           -1.00       4.368303
    2017-12-30           -1.00       4.343699
    2017-12-31           -1.00       4.319275
    [13515 rows x 2 columns]
                QOMEAS_05BA001  QOSIM_05BA001
    1980-12-31            -1.0       1.006860
    1981-01-01            -1.0       1.001954
    1981-01-02            -1.0       0.997078
    1981-01-03            -1.0       0.992233
    1981-01-04            -1.0       0.987417
    ...                    ...            ...
    2017-12-27            -1.0       1.380227
    2017-12-28            -1.0       1.372171
    2017-12-29            -1.0       1.364174
    2017-12-30            -1.0       1.356237
    2017-12-31            -1.0       1.348359

    [13515 rows x 2 columns]

    """

    # validate inputs
    hlp.validate_data(observed, simulated)

    Stations = []
    if not stations:
        for j in range(0, observed.columns.size):
            station_df =  observed.copy()
            station_df.drop(station_df.iloc[:, 0:], inplace=True, axis=1)
            station_df = pd.concat([station_df, observed.iloc[:, j], simulated.iloc[:, j]], axis = 1)
            Stations.append(station_df)
        return Stations
    else:
        for j in stations:
            # Adjust for zero indexing
            j -= 1

            station_df =  observed.copy()
            station_df.drop(station_df.iloc[:, 0:], inplace=True, axis=1)
            station_df = pd.concat([station_df, observed.iloc[:, j], simulated.iloc[:, j]], axis = 1)
            Stations.append(station_df)
        return Stations


    
#### aggregation(weekly, monthly, yearly)(check hydrostats)
# (median, mean, min, max, sum, instantaenous values options)

def seasonal_period(df: pd.DataFrame, daily_period: tuple[str, str],
                              time_range: tuple[str, str]=None) -> pd.DataFrame:
    """Creates a dataframe with a specified seasonal period

    Parameters
    ----------
    merged_dataframe: DataFrame
        A pandas DataFrame with a datetime index and columns containing float type values.
    daily_period: tuple(str, str)
        A list of length two with strings representing the start and end dates of the seasonal period (e.g.
        (01-01, 01-31) for Jan 1 to Jan 31.
    time_range: tuple(str, str)
        A tuple of string values representing the start and end dates of the time range. Format is YYYY-MM-DD.

    Returns
    -------
    pd.Dataframe
        Pandas dataframe that has been truncated to fit the parameters specified for the seasonal period.
    
    Examples
    --------
    Extraction of a Seasonal period

    >>> from postprocessinglib.evaluation import data
    >>> DATAFRAMES = data.generate_dataframes(csv_fpath=path, warm_up=365)
    >>> merged_df = DATAFRAMES["DF"]
    >>> print(merged_df)
                QOMEAS_05BB001  QOSIM_05BB001  QOMEAS_05BA001  QOSIM_05BA001
    1980-12-31           10.20       2.530770            -1.0       1.006860
    1981-01-01            9.85       2.518999            -1.0       1.001954
    1981-01-02           10.20       2.507289            -1.0       0.997078
    1981-01-03           10.00       2.495637            -1.0       0.992233
    1981-01-04           10.10       2.484073            -1.0       0.987417
    ...                    ...            ...             ...            ...
    2017-12-27           -1.00       4.418050            -1.0       1.380227
    2017-12-28           -1.00       4.393084            -1.0       1.372171
    2017-12-29           -1.00       4.368303            -1.0       1.364174
    2017-12-30           -1.00       4.343699            -1.0       1.356237
    2017-12-31           -1.00       4.319275            -1.0       1.348359
    
    >>> # Extract the time period
    >>> seasonal_p = data.seasonal_period(df=merged_df, daily_period=('01-01', '01-31'),
                            time_range = ('1981-01-01', '1985-12-31'))
    >>> print(seasonal_p)
                QOMEAS_05BB001  QOSIM_05BB001  QOMEAS_05BA001  QOSIM_05BA001
    1981-01-01            9.85       2.518999            -1.0       1.001954
    1981-01-02           10.20       2.507289            -1.0       0.997078
    1981-01-03           10.00       2.495637            -1.0       0.992233
    1981-01-04           10.10       2.484073            -1.0       0.987417
    1981-01-05            9.99       2.472571            -1.0       0.982631
    ...                    ...            ...             ...            ...
    1985-01-27           11.40       2.734883            -1.0       0.809116
    1985-01-28           11.60       2.721414            -1.0       0.805189
    1985-01-29           11.70       2.708047            -1.0       0.801287
    1985-01-30           11.60       2.694749            -1.0       0.797410
    1985-01-31           11.60       2.681550            -1.0       0.793556
    
    """
    # Making a copy to avoid changing the original df
    df_copy = df.copy()

    if time_range:
        # Setting the time range
        df_copy = df_copy.loc[time_range[0]: time_range[1]]
    
    # Setting a placeholder for the datetime string values
    df_copy.insert(loc=0, column='placeholder', value=df_copy.index.strftime('%m-%d'))

    # getting the start and end of the seasonal period
    start = daily_period[0]
    end = daily_period[1]

    # Getting the seasonal period
    if start < end:
        df_copy = df_copy.loc[(df_copy['placeholder'] >= start) &
                              (df_copy['placeholder'] <= end)]
    else:
        df_copy = df_copy.loc[(df_copy['placeholder'] >= start) |
                              (df_copy['placeholder'] <= end)]
    # Dropping the placeholder
    df_copy = df_copy.drop(columns=['placeholder'])
    
    return df_copy

def daily_aggregate(df: pd.DataFrame, method: str="mean") -> pd.DataFrame:
    """ Returns the daily aggregate value of a given dataframe based
        on the chosen method 

    Parameters
    ---------- 
    df: pd.DataFrame
            A pandas DataFrame with a datetime index and columns containing float type values.
    method: string
            string indicating the method of aggregation
            i.e, mean, min, max, median, sum and instantaenous
            - default is mean

    Returns
    -------
    pd.DataFrame:
            The new dataframe with the values aggregated by day 

    Examples
    --------
    Extraction of a Daily Aggregate

    >>> from postprocessinglib.evaluation import data
    >>> DATAFRAMES = data.generate_dataframes(csv_fpath=path, warm_up=365)
    >>> merged_df = DATAFRAMES["DF"]
    >>> print(merged_df)
                QOMEAS_05BB001  QOSIM_05BB001  QOMEAS_05BA001  QOSIM_05BA001
    1980-12-31           10.20       2.530770            -1.0       1.006860
    1981-01-01            9.85       2.518999            -1.0       1.001954
    1981-01-02           10.20       2.507289            -1.0       0.997078
    1981-01-03           10.00       2.495637            -1.0       0.992233
    1981-01-04           10.10       2.484073            -1.0       0.987417
    ...                    ...            ...             ...            ...
    2017-12-27           -1.00       4.418050            -1.0       1.380227
    2017-12-28           -1.00       4.393084            -1.0       1.372171
    2017-12-29           -1.00       4.368303            -1.0       1.364174
    2017-12-30           -1.00       4.343699            -1.0       1.356237
    2017-12-31           -1.00       4.319275            -1.0       1.348359
    
    >>> # Extract the daily aggregate by mean(default aggregation method)
    >>> daily_agg = data.daily_aggregate(df=merged_df)
    >>> print(daily_agg)
                QOMEAS_05BB001  QOSIM_05BB001  QOMEAS_05BA001  QOSIM_05BA001
    1980-12-31           10.20       2.530770            -1.0       1.006860
    1981-01-01            9.85       2.518999            -1.0       1.001954
    1981-01-02           10.20       2.507289            -1.0       0.997078
    1981-01-03           10.00       2.495637            -1.0       0.992233
    1981-01-04           10.10       2.484073            -1.0       0.987417
    ...                    ...            ...             ...            ...
    2017-12-27           -1.00       4.418050            -1.0       1.380227
    2017-12-28           -1.00       4.393084            -1.0       1.372171
    2017-12-29           -1.00       4.368303            -1.0       1.364174
    2017-12-30           -1.00       4.343699            -1.0       1.356237
    2017-12-31           -1.00       4.319275            -1.0       1.348359

    """
    # Check that there is a chosen method else return error
    if not method:
        raise RuntimeError("ERROR: A method of aggregation is required")
    else:
        # Making a copy to avoid changing the original df
        df_copy = df.copy()

        if method == "sum":
            daily_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%m/%d")).sum()
        if method == "mean":
            daily_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%m/%d")).mean()
        if method == "median":
            daily_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%m/%d")).median()
        if method == "min":
            daily_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%m/%d")).min()
        if method == "max":
            daily_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%m/%d")).max()
        if method == "inst":
            daily_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%m/%d")).last() 
    
    return daily_aggr

def weekly_aggregate(df: pd.DataFrame, method: str="mean") -> pd.DataFrame:
    """ Returns the weekly aggregate value of a given dataframe based
        on the chosen method 

    Parameters
    ---------- 
    df: pd.DataFrame
            A pandas DataFrame with a datetime index and columns containing float type values.
    method: string
            string indicating the method of aggregation
            i.e, mean, min, max, median, sum and instantaenous
            - default is mean

    Returns
    -------
    pd.DataFrame:
            The new dataframe with the values aggregated by week 

    Examples
    --------
    Extraction of a Weekly Aggregate

    >>> from postprocessinglib.evaluation import data
    >>> DATAFRAMES = data.generate_dataframes(csv_fpath=path, warm_up=365)
    >>> merged_df = DATAFRAMES["DF"]
    >>> print(merged_df)
                QOMEAS_05BB001  QOSIM_05BB001  QOMEAS_05BA001  QOSIM_05BA001
    1980-12-31           10.20       2.530770            -1.0       1.006860
    1981-01-01            9.85       2.518999            -1.0       1.001954
    1981-01-02           10.20       2.507289            -1.0       0.997078
    1981-01-03           10.00       2.495637            -1.0       0.992233
    1981-01-04           10.10       2.484073            -1.0       0.987417
    ...                    ...            ...             ...            ...
    2017-12-27           -1.00       4.418050            -1.0       1.380227
    2017-12-28           -1.00       4.393084            -1.0       1.372171
    2017-12-29           -1.00       4.368303            -1.0       1.364174
    2017-12-30           -1.00       4.343699            -1.0       1.356237
    2017-12-31           -1.00       4.319275            -1.0       1.348359
    
    >>> # Extract the weekly aggregate by taking the minumum value per week
    >>> weekly_agg = data.weekly_aggregate(df=merged_df, method="min")
    >>> print(weekly_agg)
             QOMEAS_05BB001  QOSIM_05BB001  QOMEAS_05BA001  QOSIM_05BA001
    1980/52           10.20       2.530770            -1.0       1.006860
    1981/00            9.85       2.484073            -1.0       0.987417
    1981/01            8.70       2.404939            -1.0       0.954524
    1981/02            8.24       2.328990            -1.0       0.923008
    1981/03            7.86       2.256059            -1.0       0.892801
    ...                 ...            ...             ...            ...
    2017/48           -1.00       5.074800            -1.0       1.592361
    2017/49           -1.00       4.871694            -1.0       1.526912
    2017/50           -1.00       4.677985            -1.0       1.464212
    2017/51           -1.00       4.494041            -1.0       1.404762
    2017/52           -1.00       4.319275            -1.0       1.348359

    """
    # Check that there is a chosen method else return error
    if not method:
        raise RuntimeError("ERROR: A method of aggregation is required")
    else:
        # Making a copy to avoid changing the original df
        df_copy = df.copy()

        if method == "sum":
            weekly_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%W")).sum()
        if method == "mean":
            weekly_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%W")).mean()
        if method == "median":
            weekly_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%W")).median()
        if method == "min":
            weekly_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%W")).min()
        if method == "max":
            weekly_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%W")).max()
        if method == "inst":
            weekly_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%W")).last()    
    
    return weekly_aggr

def monthly_aggregate(df: pd.DataFrame, method: str="mean") -> pd.DataFrame:
    """ Returns the weekly aggregate value of a given dataframe based
        on the chosen method 

    Parameters
    ---------- 
    df: pd.DataFrame
            A pandas DataFrame with a datetime index and columns containing float type values.
    method: string
            string indicating the method of aggregation
            i.e, mean, min, max, median, sum and instantaenous
            - default is mean

    Returns
    -------
    pd.DataFrame:
            The new dataframe with the values aggregated by 
            months of the year 

    Examples
    --------
    Extraction of a Monthly Aggregate

    >>> from postprocessinglib.evaluation import data
    >>> DATAFRAMES = data.generate_dataframes(csv_fpath=path, warm_up=365)
    >>> merged_df = DATAFRAMES["DF"]
    >>> print(merged_df)
                QOMEAS_05BB001  QOSIM_05BB001  QOMEAS_05BA001  QOSIM_05BA001
    1980-12-31           10.20       2.530770            -1.0       1.006860
    1981-01-01            9.85       2.518999            -1.0       1.001954
    1981-01-02           10.20       2.507289            -1.0       0.997078
    1981-01-03           10.00       2.495637            -1.0       0.992233
    1981-01-04           10.10       2.484073            -1.0       0.987417
    ...                    ...            ...             ...            ...
    2017-12-27           -1.00       4.418050            -1.0       1.380227
    2017-12-28           -1.00       4.393084            -1.0       1.372171
    2017-12-29           -1.00       4.368303            -1.0       1.364174
    2017-12-30           -1.00       4.343699            -1.0       1.356237
    2017-12-31           -1.00       4.319275            -1.0       1.348359
    
    >>> # Extract the monthly aggregate by taking the instantaenous value of each month
    >>> monthly_agg = data.monthly_aggregate(df=merged_df, method="inst")
    >>> print(monthly_agg)
             QOMEAS_05BB001  QOSIM_05BB001  QOMEAS_05BA001  QOSIM_05BA001
    1980/12           10.20       2.530770            -1.0       1.006860
    1981/01            8.62       2.195846            -1.0       0.867900
    1981/02            7.20       1.940355            -1.0       0.762678
    1981/03            7.25       1.699932            -1.0       0.664341
    1981/04           15.30       3.859564            -1.0       0.584523
    ...                 ...            ...             ...            ...
    2017/08           -1.00      31.050230            -1.0      17.012710
    2017/09           -1.00      16.144130            -1.0      11.127440
    2017/10           -1.00       6.123822            -1.0       1.938875
    2017/11           -1.00       5.164804            -1.0       1.621027
    2017/12           -1.00       4.319275            -1.0       1.348359

    """
    # Check that there is a chosen method else return error
    if not method:
        raise RuntimeError("ERROR: A method of aggregation is required")
    else:
        # Making a copy to avoid changing the original df
        df_copy = df.copy()

        if method == "sum":
            monthly_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%m")).sum()
        if method == "mean":
            monthly_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%m")).mean()
        if method == "median":
            monthly_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%m")).median()
        if method == "min":
            monthly_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%m")).min()
        if method == "max":
            monthly_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%m")).max()
        if method == "inst":
            monthly_aggr = df_copy.groupby(df_copy.index.strftime("%Y/%m")).last()    
    
    return monthly_aggr

def yearly_aggregate(df: pd.DataFrame, method: str="mean") -> pd.DataFrame:
    """ Returns the weekly aggregate value of a given dataframe based
        on the chosen method 

    Parameters
    ---------- 
    df: pd.DataFrame
            A pandas DataFrame with a datetime index and columns containing float type values.
    method: string
            string indicating the method of aggregation
            i.e, mean, min, max, median, sum and instantaenous
            - default is mean

    Returns
    -------
    pd.DataFrame:
            The new dataframe with the values aggregated by years 

    Examples
    --------
    Extraction of a Yearly Aggregate

    >>> from postprocessinglib.evaluation import data
    >>> DATAFRAMES = data.generate_dataframes(csv_fpath=path, warm_up=365)
    >>> merged_df = DATAFRAMES["DF"]
    >>> print(merged_df)
                QOMEAS_05BB001  QOSIM_05BB001  QOMEAS_05BA001  QOSIM_05BA001
    1980-12-31           10.20       2.530770            -1.0       1.006860
    1981-01-01            9.85       2.518999            -1.0       1.001954
    1981-01-02           10.20       2.507289            -1.0       0.997078
    1981-01-03           10.00       2.495637            -1.0       0.992233
    1981-01-04           10.10       2.484073            -1.0       0.987417
    ...                    ...            ...             ...            ...
    2017-12-27           -1.00       4.418050            -1.0       1.380227
    2017-12-28           -1.00       4.393084            -1.0       1.372171
    2017-12-29           -1.00       4.368303            -1.0       1.364174
    2017-12-30           -1.00       4.343699            -1.0       1.356237
    2017-12-31           -1.00       4.319275            -1.0       1.348359
    
    >>> # Extract the yearly aggregate by taking the sum of the entire year's values
    >>> yearly_agg = data.yearly_aggregate(df=merged_df, method="sum")
    >>> print(yearly_agg)
          QOMEAS_05BB001  QOSIM_05BB001  QOMEAS_05BA001  QOSIM_05BA001
    1980           10.20       2.530770           -1.00       1.006860
    1981        10386.27    9273.383180         2243.26    4007.949313
    1982        12635.47    8874.369067         2982.23    4123.606233
    1983        11909.23    8214.793557         3017.17    3810.515038
    1984        13298.33    7459.351671         2517.42    3431.981225
    1985        13730.50    8487.241498         2811.40    3756.822014
    1986        12576.84   10651.883689         2922.15    4794.825198
    1987        15066.57    8947.025052         3418.74    4260.917801
    1988        12642.53   10377.241643         2790.87    4614.234614
    1989        10860.93   11118.336160         2443.79    5193.322199
    1990        11129.76   11226.011936         2469.50    5273.448490
    1991        14354.61   12143.013205         3034.89    5732.371571
    1992        17033.16    9919.064629         3703.72    4566.044810
    1993        15238.65   10265.868953         3417.67    4700.055333
    1994        15623.13    8064.390172         3596.16    4053.331783
    1995        12892.89   10526.186570         3640.08    5006.592916
    1996        12551.39    9191.247302         3073.36    4195.638177
    1997         -352.80    9078.253847         -365.00    4469.825844
    1998         -365.00    9421.178402         3418.21    4650.819283
    1999         -365.00    8683.319193         3039.62    4032.381482
    2000         -366.00   10181.718825         -366.00    4921.033689
    2001         -365.00    7076.942619         -365.00    3525.593143
    2002         -365.00    8046.998223         -365.00    4048.992212
    2003         -365.00    9017.711719         -365.00    4517.088194
    2004         -366.00   11726.707770         -366.00    4941.582065
    2005         -365.00   11975.002047         -365.00    4700.295391
    2006         -365.00    8972.956022         -365.00    4038.214837
    2007         -365.00   11089.242586         -365.00    5035.426223
    2008         -366.00    9652.958064         -366.00    4630.531909
    2009         -365.00    8762.313253         -365.00    3659.265122
    2010         -365.00    8006.621137         -365.00    3475.115315
    2011         -365.00   10158.521707         -365.00    4748.153725
    2012         -366.00   13141.668859         -366.00    5847.670810
    2013         -365.00   11389.072535         -365.00    4769.917090
    2014         -365.00   12719.851800         -365.00    5298.904086
    2015         -365.00   12258.178724         -365.00    5362.497143
    2016         -366.00    9989.779678         -366.00    4269.909376
    2017         -365.00    8801.897128         -365.00    4226.258100

    """
    # Check that there is a chosen method else return error
    if not method:
        raise RuntimeError("ERROR: A method of aggregation is required")
    else:
        # Making a copy to avoid changing the original df
        df_copy = df.copy()
        
        if method == "sum":
            yearly_aggr = df_copy.groupby(df_copy.index.strftime("%Y")).sum()
        if method == "mean":
            yearly_aggr = df_copy.groupby(df_copy.index.strftime("%Y")).mean()
        if method == "median":
            yearly_aggr = df_copy.groupby(df_copy.index.strftime("%Y")).median()
        if method == "min":
            yearly_aggr = df_copy.groupby(df_copy.index.strftime("%Y")).min()
        if method == "max":
            yearly_aggr = df_copy.groupby(df_copy.index.strftime("%Y")).max()  
        if method == "inst":
            yearly_aggr = df_copy.groupby(df_copy.index.strftime("%Y")).last()  
    
    return yearly_aggr  

def generate_dataframes(csv_fpath: str='', sim_fpath: str='', obs_fpath: str='', warm_up: int = 0, start_date :str = "", end_date: str = "",
                        daily_agg:bool=False, da_method:str="", weekly_agg:bool=False, wa_method:str="",
                        monthly_agg:bool=False, ma_method:str="", yearly_agg:bool=False, ya_method:str="",
                        seasonal_p:bool=False, sp_dperiod:tuple[str, str]=[], sp_time_range:tuple[str, str]=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ Function to Generate the required dataframes

    Parameters
    ----------
    csv_fpath : string
            the path to the csv file. It can be relative or absolute
    sim_fpath: str
        The filepath to the simulated csv of data.
    obs_fpath: str
        The filepath to the observed csv of the data.
    num_min: int 
            number of days required to "warm up" the system
    start_date: str 
            The date at which you want to start calculating the metric in the
            format yyyy-mm-dd
    end_date: str
            The date at which you want the calculations to end in the
            format yyyy-mm-dd
    daily_agg: bool = False
            If True calculate and return the daily aggregate of the combined dataframes
            using da_method if its available
    da_method: str = ""
            If provided, it determines the method of daily aggregation. It 
            is "mean" by default, see daily_aggregate() function
    weekly_agg: bool = False
            If True calculate and return the weekly aggregate of the combined dataframes
            using wa_method if its available
    wa_method: str = ""
            If provided, it determines the method of weekly aggregation. It 
            is "mean" by default, see weekly_aggregate() function
    monthly_agg: bool = False
            If True calculate and return the monrhly aggregate of the combined dataframes
            using ma_method if its available
    ma_method: str = ""
            If provided, it determines the method of monthly aggregation. It 
            is "mean" by default, see monthly_aggregate() function
    yearly_agg: bool = False
            If True calculate and return the yearly aggregate of the combined dataframes
            using ya_method if its available
    ya_method: str = ""
            If provided, it determines the method of yearly aggregation. It 
            is "mean" by default, see yearly_aggregate() function
    seasonal_p: bool = False
            If True calculate and return a dataframe truncated to fit the parameters specified
            for the seasonal period 
            Requirement:- sp_dperiod.
    sp_dperiod: tuple(str, str)
            A list of length two with strings representing the start and end dates of the seasonal period (e.g.
            (01-01, 01-31) for Jan 1 to Jan 31.
    sp_time_range: tuple(str, str)
            A tuple of string values representing the start and end dates of the time range. Format is YYYY-MM-DD.

    Returns
    -------
    dict{str: pd.dataframe}
            A dictionary containing each Dataframe requested. Its default content is:
            - DF = merged dataframe
            - DF_SIMULATED = all simulated data
            - DF_OBSERVED = all observed data
            
            Depending on which you requested it can also contain:
            - DF_DAILY = dataframe aggregated by days of the year
            - DF_WEEKLY = dataframe aggregated by the weeks of the year
            - DF_MONTHLY = dataframe aggregated by months of the year
            - DF_YEARLY = dataframe aggregated by all the years in the data
            - DF_CUSTOM = dataframe truncated as per the seasonal period parameters

    See jupyter notebook file linked below for usage instances
            
    """

    DATAFRAMES = {}
    if csv_fpath:
        # read the combined csv file into a dataframe
        df = pd.read_csv(csv_fpath, skipinitialspace = True, index_col = [0, 1])
        # if there are any extra columns at the end of the csv file, remove them
        if len(df.columns) % 2 != 0:
            df.drop(columns=df.columns[-1], inplace = True)        
        # Convert the year and jday index to datetime indexing
        start_day = hlp.MultiIndex_to_datetime(df.index[0])
        df.index = pd.to_datetime([i for i in range(len(df.index))], unit='D',origin=pd.Timestamp(start_day))
        # replace all invalid values with NaN
        df = df.replace([-1, 0], np.nan)   
        
        # Take off the warm up time
        DATAFRAMES["DF"] = df[warm_up:]    
        simulated = observed = df[warm_up:].copy()
        simulated.drop(simulated.iloc[:, 0:], inplace=True, axis=1)
        observed.drop(observed.iloc[:, 0:], inplace=True, axis=1)
        for j in range(0, len(df.columns), 2):
            arr1 = df.iloc[warm_up:, j]
            arr2 = df.iloc[warm_up:, j+1]
            observed = pd.concat([observed, arr1], axis = 1)
            simulated = pd.concat([simulated, arr2], axis = 1)

    elif sim_fpath and obs_fpath:
        # read the simulated and observed csv files into dataframes
        sim_df = pd.read_csv(sim_fpath, skipinitialspace = True, index_col=[0, 1])
        obs_df = pd.read_csv(obs_fpath, skipinitialspace = True, index_col=[0, 1])

        # Convert the year and jday index to datetime indexing
        # simulated
        start_day = hlp.MultiIndex_to_datetime(sim_df.index[0])
        sim_df.index = pd.to_datetime([i for i in range(len(sim_df.index))], unit='D',origin=pd.Timestamp(start_day))
        
        
        # observed
        start_day = hlp.MultiIndex_to_datetime(obs_df.index[0])
        obs_df.index = pd.to_datetime([i for i in range(len(obs_df.index))], unit='D',origin=pd.Timestamp(start_day))

        # replace all invalid values with NaN
        sim_df = sim_df.replace([-1, 0], np.nan)
        obs_df = obs_df.replace([-1, 0], np.nan)
        df = pd.DataFrame(index = obs_df.index)
        for j in range(0, len(obs_df.columns)):
            arr1 = obs_df.iloc[:, j]
            arr2 = sim_df.iloc[:, j]
            df = pd.concat([df, arr1, arr2], axis = 1)

        # Take off the warm up time
        simulated = sim_df[warm_up:]
        observed = obs_df[warm_up:]                
        DATAFRAMES["DF"] = df[warm_up:] 

    else:
        raise RuntimeError('either sim_fpath and obs_fpath or csv_fpath are required inputs.')
       

    # splice the dataframes according to the time frame
    if not start_date and end_date:
        # there's an end date but no start date
        simulated = simulated.loc[:end_date]
        observed = observed.loc[:end_date]
        DATAFRAMES["DF"] = DATAFRAMES["DF"][:end_date]
    elif not end_date and start_date:
        # there's and end date but no start date
        simulated = simulated.loc[start_date:]
        observed = observed.loc[start_date:]
        DATAFRAMES["DF"] = DATAFRAMES["DF"][start_date:]
    elif start_date and end_date:
        # there's a start and end date
        simulated = simulated.loc[start_date:end_date]
        observed = observed.loc[start_date:end_date]
        DATAFRAMES["DF"] = DATAFRAMES["DF"][start_date:end_date]
    
    # validate inputs
    hlp.validate_data(observed, simulated)
    
    DATAFRAMES["DF_SIMULATED"] = simulated
    DATAFRAMES["DF_OBSERVED"] = observed

    # Creating the remaining dataframes based on input
    # 1. Daily aggregate
    if daily_agg and da_method:
        DATAFRAMES["DF_DAILY"] = daily_aggregate(df = DATAFRAMES["DF"], method=da_method)
    elif daily_agg:
        # mean by default
        DATAFRAMES["DF_DAILY"] = daily_aggregate(df = DATAFRAMES["DF"])

    # 2. Weekly aggregate
    if weekly_agg and wa_method:
        DATAFRAMES["DF_WEEKLY"] = weekly_aggregate(df = DATAFRAMES["DF"], method=wa_method)
    elif weekly_agg:
        # mean by default
        DATAFRAMES["DF_WEEKLY"] = weekly_aggregate(df = DATAFRAMES["DF"])

    # 3. Monthly aggregate
    if monthly_agg and ma_method:
        DATAFRAMES["DF_MONTHLY"] = monthly_aggregate(df = DATAFRAMES["DF"], method=ma_method)
    elif monthly_agg:
        # mean by default
        DATAFRAMES["DF_MONTHLY"] = monthly_aggregate(df = DATAFRAMES["DF"])

    # 4.Yearly aggregate
    if yearly_agg and ya_method:
        DATAFRAMES["DF_YEARLY"] = yearly_aggregate(df = DATAFRAMES["DF"], method=ya_method)
    elif yearly_agg:
        # mean by default
        DATAFRAMES["DF_YEARLY"] = yearly_aggregate(df = DATAFRAMES["DF"])

    # 5. Seasonal Period
    if seasonal_p and sp_dperiod == []:
        raise RuntimeError("You cannot calculate a seasonal period without a daily period")
    elif seasonal_p and sp_dperiod and sp_time_range:
        DATAFRAMES["DF_CUSTOM"] = seasonal_period(df = DATAFRAMES["DF"], daily_period=sp_dperiod,
                                                  time_range=sp_time_range)    
    elif seasonal_p and sp_dperiod:
        DATAFRAMES["DF_CUSTOM"] = seasonal_period(df = DATAFRAMES["DF"], daily_period=sp_dperiod)
    
    
    return DATAFRAMES