"""
The metrics module contains all of the metrics required during the prost processing process.
Each metric has a function and there are hepler functions that help with error checking and 
reading and preparation of data to be evaluated.
This module contains functions evaluate single sample data  such as the 

- The Slope of Flow duration curve ``slope_fdc``,
- Time to peak ``ttp``, 
- Time to centre of mass ``ttcom``, and,
- Spring Pulse Onset Delay ``spod``.

as well as comparison samples between the measured and predicted data such as 

- Mean Square Error ``mse``,
- Percentage Bias ``bias``, and ,
- Kling Gupta Efficiency ``kge``, amongst others.  

Finally we are also able to calculate a number of metrices at the same time or all of them at the same
time using one of the two function below:

- ``calculate_metrics([list of metrics])``
- ``calculate_all_metrics()``

All these functions along side their expected inputs and outputs are shown below:

"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple


from postprocessinglib.utilities import _helper_functions as hlp

def available_metrics() -> list[int]:
    """ Get a list of currently available metrics

    Returns
    -------
    List[str]
        List of implemented metric names.

    Example
    -------
    
    >>> from postprocessinglib.evaluation import metrics
    >>> print(metrics.available_metrics())
        ["MSE - Mean Square Error", "RMSE - Roor Mean Square Error", 
        "MAE - Mean Average Error", "NSE - Nash-Sutcliffe Efficiency ", 
        "NegNSE - Nash-Sutcliffe Efficiency * -1", "LogNSE - Log of Nash-Sutcliffe Efficiency", 
        "NegLogNSE - Log of Nash-Sutcliffe Efficiency * -1",
        "KGE - Kling-Gupta Efficiency", "NegKGE - Kling-Gupta Efficiency * -1", 
        "KGE 2012 - Kling-Gupta Efficiency modified as of 2012", "BIAS- Prcentage Bias", 
        "AbsBIAS - Absolute Value of the Percentage Bias", "TTP - Time to Peak", 
        "TTCoM - Time to Centre of Mass", "SPOD - Spring Pulse ONset Delay", 
        'FDC Slope - Slope of the Flow Duration Curve' ]

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-metrics.ipynb>`_

    """
    metrics = [
        "MSE - Mean Square Error", "RMSE - Roor Mean Square Error", 
        "MAE - Mean Average Error", "NSE - Nash-Sutcliffe Efficiency ", 
        "NegNSE - Nash-Sutcliffe Efficiency * -1", "LogNSE - Log of Nash-Sutcliffe Efficiency", 
        "NegLogNSE - Log of Nash-Sutcliffe Efficiency * -1",
        "KGE - Kling-Gupta Efficiency", "NegKGE - Kling-Gupta Efficiency * -1", 
        "KGE 2012 - Kling-Gupta Efficiency modified as of 2012", "BIAS- Prcentage Bias", 
        "AbsBIAS - Absolute Value of the Percentage Bias", "TTP - Time to Peak", 
        "TTCoM - Time to Centre of Mass", "SPOD - Spring Pulse ONset Delay", 
        'FDC Slope - Slope of the Flow Duration Curve'
    ]
    
    return metrics

def mse(observed: pd.DataFrame, simulated: Union[pd.DataFrame, List[pd.DataFrame]], stations: list[int]=[]) -> float:
    """ Calculates the Mean Square value of the data

    Parameters
    ---------- 
    observed: pd.DataFrame
            Observed values[1: Datetime ; 2+: Streamflow Values]
    simulated: pd.DataFrame or list[pd.DataFrame]
            Simulated values[1: Datetime ; 2+: Streamflow Values]
    stations: list[int]
            numbers pointing to the location of the stations in the list of stations.
            Values can be any number from 1 to number of stations in the data

    Returns
    -------
    pd.DataFrame:
        the mean square value of the data

    Example
    -------
    Calculate the Mean Square Error
        
    >>> import numpy as np
    >>> import pandas as pd
    >>> from postprocessinglib.evaluation import metrics
    >>> # Create your index as an array
    >>> index = np.array([1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990])
    >>> .
    >>> # Create a test dataframe
    >>> test_df = pd.DataFrame(data = np.random.rand(10, 4), columns = ("obs1", "sim1", "obs2", "sim2"), index = index)
    >>> print(test_df)
              obs1      sim1      obs2      sim2
    1981  0.869720  0.914777  0.701577  0.034410
    1982  0.126930  0.150236  0.217605  0.283580
    1983  0.082436  0.066993  0.281314  0.706240
    1984  0.865263  0.720315  0.445746  0.902906
    1985  0.042514  0.702998  0.451351  0.421407
    1986  0.400267  0.756454  0.084404  0.720665
    1987  0.352093  0.178805  0.197526  0.300795
    1988  0.154050  0.027170  0.020469  0.621782
    1989  0.153899  0.492885  0.870073  0.013124
    1990  0.255068  0.559826  0.244888  0.579176

    >>> # Generate the observed and simulated Dataframes
    >>> obs = test_df.iloc[:, [0, 2]]
    >>> sim = test_df.iloc[:, [1, 3]]
    >>> .
    >>> Calculate the Mean Square Error
    >>> mse = metrics.mse(observed = obs, simulated = sim)
    >>> print(mse)
                    model1
        Station 1  0.22170
        Station 2  0.08079

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-metrics.ipynb>`_

    """     
    if (isinstance(simulated, pd.DataFrame)):
        # If simulated is a single DataFrame, convert it to a list of DataFrames
        simulated = [simulated]
    
    # validate inputs
    for i in range(len(simulated)):
        if not isinstance(simulated[i], pd.DataFrame):
            raise ValueError(f"Simulated data at index {i} is not a DataFrame.")
        hlp.validate_data(observed, simulated[i])

    stations_to_process = [s - 1 for s in stations] if stations else list(range(observed.shape[1]))

    mse_results = {}

    for j in stations_to_process:
        valid_observed = hlp.filter_valid_data(observed, station_num=j)

        station_mse = {}
        for k, sim in enumerate(simulated):
            sim_values = sim.loc[valid_observed.index].iloc[:, j]
            obs_values = valid_observed.iloc[:, j]

            mse_val = np.mean((obs_values - sim_values) ** 2)
            station_mse[f"model{k+1}"] = hlp.sig_figs(mse_val, 4)

        # Store per-station result (station index is 1-based)
        mse_results[f"Station {j+1}"] = station_mse

    return pd.DataFrame(mse_results).T  # Transpose so stations are rows


def rmse(observed: pd.DataFrame, simulated: Union[pd.DataFrame, List[pd.DataFrame]], stations: list[int]=[]) -> float:
    """ Calculates the Root Mean Square value of the data

    Parameters
    ---------- 
    observed: pd.DataFrame
            Observed values[1: Datetime ; 2+: Streamflow Values]
    simulated: pd.DataFrame or list[pd.DataFrame]
            Simulated values[1: Datetime ; 2+: Streamflow Values]
    stations: list[int]
            numbers pointing to the location of the stations in the list of stations.
            Values can be any number from 1 to number of stations in the data

    Returns
    -------
    pd.DataFrame:
        the root mean square value of the data

    Example
    -------
    Calculate the Root Mean Square Error
        
    >>> import numpy as np
    >>> import pandas as pd
    >>> from postprocessinglib.evaluation import metrics
    >>> # Create your index as an array
    >>> index = np.array([1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990])
    >>> .
    >>> # Create a test dataframe
    >>> test_df = pd.DataFrame(data = np.random.rand(10, 4), columns = ("obs1", "sim1", "obs2", "sim2"), index = index)
    >>> print(test_df)
              obs1      sim1      obs2      sim2
    1981  0.966878  0.348580  0.053977  0.043133
    1982  0.188252  0.739990  0.941848  0.580866
    1983  0.430902  0.292824  0.963190  0.798885
    1984  0.718644  0.098746  0.031072  0.446317
    1985  0.586581  0.479616  0.541689  0.639898
    1986  0.380978  0.193639  0.737498  0.025509
    1987  0.072452  0.095210  0.188173  0.357554
    1988  0.833037  0.542694  0.913704  0.963027
    1989  0.434239  0.817284  0.425448  0.865841
    1990  0.698412  0.484796  0.693588  0.981778

    >>> # Generate the observed and simulated Dataframes
    >>> obs = test_df.iloc[:, [0, 2]]
    >>> sim = test_df.iloc[:, [1, 3]]
    >>> .
    >>> Calculate the Root Mean Square Error
    >>> rmse = metrics.rmse(observed = obs, simulated = sim)
    >>> print(rmse)
                   model1
        Station 1  0.3760
        Station 2  0.3398

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-metrics.ipynb>`_

    """   
    if (isinstance(simulated, pd.DataFrame)):
        # If simulated is a single DataFrame, convert it to a list of DataFrames
        simulated = [simulated]
    
    # validate inputs
    for i in range(len(simulated)):
        if not isinstance(simulated[i], pd.DataFrame):
            raise ValueError(f"Simulated data at index {i} is not a DataFrame.")
        hlp.validate_data(observed, simulated[i])

    stations_to_process = [s - 1 for s in stations] if stations else list(range(observed.shape[1]))

    rmse_results = {}    
    
    for j in stations_to_process:
        valid_observed = hlp.filter_valid_data(observed, station_num=j)

        station_rmse = {}
        for k, sim in enumerate(simulated):
            sim_values = sim.loc[valid_observed.index].iloc[:, j]
            obs_values = valid_observed.iloc[:, j]

            rmse_val = np.sqrt(np.mean((obs_values - sim_values) ** 2))
            station_rmse[f"model{k+1}"] = hlp.sig_figs(rmse_val, 4)

        # Store per-station result (station index is 1-based)
        rmse_results[f"Station {j+1}"] = station_rmse

    return pd.DataFrame(rmse_results).T  # Transpose so stations are rows


def mae(observed: pd.DataFrame, simulated: Union[pd.DataFrame, List[pd.DataFrame]], stations: list[int]=[]) -> float:
    """ Calculates the Mean Average value of the data

    Parameters
    ---------- 
    observed: pd.DataFrame
            Observed values[1: Datetime ; 2+: Streamflow Values]
    simulated: pd.DataFrame or list[pd.DataFrame]
            Simulated values[1: Datetime ; 2+: Streamflow Values]
    stations: list[int]
            numbers pointing to the location of the stations in the list of stations.
            Values can be any number from 1 to number of stations in the data

    Returns
    -------
    pd.DataFrame:
        the mean average value of the data

    Example
    -------
    Calculate the Mean Average Error
        
    >>> import numpy as np
    >>> import pandas as pd
    >>> from postprocessinglib.evaluation import metrics
    >>> # Create your index as an array
    >>> index = np.array([1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990])
    >>> .
    >>> # Create a test dataframe
    >>> test_df = pd.DataFrame(data = np.random.rand(10, 4), columns = ("obs1", "sim1", "obs2", "sim2"), index = index)
    >>> print(test_df)
              obs1      sim1      obs2      sim2
    1981  0.966878  0.348580  0.053977  0.043133
    1982  0.188252  0.739990  0.941848  0.580866
    1983  0.430902  0.292824  0.963190  0.798885
    1984  0.718644  0.098746  0.031072  0.446317
    1985  0.586581  0.479616  0.541689  0.639898
    1986  0.380978  0.193639  0.737498  0.025509
    1987  0.072452  0.095210  0.188173  0.357554
    1988  0.833037  0.542694  0.913704  0.963027
    1989  0.434239  0.817284  0.425448  0.865841
    1990  0.698412  0.484796  0.693588  0.981778

    >>> # Generate the observed and simulated Dataframes
    >>> obs = test_df.iloc[:, [0, 2]]
    >>> sim = test_df.iloc[:, [1, 3]]
    >>> .
    >>> Calculate the Mean Average Error
    >>> mae = metrics.mae(observed = obs, simulated = sim)
    >>> print(mae)
                   model1
        Station 1  0.3760
        Station 2  0.3398

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-metrics.ipynb>`_

    """
    if (isinstance(simulated, pd.DataFrame)):
        # If simulated is a single DataFrame, convert it to a list of DataFrames
        simulated = [simulated]
    
    # validate inputs
    for i in range(len(simulated)):
        if not isinstance(simulated[i], pd.DataFrame):
            raise ValueError(f"Simulated data at index {i} is not a DataFrame.")
        hlp.validate_data(observed, simulated[i])

    stations_to_process = [s - 1 for s in stations] if stations else list(range(observed.shape[1]))

    mae_results = {}    
    
    for j in stations_to_process:
        valid_observed = hlp.filter_valid_data(observed, station_num=j)

        station_mae = {}
        for k, sim in enumerate(simulated):
            sim_values = sim.loc[valid_observed.index].iloc[:, j]
            obs_values = valid_observed.iloc[:, j]

            mae_val = np.mean(np.sum(abs(obs_values - sim_values)))
            station_mae[f"model{k+1}"] = hlp.sig_figs(mae_val, 4)

        # Store per-station result (station index is 1-based)
        mae_results[f"Station {j+1}"] = station_mae

    return pd.DataFrame(mae_results).T  # Transpose so stations are rows


def nse(observed: pd.DataFrame, simulated: Union[pd.DataFrame, List[pd.DataFrame]], stations: list[int]=[]) -> float:
    """ Calculates the Nash-Sutcliffe Efficiency of the data

    Parameters
    ---------- 
    observed: pd.DataFrame
            Observed values[1: Datetime ; 2+: Streamflow Values]
    simulated: pd.DataFrame or list[pd.DataFrame]
            Simulated values[1: Datetime ; 2+: Streamflow Values]
    stations: list[int]
            numbers pointing to the location of the stations in the list of stations.
            Values can be any number from 1 to number of stations in the data

    Returns
    -------
    pd.DataFrame:
        the Nash-Sutcliffe Efficiency of the data

    Example
    -------
    Calculate the Nash-Sutcliffe Efficiency
        
    >>> import numpy as np
    >>> import pandas as pd
    >>> from postprocessinglib.evaluation import metrics
    >>> # Create your index as an array
    >>> index = np.array([1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990])
    >>> .
    >>> # Create a test dataframe
    >>> test_df = pd.DataFrame(data = np.random.rand(10, 4), columns = ("obs1", "sim1", "obs2", "sim2"), index = index)
    >>> print(test_df)
              obs1      sim1      obs2      sim2
    1981  0.966878  0.348580  0.053977  0.043133
    1982  0.188252  0.739990  0.941848  0.580866
    1983  0.430902  0.292824  0.963190  0.798885
    1984  0.718644  0.098746  0.031072  0.446317
    1985  0.586581  0.479616  0.541689  0.639898
    1986  0.380978  0.193639  0.737498  0.025509
    1987  0.072452  0.095210  0.188173  0.357554
    1988  0.833037  0.542694  0.913704  0.963027
    1989  0.434239  0.817284  0.425448  0.865841
    1990  0.698412  0.484796  0.693588  0.981778

    >>> # Generate the observed and simulated Dataframes
    >>> obs = test_df.iloc[:, [0, 2]]
    >>> sim = test_df.iloc[:, [1, 3]]
    >>> .
    >>> Calculate the Nash-Sutcliffe Efficiency 
    >>> nse = metrics.nse(observed = obs, simulated = sim)
    >>> print(nse)
                    model1
        Station 1  -0.9713
        Station 2  0.01669

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-metrics.ipynb>`_

    """       
    if (isinstance(simulated, pd.DataFrame)):
        # If simulated is a single DataFrame, convert it to a list of DataFrames
        simulated = [simulated]
    
    # validate inputs
    for i in range(len(simulated)):
        if not isinstance(simulated[i], pd.DataFrame):
            raise ValueError(f"Simulated data at index {i} is not a DataFrame.")
        hlp.validate_data(observed, simulated[i])

    stations_to_process = [s - 1 for s in stations] if stations else list(range(observed.shape[1]))

    nse_results = {}

    for j in stations_to_process:
        valid_observed = hlp.filter_valid_data(observed, station_num=j)
        obs_values = valid_observed.iloc[:, j]
        obs_mean = obs_values.mean()

        station_nse = {}
        for k, sim in enumerate(simulated):
            sim_values = sim.loc[valid_observed.index].iloc[:, j]
            numerator = np.sum((obs_values - sim_values) ** 2)
            denominator = np.sum((obs_values - obs_mean) ** 2)
            nse_val = 1 - (numerator / denominator)
            station_nse[f"model{k+1}"] = hlp.sig_figs(nse_val, 4)

        nse_results[f"Station {j+1}"] = station_nse

    return pd.DataFrame(nse_results).T


def lognse(observed: pd.DataFrame, simulated: Union[pd.DataFrame, List[pd.DataFrame]], stations: list[int]=[]) -> float:
    """ Calculates the Logarithmic Nash-Sutcliffe Efficiency of the data

    Parameters
    ---------- 
    observed: pd.DataFrame
            Observed values[1: Datetime ; 2+: Streamflow Values]
    simulated: pd.DataFrame or list[pd.DataFrame]
            Simulated values[1: Datetime ; 2+: Streamflow Values]
    stations: list[int]
            numbers pointing to the location of the stations in the list of stations.
            Values can be any number from 1 to number of stations in the data

    Returns
    -------
    pd.DataFrame:
        the Logarithmic Nash-Sutcliffe Efficiency of the data

    Example
    -------
    Calculate the Logarithmic Value of Nash-Sutcliffe Efficiency
        
    >>> import numpy as np
    >>> import pandas as pd
    >>> from postprocessinglib.evaluation import metrics
    >>> # Create your index as an array
    >>> index = np.array([1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990])
    >>> .
    >>> # Create a test dataframe
    >>> test_df = pd.DataFrame(data = np.random.rand(10, 4), columns = ("obs1", "sim1", "obs2", "sim2"), index = index)
    >>> print(test_df)
              obs1      sim1      obs2      sim2
    1981  0.966878  0.348580  0.053977  0.043133
    1982  0.188252  0.739990  0.941848  0.580866
    1983  0.430902  0.292824  0.963190  0.798885
    1984  0.718644  0.098746  0.031072  0.446317
    1985  0.586581  0.479616  0.541689  0.639898
    1986  0.380978  0.193639  0.737498  0.025509
    1987  0.072452  0.095210  0.188173  0.357554
    1988  0.833037  0.542694  0.913704  0.963027
    1989  0.434239  0.817284  0.425448  0.865841
    1990  0.698412  0.484796  0.693588  0.981778

    >>> # Generate the observed and simulated Dataframes
    >>> obs = test_df.iloc[:, [0, 2]]
    >>> sim = test_df.iloc[:, [1, 3]]
    >>> .
    >>> Calculate the Log of Nash-Sutcliffe Efficiency 
    >>> lognse = metrics.lognse(observed = obs, simulated = sim)
    >>> print(lognse)
                    model1
        Station 1  -0.4923
        Station 2  -0.4228

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-metrics.ipynb>`_

    """       
    if (isinstance(simulated, pd.DataFrame)):
        # If simulated is a single DataFrame, convert it to a list of DataFrames
        simulated = [simulated]
    
    # validate inputs
    for i in range(len(simulated)):
        if not isinstance(simulated[i], pd.DataFrame):
            raise ValueError(f"Simulated data at index {i} is not a DataFrame.")
        hlp.validate_data(observed, simulated[i])

    stations_to_process = [s - 1 for s in stations] if stations else list(range(observed.shape[1]))

    lognse_results = {}

    for j in stations_to_process:
        valid_observed = hlp.filter_valid_data(observed, station_num=j)
        obs_values = np.log(valid_observed.iloc[:, j])
        obs_mean = obs_values.mean()

        station_lognse = {}
        for k, sim in enumerate(simulated):
            sim_values = np.log(sim.loc[valid_observed.index].iloc[:, j])
            numerator = np.sum((obs_values - sim_values) ** 2)
            denominator = np.sum((obs_values - obs_mean) ** 2)
            lognse_val = 1 - (numerator / denominator)
            station_lognse[f"model{k+1}"] = hlp.sig_figs(lognse_val, 4)

        lognse_results[f"Station {j+1}"] = station_lognse

    return pd.DataFrame(lognse_results).T


def kge(observed: pd.DataFrame, simulated: Union[pd.DataFrame, List[pd.DataFrame]],
            stations: list[int]=[], scale: list[float]=[1. ,1. ,1.]) -> float:
    """ Calculates the Kling-Gupta Efficiency of the data

    Parameters
    ---------- 
    observed: pd.DataFrame
            Observed values[1: Datetime ; 2+: Streamflow Values]
    simulated: pd.DataFrame or list[pd.DataFrame]
            Simulated values[1: Datetime ; 2+: Streamflow Values]
    stations: list[int]
            numbers pointing to the location of the stations in the list of stations.
            Values can be any number from 1 to number of stations in the data
    scale: list[float, float, float]
            Scale factor for correlation[0], alpha[1], and beta[2] components 
            in the calculation of KGE

    Returns
    -------
    pd.DataFrame:
        the Kling-Gupta Efficiency of the data

    Example
    -------
    Calculate the Kling-Gupta Efficiency
        
    >>> import numpy as np
    >>> import pandas as pd
    >>> from postprocessinglib.evaluation import metrics
    >>> # Create your index as an array
    >>> index = np.array([1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990])
    >>> .
    >>> # Create a test dataframe
    >>> test_df = pd.DataFrame(data = np.random.rand(10, 4), columns = ("obs1", "sim1", "obs2", "sim2"), index = index)
    >>> print(test_df)
              obs1      sim1      obs2      sim2
    1981  0.966878  0.348580  0.053977  0.043133
    1982  0.188252  0.739990  0.941848  0.580866
    1983  0.430902  0.292824  0.963190  0.798885
    1984  0.718644  0.098746  0.031072  0.446317
    1985  0.586581  0.479616  0.541689  0.639898
    1986  0.380978  0.193639  0.737498  0.025509
    1987  0.072452  0.095210  0.188173  0.357554
    1988  0.833037  0.542694  0.913704  0.963027
    1989  0.434239  0.817284  0.425448  0.865841
    1990  0.698412  0.484796  0.693588  0.981778

    >>> # Generate the observed and simulated Dataframes
    >>> obs = test_df.iloc[:, [0, 2]]
    >>> sim = test_df.iloc[:, [1, 3]]
    >>> .
    >>> Calculate the Kling-Gupta Efficiency 
    >>> kge = metrics.kge(observed = obs, simulated = sim)
    >>> print(kge)
                    model1
        Station 1  -0.02108
        Station 2  -0.4929
    
    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-metrics.ipynb>`_

    """
    if isinstance(simulated, pd.DataFrame):
        simulated = [simulated]

    for i, sim in enumerate(simulated):
        if not isinstance(sim, pd.DataFrame):
            raise ValueError(f"Simulated data at index {i} is not a DataFrame.")
        hlp.validate_data(observed, sim)

    stations_to_process = [s - 1 for s in stations] if stations else list(range(observed.shape[1]))
    kge_results = {}

    for j in stations_to_process:
        valid_observed = hlp.filter_valid_data(observed, station_num=j)
        obs_values = valid_observed.iloc[:, j]
        mean_obs = obs_values.mean()
        std_obs = obs_values.std(ddof=1)

        station_kge = {}
        for k, sim in enumerate(simulated):
            sim_values = sim.loc[valid_observed.index].iloc[:, j]
            mean_sim = sim_values.mean()
            std_sim = sim_values.std(ddof=1)

            r_numerator = np.sum((obs_values - mean_obs) * (sim_values - mean_sim))
            r_denominator = np.sqrt(np.sum((obs_values - mean_obs) ** 2) * np.sum((sim_values - mean_sim) ** 2))
            r = r_numerator / r_denominator if r_denominator != 0 else np.nan

            b = mean_sim / mean_obs if mean_obs != 0 else np.nan
            a = std_sim / std_obs if std_obs != 0 else np.nan

            kge_val = 1 - np.sqrt(
                (scale[0] * (r - 1)) ** 2 +
                (scale[1] * (a - 1)) ** 2 +
                (scale[2] * (b - 1)) ** 2
            )
            station_kge[f"model{k+1}"] = hlp.sig_figs(kge_val, 4)

        kge_results[f"Station {j+1}"] = station_kge

    return pd.DataFrame(kge_results).T


def kge_2012(observed: pd.DataFrame, simulated: Union[pd.DataFrame, List[pd.DataFrame]],
             stations: list[int]=[], scale: list[float]=[1. ,1. ,1.]) -> float:
    """ Calculates the Kling-Gupta Efficiency of the data

    Parameters
    ---------- 
    observed: pd.DataFrame
            Observed values[1: Datetime ; 2+: Streamflow Values]
    simulated: pd.DataFrame or list[pd.DataFrame]
            Simulated values[1: Datetime ; 2+: Streamflow Values]
    stations: list[int]
            numbers pointing to the location of the stations in the list of stations.
            Values can be any number from 1 to number of stations in the data
    scale: list[float, float, float]
            Scale factor for correlation[0], alpha[1], and beta[2] components 
            in the calculation of KGE

    Returns
    -------
    pd.DataFrame:
        the Kling-Gupta Efficiency of the data

    Note
    ----
    This is different from the regular kge in that this uses the coefficient of Variation
    as its bias term (i.e., std/mean) as opposed to just the mean

    Example
    -------
    Calculate the Kling-Gupta Efficiency
        
    >>> import numpy as np
    >>> import pandas as pd
    >>> from postprocessinglib.evaluation import metrics
    >>> # Create your index as an array
    >>> index = np.array([1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990])
    >>> .
    >>> # Create a test dataframe
    >>> test_df = pd.DataFrame(data = np.random.rand(10, 4), columns = ("obs1", "sim1", "obs2", "sim2"), index = index)
    >>> print(test_df)
              obs1      sim1      obs2      sim2
    1981  0.966878  0.348580  0.053977  0.043133
    1982  0.188252  0.739990  0.941848  0.580866
    1983  0.430902  0.292824  0.963190  0.798885
    1984  0.718644  0.098746  0.031072  0.446317
    1985  0.586581  0.479616  0.541689  0.639898
    1986  0.380978  0.193639  0.737498  0.025509
    1987  0.072452  0.095210  0.188173  0.357554
    1988  0.833037  0.542694  0.913704  0.963027
    1989  0.434239  0.817284  0.425448  0.865841
    1990  0.698412  0.484796  0.693588  0.981778

    >>> # Generate the observed and simulated Dataframes
    >>> obs = test_df.iloc[:, [0, 2]]
    >>> sim = test_df.iloc[:, [1, 3]]
    >>> .
    >>> Calculate the Kling-Gupta Efficiency 
    >>> kge_2012 = metrics.kge_2012(observed = obs, simulated = sim)
    >>> print(kge_2012)
                    model1
        Station 1  -0.0210
        Station 2  0.48940

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-metrics.ipynb>`_

    """
    if isinstance(simulated, pd.DataFrame):
        simulated = [simulated]

    for i, sim in enumerate(simulated):
        if not isinstance(sim, pd.DataFrame):
            raise ValueError(f"Simulated data at index {i} is not a DataFrame.")
        hlp.validate_data(observed, sim)

    stations_to_process = [s - 1 for s in stations] if stations else list(range(observed.shape[1]))
    kge_results = {}

    for j in stations_to_process:
        valid_observed = hlp.filter_valid_data(observed, station_num=j)
        obs_values = valid_observed.iloc[:, j]
        mean_obs = obs_values.mean()
        std_obs = obs_values.std(ddof=1)

        station_kge = {}
        for k, sim in enumerate(simulated):
            sim_values = sim.loc[valid_observed.index].iloc[:, j]
            mean_sim = sim_values.mean()
            std_sim = sim_values.std(ddof=1)

            r_numerator = np.sum((obs_values - mean_obs) * (sim_values - mean_sim))
            r_denominator = np.sqrt(np.sum((obs_values - mean_obs) ** 2) * np.sum((sim_values - mean_sim) ** 2))
            r = r_numerator / r_denominator if r_denominator != 0 else np.nan

            b = mean_sim / mean_obs if mean_obs != 0 else np.nan
            a = (std_sim/mean_sim) / (std_obs/mean_obs) if (std_obs/mean_obs) != 0 else np.nan

            kge_val = 1 - np.sqrt(
                (scale[0] * (r - 1)) ** 2 +
                (scale[1] * (a - 1)) ** 2 +
                (scale[2] * (b - 1)) ** 2
            )
            station_kge[f"model{k+1}"] = hlp.sig_figs(kge_val, 4)

        kge_results[f"Station {j+1}"] = station_kge

    return pd.DataFrame(kge_results).T


def bias(observed: pd.DataFrame, simulated: Union[pd.DataFrame, List[pd.DataFrame]], stations: list[int]=[]) -> float:
    """ Calculates the Percentage Bias of the data

    Parameters
    ---------- 
    observed: pd.DataFrame
            Observed values[1: Datetime ; 2+: Streamflow Values]
    simulated: pd.DataFrame or list[pd.DataFrame]
            Simulated values[1: Datetime ; 2+: Streamflow Values]
    stations: list[int]
            numbers pointing to the location of the stations in the list of stations.
            Values can be any number from 1 to number of stations in the data

    Returns
    -------
    pd.DataFrame:
        the Percentage Bias of the data

    Example
    -------
    Calculate the Percentage Bias
    
    >>> import numpy as np
    >>> import pandas as pd
    >>> from postprocessinglib.evaluation import metrics
    >>> # Create your index as an array
    >>> index = np.array([1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990])
    >>> .
    >>> # Create a test dataframe
    >>> test_df = pd.DataFrame(data = np.random.rand(10, 4), columns = ("obs1", "sim1", "obs2", "sim2"), index = index)
    >>> print(test_df)
              obs1      sim1      obs2      sim2
    1981  0.966878  0.348580  0.053977  0.043133
    1982  0.188252  0.739990  0.941848  0.580866
    1983  0.430902  0.292824  0.963190  0.798885
    1984  0.718644  0.098746  0.031072  0.446317
    1985  0.586581  0.479616  0.541689  0.639898
    1986  0.380978  0.193639  0.737498  0.025509
    1987  0.072452  0.095210  0.188173  0.357554
    1988  0.833037  0.542694  0.913704  0.963027
    1989  0.434239  0.817284  0.425448  0.865841
    1990  0.698412  0.484796  0.693588  0.981778

    >>> # Generate the observed and simulated Dataframes
    >>> obs = test_df.iloc[:, [0, 2]]
    >>> sim = test_df.iloc[:, [1, 3]]
    >>> .
    >>> # Calculate the percentage Bias
    >>> bias = metrics.bias(observed = obs, simulated = sim)
    >>> print(bias)
                   model1
        Station 1  -22.92
        Station 2  3.873

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-metrics.ipynb>`_

    """    
    if isinstance(simulated, pd.DataFrame):
        simulated = [simulated]

    for i, sim in enumerate(simulated):
        if not isinstance(sim, pd.DataFrame):
            raise ValueError(f"Simulated data at index {i} is not a DataFrame.")
        hlp.validate_data(observed, sim)

    stations_to_process = [s - 1 for s in stations] if stations else list(range(observed.shape[1]))
    pbias_results = {}

    for j in stations_to_process:
        valid_observed = hlp.filter_valid_data(observed, station_num=j)
        obs_values = valid_observed.iloc[:, j]
        station_pbias = {}

        for k, sim in enumerate(simulated):
            sim_values = sim.loc[valid_observed.index].iloc[:, j]
            numerator = np.sum(sim_values - obs_values)
            denominator = np.sum(obs_values)

            pbias_val = 100 * numerator / denominator if denominator != 0 else np.nan
            station_pbias[f"model{k+1}"] = hlp.sig_figs(pbias_val, 4)

        pbias_results[f"Station {j+1}"] = station_pbias

    return pd.DataFrame(pbias_results).T
        

def time_to_peak(df: pd.DataFrame, stations: list[int]=[], use_jday:bool=False)->float:
    """ Calculates the time to peak of a given series of data whether observed 
        or simulated

    Parameters
    ----------
    df: pd.DataFrame
            the observed or simulated dataframe
    stations: list[int]
            numbers pointing to the location of the stations in the list of stations.
            Values can be any number from 1 to number of stations in the data
    use_jday : bool, default False
        If True, treats data as JDAY-style (fixed 366-day years).
        If False, uses datetime index logic.

    Returns
    -------
    pd.DataFrame
        DataFrame of the average time to peak value of the given data

    Example
    -------
    Calculation of the Time to Peak

    >>> from postprocessinglib.evaluation import metrics, data
    >>> path = 'MESH_output_streamflow_1.csv'
    >>> DATAFRAMES = data.generate_dataframes(csv_fpath=path, warm_up=365)
    >>> observed = DATAFRAMES["DF_OBSERVED"] 
    >>> simulated = DATAFRAMES["DF_SIMULATED"]
    >>> print(observed)
                QOMEAS_05BB001  QOMEAS_05BA001
    YEAR JDAY
    1980 366            10.20            NaN
    1981 1               9.85            NaN
         2              10.20            NaN
         3              10.00            NaN
         4              10.10            NaN
    ...                   ...             ...
    2017 361              NaN            NaN
         362              NaN            NaN
         363              NaN            NaN
         364              NaN            NaN
         365              NaN            NaN
    >>> .
    >>> print(simulated)
           QOSIM_05BB001  QOSIM_05BA001
    YEAR JDAY
    1980 366        2.530770       1.006860
    1981 1          2.518999       1.001954
         2          2.507289       0.997078
         3          2.495637       0.992233
         4          2.484073       0.987417
    ...                  ...            ...
    2017 361        4.418050       1.380227
         362        4.393084       1.372171
         363        4.368303       1.364174
         364        4.343699       1.356237
         365        4.319275       1.348359

    >>> # Calculating the time to peak
    >>> ttp = metrics.time_to_peak(df=observed)
    >>> print(ttp)
                      ttp
        Station 1   171.0
        Station 2   177.0  

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-metrics.ipynb>`_ 
    
    """

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    
    if not stations:
        stations = df.columns.tolist()

    results = {}
    if use_jday:
        # For JDAY-style: fixed 366-day blocks
        n = df.shape[0]
        days_per_year = 366

        for station in stations:
            data = df[station].values
            tpd = 0.0
            ycount = 0

            for i in range(0, n, days_per_year):
                j = i + days_per_year
                if j > n:
                    break  # Incomplete final year; skip

                year_chunk = data[i:j]

                if np.sum(year_chunk) > 0.0 and (j - i) > 200:
                    peak_day = np.argmax(year_chunk) + 1
                    tpd += peak_day
                    ycount += 1

            avg_ttp = tpd / ycount if ycount > 0 else np.nan
            results[station] = round(avg_ttp, 3)

    else:
        last_year = df.index[-1].year
        for station in stations:
            station_data = df[station]
            year = df.index[0].year
            start = 0
            yearly_peaks = []

            while year < last_year:
                num_days = 366 if hlp.is_leap_year(year) else 365
                valid_days = np.sum(np.fromiter((df.index[i].year == year for i in range(start, start + num_days)), int))

                if valid_days > 200:
                    data = station_data.iloc[start:start + valid_days]
                    if np.nansum(data) > 0:
                        peak_day = np.nanargmax(data.values) + 1
                        yearly_peaks.append(peak_day)

                start += valid_days
                year += 1

            avg_peak = np.mean(yearly_peaks) if yearly_peaks else np.nan
            results[station] = hlp.sig_figs(avg_peak, 3)

    df_out = pd.DataFrame.from_dict(results, orient='index', columns=['ttp'])
    # Rename the index to be more descriptive and to match with the other metrics
    df_out.index = [f"Station {i}" for i in df_out.index.str.extract('(\d+)').astype(int)[0]]
    df_out.index.name = "Station"
    return df_out


def time_to_centre_of_mass(df: pd.DataFrame, stations: list[int]=[], use_jday:bool=False)->float:
    """ Calculates the time it takes to obtain 50% of the stream flow in a given year

    Parameters
    ----------
    df: pd.DataFrame
            the observed or simulated dataframe
    stations: list[int]
            numbers pointing to the location of the stations in the list of stations.
            Values can be any number from 1 to number of stations in the data
    use_jday : bool, default False
        If True, treats data as JDAY-style (fixed 366-day years).
        If False, uses datetime index logic.

    Returns
    -------
   pd.DataFrame
        Dataframe containing the average time to the centre of mass for the stations

    Example
    -------
    Calculation of the time to center of mass

    >>> from postprocessinglib.evaluation import metrics, data
    >>> path = 'MESH_output_streamflow_1.csv'
    >>> DATAFRAMES = data.generate_dataframes(csv_fpath=path, warm_up=365)
    >>> observed = DATAFRAMES["DF_OBSERVED"] 
    >>> simulated = DATAFRAMES["DF_SIMULATED"]
    >>> print(observed)
                QOMEAS_05BB001  QOMEAS_05BA001
    YEAR JDAY
    1980 366            10.20            NaN
    1981 1               9.85            NaN
         2              10.20            NaN
         3              10.00            NaN
         4              10.10            NaN
    ...                   ...             ...
    2017 361              NaN            NaN
         362              NaN            NaN
         363              NaN            NaN
         364              NaN            NaN
         365              NaN            NaN
    >>> .
    >>> print(simulated)
           QOSIM_05BB001  QOSIM_05BA001
    YEAR JDAY
    1980 366        2.530770       1.006860
    1981 1          2.518999       1.001954
         2          2.507289       0.997078
         3          2.495637       0.992233
         4          2.484073       0.987417
    ...                  ...            ...
    2017 361        4.418050       1.380227
         362        4.393084       1.372171
         363        4.368303       1.364174
         364        4.343699       1.356237
         365        4.319275       1.348359

    >>> # Calculating the time to center of mass
    >>> ttcom = metrics.time_to_centre_of_mass(df=observed)
    >>> print(ttcom)
                    ttcom
        Station 1   185.0
        Station 2   166.0

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-metrics.ipynb>`_

    """
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    
    if not stations:
        stations = df.columns.tolist()

    results = {}
    if use_jday:
        n = df.shape[0]
        days_per_year = 366

        for station in stations:
            data = df[station].values
            total = 0.0
            count = 0

            for i in range(0, n, days_per_year):
                j = i + days_per_year
                if j > n:
                    break

                year_data = data[i:j]
                if np.nansum(year_data) > 0 and (j - i) > 200:
                    days = np.arange(1, days_per_year + 1)
                    com = np.sum(days * year_data) / np.nansum(year_data)
                    total += com
                    count += 1

            avg_com = total / count if count > 0 else np.nan
            results[station] = round(avg_com, 3)

    else:
        last_year = df.index[-1].year
        for station in stations:
            station_data = df[station]
            year = df.index[0].year
            start = 0
            yearly_com = []

            while year < last_year:
                num_days = 366 if hlp.is_leap_year(year) else 365
                valid_days = np.sum(np.fromiter((df.index[i].year == year for i in range(start, start + num_days)), int))

                if valid_days > 200:
                    data = station_data.iloc[start:start + valid_days]
                    if np.nansum(data) > 0:
                        days = np.arange(1, valid_days + 1)
                        com = np.sum(days * data.values) / np.nansum(data)
                        yearly_com.append(com)

                start += valid_days
                year += 1

            avg_com = np.mean(yearly_com) if yearly_com else np.nan
            results[station] = hlp.sig_figs(avg_com, 3)

    df_out = pd.DataFrame.from_dict(results, orient='index', columns=['ttcom'])
    # Rename the index to be more descriptive and to match with the other metrics
    df_out.index = [f"Station {i}" for i in df_out.index.str.extract('(\d+)').astype(int)[0]]
    df_out.index.name = "Station"
    return df_out



def SpringPulseOnset(df: pd.DataFrame, stations: list[int]=[], use_jday:bool=False)->int:
    """ Calculates the average day of year when the spring pulse (snowmelt) begins for each station.

    Parameters
    ----------
    df : pd.DataFrame
        Observed or simulated streamflow data with a MultiIndex (YEAR, JDAY).
    stations : list[int], optional
        List of 1-indexed station numbers to evaluate. If empty, all stations are used.
    use_jday : bool, default False
        If True, treats data as JDAY-style (fixed 366-day years).
        If False, uses datetime index logic.

    Returns
    -------
    pd.DataFrame
        DataFrame with station indices and average spring pulse onset day.

    Example
    -------
    Calculation of the SpringPulseOnset

    >>> from postprocessinglib.evaluation import metrics, data
    >>> path = 'MESH_output_streamflow_1.csv'
    >>> DATAFRAMES = data.generate_dataframes(csv_fpath=path, warm_up=365)
    >>> observed = DATAFRAMES["DF_OBSERVED"] 
    >>> simulated = DATAFRAMES["DF_SIMULATED"]
    >>> print(observed)
                QOMEAS_05BB001  QOMEAS_05BA001
    YEAR JDAY
    1980 366            10.20            NaN
    1981 1               9.85            NaN
         2              10.20            NaN
         3              10.00            NaN
         4              10.10            NaN
    ...                   ...             ...
    2017 361              NaN            NaN
         362              NaN            NaN
         363              NaN            NaN
         364              NaN            NaN
         365              NaN            NaN
    >>> .
    >>> print(simulated)
           QOSIM_05BB001  QOSIM_05BA001
    YEAR JDAY
    1980 366        2.530770       1.006860
    1981 1          2.518999       1.001954
         2          2.507289       0.997078
         3          2.495637       0.992233
         4          2.484073       0.987417
    ...                  ...            ...
    2017 361        4.418050       1.380227
         362        4.393084       1.372171
         363        4.368303       1.364174
         364        4.343699       1.356237
         365        4.319275       1.348359

    >>> # Calculating the spring pulse onset day
    >>> spod = metrics.SpringPulseOnset(df=simulated)
    >>> print(spod)
        [136, 143]

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-metrics.ipynb>`_

    """
    if not stations:
        stations = list(range(1, df.shape[1] + 1))  # 1-based indexing

    results = []

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    if use_jday:
        # Expecting a flat 366-day-per-year layout
        days_per_year = 366
        num_years = df.shape[0] // days_per_year

        for j in stations:
            col_idx = j - 1
            station_col = df.columns[col_idx]
            data = df[station_col].values
            yearly_spod = []

            for year in range(num_years):
                start = year * days_per_year
                end = start + days_per_year
                year_data = data[start:end]

                if len(year_data) > 200 and np.nansum(year_data) > 0:
                    valid = ~np.isnan(year_data)
                    mean_val = np.nanmean(year_data)
                    cumulative = 0.0
                    min_cumulative = float("inf")
                    onset_day = 0

                    for idx, val in enumerate(year_data, start=1):
                        if np.isnan(val):
                            continue
                        cumulative += val - mean_val
                        if cumulative < min_cumulative:
                            min_cumulative = cumulative
                            onset_day = idx

                    yearly_spod.append(onset_day)

            avg_spod = hlp.sig_figs(np.mean(yearly_spod), 3) if yearly_spod else np.nan
            results.append({"Station": station_col, "SPOD": avg_spod})    
    else:
        # Handle MultiIndex (YEAR, JDAY) or DatetimeIndex
        years = df.index.get_level_values(0).unique() if isinstance(df.index, pd.MultiIndex) else df.index.year.unique()

        for j in stations:
            col_idx = j - 1
            station_col = df.columns[col_idx]
            yearly_spod = []

            for year in years:
                try:
                    # Get the year's data
                    data_slice = df.loc[year, station_col] if isinstance(df.index, pd.MultiIndex) else df[df.index.year == year][station_col]
                except KeyError:
                    continue

                data_slice = data_slice.dropna()

                if len(data_slice) > 200 and data_slice.sum() > 0:
                    mean_val = data_slice.mean()
                    cumulative = 0.0
                    min_cumulative = float("inf")
                    onset_day = 0

                    for idx, val in enumerate(data_slice, start=1):
                        cumulative += val - mean_val
                        if cumulative < min_cumulative:
                            min_cumulative = cumulative
                            onset_day = idx

                    yearly_spod.append(onset_day)

            if yearly_spod:
                spod = hlp.sig_figs(np.mean(yearly_spod), 3)
            else:
                spod = np.nan

            # results[f"Station {j}"] = spod  # Store in results dictionary with station name as key
            results.append({"Station": f"Station {j}", "SPOD": spod})  
            

    # Create DataFrame from results dictionary and set the index as 'Station'
    df_out = pd.DataFrame(results, columns=['Station', 'SPOD'])
    df_out.set_index('Station', inplace=True)
    
    return df_out


def slope_fdc(df: pd.DataFrame, percentiles: tuple[float, float] = [33, 66], stations: list[int] = []) -> list[float]:
    """
    Calculates the slope of the Flow Duration Curve (FDC).

    Parameters
    ----------
    df: pd.DataFrame
        Streamflow values for calculating the FDC slope. Each column corresponds to a station.
    percentiles: tuple[float, float]
        Percentiles used for slope calculation (e.g., (33, 66) for 33rd and 66th percentiles).
    stations: list[int]
        List of station indices (1-indexed) for which to calculate the slope. If empty, all stations are processed.

    Returns
    -------
    pd.DataFrame
        DataFrame with the slope of the FDC for each station.

    Example
    -------
    >>> import pandas as pd
    >>> from postprocessinglib.evaluation import metrics
    >>> from metrics import slope_fdc
    >>> data = pd.DataFrame({
    >>>     "Station1": [1.2, 0.8, 0.6, 0.4, 0.2],
    >>>     "Station2": [2.0, 1.5, 1.0, 0.5, 0.2]
    >>> })
    >>> slope_fdc(df=data, percentiles=(33, 66))
                    fdc_Slope
        Station 1      3.1566
        Station 2      2.3474
    """
    if not stations:
        stations = list(range(1, df.shape[1] + 1))

    results = {}
    for station in stations:
        col = df.iloc[:, station - 1]
        q_low = col.quantile(percentiles[0] / 100)
        q_high = col.quantile(percentiles[1] / 100)
        slope = (np.log(q_high) - np.log(q_low)) / ((percentiles[1] - percentiles[0]) / 100)
        results[station] = round(slope, 4)

    df_out = pd.DataFrame.from_dict(results, orient='index', columns=['fdc_Slope']).sort_index()
    # Rename the index to be more descriptive and to match with the other metrics
    df_out.index = [f"Station {i}" for i in df_out.index]
    return df_out


def calculate_all_metrics(observed: pd.DataFrame, simulated: Union[pd.DataFrame, List[pd.DataFrame]], stations: list[int]=[],
                          format: str="", out: str='metrics_out') -> dict[str, float]:
    """Calculate all metrics.

    Parameters
    ---------- 
    observed: pd.DataFrame
            Observed values[1: Datetime ; 2+: Streamflow Values]
    simulated: pd.DataFrame or list[pd.DataFrame]
            Simulated values[1: Datetime ; 2+: Streamflow Values]
    stations: list[int]
            numbers pointing to the location of the stations in the list of stations.
            Values can be any number from 1 to number of stations in the data
    format: str
            used to indicate that you want the output to be saved to a output file who's
            name is specified by the 'out' parameter
    out: str
            used in tandem with the 'format' parameter to specify the name of the output file.
            it is 'metrics_out.{format}' by default


    Returns
    -------
    pd.DataFrame
        DataFrame containing every metric that can be evaluated and
            its result

    Example
    -------
    Calculation of all available metrics

    >>> from postprocessinglib.evaluation import metrics, data
    >>> path = 'MESH_output_streamflow_1.csv'
    >>> DATAFRAMES = data.generate_dataframes(csv_fpath=path, warm_up=365)
    >>> print(metrics.calculate_all_metrics(observed=DATAFRAMES["DF_OBSERVED"], simulated=DATAFRAMES["DF_SIMULATED"]))
                    MSE	        RMSE	MAE	        NSE	       NegNSE	LogNSE	NegLogNSE	KGE	NegKGE	KGE 2012	BIAS	AbsBIAS	TTP_obs	TTCoM_obs	SPOD_obs	TTP_sim_model1	TTCoM_sim_model1	SPOD_sim_model1
                    model1	    model1	model1	    model1	   model1	model1	model1	model1	model1	model1	model1	model1	ttp	ttcom	SPOD	ttp	ttcom	SPOD
        Station 1	1299.000	36.050	209200.0	0.51660	  -0.51660	-0.25110	0.25110	0.50940	-0.50940	0.56060	34.160	34.160	157.0	NaN	113.0	171.0	185.0	128.0
        Station 2	780.600	    27.940	29480.0	    -1.67500   1.67500	-0.16920	0.16920	-0.11130	0.11130	0.08006	-11.500	11.500	157.0	NaN	NaN	177.0	166.0	115.0

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-metrics.ipynb>`_
            
    """
    if isinstance(simulated, pd.DataFrame):
        simulated = [simulated]
    parameters = (observed, simulated, stations)

    # Define all metric functions and any tweaks
    metric_funcs = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "NSE": nse,
        "NegNSE": lambda *args: -nse(*args),
        "LogNSE": lognse,
        "NegLogNSE": lambda *args: -lognse(*args),
        "KGE": kge,
        "NegKGE": lambda *args: -kge(*args),
        "KGE 2012": kge_2012,
        "BIAS": bias,
        "AbsBIAS": lambda *args: bias(*args).abs()
    }
    
    metric_dfs = []

    for name, func in metric_funcs.items():
        df = func(*parameters)  # Expecting a DataFrame with index=station, columns=model1, model2...
        df.columns = pd.MultiIndex.from_product([[name], df.columns])  # e.g., ('KGE', 'model1')
        metric_dfs.append(df)
    
    # Observed-only single-DF metrics
    single_obs_metrics = {
        "TTP_obs": time_to_peak,
        "TTCoM_obs": time_to_centre_of_mass,
        "SPOD_obs": SpringPulseOnset,
        # "FDC_Slope_obs": slope_fdc,
    }

    for name, func in single_obs_metrics.items():
        df = func(observed, stations)  # Each returns a single DF
        df.columns = pd.MultiIndex.from_product([[name], df.columns])
        metric_dfs.append(df)

    # Per-model single-DF metrics
    for idx, sim_df in enumerate(simulated):
        model_name = f"model{idx+1}"

        for prefix, func in single_obs_metrics.items():
            name = f"{prefix.replace('_obs', f'_sim_{model_name}')}"
            df = func(sim_df, stations)
            df.columns = pd.MultiIndex.from_product([[name], df.columns])
            metric_dfs.append(df)

    final_df = pd.concat(metric_dfs, axis=1)

    # Check for a specified format, else print to screen
    if format:
        if format == "txt":
            final_df_txt = final_df.to_csv(sep='\t', index=False, lineterminator='\n')
            lines = final_df_txt.split('\n')
            columns = [line.split('\t') for line in lines if line]
            formatted_lines = []
            for line in columns:
                formatted_line = ''.join(f'{col:<{12}}' for col in line)
                formatted_lines.append(formatted_line)            
            # Join the formatted lines into a single string
            formatted_str = '\n'.join(formatted_lines)
            with open(out+"."+format, "w") as file:
                file.write(formatted_str)
            file.close()
            print(f"See {out}.{format} file in directory")
        elif format == "csv":
            final_df.to_csv(f"{out}.{format}")
            print(f"See {out}.{format} file in directory")            
        else:
            print("unknown or uncoded format - " + format)
    else:
        return final_df


def calculate_metrics(observed: pd.DataFrame, simulated: Union[pd.DataFrame, List[pd.DataFrame]], metrices: list[str],
                      stations: list[int]=[], format: str="", out: str='metrics_out') -> dict[str, float]:
    """Calculate the requested metrics.

    Parameters
    ---------- 
    observed: pd.DataFrame
            Observed values[1: Datetime ; 2+: Streamflow Values]
    simulated: pd.DataFrame
            Simulated values[1: Datetime ; 2+: Streamflow Values]
    stations: list[int]
            numbers pointing to the location of the stations in the list of stations.
            Values can be any number from 1 to number of stations in the data
    metrices: List[str]
            List of metrics to be calculated
    format: str
            used to indicate that you want the output to be saved to a output file who's
            name is specified by the 'out' parameter
    out: str
            used in tandem with the 'format' parameter to specify the name of the output file.
            it is 'metrics_out.{format}' by default

    Returns
    -------
    pd.DataFrame
        Dataframe containing each metric to be evaluated and its result 

    Example
    -------
    Calculation of a list of metrics

    >>> from postprocessinglib.evaluation import metrics, data
    >>> path = 'MESH_output_streamflow_1.csv'
    >>> DATAFRAMES = data.generate_dataframes(csv_fpath=path, warm_up=365)
    >>> list_of_metrices = ["MSE", "NSE", "KGE 2012"]
    >>> print(metrics.calculate_metrics(observed=DATAFRAMES["DF_OBSERVED"], simulated=DATAFRAMES["DF_SIMULATED"], metrices=list_of_metrics))
        {'MSE': [1890, 665.9], 'NSE': [0.09948, -3.583], 'KGE 2012': [0.3130, -0.1483]} 

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-metrics.ipynb>`_
               
    """
    # validate inputs
    if isinstance(simulated, pd.DataFrame):
        simulated = [simulated]
    
    parameters_list = [(observed, sim_df, stations) for sim_df in simulated]

    # Mapping of metrics to their functions
    metric_funcs = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "nse": nse,
        "negnse": lambda obs, sim, st: -nse(obs, sim, st),
        "lognse": lognse,
        "neglognse": lambda obs, sim, st: -lognse(obs, sim, st),
        "kge": kge,
        "negkge": lambda obs, sim, st: -kge(obs, sim, st),
        "kge 2012": kge_2012,
        "bias": bias,
        "absbias": lambda obs, sim, st: bias(obs, sim, st).abs(),
    }

    # Single-input metrics
    single_input_metrics = {
        "ttp_obs": lambda: time_to_peak(observed, stations),
        "ttcom_obs": lambda: time_to_centre_of_mass(observed, stations),
        "spod_obs": lambda: SpringPulseOnset(observed, stations),
        "fdc_obs": lambda: slope_fdc(observed, stations),

        "ttp_sim": lambda sim: time_to_peak(sim, stations),
        "ttcom_sim": lambda sim: time_to_centre_of_mass(sim, stations),
        "spod_sim": lambda sim: SpringPulseOnset(sim, stations),
        "fdc_sim": lambda sim: slope_fdc(sim, stations),
    }

    metric_dfs = []

    for metric in metrices:
        metric_lower = metric.lower()

        # Sim-obs comparison metrics (computed per model)
        if metric_lower in metric_funcs:
            for idx, (obs, sim, st) in enumerate(parameters_list):
                result = metric_funcs[metric_lower](obs, sim, st)
                model_name = f"model{idx+1}"
                df = result.copy()
                df.columns = pd.MultiIndex.from_product([[metric.upper()], [model_name]])
                metric_dfs.append(df)

        # Obs-only metrics
        elif metric_lower in ["ttp_obs", "ttcom_obs", "spod_obs", "fdc_obs"]:
            result = single_input_metrics[metric_lower]()
            df = result.copy()
            df.columns = pd.MultiIndex.from_product([[metric.upper()], result.columns])
            metric_dfs.append(df)

        # Sim-only metrics (computed per model)
        elif metric_lower in ["ttp_sim", "ttcom_sim", "spod_sim", "fdc_sim"]:
            for idx, sim in enumerate(simulated):
                result = single_input_metrics[metric_lower](sim)
                model_name = f"model{idx+1}"
                df = result.copy()
                df.columns = pd.MultiIndex.from_product([[f"{metric.upper()}_{model_name}"], result.columns])
                metric_dfs.append(df)

        else:
            raise RuntimeError(f"Unknown metric '{metric}'")

    final_df = pd.concat(metric_dfs, axis=1)

    if format:
        if format == "txt":
            txt_data = final_df.to_csv(sep='\t', index=False, lineterminator='\n')
            lines = txt_data.split('\n')
            columns = [line.split('\t') for line in lines if line]
            formatted_lines = [''.join(f'{col:<{12}}' for col in row) for row in columns]
            with open(f"{out}.{format}", "w") as file:
                file.write('\n'.join(formatted_lines))
            print(f"See {out}.{format} file in directory")

        elif format == "csv":
            final_df.to_csv(f"{out}.{format}", index=False)
            print(f"See {out}.{format} file in directory")

        else:
            print("Unknown or unsupported format:", format)

    else:
        return final_df
