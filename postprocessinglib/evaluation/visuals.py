"""
The visual module contains different plotting functions for time series visualization.
It allows users to plot hydrographs per station for each stations to allow us visualize
the time-series data. These graphs provide a simple and clear way to immeditely identify
patterns and discrepancies with model operation.
They are also made to very customizable with a lot of options to suit the need of many types
of users.   
Some of them also allow their metrics to be placed beside the plots as shown below:

.. image:: Figures/Visuals.png
  :alt: graphs showing graph types
.. image:: Figures/Visuals_m.png
  :alt: line plot showing metrics

"""

from typing import Union, List, Tuple
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
from shapely.geometry import Point
import matplotlib.colors as mc
import colorsys

from postprocessinglib.evaluation import metrics
from postprocessinglib.utilities import _helper_functions as hlp

def _parse_linestyle(linestyle):
    """
    Parses a linestyle string to extract color and style.
    Handles both named colors and RGB tuples as well as style symbols and words.
    """
    STYLE_MAP = {
        '--': '--',
        '-.': '-.',
        '-': '-',
        ':': ':',
        'dotted': ':',
        'dashed': '--',
        'solid': '-',
        '.':'.'
    }

    # Sort styles by length descending to match longer ones first (e.g., '--' before '-')
    sorted_styles = sorted(STYLE_MAP.keys(), key=len, reverse=True)
    for style_key in sorted_styles:
        if linestyle.endswith(style_key):
            color_part = linestyle[:-len(style_key)].strip()
            style = STYLE_MAP[style_key]

            if color_part.startswith('('):  # RGB tuple as string
                from ast import literal_eval
                try:
                    color = literal_eval(color_part)
                except Exception:
                    raise ValueError(f"Invalid RGB color format: {color_part}")
            else:
                color = color_part if color_part else None  # allow default

            return color, style

    # If no line style is found
    return linestyle, '-'  # Default to solid line

def _parse_markerstyle(markerstyle: str):
    """
    Parses a markerstyle string like 'r.' or '(0.2, 0.4, 0.6, 1.0)^'
    into a (color, marker) tuple.

    Returns:
        (color, marker)
    """
    valid_markers = {'.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4',
                     's', 'p', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_'}

    # Try each possible marker at the end of the string
    for marker in sorted(valid_markers, key=len, reverse=True):
        if markerstyle.endswith(marker):
            color_part = markerstyle[:-len(marker)].strip()

            # Handle RGB tuples
            if color_part.startswith('('):
                from ast import literal_eval
                try:
                    color = literal_eval(color_part)
                except Exception:
                    raise ValueError(f"Invalid RGB color format: {color_part}")
            else:
                color = color_part if color_part else None

            return color, marker

    raise ValueError(f"Invalid markerstyle: '{markerstyle}'. No valid marker found.")


def _save_or_display_plot(fig, save: bool, save_as: Union[str, List[str]], dir: str, i: int, type: str):
    """
    Save the plot to a file or display it based on user preferences.

    This helper function determines whether to save the plot to a specified directory or display 
    it on the screen. If saving, the plot is saved as a PNG file with a specified or default filename.
    If not saving, the plot is displayed interactively using Matplotlib's `plt.show()`.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The Matplotlib figure instance to be saved or displayed.

    save : bool
        Whether to save the plot to a file. If `True`, the plot is saved. If `False`, the plot is displayed.

    save_as : str or list of str
        The name or list of names to save the plot as. If provided, the plot is saved with this name(s). 
        If `save_as` is a list, the plot is saved using the corresponding name for each figure. Default is None.

    dir : str
        The directory where the plot will be saved. If the directory does not exist, it will be created. 
        Default is the current working directory.

    i : int
        The index for generating unique filenames when saving multiple plots. Used when `save_as` is a list.

    type : str
        The type of the plot (e.g., 'scatter-plot'). This is used to generate a default filename if 
        `save_as` is not provided.

    Returns
    -------
    None
        This function does not return anything. It either saves or displays the plot.

    Example
    -------
    Save a plot with a custom filename:

    >>> fig = plt.figure()
    >>> # Plotting code...
    >>> _save_or_display_plot(fig, save=True, save_as="my_plot", dir="./plots", i=0, type="scatter")

    Display a plot:

    >>> _save_or_display_plot(fig, save=False, save_as="my_plot", dir="./plots", i=0, type="scatter")

    Notes
    -----
    - If `save_as` is a string, the plot will be saved with that name.
    - If `save_as` is a list, the plot will be saved with the corresponding name from the list, 
      using the index `i` to select the correct filename.
    - The `plt.tight_layout()` function is called to ensure the plot layout is adjusted before saving.

    """
    if save:
        plt.tight_layout()
        if not os.path.exists(dir):
            os.makedirs(dir)
        if isinstance(save_as, str):
            save_as = [save_as]
        filename = f"{save_as[i]}.png" if save_as and i < len(save_as) else f"{type}_{i + 1}.png"
        fig.savefig(os.path.join(dir, filename), bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def _normalize_bounds(bounds: Union[pd.DataFrame, List[pd.DataFrame], List[List[pd.DataFrame]]], lines=None):
    """
    Normalize bounds input into a consistent list-of-lists structure.

    Parameters
    ----------
    bounds : Union[pd.DataFrame, List[pd.DataFrame], List[List[pd.DataFrame]]]
        The bounds input can be:
            - None
            - A single DataFrame
            - A list of DataFrames
            - A list of lists of DataFrames

    lines : Optional[List[pd.DataFrame]]
        The list of lines. Used to determine if bounds should be aligned one-to-one with lines.

    Returns
    -------
    List[List[pd.DataFrame]]
        A nested list of bounds per line.

    Raises
    ------
    ValueError:
        If the input format is invalid or if bounds are ambiguous with respect to lines.
    """
    if bounds is None:
        return None

    if isinstance(bounds, pd.DataFrame):
        return [[bounds]]

    if isinstance(bounds, list):
        if all(isinstance(b, pd.DataFrame) for b in bounds):
            # Check if number of bounds matches number of lines (if lines are provided)
            if lines and len(lines) == len(bounds):
                return [[b] for b in bounds]  # Each bound belongs to one line
            elif not lines or len(lines) == 1:
                return [bounds]  # Multiple bounds for single line
            else:
                raise ValueError("Ambiguous bounds list: wrap them in sublists to indicate grouping per line.")
        
        if all(isinstance(b, list) and all(isinstance(df, pd.DataFrame) for df in b) for b in bounds):
            return bounds  # Already normalized

    raise ValueError("Bounds must be a DataFrame, a list of DataFrames, or a list of lists of DataFrames.")

def _prepare_bounds(bounds, line_index, column_index):
    """
    Extracts the column-wise bounds for a specific line and column.

    Parameters
    ----------
    bounds: Union[List[pd.DataFrame], List[List[pd.DataFrame]]] 
            Nested list of DataFrames, structured as List[List[pd.DataFrame]]
    line_index: int
            Index of the current line in `lines`
    column_index: int
             Index of the current column within that line's DataFrame

    Returns:
    List[pd.Series]:
            A list of Series objects representing the bounds for the specified line and column.
    """
    if bounds is None:
        return []
    line_bounds = bounds[line_index]  # List of DataFrames for this line
    return [b.iloc[:, column_index] for b in line_bounds]

def _finalize_plot(ax, grid, minor_grid, font_size, labels, title, name, i):
    """
    Finalizes the plot by setting labels, title, and grid options.
    """
    plt.legend(loc='best')

    plt.tight_layout()    
    plt.xticks(fontsize=font_size, rotation=45)
    plt.yticks(fontsize=font_size)

    if labels:
        plt.xlabel(labels[0], fontsize=font_size)
        plt.ylabel(labels[1], fontsize=font_size)

    if title:
        title_dict = {'family': 'sans-serif', 'color': 'black', 'weight': 'normal', 'size': font_size}
        if isinstance(title, list):
            ax.set_title(title[i] if i < len(title) else f"{name}_{i + 1}", fontdict=title_dict, pad=15)
        elif isinstance(title, str):
            ax.set_title(label=title, fontdict=title_dict, pad=15)

    if grid:
        plt.grid(True, which='major', linestyle='--') 

    if minor_grid:
        plt.minorticks_on()
        ax.grid(which='minor', linewidth='0.5', color='gray', linestyle='--')
        


def plot(
    merged_df: pd.DataFrame = None, 
    df: pd.DataFrame = None, 
    sim_df: pd.DataFrame = None,
    step: bool = False,
    where: str = 'pre',
    legend: tuple[str, str] = ('Data',), 
    metrices: list[str] = None,
    mode:str = 'median',
    models:List[str] = None,
    grid: bool = False,
    minor_grid: bool = False, 
    title: str = None, 
    labels: list[str] = None, 
    padding: bool = False,
    linestyles: tuple[str, str] = ('r-',), 
    linewidth: tuple[float, float] = (1.5,),
    fig_size: tuple[float, float] = (10, 6), 
    font_size: int = 12,
    text_size: int = 12,
    metrics_adjust: tuple[float, float] = (1.05, 0.5),
    plot_adjust: float = 0.2,
    save: bool = False, 
    save_as: str = None, 
    dir: str = os.getcwd()
    ) -> plt.figure:
    """ Create a comparison time series line plot of simulated and optionally,  observed time series data

    This function generates line plots for any number of simulated and optionally observed data
    
    The function can handle data provided in three formats:
    - A merged DataFrame containing both observed and simulated data.
    - A Single DataFrame of your choosing.
    - A DataFrame containing only simulated data.

    The plot allows customization of various visual elements like line style, colors, axis labels, and title. 
    The resulting figure can be displayed or saved to a specified directory and file name.

    Parameters
    ----------
    merged_df : pd.DataFrame, optional
        The dataframe containing the series of observed and simulated values. It must have a datetime index.
        To be use when the data contains both observed and simulated values.

    sim_df : pd.DataFrame, optional
        A DataFrame containing only the simulated data series

    df : pd.DataFrame, optional
        A Single DataFrame usually containing only one of either simulated or observed data... or any data.
    
    step : bool, optional
        Whether to plot the data as a step plot. Default is False, which plots a regular line plot.

    where : str, optional
        The location of the step in the step plot. Default is 'pre', which means the step occurs before the x value.

    legend : tuple of str, optional
        A tuple containing the labels for the data being plotted

    metrices : list of str, optional
        A list of metrics to display on the plot, default is None.

    grid : bool, optional
        Whether to display a grid on the plot, default is False.
    
    minor_grid : bool, optional
        Whether to display a minor grid on the plot, default is False.

    title : str, optional
        The title of the plot.

    labels : list of strs, optional
        A tuple containing the labels for the x and y axes.

    padding : bool, optional
        Whether to add padding to the x-axis limits for a tighter plot, default is False.

    linestyles : tuple of str, optional
        A tuple specifying the line styles for the simulated and observed data.

    linewidth : tuple of float, optional
        A tuple specifying the line widths for the simulated and observed data.

    fig_size : tuple of float, optional
        A tuple specifying the size of the figure.

    font_size : int, optional
        The font size for the plot text, default is 12.

    text_size : int, optional
        The font size for the metrics text on the plot, default is 12.

    metrics_adjust : tuple of float, optional
        A tuple specifying the position for the metrics on the plot.

    plot_adjust : float, optional
        A value to adjust the plot layout to avoid clipping.
    
    mode: str, optional
        The mode used to calculate the metric for the scatter plot. Default is 'median'. But it can be 'models' or 'mean'.
        It can also be models used to indicate that the metric is to be calculated for each model as specified in the models list.

    models: list of str, optional
        A list of model names to be used when calculating the metric for the scatter plot. Default is None.
        It is only used when mode is 'models'.

    save : bool, optional
        Whether to save the plot to a file, default is False.

    save_as : str or list of str, optional
        The name or list of names to save the plot as. If a list is provided, each plot will be saved with the corresponding name.

    dir : str, optional
        The directory to save the plot to, default is the current working directory.

    Returns
    -------
    fig : Matplotlib figure instance and/or png files of the figures.
    
    Examples
    --------

    >>> from postprocessinglib.evaluation import visuals
    >>> # Example 1: Plotting merged data with simulated and observed values
    >>> merged_data = pd.DataFrame({...})  # Your merged dataframe
    >>> visuals.plot(merged_df = merged_data,
                    title='Simulated vs Observed',
                    labels=['Time', 'Value'], grid=True,
                    metrices = ['KGE','RMSE'])

    .. image:: ../Figures/plot1_example.png

    >>> # Example 2: Plotting only observed and simulated data with custom linestyles and saving the plot
    >>> obs_data = pd.DataFrame({...})  # Your observed data
    >>> sim_data = pd.DataFrame({...})  # Your simulated data
    >>> visuals.plot(obs_df = obs_data, sim_df = sim_data, linestyles=('g-', 'b-'),
                    save=True, save_as="plot2_example", dir="../Figures")

    .. image:: ../Figures/plot2_example.png

    >>> # Example 3: Plotting a single dataframe
    >>> single_data = pd.DataFrame({...})  # Your single dataframe (either simulated or observed)
    >>> visuals.plot(df=single_data, grid=True, title="Single Line Plot", labels=("Time", "Value"))

    .. image:: ../Figures/plot3_example.png

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-visualizations.ipynb>`_

    Notes
    -----
    - The function requires at least one valid data input (merged_df, sim_df, or df).
    - The time index of the input DataFrames must be a datetime index or convertible to datetime.
    - If the number of columns in the `obs_df` or `sim_df` exceeds five, the plot will be automatically saved.
    - Metrics will be displayed on the plot if specified in the `metrices` parameter.
         
    """
    if df is None:
        # Get the number of simulated data columns
        num_sim = sum(1 for col in  merged_df.columns if col[0] == merged_df.columns[0][0])-1 if merged_df is not None else sum(1 for col in  sim_df.columns if col[0] == sim_df.columns[0][0])
        print(f"Number of simulated data columns: {num_sim}")
        # Line width generation
        if len(linewidth) < num_sim + 1:
            print("Number of linewidths provided is less than the number of columns. "
                    "Number of columns : " + str(num_sim + 1) + ". Number of linewidths provided is: ", str(len(linewidth)) +
                    ". Defaulting to 1.5")
            linewidth = linewidth + (1.5,) * (num_sim + 1 if merged_df is not None else num_sim)
        
        # Generate colors dynamically using Matplotlib colormap
        cmap = plt.cm.get_cmap("tab10", num_sim + 1)  # +1 for Observed
        colors = [cmap(i) for i in range(num_sim + 1)]

        # Available line styles
        # base_linestyles = ["-", "--", "-.", ":"]
        style = ('-',) * (num_sim + 1) # default to solid lines unless overwritten

        # Generate linestyles dynamically
        if len(linestyles) < num_sim + 1:
            print("Number of linestyles provided is less than the number of columns. "
                    "Number of columns : " + str(num_sim + 1) + ". Number of linestyles provided is: ", str(len(linestyles)) +
                    ". Defaulting to solid lines (-)")
            linestyles = linestyles + tuple(f"{colors[i % len(colors)]}{style[i % len(style)]}" 
                            for i in range(num_sim + 1 if merged_df is not None else num_sim))
            
        # Generate Legends dynamically
        if len(legend) < num_sim + 1:
            print("Number of legends provided is less than the number of columns. "
                    "Number of columns : " + str(num_sim + 1) + ". Number of legends provided is: ", str(len(legend)) +
                    ". Applying Default legend names")
            legend = (["Observed"] + [f"Simulated {i+1}" for i in range(num_sim)] if merged_df is not None else [f"Simulated {i+1}" for i in range(num_sim)])           
            

    # Assign the data based on inputs
    sims = {}
    obs = None
    if merged_df is not None:
        # If merged_df is provided, separate observed and simulated data
        obs = merged_df.iloc[:, ::num_sim+1]
        for i in range(1, num_sim+1):
            sims[f"sim_{i}"] = merged_df.iloc[:, i::num_sim+1]
        time = merged_df.index
    elif sim_df is not None:
        # If sim_df is provided, that means theres no observed.
        for i in range(0, num_sim):
            sims[f"sim_{i+1}"] = sim_df.iloc[:, i::num_sim]
        time = sim_df.index
    elif df is not None:
        # If only df is provided, it could be either obs, simulated or just random data.
        # obs = df # to keep the future for loop valid
        line_df = df
        time = df.index
    else:
        raise RuntimeError('Please provide valid data (merged_df, sim_df, or df)')

    # Convert time index to float or int if not datetime
    if not isinstance(time, pd.DatetimeIndex):
        # if the index is not a datetime, then it was converted during aggregation to
        # either a string (most likely) or an int or float
        if (isinstance(time[0], int)) or (isinstance(time[0], float)) or (isinstance(time[0],np.int64)):
            pass
        else:
            if '_' in time[0]:
                parts = time[0].split('_')
                if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                    # 12-hourly: YYYY_DDD_HH
                    time = [datetime.datetime.strptime(day, "%Y_%j_%H") for day in time]
                else:
                    raise ValueError("Invalid format for 12-hourly timestamp. Expected format: YYYY_DDD_HH")
            elif '/' in time[0]:
                # daily
                time = [pd.Timestamp(datetime.datetime.strptime(day, '%Y/%j').date()) for day in time]
            elif '.' in time[0]:
                # weekly
                # datetime ignores the week specifier unless theres a weekday attached,
                # so we extrct the week number and attach Monday - day 1
                time = [pd.to_datetime(f"{int(float(week))}-W{int((float(week) - int(float(week))) * 100):02d}-4", format="%Y-W%U-%w") for week in time]
            elif '-' in time[0]:
                # monthly
                time = [pd.to_datetime(f"{month}-15") for month in time]
            else: # yearly
                time = [pd.to_datetime(f"{year}-07-15") for year in time]
    
    if df is not None:
        for i in range (0, len(line_df.columns)) if isinstance(line_df, pd.DataFrame) else range (0, len(line_df)):
            # Plotting the Data     
            fig, ax = plt.subplots(figsize=fig_size, facecolor='w', edgecolor='k')
            color, style = _parse_linestyle(linestyles[0])  # parse once for df
            if step:
                ax.step(time, line_df.iloc[:, i], where=where, color=color, linestyle=style,
                        label=legend[0], linewidth=linewidth[0])
            else:
                ax.plot(time, line_df.iloc[:, i], color=color, linestyle=style,
                        label=legend[0], linewidth=linewidth[0])

            if padding:
                plt.xlim(time[0], time[-1])
            _finalize_plot(ax, grid, minor_grid, font_size, labels, title, "plot", i)
            auto_save = len(line_df.columns) > 5
            _save_or_display_plot(fig, save or auto_save, save_as, dir, i, "plot")
    else:
        # In either case of merged or sim_df, we will alwaays have simulated data, so we plot the obs first if we have it.
        for i in range (0, len(sims["sim_1"].columns)):
            # Plotting the Data     
            fig, ax = plt.subplots(figsize=fig_size, facecolor='w', edgecolor='k')
            if obs is not None:                
                color, style = _parse_linestyle(linestyles[0])
                if step:
                    ax.step(time, obs.iloc[:, i], where=where, color=color, linestyle=style,
                            label=legend[0], linewidth=linewidth[0])
                else:
                    ax.plot(time, obs.iloc[:, i], color=color, linestyle=style,
                            label=legend[0], linewidth=linewidth[0])
            for j in range(1, num_sim+1):
                color, style = _parse_linestyle(linestyles[j])  # Parse the first linestyle for simulated data
                if step:
                    ax.step(time, sims[f"sim_{j}"].iloc[:, i], where=where, color=color,
                            linestyle=style, label=legend[j], linewidth=linewidth[j])
                else:
                    ax.plot(time, sims[f"sim_{j}"].iloc[:, i], color=color,
                            linestyle=style, label=legend[j], linewidth=linewidth[j])           
            if padding:
                plt.xlim(time[0], time[-1])
            _finalize_plot(ax, grid, minor_grid, font_size, labels, title, "plot", i)

            # Placing Metrics on the Plot if requested
            if obs is not None and metrices:
                observed = obs.iloc[:, [i]]
                sim_list = [sims[f"sim_{j}"].iloc[:,[i]] for j in range(1, num_sim+1)]
                metr = metrics.calculate_metrics(observed=observed, simulated=sim_list, metrices=metrices)
                formatted_metrics = "Metrics:\n"

                if mode == "median":
                    assert isinstance(metr, pd.DataFrame)
                    for metric_name in metrices:
                        median_series = metr[metric_name].median(axis=1)
                        formatted_metrics += f"{metric_name} : {median_series.iloc[0]:.3f}\n"

                elif mode == "models" and models:
                    assert isinstance(metr, pd.DataFrame)
                    for metric_name in metrices:
                        formatted_metrics += f"{metric_name}:\n"
                        for model in models:
                            try:
                                value = metr[(metric_name, model)].iloc[0]
                                formatted_metrics += f"  {model} : {value:.3f}\n"
                            except KeyError:
                                formatted_metrics += f"  {model} : N/A\n"
                else:
                    raise ValueError("Invalid mode or missing models list.")

                # Plot the text on the figure
                font = {'family': 'sans-serif', 'weight': 'normal', 'size': text_size}
                plt.text(metrics_adjust[0], metrics_adjust[1], formatted_metrics,
                        ha='left', va='center', transform=ax.transAxes, fontdict=font,
                        bbox=dict(boxstyle="round, pad=0.5,rounding_size=0.3", facecolor="0.8", edgecolor="k"))

                plt.subplots_adjust(right = 1-plot_adjust)

            # Save or auto-save for large column counts
            auto_save = len(sims["sim_1"].columns) > 5 
            _save_or_display_plot(fig, save or auto_save, save_as, dir, i, "plot")


def bounded_plot(
    lines: Union[List[pd.DataFrame], pd.DataFrame],
    extra_lines: List[pd.DataFrame] = None,
    upper_bounds: List[pd.DataFrame] = None,
    lower_bounds: List[pd.DataFrame] = None,
    step: bool = False,
    where: str = 'pre',
    bound_legend: List[str] = None,
    legend: Tuple[str, str] = None,
    grid: bool = False,
    minor_grid: bool = False,
    title: Union[str, List[str]] = None,
    labels: Tuple[str, str] = None,
    linestyles: Tuple[str, str] = ('r-',),
    linewidth: tuple[float, float] = (1.5,),
    padding: bool = False,
    fig_size: Tuple[float, float] = (10, 6),
    font_size: int = 12,
    text_size: int = 5,
    metrices: List[str] = None,
    transparency: Tuple[float, float] = [0.4],
    save: bool = False,
    save_as: Union[str, List[str]] = None,
    dir: str = os.getcwd()
    ) -> plt.figure:
    """ 
    Plots time-series data with optional confidence bounds.
    Generate a bounded time-series plot comparing model streamflow values with confidence intervals.

    A bounded plot is a time-series visualization that compares observed and simulated hydrological data while incorporating confidence bounds to represent uncertainty.
    This function plots the streamflow data against Julian days/Datetime, providing insights into seasonal variations and model performance over time. 
    The confidence bounds, which can be defined using minimum-maximum ranges or percentiles (e.g., 5th-95th or 25th-75th percentiles), highlight the range of variability in the observed and simulated datasets. 
    The function allows for flexible customization of labels, legends, transparency, and line styles. 
    This visualization is particularly useful for evaluating hydrological models, identifying systematic biases, and assessing the reliability of simulated streamflow under different flow conditions. 

    Parameters
    ----------
    lines : list of pd.DataFrame
        A list of DataFrames containing the observed and/or simulated data series to be plotted. Each DataFrame must have a datetime index.

    upper_bounds : list of pd.DataFrame, optional
        A list of DataFrames containing the upper bounds for each series. If not provided, no upper bounds are plotted. Each bound must be its own list.

    lower_bounds : list of pd.DataFrame, optional
        A list of DataFrames containing the lower bounds for each series. If not provided, no lower bounds are plotted. Each bound must be its own list.

    extra_lines : list of pd.DataFrame, optional
        A list of DataFrames containing additional lines to be plotted on the same graph. These lines will be plotted with dashed lines.
        They have no associated bounds. If included though, it will take on the first item in all plot customization options i.e.,
        - linestyles[0]
        - legend[0]
        - colors[0], etc.
        Take note of this when plotting.  

    step : bool, optional
        Whether to plot the data as a step plot. Default is False, which plots a regular line plot.

    where : str, optional
        The location of the step in the step plot. Default is 'pre', which means the step occurs before the x value.
    
    bound_legend : list of str, optional
        A list containing the labels for the upper and lower bounds.

    legend : tuple of str, optional
        A tuple containing the labels for the simulated and observed data, default is ('Simulated Data', 'Observed Data').

    grid : bool, optional
        Whether to display a grid on the plot, default is False.

    minor_grid : bool, optional
        Whether to display a minor grid on the plot, default is False.

    title : str, optional
        The title of the plot.

    labels : tuple of str, optional
        A tuple containing the labels for the x and y axes.

    linestyles : tuple of str, optional
        A tuple specifying the line styles for the simulated and observed data.

    linewidth : tuple of float, optional
        A tuple specifying the line widths for each line or extra line. Default is (1.5,).

    padding : bool, optional
        Whether to add padding to the x-axis limits for a tighter plot, default is False.

    fig_size : tuple of float, optional
        A tuple specifying the size of the figure.

    font_size : int, optional
        The font size for the plot text, default is 12.
    
    text_size : int, optional
        The font size for the metrics text on the plot, default is 5.

    transparency : list of float, optional
        A list specifying the transparency levels for the upper and lower bounds, default is [0.4, 0.4].
    
    metrices : list of str, optional
        A list of metrics to display on the plot, default is None. Because its a single line being plotted each time,
        Only single line metrics are calculated and displayed i.e., TTCOM, TTP, SPOD, etc. 

    save : bool, optional
        Whether to save the plot to a file, default is False.

    save_as : str or list of str, optional
        The name or list of names to save the plot as. If a list is provided, each plot will be saved with the corresponding name.

    dir : str, optional
        The directory to save the plot to, default is the current working directory.

    Returns
    -------
    fig : Matplotlib figure instance
    
    Example
    -------
    Generate a bounded plot with simulated and observed data, along with upper and lower bounds.

    >>> import pandas as pd
    >>> import numpy as np
    >>> from postprocessinglib.evaluation import visuals

    >>> # Create an index for the data
    >>> time_index = pd.date_range(start='2025-01-01', periods=50, freq='D')
    >>> # Generate sample observed and simulated data
    >>> obs_data = pd.DataFrame({
    ...     "Station1_Observed": np.random.rand(50),
    ...     "Station2_Observed": np.random.rand(50)
    ... }, index=time_index)
    >>> sim_data = pd.DataFrame({
    ...     "Station1_Simulated": np.random.rand(50),
    ...     "Station2_Simulated": np.random.rand(50)
    ... }, index=time_index)

    >>> # Combine observed and simulated data
    >>> data = pd.concat([obs_data, sim_data], axis=1)
    >>> # Generate sample bounds
    >>> upper_bounds = [
    ...     pd.DataFrame({
    ...         "Station1_Upper": np.random.rand(50) + 0.5,
    ...         "Station2_Upper": np.random.rand(50) + 0.5
    ...     }, index=time_index)
    ... ]
    >>> lower_bounds = [
    ...     pd.DataFrame({
    ...         "Station1_Lower": np.random.rand(50) - 0.5,
    ...         "Station2_Lower": np.random.rand(50) - 0.5
    ...     }, index=time_index)
    ... ]

    >>> # Plot the data with bounds
    >>> visuals.bounded_plot(
    ...     lines=data,
    ...     upper_bounds=upper_bounds,
    ...     lower_bounds=lower_bounds,
    ...     legend=('Simulated Data', 'Observed Data'),
    ...     labels=('Datetime', 'Streamflow'),
    ...     transparency = [0.4, 0.3],
    ...     grid=True,
    ...     save=True,
    ...     save_as = 'bounded_plot_example',
    ...     dir = '../Figures'
    ... )

    .. image:: ../Figures/bounded_plot_example_1.png

    >>> # Adjust a few other metrics
    >>> visuals.bounded_plot(
    ...     lines = merged_df,
    ...     upper_bounds = upper_bounds,
    ...     lower_bounds = lower_bounds,
    ...     title=['Long Term Aggregation by days of the Year'],
    ...     legend = ['Predicted Streamflow','Recorded Streamflow'],
    ...     linestyles=['k', 'r-'],
    ...     labels=['Days of the year', 'Streamflow Values'],
    ...     transparency = [0.4, 0.7],
    ... )

    .. image:: ../Figures/bounded_plot_example_2.png

    >>> # Example 3: Plotting with extra lines
    >>> extra_lines = SingleDataFrame()  # Your extra lines DataFrame
    >>> visuals.bounded_plot(
    ...     lines = merged_df,
    ...     upper_bounds = upper_bounds,
    ...     lower_bounds = lower_bounds,
    ...     extra_lines = extra_lines,
    ...     title=['Long Term Aggregation by days of the Year'],
    ...     legend = ['Extra Line','Predicted Streamflow'],
    ...     linestyles=['b--', 'r-'],
    ...     labels=['Datetime', 'Streamflow Values'],
    ... )

    .. image:: ../Figures/bounded_plot_example_3.png

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-visualizations.ipynb>`_
    
    """

    ## Check that the lines inputs are DataFrames
    if isinstance(lines, pd.DataFrame):
        lines = [lines]
    elif not isinstance(lines, list):
        raise ValueError("Argument must be a dataframe or a list of dataframes.")
    
    ## Check that the extra line inputs are DataFrames
    if extra_lines is not None:
        if isinstance(extra_lines, pd.DataFrame):
            extra_lines = [extra_lines]
        elif not isinstance(extra_lines, list):
            raise ValueError("Argument must be a dataframe or a list of dataframes.")
    
    upper_bounds = _normalize_bounds(upper_bounds, lines=lines)
    lower_bounds = _normalize_bounds(lower_bounds, lines=lines)

    if upper_bounds is not None and lower_bounds is not None:
        if len(upper_bounds) != len(lower_bounds):
            raise ValueError("Upper and lower bounds lists must have the same length.")

    # Available line styles
    # Generate colors dynamically using Matplotlib colormap
    cmap = plt.cm.get_cmap("tab10", len(lines)+len(extra_lines) if extra_lines else len(lines))  # +1 for extra line
    colors = [cmap(i) for i in range(len(lines)+len(extra_lines) if extra_lines else len(lines))]
    # base_linestyles = ["-", "--", "-.", ":"]
    style = ('-',) * (len(lines)+len(extra_lines) if extra_lines else len(lines)) # default to solid lines unless overwritten

    # Generate linestyles dynamically
    if len(linestyles) < (len(lines)+len(extra_lines) if extra_lines else len(lines)):
        print("Number of linestyles provided is less than the minimum required. "
                "Number of Lines : " + str(len(lines)+ len(extra_lines) if extra_lines else len(lines)) +
                ". Number of linestyles provided is: ", str(len(linestyles)) +
                ". Defaulting to solid lines (-)")
        linestyles = linestyles + tuple(f"{colors[i % len(colors)]}{style[i % len(style)]}" 
                        for i in range(len(lines)+len(extra_lines) if extra_lines else len(lines)))
        
    # Line width generation
    if len(linewidth) < (len(lines)+len(extra_lines) if extra_lines else len(lines)):
        print("Number of linewidths provided is less than the number of lines to plot. "
                "Number of lines : " + str(len(lines)+len(extra_lines) if extra_lines else len(lines)) + 
                ". Number of linewidths provided is: ", str(len(linewidth)) +
                ". Defaulting to 1.5")
        linewidth = linewidth + (1.5,) * (len(lines)+len(extra_lines) if extra_lines else len(lines))


    # Plotting
    num_columns = lines[0].shape[1]  # all lines have same number of columns
    transparency = transparency * len(lines) # Extend transparency list to match number of lines

    for i in range(num_columns):
        fig, ax = plt.subplots(figsize=fig_size, facecolor='w', edgecolor='k')

        # Plot extra lines (if any), column i
        if extra_lines:
            # z = 0 # Just to avoid erroring out if extra_lines is None
            for z, extra_line in enumerate(extra_lines):

                color, style = _parse_linestyle(linestyles[z])

                if step:
                    ax.step(
                        extra_line.index,
                        extra_line.iloc[:, i],
                        color=color, 
                        linestyle=style,
                        label=legend[z] if legend else "Extra Line",
                        linewidth=linewidth[z],
                        where=where,
                    )
                else:
                    ax.plot(
                        extra_line.index,
                        extra_line.iloc[:, i],
                        color=color, 
                        linestyle=style,
                        label=legend[z] if legend else "Extra Line",
                        linewidth=linewidth[z]
                    )

        # Plot each main line and its bounds
        for line_index, line in enumerate(lines):
            if not isinstance(line, pd.DataFrame):
                raise ValueError("All items in 'lines' must be a DataFrame.")
            
            color, style = _parse_linestyle(linestyles[line_index+z+1] if extra_lines else linestyles[line_index])
            
            if step:
                ax.step(
                    line.index,
                    line.iloc[:, i],
                    color = color, 
                    linestyle = style,
                    label = (
                            legend[line_index+z+1] if legend and extra_lines
                            else legend[line_index] if legend
                            else f"Line {line_index+1}"
                        ),
                    linewidth= linewidth[line_index+z+1] if extra_lines else linewidth[line_index],
                    where=where,
                )
            else:
                ax.plot(
                    line.index,
                    line.iloc[:, i],
                    color = color, 
                    linestyle = style,
                    label = (
                            legend[line_index+z+1] if legend and extra_lines
                            else legend[line_index] if legend
                            else f"Line {line_index+1}"
                        ),
                    linewidth=linewidth[line_index+z+1] if extra_lines else linewidth[line_index]
                )

            upper_obs = _prepare_bounds(upper_bounds, line_index, i)
            lower_obs = _prepare_bounds(lower_bounds, line_index, i)

            for upper_index, upper in enumerate(upper_obs):
                ax.fill_between(
                    line.index,
                    lower_obs[upper_index],
                    upper,
                    alpha=transparency[line_index],
                    color = color,
                    label=bound_legend[upper_index] if bound_legend and line_index < len(bound_legend) else None
                )

            # Add single metrics calculation if requested
            possible_metrices = ["SPOD", "TTP", "TTCOM"]
            if metrices is not None:
                if not isinstance(metrices, list):
                    raise TypeError("Metrices must be a list.")
                invalid = [x for x in metrices if x not in possible_metrices]
                if invalid:
                    raise ValueError(f"Invalid metrics: {', '.join(invalid)}. Valid options are: {', '.join(possible_metrices)}.")

                # Mapping from metric name to the actual function
                metric_funcs = {
                    "SPOD": metrics.SpringPulseOnset,
                    "TTP": metrics.time_to_peak,
                    "TTCOM": metrics.time_to_centre_of_mass,
                }

                # Calculate and format metric values
                text_lines = []
                for metric in metrices:
                    result_df = metric_funcs[metric](line.iloc[:, [i]], use_jday = True)
                    # Assume single-row result, get the first value
                    value = result_df.iloc[0, 0]
                    text_lines.append(f"{metric}: {value}")

                # Join all metric results into one multiline string
                text_block = '\n'.join(text_lines)

                # Display the text in the plot (top-left corner)
                plt.text(
                    0.01, 0.95 - 0.10 * line_index,  # Offset based on line index
                    s=text_block,
                    transform=plt.gca().transAxes,
                    fontsize=text_size,
                    verticalalignment='top',
                    bbox=dict(
                        boxstyle='round,pad=0.3',
                        facecolor=color,
                        edgecolor='gray',
                        alpha=0.7
                    )
                )

                

        if padding:
            plt.xlim(lines[0].index[0], lines[0].index[-1])

        _finalize_plot(ax, grid, minor_grid, font_size, labels, title, "bounded-plot", i)

        auto_save = num_columns > 5
        _save_or_display_plot(fig, save or auto_save, save_as, dir, i, "bounded-plot")


def histogram(
    merged_df: pd.DataFrame = None, 
    df: pd.DataFrame = None, 
    sim_df: pd.DataFrame = None,
    bins: int = 100,
    legend: Tuple[str, str] = ('Simulated Data', 'Observed Data'),
    colors: list[str] = ['r', 'b'],
    transparency: float = 0.6,
    z_norm=False,
    prob_dens=False,
    fig_size: Tuple[float, float] = (12, 6),
    font_size: int = 12,
    title: str = None,
    labels: Tuple[str, str] = ('Value', 'Frequency'),
    grid: bool = False,
    minor_grid: bool = False,
    save: bool = False,
    save_as: str = None,
    dir: str = os.getcwd()
    ) -> plt.figure:
    """
    Plots Histogram for Observed and Simulated Data with Optional Normalization

    This function generates a histogram comparing the distribution of observed and simulated data, providing insights into their statistical characteristics and variability.
    The histogram allows users to analyze the frequency distribution of hydrological data, assess model performance, and identify biases in the simulated dataset.
    The function supports Z-score normalization, which transforms the data into standard deviations from the mean, enabling comparison of datasets with different scales. 
    It also includes an option to plot the histogram as a probability density function (PDF), ensuring that the area under the histogram sums to one, making it easier to compare distributions.
    Users can customize the number of bins, colors, legend labels, and transparency levels to enhance visualization clarity. The function also allows for gridlines, axis labeling,
    and automatic or manual saving of plots.
    This visualization is particularly useful for hydrological modeling, statistical analysis, and understanding deviations between observed and simulated streamflow distributions under various conditions.

    Parameters
    ----------
    merged_df : pd.DataFrame, optional
        The dataframe containing the series of observed and simulated values. It must have a datetime index.
        
    obs_df : pd.DataFrame, optional
        A DataFrame containing the observed data series if using separate observed and simulated data.

    sim_df : pd.DataFrame, optional
        A DataFrame containing the simulated data series if using separate observed and simulated data.

    df : pd.DataFrame, optional
        A DataFrame containing the data to be plotted if no merged or separate observed/simulated data are provided.

    legend : tuple of str, optional
        A tuple containing the labels for the simulated and observed data, default is ('Simulated Data', 'Observed Data').

    bins: int
        Specifies the number of bins in the histogram.

    z_norm: bool
        If True, the data will be Z-score normalized.
    
    prob_dens: bool
        If True, normalizes both histograms to form a probability density, i.e., the area
        (or integral) under each histogram will sum to 1.

    legend: tuple of str
        Tuple of length two with str inputs. Adds a Legend in the 'best' location determined by
        matplotlib. The entries in the tuple label the simulated and observed data
        (e.g. ['Simulated Data', 'Predicted Data']).

    grid: bool
        If True, adds a grid to the plot.

    minor_grid: bool
        If True, adds a minor grid to the plot.

    title: str
        If given, sets the title of the plot.

    labels: tuple of str
        Tuple of two string type objects to set the x-axis labels and y-axis labels, respectively.

    figsize: tuple of float
        Tuple of length two that specifies the horizontal and vertical lengths of the plot in
        inches, respectively.
    
    font_size: int, optional
        Font size for the plot text elements, default is 12.

    colors : tuple of str, optional
        Colors for the simulated and observed histograms.

    transparency : float, optional
        Transparency level for the histograms, default is 0.6.

    save : bool, optional
        Whether to save the plot to a file, default is False.

    save_as : str or list of str, optional
        The name or list of names to save the plot as. If a list is provided, each plot will be saved with the corresponding name.

    dir : str, optional
        The directory to save the plot to, default is the current working directory.   

    Returns
    -------
    fig : Matplotlib figure instance and/or png files of the figures.

    Examples
    --------

    >>> from postprocessinglib.evaluation import visuals
    >>> # Example 1: Plotting merged data with simulated and observed values
    >>> merged_data = pd.DataFrame({...})  # Your merged dataframe
    >>> visuals.plot(merged_df = merged_data,
                    title='Simulated vs Observed',
                    bins = 100,
                    labels=['Frequency', 'Value'], grid=True)

    .. image:: ../Figures/hist1_Example.png

    >>> # Example 2: Plotting observed and simulated data with custom linestyles and saving the plot
    >>> obs_data = pd.DataFrame({...})  # Your observed data
    >>> sim_data = pd.DataFrame({...})  # Your simulated data
    >>> visuals.plot(obs_df = obs_data, sim_df = sim_data, colors=('g', 'c'), bins = 100, z_norm = True, prob_dens = True,
                    save=True, save_as="hist2_example", dir="../Figures")

    .. image:: ../Figures/hist2_Example.png

    >>> # Example 3: Plotting a single dataframe
    >>> single_data = pd.DataFrame({...})  # Your single dataframe (either simulated or observed)
    >>> visuals.plot(df=single_data, grid=True, title="Single Histogram Plot", labels=("Time", "Frequency"))

    .. image:: ../Figures/hist3_Example.png

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-visualizations.ipynb>`_

    """
    if df is None:
        # Get the number of simulated data columns
        num_sim = sum(1 for col in  merged_df.columns if col[0] == merged_df.columns[0][0])-1 if merged_df is not None else sum(1 for col in  sim_df.columns if col[0] == sim_df.columns[0][0])
        print(f"Number of simulated data columns: {num_sim}")
        
        # Generate colors dynamically using Matplotlib colormap
        cmap = plt.cm.get_cmap("tab10", num_sim + 1)  # +1 for Observed
        colors = colors + [cmap(i) for i in range(num_sim + 1)]
            
        # Generate Legends dynamically
        if len(legend) < num_sim + 1:
            print("Number of legends provided is less than the number of columns. "
                    "Number of columns : " + str(num_sim + 1) + ". Number of legends provided is: ", str(len(legend)) +
                    ". Applying Default legend names")
            legend = (["Observed"] + [f"Simulated {i+1}" for i in range(num_sim)] if merged_df is not None else [f"Simulated {i+1}" for i in range(num_sim)])           
            

    # Assign the data based on inputs
    sims = {}
    obs = None
    if merged_df is not None:
        # If merged_df is provided, separate observed and simulated data
        obs = merged_df.iloc[:, ::num_sim+1]
        for i in range(1, num_sim+1):
            sims[f"sim_{i}"] = merged_df.iloc[:, i::num_sim+1]
    elif sim_df is not None:
        # If sim_df is provided, that means theres no observed.
        for i in range(0, num_sim):
            sims[f"sim_{i+1}"] = sim_df.iloc[:, i::num_sim]
    elif df is not None:
        # If only df is provided, it could be either obs, simulated or just random data.
        # obs = df # to keep the future for loop valid
        line_df = df
    else:
        raise RuntimeError('Please provide valid data (merged_df, sim_df, or df)')
    
    # Plotting
    if df is not None:
        for i in range (0, len(line_df.columns)) if isinstance(line_df, pd.DataFrame) else range (0, len(line_df)):
            # Manipulating and generating the Data
            if z_norm:
                # calculating the z-score for the observed data
                line_df.iloc[:, i] = (line_df.iloc[:, i] - line_df.iloc[:, i].mean()) / line_df.iloc[:, i].std()

            # finding the mimimum and maximum z-scores
            total_max = line_df.iloc[:, i].max()
            total_min = line_df.iloc[:, i].min()

            # creating the bins based on the max and min
            num_bins = np.linspace(total_min - 0.01, total_max + 0.01, bins) 
            
            # Plotting the Data     
            fig = plt.figure(figsize=fig_size, facecolor='w', edgecolor='k')
            ax = fig.add_subplot(111)
            
            ax.hist(line_df.iloc[:, i],
                bins=num_bins,
                alpha=transparency,
                label=legend[0],
                color=colors[0],
                edgecolor='black',
                linewidth=0.5,
                density=prob_dens)
            
            _finalize_plot(ax, grid, minor_grid, font_size, labels, title, "plot", i)
            auto_save = len(line_df.columns) > 5
            _save_or_display_plot(fig, save or auto_save, save_as, dir, i, "plot")
    else:
        # In either case of merged or sim_df, we will alwaays have simulated data, so we plot the obs first if we have it.
        for i in range (0, len(sims["sim_1"].columns)):
            obs_series = obs.iloc[:, i] if obs is not None else None
            sim_series_list = []
            if z_norm:
                if obs is not None:
                    obs_series = (obs_series - obs_series.mean()) / obs_series.std()
                for j in range(1, num_sim + 1):
                    sim_j = sims[f"sim_{j}"].iloc[:, i]
                    sim_j = (sim_j - sim_j.mean()) / sim_j.std()
                    sim_series_list.append(sim_j)
            else:
                if obs is not None:
                    obs_series = obs_series.copy()
                for j in range(1, num_sim + 1):
                    sim_series_list.append(sims[f"sim_{j}"].iloc[:, i])       

            # Combine all relevant series to determine global min/max
            combined_series = sim_series_list.copy()
            if obs_series is not None:
                combined_series.append(obs_series)

            total_min = min(s.min() for s in combined_series)
            total_max = max(s.max() for s in combined_series)
            num_bins = np.linspace(total_min - 0.01, total_max + 0.01, bins)

            # Plotting the Data     
            fig = plt.figure(figsize=fig_size, facecolor='w', edgecolor='k')
            ax = fig.add_subplot(111)
            
            ax.hist(obs_series,
                bins=num_bins,
                alpha=transparency,
                label=legend[0],
                color=colors[0],
                edgecolor='black',
                linewidth=0.5,
                density=prob_dens
                )
            for j, sim_series in enumerate(sim_series_list, start=1):
                ax.hist(sim_series,
                        bins=num_bins,
                        alpha=transparency,
                        label=legend[j],
                        color=colors[j],
                        edgecolor='black',
                        linewidth=0.5,
                        density=prob_dens
                        )
            
            _finalize_plot(ax, grid, minor_grid, font_size, labels, title, "plot", i)
            auto_save = len(sims["sim_1"].columns) > 5
            _save_or_display_plot(fig, save or auto_save, save_as, dir, i, "plot")


def scatter(
  grid: bool = False, 
  minor_grid: bool = False,
  title: str = None, 
  legend: tuple[str, str] = None,
  labels: tuple[str, str] = ('Simulated Data', 'Observed Data'),
  fig_size: tuple[float, float] = (10, 6), 
  best_fit: bool = False, 
  line45: bool = False,
  mode:str = 'median',
  models:List[str] = None,
  font_size: int = 12,
  text_size: int = 12,

  merged_df: pd.DataFrame = None, 
  obs_df: pd.DataFrame = None, 
  sim_df: pd.DataFrame = None,
  metrices: list[str] = None, 
  markerstyle: list[str] = ['bo'], 
  save: bool = False, 
  plot_adjust: float = 0.2,
  save_as: str = None, 
  metrics_adjust: tuple[float, float] = (1.05, 0.5), 
  dir: str = os.getcwd(),

  shapefile_path: str = "", 
  x_axis: pd.DataFrame = None, 
  y_axis: pd.DataFrame = None,
  metric: str = "", 
  observed: pd.DataFrame = None, 
  simulated: pd.DataFrame = None,
  markersize: int = 10,
  cmap: str='jet',
  vmin: float=None,
  vmax:float=None
  ) -> plt.figure:
    """
    Creates a scatter plot comparing observed and simulated data, with optional features like 
    best fit lines, 45-degree reference lines, and metric annotations.

    This function can handle both merged data (observed and simulated in a single DataFrame) and 
    separate observed and simulated data DataFrames. Additionally, it can plot scatter plots over 
    shapefiles for geographic data visualization.

    The plot can be customized with various visual features, such as the color map, gridlines, 
    markers, and axis labels. The function also allows adding a linear regression best-fit line, 
    a 45-degree line, and annotations for metrics. The plot can be saved to a file if desired.

    Parameters
    ----------
    grid : bool, optional
        Whether to display a grid on the plot, default is False.

    minor_grid : bool, optional
        Whether to display a minor grid on the plot, default is False.

    title : str, optional
        The title of the plot.

    labels: tuple of str, optional
        A tuple containing the labels for the simulated and observed data, default is ('Simulated Data', 'Observed Data').

    legend : tuple of str, optional
        A tuple containing the labels for the x and y axes.

    fig_size : tuple of float, optional
        A tuple specifying the size of the figure.

    font_size : int, optional
        Font size for the plot text, default is 12.

    merged_df : pd.DataFrame, optional
        The dataframe containing the series of observed and simulated values. It must have a datetime index.
        
    obs_df : pd.DataFrame, optional
        A DataFrame containing the observed data series if using separate observed and simulated data.

    sim_df : pd.DataFrame, optional
        A DataFrame containing the simulated data series if using separate observed and simulated data.

    metrices : list of str, optional
        A list of metrics to display on the plot, default is None.

    markerstyle: str
        List of two strings that determine the point style and shape of the data being plotted 

    metrics_adjust : tuple of float, optional
        A tuple specifying the position for the metrics on the plot.

    plot_adjust : float, optional
        A value to adjust the plot layout to avoid clipping. 

    best_fit: bool
        If True, adds a best linear regression line on the graph with the equation for the line in the legend. 
        If there are multiple columns, the best fit line will be added to each column i.e, multiple best fit lines.

    line45: bool
        IF True, adds a 45 degree line to the plot and the legend. There is only one 45 degree line for all columns.
        
    save : bool, optional
        Whether to save the plot to a file, default is False.

    save_as : str or list of str, optional
        The name or list of names to save the plot as. If a list is provided, each plot will be saved with the corresponding name.

    dir : str, optional
        The directory to save the plot to, default is the current working directory.

    shapefile_path : str, optional
        The path to a shapefile on top of which the scatter plot will be drawn.

    x_axis : pd.DataFrame, optional
        Used when plotting with a shapefile to determine the x-axis values.

    y_axis : pd.DataFrame, optional
        Used when plotting with a shapefile to determine the y-axis values.

    metric : str, optional
        The metric used to generate the color map for the scatter plot.

    observed : pd.DataFrame, optional
        Used to calculate the metric for the scatter plot.

    simulated : pd.DataFrame, optional
        Used to calculate the metric for the scatter plot.

    markersize: int, optional
        Size of the markers in the shapefile scatter plot. Default is 10.

    mode: str, optional
        The mode used to calculate the metric for the scatter plot. Default is 'median'. But it can be 'models' or 'mean'.
        It can also be models used to indicate that the metric is to be calculated for each model as specified in the models list.

    models: list of str, optional
        A list of model names to be used when calculating the metric for the scatter plot. Default is None.
        It is only used when mode is 'models'.    
    
    cmap: string, optional
        Used to determine the color scheme of the color map for the shapefile plot 

    vmin: float, optional
        Minimum colormap value
    
    vmax: float, optional
        Maximum colormap value
    
    Returns
    -------
    fig : Matplotlib figure instance

    Example
    -------
    Generate a scatter plot using observed and simulated data:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from postprocessinglib.evaluation import visuals
    >>> #
    >>> # Create test data
    >>> index = pd.date_range(start="2022-01-01", periods=10, freq="D")
    >>> obs_df = pd.DataFrame({
    >>>     "Station1": np.random.rand(10),
    >>>     "Station2": np.random.rand(10)
    >>> }, index=index)
    >>> #
    >>> sim_df = pd.DataFrame({
    >>>     "Station1": np.random.rand(10),
    >>>     "Station2": np.random.rand(10)
    >>> }, index=index)
    >>> #
    >>> # Call the scatter plot function
    >>> visuals.scatter(
    >>>     obs_df=obs_df,
    >>>     sim_df=sim_df,
    >>>     labels=("Observed", "Simulated"),
    >>>     title="Scatter Plot Example",
    >>>     grid=True,
    >>>     metrices = ['KGE','RMSE'],
    >>>     line45=True,
    >>>     markerstyle = 'b.',
    >>>     save=True,
    >>>     save_as="scatter_plot_example_1.png"
    >>> )

    .. image:: ../Figures/scatter_plot_example_1.png

    >>> visuals.scatter(
    >>>     merged_df=merged_df,
    >>>     labels=("Observed Data", "Simulated Data"),
    >>>     title="Scatterplot of the data of 2015",
    >>>     grid=True,
    >>>     line45=True,
    >>>     markerstyle = 'cx',
    >>> )

    .. image:: ../Figures/scatter_plot_example_2.png

    >>> shapefile_path = r"SaskRB_SubDrainage2.shp"
    >>> stations_path = 'Station_data.xlsx'
    >>> Station_info = pd.read_excel(io=stations_path)
    >>> .
    >>> # plot of a few stations in the SRB showing the disparities in their KGE
    >>> visuals.scatter(shapefile_path = shapefile_path,
                        title = "SRB SubDrainage and KGE",
                        x_axis = Station_info["Lon"],
                        y_axis = Station_info["Lat"],
                        metric = "KGE",
                        fig_size = (24, 20),
                        observed = DATA_2["DF_OBSERVED"],
                        simulated = DATA_2["DF_SIMULATED"],
                        labels=['Longitude', 'Latitude'],
                        cmap = 'jet'
                    )

    .. image:: ../Figures/SRB_subDrainage_showing_KGE.png

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-visualizations.ipynb>`_

    """
    # Plotting the Data
    if not shapefile_path:
        # Get the number of simulated data columns
        num_sim = sum(1 for col in  merged_df.columns if col[0] == merged_df.columns[0][0])-1 if merged_df is not None else sum(1 for col in  sim_df.columns if col[0] == sim_df.columns[0][0])
        print(f"Number of simulated data columns: {num_sim}")

        # Generate colors dynamically using Matplotlib colormap
        color_map = plt.cm.get_cmap("tab10", num_sim)  # +1 for Observed
        colors = [color_map(i) for i in range(num_sim)]

        # Available marker styles
        style = [".", "1", "v", "x", "*", "+", "X", "3", "^", "s", "D"] # default unless overwritten

        # Generate linestyles dynamically
        if len(markerstyle) < num_sim:
            print("Number of markerstyles provided is less than the number of columns. "
                    "Number of columns : " + str(num_sim) + ". Number of markerstyles provided is: ", str(len(markerstyle)) +
                    ". Using Default Markerstyles.")
            markerstyle = markerstyle + [f"{colors[i % len(colors)]}{style[i % len(style)]}" 
                            for i in range(num_sim)]    

        # Generate Legends dynamically
        if legend is None:
            legend = [f"Simulated {i}" for i in range(1, num_sim+1)]
        elif len(legend) < num_sim:
            print("Number of legends provided is less than the number of columns. "
                    "Number of columns : " + str(num_sim) + ". Number of legends provided is: ", str(len(legend)) +
                    ". Applying Default labels")
            legend = legend + [f"Simulated {len(legend)+i}" for i in range(1, num_sim+1)]

        sims = {}
        obs = None
        if merged_df is not None:
            # If merged_df is provided, separate observed and simulated data
            obs = merged_df.iloc[:, ::num_sim+1]
            for i in range(1, num_sim+1):
                sims[f"sim_{i}"] = merged_df.iloc[:, i::num_sim+1]
        elif sim_df is not None and obs_df is not None:
            # If both sim_df and obs_df are provided
            obs = obs_df
            for i in range(0, num_sim):
                sims[f"sim_{i+1}"] = sim_df.iloc[:, i::num_sim]
        else:
            raise RuntimeError('Please provide valid data (merged_df, obs_df or sim_df)')

        for i in range (0, len(obs.columns)):
            max_obs = obs.iloc[:, i].max()
            min_obs = obs.iloc[:, i].min()
            max_sim, min_sim  = 0, 0

            # Plotting the Data
            fig, ax = plt.subplots(figsize=fig_size, facecolor='w', edgecolor='k')
            for j in range(1, num_sim+1):
                color, marker = _parse_markerstyle(markerstyle[j-1])  # Parse the first linestyle for simulated data
                ax.plot(sims[f"sim_{j}"].iloc[:, i], obs.iloc[:, i],
                        color = color,
                        marker = marker, 
                        label=legend[j-1] if labels else f"Sim {j}", 
                        linestyle='None')
                max_sim = np.max([max_sim, sims[f"sim_{j}"].iloc[:, i].max()])
                min_sim = np.min([min_sim, sims[f"sim_{j}"].iloc[:, i].min()]) 

                if best_fit:
                    # Getting a polynomial fit and defining a function with it
                    p = np.polyfit(sims[f"sim_{j}"].iloc[:, i], obs.iloc[:, i], 1)
                    f = np.poly1d(p)

                    # Calculating new x's and y's
                    x_new = np.linspace(sims[f"sim_{j}"].iloc[:, i].min(), sims[f"sim_{j}"].iloc[:, i].max(), sims[f"sim_{j}"].iloc[:, i].size)
                    y_new = f(x_new)

                    # Formatting the best fit equation to be able to display in latex
                    equation = "{} x + {}".format(np.round(p[0], 4), np.round(p[1], 4))

                    # Plotting the best fit line with the equation as a legend in latex
                    ax.plot(x_new, y_new,
                            color = eval(markerstyle[j-1][:-1]) if markerstyle[j-1][:-1].startswith("(") else markerstyle[j-1][:-1], 
                            label="${}$".format(equation))

            
            if line45:
                max = np.nanmax([max_sim, max_obs])
                min = np.nanmin([min_sim, min_obs])
                # Plotting the 45 degree line
                ax.plot(np.arange(int(min), int(max) + 1), np.arange(int(min), int(max) + 1), 'r--', label='45$^\u00b0$ Line')

            
            if best_fit or line45:
                ax.legend(fontsize=font_size, loc='best')
            
            _finalize_plot(ax, grid, minor_grid, font_size, labels, title, "scatter-plot", i)               

            # Placing Metrics on the Plot if requested
            if metrices:
                observed = obs.iloc[:, [i]]
                sim_list = [sims[f"sim_{j}"].iloc[:,[i]] for j in range(1, num_sim+1)]
                metr = metrics.calculate_metrics(observed=observed, simulated=sim_list, metrices=metrices)
                formatted_metrics = "Metrics:\n"

                if mode == "median":
                    assert isinstance(metr, pd.DataFrame)
                    for metric_name in metrices:
                        median_series = metr[metric_name].median(axis=1)
                        formatted_metrics += f"{metric_name} : {median_series.iloc[0]:.3f}\n"

                elif mode == "models" and models:
                    assert isinstance(metr, pd.DataFrame)
                    for metric_name in metrices:
                        formatted_metrics += f"{metric_name}:\n"
                        for model in models:
                            try:
                                value = metr[(metric_name, model)].iloc[0]
                                formatted_metrics += f"  {model} : {value:.3f}\n"
                            except KeyError:
                                formatted_metrics += f"  {model} : N/A\n"
                else:
                    raise ValueError("Invalid mode or missing models list.")

                # Plot the text on the figure
                font = {'family': 'sans-serif', 'weight': 'normal', 'size': text_size}
                plt.text(metrics_adjust[0], metrics_adjust[1], formatted_metrics,
                        ha='left', va='center', transform=ax.transAxes, fontdict=font,
                        bbox=dict(boxstyle="round, pad=0.5,rounding_size=0.3", facecolor="0.8", edgecolor="k"))

                plt.subplots_adjust(right = 1-plot_adjust)

            # Save or auto-save for large column counts
            auto_save = len(obs.columns) > 5
            _save_or_display_plot(fig, save or auto_save, save_as, dir, i, "scatter-plot")
    else:
        # use simultated to determine if single or multiple models
        if isinstance(simulated, pd.DataFrame):
            simulated = [simulated] 
        # Calculate metrics: returns MultiIndex column DataFrame
        metr = metrics.calculate_metrics(observed=observed, simulated=simulated, metrices=[metric])
        # metric_df = metr[metric]  # e.g., metr["KGE"] gives all models for KGE
        # print(f"Metrics to plot: \n{metr}\n")

        # Determine values to plot
        mode_list = ("median", "mean", "mode", "max", "min", "std", "sum")
        if mode in mode_list:
            # Map mode strings to appropriate aggregation functions
            agg_func_map = {
                "median": lambda df: df.median(axis=1),
                "mean": lambda df: df.mean(axis=1),
                "mode": lambda df: df.mode(axis=1).iloc[:, 0] if not df.mode(axis=1).empty else np.nan,
                "max": lambda df: df.max(axis=1),
                "min": lambda df: df.min(axis=1),
                "std": lambda df: df.std(axis=1),
                "sum": lambda df: df.sum(axis=1)
            }

            # Handle multi-index column DataFrame
            if isinstance(metr.columns, pd.MultiIndex):
                try:
                    data_block = metr.loc[:, metric]  # extract the DataFrame (e.g., KGE for all models)
                except KeyError:
                    raise ValueError(f"Metric '{metric}' not found in metrics DataFrame.")
            else:
                data_block = metr  # Fallback if not multi-indexed

            # Apply the appropriate aggregation
            agg_func = agg_func_map[mode]
            agg_result = agg_func(data_block)

            values = pd.DataFrame({f"{metric}_{mode}": agg_result})
            columns = [f"{metric}_{mode}"]

        elif mode == "models" and models:
            # Assume columns are like: MultiIndex([("MSE", "model1"), ("MSE", "model2"), ...])
            values = metr.loc[:, (metric, models)]
            values.columns = models  # Flatten columns for plotting
            columns = values.columns.tolist()

        else:
            raise ValueError("Invalid mode or missing model list for 'models' mode")


        # print(f"Values to plot: \n{values}\n")

        # print(f"Columns to plot: \n{columns}\n")
    
        # Create DataFrame for coordinates
        base_data = pd.DataFrame({
            'latitude': y_axis.values,
            'longitude': x_axis.values
        })

        # Read the shapefile using GeoPandas
        gdf_shapefile = gpd.read_file(shapefile_path)
 
        # Plotting logic
        for idx, col in enumerate(columns):
            fig = plt.figure(figsize=fig_size, dpi=150, frameon=True)
            ax = fig.add_axes([0, 0., .4, .1])

            # Create per-plot DataFrame
            data = base_data.copy()
            values_reset = values.reset_index(drop=True)
            data[col] = values_reset[col]

            geometry = [Point(xy) for xy in zip(data['longitude'], data['latitude'])]
            gdf_points = gpd.GeoDataFrame(data, geometry=geometry)

            # print(f"Metric data head:\n{data.head()}")
            # print(f"Unique metric values: {data[col].unique()}")
            # print(f"Geometry head:\n{geometry[:5]}")

            # Base map
            gdf_shapefile.plot(ax=ax, edgecolor='black', facecolor="None", linewidth=0.5)

            # Plot metric values
            legend_label = f"{col}" if mode == "models" else f"{metric} ({mode})" if len(simulated) > 1 else metric
            gdf_points.plot(ax=ax, column=col, cmap=cmap, vmin=vmin, vmax=vmax,
                            legend=True, markersize=markersize, label=legend_label,
                            legend_kwds={'label': legend_label, "orientation": "vertical"})
            
            # Manually get the colorbar from the figure
            cbar = plt.gcf().get_axes()[-1]  # Get the colorbar axis

            # ✅ Adjust tick label font size
            cbar.tick_params(labelsize=font_size)


            _finalize_plot(ax, grid, minor_grid, font_size, labels, title, "shapefile-plot", idx)
        
            # Save or auto-save for large column counts
            _save_or_display_plot(fig, save, save_as, dir, i=idx, type="shapefile-plot")


def qqplot(
    grid: bool = False, 
    minor_grid: bool = False,
    title: str = None, 
    labels: tuple[str, str] = None, 
    fig_size: tuple[float, float] = (10, 6),
    font_size: int = 12,
    method: str = "linear", 
    legend: list[str] = None, 
    linewidth: tuple[float, float] = (1.5, 2.5),
    merged_df: pd.DataFrame = None, 
    obs_df: pd.DataFrame = None, 
    sim_df: pd.DataFrame = None,
    linestyle: tuple[str, str, str] = ('b:', 'r-.', 'r-'), 
    quantile: tuple[int, int] = (25, 75),
    q_labels: tuple[str, str, str] = ('Range of Quantiles', 'IQR'),
    save: bool = False, 
    save_as: str = None, 
    dir: str = os.getcwd()
    ) -> plt.figure:
    """
    Generate a Quantile-Quantile (QQ) plot to compare the statistical distribution of simulated and observed data.

    A Quantile-Quantile (QQ) plot is a graphical technique for assessing whether two datasets come from the same distribution by plotting their quantiles against each other. 
    If the datasets have identical distributions, the points should fall along the 1:1 line. This function calculates and visualizes the quantiles of observed and simulated streamflow data, interpolating if necessary, and marks key statistical features such as the interquartile range. 
    By comparing the empirical quantiles of simulated and observed data, the QQ plot helps evaluate the performance of hydrological models in reproducing streamflow distributions, highlighting potential biases and differences in variability.
    The function also allows for flexible customization of labels, legends, transparency, and line styles.
    It is an essential tool in hydrology and environmental sciences for assessing the agreement between measured and modeled hydrological variables.

    Parameters
    ----------
    grid : bool, optional
        Whether to display a grid on the plot, default is False.
    
    minor_grid : bool, optional
        Whether to display a minor grid on the plot, default is False.

    title : str, optional
        The title of the plot.

    labels : tuple of str, optional
        A tuple containing the labels for the x and y axes

    fig_size: tuple[float, float]
        Tuple of length two that specifies the horizontal and vertical lengths of the plot in
        inches, respectively.

    font_size: int, optional
        Font size for the plot text, default is 12.

    method: str
        Determines whether the quantiles should be interpolated when the data length differs.
        If True, the quantiles are interpolated to align the data lengths between the observed
        and simulated data, ensuring accurate comparison.
        Default is Linear.

    legend: bool
        Whether to display the legend or not. Default is False

    llinewidth : tuple of float, optional
        A tuple specifying the line widths for the simulated and observed data.

    merged_df : pd.DataFrame, optional
        The dataframe containing the series of observed and simulated values. It must have a datetime index.
        
    obs_df : pd.DataFrame, optional
        A DataFrame containing the observed data series if using separate observed and simulated data.

    sim_df : pd.DataFrame, optional
        A DataFrame containing the simulated data series if using separate observed and simulated data.

    linestyle: tuple[str, str, str]
        List of three strings that determine the point style and shape of the data being plotted 

    quantile: tuple[int, int]
        Range of quantiles to plot, with values between 0 and 1. The first value is the lower quantile,
        and the second is the upper. Default is (25, 75).
    
    q_labels: tuple[str, str, str]
        Labels for the x-axis (simulated quantiles) and y-axis (observed quantiles).
        Default is ['Quantiles', 'Range of Quantiles', 'Inter Quartile Range'].

    save : bool, optional
        Whether to save the plot to a file, default is False.

    save_as : str or list of str, optional
        The name or list of names to save the plot as. If a list is provided, each plot will be saved with the corresponding name.

    dir : str, optional
        The directory to save the plot to, default is the current working directory.

    Example
    -------
    Generate a QQ plot to compare observed and simulated data distributions:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from postprocessinglib.evaluation import metrics
    >>> #
    >>> # Create test data
    >>> index = pd.date_range(start="2022-01-01", periods=50, freq="D")
    >>> obs_df = pd.DataFrame({
    >>>     "Station1": np.random.rand(50),
    >>>     "Station2": np.random.rand(50)
    >>> }, index=index)
    >>> #
    >>> sim_df = pd.DataFrame({
    >>>     "Station1": np.random.rand(50),
    >>>     "Station2": np.random.rand(50)
    >>> }, index=index)
    >>> #
    >>> # Call the QQ plot function
    >>> visuals.qqplot(
    >>>     obs_df=obs_df,
    >>>     sim_df=sim_df,
    >>>     labels=("Quantiles (Simulated)", "Quantiles (Observed)"),
    >>>     title="QQ Plot Example",
    >>>     save=True,
    >>>     save_as="qqplot_example.png"
    >>> )

    .. image:: ../Figures/qqplot_example.png

    >>> visuals.qqplot(
    >>>     merged_df=merged_df,
    >>>     labels=("Quantiles (Simulated)", "Quantiles (Observed)"),
    >>>     title="QQ Plot of the simulated Dataset compared to the observed from 2000 till 2005",
    >>>     grid=True,
    >>>     save=True,
    >>>     save_as="qqplot_example_2.png"
    >>> )

    .. image:: ../Figures/qqplot_example.png

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-visualizations.ipynb>`_

    """
    def _adjust_color_brightness(color, amount=1.0):
        """
        Adjusts the brightness of the given color.
        Input can be a matplotlib color string, hex string, or RGB tuple.
        The amount parameter >1 brightens the color, <1 darkens it.
        """
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


    # Determine the number of simulations
    if merged_df is not None:
        num_sim = sum(1 for col in merged_df.columns if col[0] == merged_df.columns[0][0]) - 1
    elif sim_df is not None:
        num_sim = sum(1 for col in sim_df.columns if col[0] == sim_df.columns[0][0])
    else:
        raise RuntimeError('Please provide valid data (merged_df, obs_df or sim_df)')

    print(f"Number of simulated data columns: {num_sim}")

    # Generate colors dynamically using Matplotlib colormap
    color_map = plt.cm.get_cmap("tab10", num_sim)
    base_colors = [color_map(i) for i in range(num_sim)]

    # Available marker styles
    style = [".", "1", "v", "x", "*", "+", "X", "3", "^", "s", "D"]

    # Generate Legends dynamically
    if legend is None:
        legend = [f"Simulated {i}" for i in range(1, num_sim+1)]
    elif len(legend) < num_sim:
        print("Number of legends provided is less than the number of simulations. "
              f"Number of simulations: {num_sim}. Number of legends provided: {len(legend)}. "
              "Applying default labels.")
        legend += [f"Simulated {i}" for i in range(len(legend)+1, num_sim+1)]

    # Separate observed and simulated data
    sims = {}
    if merged_df is not None:
        obs = merged_df.iloc[:, ::num_sim+1]
        for i in range(1, num_sim+1):
            sims[f"sim_{i}"] = merged_df.iloc[:, i::num_sim+1]
    elif sim_df is not None and obs_df is not None:
        obs = obs_df
        for i in range(num_sim):
            sims[f"sim_{i+1}"] = sim_df.iloc[:, i::num_sim]
    else:
        raise RuntimeError('Please provide valid data (merged_df, obs_df or sim_df)')

    for i in range(len(obs.columns)):
        fig, ax = plt.subplots(figsize=fig_size, facecolor='w', edgecolor='k')

        n = obs.iloc[:, i].size
        pvec = 100 * ((np.arange(1, n + 1) - 0.5) / n)
        obs_perc = np.percentile(obs.iloc[:, i], pvec, method=method)
        quant_1_obs = np.percentile(obs.iloc[:, i], quantile[0], method=method)
        quant_3_obs = np.percentile(obs.iloc[:, i], quantile[1], method=method)
        dobs = quant_3_obs - quant_1_obs

        for j in range(1, num_sim+1):
            base_color = base_colors[j-1]
            adjusted_color = _adjust_color_brightness(base_color, 0.7)  # Adjust brightness
            sim_perc = np.percentile(sims[f"sim_{j}"].iloc[:, i], pvec, method=method)
            quant_1_sim = np.percentile(sims[f"sim_{j}"].iloc[:, i], quantile[0], method=method)
            quant_3_sim = np.percentile(sims[f"sim_{j}"].iloc[:, i], quantile[1], method=method)
            dsim = quant_3_sim - quant_1_sim

            slope = dobs / dsim
            centersim = (quant_1_sim + quant_3_sim) / 2
            centerobs = (quant_1_obs + quant_3_obs) / 2
            maxsim = np.max(sims[f"sim_{j}"].iloc[:, i])
            minsim = np.min(sims[f"sim_{j}"].iloc[:, i])
            maxobs = centerobs + slope * (maxsim - centersim)
            minobs = centerobs - slope * (centersim - minsim)

            msim = np.array([minsim, maxsim])
            mobs = np.array([minobs, maxobs])
            quant_sim = np.array([quant_1_sim, quant_3_sim])
            quant_obs = np.array([quant_1_obs, quant_3_obs]) 

            ax.plot(sim_perc, obs_perc, linestyle=linestyle[0][1:], label=legend[j-1], markersize=2, color=base_color)
            ax.plot(msim, mobs, linestyle=linestyle[1][1:], label=q_labels[0], linewidth=linewidth[0], color=adjusted_color)
            ax.plot(quant_sim, quant_obs, linestyle=linestyle[2][1], label=q_labels[1], marker='o', markerfacecolor='w', linewidth=linewidth[1], color=adjusted_color)

        _finalize_plot(ax, grid, minor_grid, font_size, labels, title, "qqplot", i)

        # Save or auto-save for large column counts
        auto_save = len(obs.columns) > 5
        _save_or_display_plot(fig, save or auto_save, save_as, dir, i, "qqplot")  


def flow_duration_curve(
    merged_df: pd.DataFrame = None, 
    sim_df: pd.DataFrame = None,
    df: pd.DataFrame = None, 
    legend: tuple[str, str] = ('Data',), 
    grid: bool = False,
    minor_grid: bool = False, 
    title: str = None, 
    labels: tuple[str, str] = ('Exceedance Probability (%)', 'Flow'),
    linestyles: tuple[str, str] = ('r-',), 
    linewidth: tuple[float, float] = (1.5,),
    fig_size: tuple[float, float] = (10, 6), 
    font_size: int = 12,
    save: bool = False, 
    save_as: str = None, 
    dir: str = os.getcwd()
) -> plt.figure:
    """
    Generate a Flow Duration Curve (FDC) comparing observed and simulated streamflow.
    
    A Flow Duration Curve (FDC) is a graphical representation of the percentage of time that streamflow is equal to or exceeds a particular value over a given period. 
    It provides insights into the variability and availability of water in a river system, capturing both high and low flow conditions. 
    This function calculates the exceedance probability of observed and simulated streamflow, ranks the values from highest to lowest, and plots them on a probability scale.
    The function allows for flexible customization of labels, legends, transparency, and line styles. 
    The FDC is a crucial tool in hydrology for assessing water availability, evaluating hydrological model performance, and understanding flow regime characteristics.

    
    Parameters
    ----------
    merged_df : pd.DataFrame, optional
        A DataFrame containing both observed and simulated streamflow data. The observed data should be 
        in the even-numbered columns, and the simulated data in the odd-numbered columns.
        
    sim_df : pd.DataFrame, optional
        A DataFrame containing simulated streamflow data. This is used if `merged_df` is not provided.
        
    legend : tuple of str, optional
        A tuple with two string labels for the legend: the first for the simulated data and the second 
        for the observed data. Defaults to ('Simulated Data', 'Observed Data').

    grid : bool, optional
        Whether to display a grid on the plot. Defaults to False.
    
    minor_grid : bool, optional
        Whether to display a minor grid on the plot. Defaults to False.

    title : str, optional
        Title of the plot. If not provided, no title will be displayed.

    labels : tuple of str, optional
        A tuple with two string labels for the x and y axes. Defaults to 
        ('Exceedance Probability (%)', 'Flow (m³/s)').

    linestyles : tuple of str, optional
        A tuple with two strings specifying the line styles for the simulated and observed data, respectively. 
        Defaults to ('r-', 'b-').

    linewidth : tuple of float, optional
        A tuple with two floats specifying the line widths for the simulated and observed data, respectively. 
        Defaults to (1.5, 1.25).

    fig_size : tuple of float, optional
        A tuple with two floats specifying the width and height of the figure in inches. Defaults to (10, 6).

    font_size : int, optional
        Font size for the plot text. Defaults to 12.

    save : bool, optional
        Whether to save the plot as a file. Defaults to False.

    save_as : str, optional
        The file name (with extension) to save the plot as. Only used if `save=True`.

    dir : str, optional
        The directory to save the plot file. Only used if `save=True`. Defaults to the current working directory.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Matplotlib figure instance containing the FDC plot.

    Raises
    ------
    RuntimeError
        If neither `merged_df` nor both `obs_df` and `sim_df` are provided.

    Example
    -------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import postprocessinglib.evaluation.visuals as visuals
    >>> # Example observed and simulated data
    >>> observed = pd.DataFrame(np.random.randn(100, 1), columns=["Flow"])
    >>> simulated = pd.DataFrame(np.random.randn(100, 1), columns=["Flow"])
    >>> visuals.flow_duration_curve(observed=observed, simulated=simulated, title="FDC Example", grid =True)

    .. image:: ../Figures/FDC_Example.png

    >>> # Example merged dataframe
    >>> merged_df = pd.concat([observed, simulated], axis=1)
    >>> visuals.flow_duration_curve(merged_df = merged_df, title="Flow Duration Curve of the Model Result", grid =True)

    .. image:: ../Figures/FDC_Example_2.png

    `JUPYTER NOTEBOOK Examples <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/docs/source/notebooks/tutorial-visualizations.ipynb>`_
    """

    if df is None:
        # Get the number of simulated data columns
        num_sim = sum(1 for col in  merged_df.columns if col[0] == merged_df.columns[0][0])-1 if merged_df is not None else sum(1 for col in  sim_df.columns if col[0] == sim_df.columns[0][0])
        print(f"Number of simulated data columns: {num_sim}")
        # Line width generation
        if len(linewidth) < num_sim + 1:
            print("Number of linewidths provided is less than the number of columns. "
                    "Number of columns : " + str(num_sim + 1) + ". Number of linewidths provided is: ", str(len(linewidth)) +
                    ". Defaulting to 1.5")
            linewidth = linewidth + (1.5,) * (num_sim + 1 if merged_df is not None else num_sim)
        
        # Generate colors dynamically using Matplotlib colormap
        cmap = plt.cm.get_cmap("tab10", num_sim + 1)  # +1 for Observed
        colors = [cmap(i) for i in range(num_sim + 1)]

        # Available line styles
        # base_linestyles = ["-", "--", "-.", ":"]
        style = ('-',) * (num_sim + 1) # default to solid lines unless overwritten

        # Generate linestyles dynamically
        if len(linestyles) < num_sim + 1:
            print("Number of linestyles provided is less than the number of columns. "
                    "Number of columns : " + str(num_sim + 1) + ". Number of linestyles provided is: ", str(len(linestyles)) +
                    ". Defaulting to solid lines (-)")
            linestyles = linestyles + tuple(f"{colors[i % len(colors)]}{style[i % len(style)]}" 
                            for i in range(num_sim + 1 if merged_df is not None else num_sim))
            
        # Generate Legends dynamically
        if len(legend) < num_sim + 1:
            print("Number of legends provided is less than the number of columns. "
                    "Number of columns : " + str(num_sim + 1) + ". Number of legends provided is: ", str(len(legend)) +
                    ". Applying Default legend names")
            legend = (["Observed"] + [f"Simulated {i+1}" for i in range(num_sim)] if merged_df is not None else [f"Simulated {i+1}" for i in range(num_sim)]) 
    
    
    # Assign the data based on inputs
    sims = {}
    obs = None
    if merged_df is not None:
        # If merged_df is provided, separate observed and simulated data
        obs = merged_df.iloc[:, ::num_sim+1]
        for i in range(1, num_sim+1):
            sims[f"sim_{i}"] = merged_df.iloc[:, i::num_sim+1]
        time = merged_df.index
    elif sim_df is not None:
        # If sim_df is provided, that means theres no observed.
        for i in range(0, num_sim):
            sims[f"sim_{i+1}"] = sim_df.iloc[:, i::num_sim]
        time = sim_df.index
    elif df is not None:
        # If only df is provided, it could be either obs, simulated or just random data.
        # obs = df # to keep the future for loop valid
        line_df = df
        time = df.index
    else:
        raise RuntimeError('Please provide valid data (merged_df, sim_df, or df)')

    if df is not None:
        for i in range (0, len(line_df.columns)):
            # Plotting the Data     
            fig, ax = plt.subplots(figsize=fig_size, facecolor='w', edgecolor='k')
            line_df_sorted = np.sort(line_df.iloc[:, i])[::-1]
            exceedance_prob = np.linspace(0, 100, len(line_df_sorted))
            ax.plot(exceedance_prob, line_df_sorted, linestyles[0], label=legend[0], linewidth=linewidth[0])

            _finalize_plot(ax, grid, minor_grid, font_size, labels, title, "fdc-plot", i)
            
            # Save or auto-save for large column counts
            auto_save = len(obs.columns) > 5
            _save_or_display_plot(fig, save or auto_save, save_as, dir, i, "fdc-plot") 
    else:
        # In either case of merged or sim_df, we will alwaays have simulated data, so we plot the obs first if we have it.
        for i in range (0, len(sims["sim_1"].columns)):
            # Plotting the Data     
            fig, ax = plt.subplots(figsize=fig_size, facecolor='w', edgecolor='k')
            if obs is not None:
                obs_sorted = np.sort(obs.iloc[:, i])[::-1]
                exceedance_prob = np.linspace(0, 100, len(obs_sorted))                
                ax.plot(exceedance_prob, obs_sorted, color = eval(linestyles[0][:-1]) if linestyles[0][:-1].startswith("(") else linestyles[0][:-1], 
                        linestyle = linestyles[0][-1],label=legend[0], linewidth = linewidth[0])
            for j in range(1, num_sim+1):
                sim_sorted = np.sort(sims[f"sim_{j}"].iloc[:, i])[::-1]  # Sorting the simulated data
                exceedance_prob = np.linspace(0, 100, len(obs_sorted))
                ax.plot(exceedance_prob, sim_sorted, color = eval(linestyles[j][:-1]) if linestyles[j][:-1].startswith("(") else linestyles[j][:-1],
                        linestyle = linestyles[j][-1], label=legend[j], linewidth = linewidth[j])            

            _finalize_plot(ax, grid, minor_grid, font_size, labels, title, "fdc-plot", i)
    
            # Save or auto-save for large column counts
            auto_save = len(obs.columns) > 5
            _save_or_display_plot(fig, save or auto_save, save_as, dir, i, "fdc-plot") 


