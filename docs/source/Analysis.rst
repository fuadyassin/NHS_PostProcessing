Analysis
========

Streamflow Analysis
-------------------
This section runs us through performing an analysis of streamflow data for various stations over three different periods: 1990-2010, 2026-2055, and 2071-2100. 
The analysis includes calculating and visualizing the minimum, maximum, and median streamflow values for each period. 
Additionally, key hydrological metrics such as the Time to Center of Mass (TtCoM), Time to Peak (TtP), and Spring Pulse Onset are computed for each period. 
The results are visualized using filled plots to highlight the variability and central tendency of streamflow over time.

Steps:
------
1. **Load the Streamflow Data**:
    - Import the necessary libraries for data manipulation and visualization.
    - Use `pandas` to select datetime period, `glob` to organize the multiple csv file and `natsort` for organization.
    - Most of evrerything is done in the `postprocessinglib` library.
    - Define the input path to the CSV files containing streamflow data.
    - Pass the path to the `generate_dataframes()` function from the `data` module to read the data into a DataFrame.
2. **Process and Aggregate the Data**:
    - use the `long_term_seasonal()` and `stat_aggregate()` functions from the `data` module to aggregate the data year day and model.
    - Compute the minimum, maximum, and median streamflow values for each station and period.
3. **Visualize the Streamflow Data**:
    - Use `bounded_plot()` function from the `visuals` section to create filled plots for each period.
    - Plot the minimum, maximum, and median streamflow values for each station.
    - Customize the plots with titles, labels, and legends for clarity.
    - **Compute Hydrological Metrics**:
        - Calculate key hydrological metrics such as Time to Center of Mass (TtCoM), Time to Peak (TtP), and Spring Pulse Onset for each period


.. note::
  - The code is designed to be run in a Jupyter Notebook or Python script with the necessary libraries installed.
  - Ensure that the input data files are correctly formatted and located in the specified directory.
  - The code assumes that the streamflow data is structured with columns for date, station ID, and streamflow values.
  - Ensure that your environment contains all the necessary dependencies, including `pandas`, `matplotlib`, and `postprocessinglib` as defined in the environment file linked here 
    `environment.yml <https://github.com/UchechukwuUdenze/NHS_PostProcessing/tree/main/environments>`_.
  - Ensure that the `postprocessinglib` library is installed and properly configured in your Python environment.


.. toctree::
   :maxdepth: 1

   notebooks/Streamflow_Analysis.ipynb


This section shows the steps involved with processing streamflow data for multiple stations over three distinct periods. 
The data is aggregated by Julian date to compute the mean streamflow for each day of the year. The minimum, maximum, and median streamflow values are calculated for each station and period. 
Key hydrological metrics (TtCoM, TtP, Spring Pulse Onset) are computed for the median values. The results are visualized using filled plots to highlight variability and central tendency, with annotations showing the calculated metrics and their differences relative to the baseline period.

Single Model Analysis
----------------------

.. toctree::
   :maxdepth: 1

   notebooks/Single-Model-Analysis.ipynb


This section provides a comprehensive guide to performing a single model analysis using the `postprocessinglib` library.
The analysis focuses on evaluating the performance of a single hydrological model across multiple stations and periods.

Multimodel Analysis
-------------------

.. toctree::
   :maxdepth: 1

   notebooks/Multi-Model-Analysis.ipynb

This section provides a comprehensive guide to performing a multimodel analysis using the `postprocessinglib` library.
The analysis focuses on evaluating the performance of multiple hydrological models across various stations and periods.


Forecast Analysis
-----------------

.. toctree::
   :maxdepth: 1

   notebooks/Forecast_Analysis.ipynb

This section provides a comprehensive guide to performing a forecast analysis using the `postprocessinglib` library.
The analysis focuses on evaluating the performance of forecasted streamflow data across multiple stations and periods.


If you have any comments, please reach out to Uchechukwu UDENZE at uchechukwu.udenze@ec.gc.ca or Fuad Yassin at fuad.yassin@ec.gc.ca.