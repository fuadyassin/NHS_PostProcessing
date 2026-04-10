Data Processing/Manipulation
=============================

This section describes the data ingestion and manipulation layer of the library.
It is the backbone of the library — its outputs feed directly into both the
:doc:`Metrics` and :doc:`Visualization` sections.

Overview
--------

There are two ways to load data into the library:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Function
     - When to use
   * - :func:`generate_dataframes`
     - You already have a ``MESH_output_streamflow.csv`` (or similar pre-built CSV
       with interleaved ``QOMEAS_`` / ``QOSIM_`` columns).
   * - :func:`generate_dataframes_from_mesh`
     - You want to go **directly** from raw MESH NetCDF outputs (``QO_D_GRD.nc``)
       and a ``.tb0`` observed file without writing an intermediate CSV first.

Both functions return the **same dictionary structure**, so all downstream
metric and visualisation calls are identical regardless of which loader you use.

.. _mesh-workflow:

Reading MESH outputs directly (``generate_dataframes_from_mesh``)
------------------------------------------------------------------

This function combines the steps that were previously handled by the
``combine_mesh_sim_obs`` notebook script:

1. Reads the MESH drainage-database NetCDF to get the ``subbasin`` → array-index
   mapping.
2. Reads the stations GeoPackage (produced by the COMID-matching pre-processing
   step) to build the station → COMID lookup table.
3. Reads simulated streamflow from one or more ``QO_D_GRD.nc`` NetCDF files.
4. Reads observed streamflow from the ``.tb0`` EnSim file, parsing the
   ``:StartTime`` header automatically.
5. Aligns observed and simulated data to their overlapping date range.
6. Returns the same ``DATAFRAMES`` dictionary as :func:`generate_dataframes`.

**Required pre-processing files** (produced once per study domain):

* ``combined_discharge_stations_comids.gpkg`` — gauge stations with COMID
  assignments (produced by the COMID-matching notebook).
* ``MESH_input_streamflow_latlon.tb0`` — observed streamflow in EnSim format
  (produced by ``GenStreamflowAsync``).
* ``MESH_drainage_database_*.nc`` — MESH drainage database NetCDF.

Single model run
~~~~~~~~~~~~~~~~

.. code-block:: python

   from postprocessinglib.evaluation import data, metrics, visuals

   DATAFRAMES = data.generate_dataframes_from_mesh(
       input_stations_comids="combined_discharge_stations_comids.gpkg",
       input_obs="MESH_input_streamflow_latlon.tb0",
       input_ddb="MESH_drainage_database.nc",
       mesh_flow="QO_D_GRD.nc",
       warm_up=365,
   )

   # Metrics — identical to the CSV-based workflow
   results = metrics.calculate_all_metrics(
       observed=DATAFRAMES["DF_OBSERVED"],
       simulated=DATAFRAMES["DF_SIMULATED"],
   )

Multiple model runs
~~~~~~~~~~~~~~~~~~~

Pass a list of NetCDF paths to compare several runs at once. The function
returns one ``DF_SIMULATED_1``, ``DF_SIMULATED_2``, … key per run:

.. code-block:: python

   DATAFRAMES = data.generate_dataframes_from_mesh(
       input_stations_comids="combined_discharge_stations_comids.gpkg",
       input_obs="MESH_input_streamflow_latlon.tb0",
       input_ddb="MESH_drainage_database.nc",
       mesh_flow=[
           "run1/QO_D_GRD.nc",
           "run2/QO_D_GRD.nc",
       ],
       warm_up=365,
   )

   results_run1 = metrics.calculate_all_metrics(
       observed=DATAFRAMES["DF_OBSERVED"],
       simulated=DATAFRAMES["DF_SIMULATED_1"],
   )
   results_run2 = metrics.calculate_all_metrics(
       observed=DATAFRAMES["DF_OBSERVED"],
       simulated=DATAFRAMES["DF_SIMULATED_2"],
   )

Station filtering and aggregation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both loaders accept the same filtering and aggregation flags:

.. code-block:: python

   DATAFRAMES = data.generate_dataframes_from_mesh(
       ...,
       keep_stations=["05BB001", "05BA001"],   # subset of stations
       monthly_agg=True, ma_method="mean",
       yearly_agg=True,  ya_method="sum",
       long_term=True,
       warm_up=365,
   )

   # Extra keys are ready to use immediately
   monthly  = DATAFRAMES["DF_MONTHLY"]
   lt_mean  = DATAFRAMES["LONG_TERM_MEAN"]

Reading from a pre-built CSV (``generate_dataframes``)
-------------------------------------------------------

If you have already exported a ``MESH_output_streamflow.csv`` (e.g. from a
previous notebook run), use :func:`generate_dataframes` directly:

.. code-block:: python

   DATAFRAMES = data.generate_dataframes(
       csv_fpaths=["MESH_output_streamflow.csv"],
       warm_up=365,
   )

For multiple CSV files (one per model run):

.. code-block:: python

   DATAFRAMES = data.generate_dataframes(
       csv_fpaths=[
           "run1/MESH_output_streamflow.csv",
           "run2/MESH_output_streamflow.csv",
       ],
       warm_up=365,
   )

Returned dictionary keys
------------------------

Both loaders return a dictionary with the following keys:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Key
     - Content
   * - ``"DF"``
     - Flat merged DataFrame (single run).
   * - ``"DF_1"``, ``"DF_2"``, …
     - Flat merged DataFrames, one per run (multiple runs).
   * - ``"DF_OBSERVED"``
     - Observed-only DataFrame (``QOMEAS_*`` columns).
   * - ``"DF_SIMULATED"``
     - Simulated-only DataFrame (single run).
   * - ``"DF_SIMULATED_1"``, …
     - Per-run simulated DataFrames (multiple runs).
   * - ``"DF_MERGED"``
     - MultiIndex-column DataFrame (station × variable).
   * - ``"DF_DAILY"``
     - Daily aggregate (when ``daily_agg=True``).
   * - ``"DF_WEEKLY"``
     - Weekly aggregate (when ``weekly_agg=True``).
   * - ``"DF_MONTHLY"``
     - Monthly aggregate (when ``monthly_agg=True``).
   * - ``"DF_YEARLY"``
     - Yearly aggregate (when ``yearly_agg=True``).
   * - ``"DF_CUSTOM"``
     - Seasonal-period subset (when ``seasonal_p=True``).
   * - ``"LONG_TERM_MIN"`` / ``"MAX"`` / ``"MEDIAN"``
     - Long-term seasonal aggregates (when ``long_term=True``).
   * - ``"DF_STATS"``
     - Cross-simulation statistics (when ``stat_agg=True``).

Aggregation helpers
-------------------

The following standalone helpers can also be applied to any DataFrame with a
``DatetimeIndex``:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Function
     - Description
   * - :func:`daily_aggregate`
     - Aggregate by day of year.
   * - :func:`weekly_aggregate`
     - Aggregate by calendar week.
   * - :func:`monthly_aggregate`
     - Aggregate by month.
   * - :func:`yearly_aggregate`
     - Aggregate by year.
   * - :func:`long_term_seasonal`
     - Long-term seasonal mean/min/max/quantile.
   * - :func:`seasonal_period`
     - Slice to a recurring calendar window.
   * - :func:`stat_aggregate`
     - Aggregate across multiple simulation columns.
   * - :func:`twelve_hour_aggregate`
     - Aggregate to 12-hour intervals.
   * - :func:`station_dataframe`
     - Extract individual station DataFrames.

.. currentmodule:: postprocessinglib.evaluation.data

.. rubric:: Loaders

.. autofunction:: generate_dataframes_from_mesh
.. autofunction:: generate_dataframes

.. rubric:: Aggregation helpers

.. autofunction:: daily_aggregate
.. autofunction:: weekly_aggregate
.. autofunction:: monthly_aggregate
.. autofunction:: yearly_aggregate
.. autofunction:: twelve_hour_aggregate
.. autofunction:: long_term_seasonal
.. autofunction:: seasonal_period
.. autofunction:: stat_aggregate
.. autofunction:: station_dataframe
