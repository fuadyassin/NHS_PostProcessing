Quick Start
============

Prerequisites
-------------
As a first step you need a Python environment with all required dependencies.
The recommended way is to use Anaconda and to create a new environment using our predefined environment files in `environments <https://github.com/fuadyassin/NHS_PostProcessing/tree/main/environments>`_.

Use:

.. code-block::

    conda env create -f environments/environment.yml

Installation
-------------
**The Library is not yet available on PyPi so it will have to be installed directly from the git repo**

To install the library use:

.. code-block::

    pip install git+https://github.com/fuadyassin/NHS_PostProcessing.git


If you want to install an editable version to implement your own models or dataset you'll have to clone the repository using:

.. code-block::

    git clone https://github.com/fuadyassin/NHS_PostProcessing.git


or just download the zip file `here <https://github.com/fuadyassin/NHS_PostProcessing/archive/refs/tags/v1.0.0.zip>`_

After this, you are then left with a directory called *NHS_PostProcessing* or *NHS_PostProcessing-main*.  Next, we'll go to that directory and install a local, editable copy of the package:

.. code-block::

    cd NHS_PostProcessing
    pip install -e .

For the MESH direct-ingestion workflow (``generate_dataframes_from_mesh``), also install:

.. code-block::

    pip install xarray geopandas

Workflows
---------

There are two main entry-point workflows depending on what files you have available.

Workflow A — CSV-based (existing outputs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use this workflow when you already have a ``MESH_output_streamflow.csv``
(produced by an earlier notebook run or a previous script).

.. code-block:: python

    from postprocessinglib.evaluation import data, metrics, visuals

    # Step 1 – load data
    DATAFRAMES = data.generate_dataframes(
        csv_fpaths=["MESH_output_streamflow.csv"],
        warm_up=365,
    )

    # Step 2 – compute metrics
    results = metrics.calculate_all_metrics(
        observed=DATAFRAMES["DF_OBSERVED"],
        simulated=DATAFRAMES["DF_SIMULATED"],
    )
    print(results)

    # Step 3 – plot
    visuals.plot(
        merged_dataframe=DATAFRAMES["DF"],
        num_stations=1,
        title="Streamflow comparison",
    )

Workflow B — MESH NetCDF direct ingestion (no intermediate CSV)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use this workflow to go **directly** from raw MESH model outputs to metrics
and plots without writing an intermediate CSV file.

**Required inputs** (produced once per study domain):

* ``combined_discharge_stations_comids.gpkg`` — gauge stations with COMID
  assignments (from the COMID-matching pre-processing step).
* ``MESH_input_streamflow_latlon.tb0`` — observed streamflow in EnSim ``.tb0``
  format (from ``GenStreamflowAsync``).
* ``MESH_drainage_database_*.nc`` — MESH drainage database NetCDF.
* ``QO_D_GRD.nc`` — MESH simulated streamflow output.

.. code-block:: python

    from postprocessinglib.evaluation import data, metrics, visuals

    # Step 1 – load directly from MESH outputs
    DATAFRAMES = data.generate_dataframes_from_mesh(
        input_stations_comids="combined_discharge_stations_comids.gpkg",
        input_obs="MESH_input_streamflow_latlon.tb0",
        input_ddb="MESH_drainage_database.nc",
        mesh_flow="QO_D_GRD.nc",
        warm_up=365,
    )

    # Step 2 – compute metrics (identical to Workflow A)
    results = metrics.calculate_all_metrics(
        observed=DATAFRAMES["DF_OBSERVED"],
        simulated=DATAFRAMES["DF_SIMULATED"],
    )
    print(results)

    # Step 3 – plot (identical to Workflow A)
    visuals.plot(
        merged_dataframe=DATAFRAMES["DF"],
        num_stations=1,
        title="Streamflow comparison",
    )

Comparing multiple model runs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass a list of NetCDF paths to :func:`~postprocessinglib.evaluation.data.generate_dataframes_from_mesh`
to evaluate several runs side-by-side:

.. code-block:: python

    DATAFRAMES = data.generate_dataframes_from_mesh(
        input_stations_comids="combined_discharge_stations_comids.gpkg",
        input_obs="MESH_input_streamflow_latlon.tb0",
        input_ddb="MESH_drainage_database.nc",
        mesh_flow=[
            "Average_GRU_Params/run_A/QO_D_GRD.nc",
            "Average_GRU_Params/run_B/QO_D_GRD.nc",
        ],
        warm_up=365,
    )

    for i in [1, 2]:
        r = metrics.calculate_all_metrics(
            observed=DATAFRAMES["DF_OBSERVED"],
            simulated=DATAFRAMES[f"DF_SIMULATED_{i}"],
        )
        print(f"Run {i}:\n", r)

With aggregations
~~~~~~~~~~~~~~~~~

Both workflows accept the same aggregation flags:

.. code-block:: python

    DATAFRAMES = data.generate_dataframes_from_mesh(
        ...,
        monthly_agg=True, ma_method="mean",
        yearly_agg=True,  ya_method="sum",
        long_term=True,
        warm_up=365,
    )

    monthly  = DATAFRAMES["DF_MONTHLY"]
    lt_mean  = DATAFRAMES["LONG_TERM_MEAN"]
    lt_min   = DATAFRAMES["LONG_TERM_MIN"]

See :doc:`Data` for the full list of returned dictionary keys and aggregation
options, and :doc:`Metrics` for all available performance metrics.
