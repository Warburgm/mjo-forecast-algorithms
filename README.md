# MJO Forecast Algorithms

Statistical temperature forecasting using the Madden–Julian Oscillation (MJO).

## Overview

This repository contains a set of statistical forecast models that predict local temperature anomalies based on the state and evolution of the Madden–Julian Oscillation (MJO).

The system is built using:

* RMM1 / RMM2 indices
* MJO amplitude
* Lagged autoregressive temperature terms
* Lead-time dependent ridge regression models

Forecasts are currently implemented for:

* Albany, NY
* Chicago, IL

## Methodology

The approach follows a physically interpretable framework:

1. Construct lagged predictors from MJO indices and local temperature
2. Define forecast targets at multiple lead times
3. Fit ridge regression models separately for each lead
4. Evaluate skill across leads and regimes

No machine learning black-box models are used — the system remains fully interpretable.

## Features

* Lead-dependent regression models (0–20 day forecasts)
* Explicit use of MJO phase-space dynamics
* Modular pipeline for adding new locations
* Reproducible workflow

## Data Sources

* MJO RMM Index (BoM)
* Temperature data (ERA5 or nClimGrid)

## Repository Structure

```
src/mjo/
    data.py        # data loading and preprocessing
    features.py    # feature engineering
    models.py      # regression models
    forecast.py    # forecast generation

scripts/
    run_albany.py
    run_chicago.py

notebooks/
    exploration notebooks
```

## Usage

Example:

```
PYTHONPATH=src python scripts/run_albany.py
```

## Future Work

* Add additional cities
* Incorporate probabilistic forecasts
* Explore nonlinear models (RF, boosting)
* Integrate real-time MJO forecasts

## License

MIT License
