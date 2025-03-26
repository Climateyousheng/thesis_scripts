
# Scripts for THESIS

Thesis is authored by Yousheng Li, who could be contacted at <ys.li@bristol.ac.uk>

We aim to add all relevant scripts to this repo. 

Local copy sits under:`~/docs/scripts/scripts_thesis`


## Chapter global climate

### 1. Model-data comparison

1. Proxy records collection and processing
2. Model data processing
3. Model-data comparison


### 2. EBM analysis

This is a fit version for local laptops with all the locals file readable.

We'll put the eocene solid version here and make further changes.

Currently, `make_EBM_fluxes_yoush.py` is the standard file to produce normal EBM analysis plots for 12 sets of expts, file `12sims_make.py` is likely a copy of it.
Files, `amoc_EBM_fluxes.py`, `co2_sensitivity_EBM_fluxes.py`, `humidity_analysis.py` are different versions of codes used to produce standard EBM analysis plots for AMOC series expts (based on PI), for CO<sub>2</sub> sensitivity expts based on the 280 ppm expt, and for humidity analysis for standard global climate expts for 4 paleogeogs under 3 CO<sub>2</sub> scenarios.

## Chapter AMOC

### 1. AMOC time series

Figure 2 in the AMOC paper (AMOC time series, waterflux bar chart and linear regression between AMOC strength and net fresh waterflux) is produced by `AMOC_time_series_topography_sensitivity_expts.py`.

### 2. Waterflux


## Chapter Ocean Biogeochemistry