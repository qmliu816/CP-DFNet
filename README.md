# CP-DFNet

This repository provides the source code for **CP-DFNet: A Critical Path Guided Dual-Branch Fusion Network for PM2.5 Forecasting in Urban Agglomerations**.

CP-DFNet is designed for multi-site and multi-horizon PM2.5 forecasting. It constructs critical pollutant transport pathways using geographic distance and historical PM2.5 correlations, and adopts a dual-branch architecture to capture local short-term fluctuations and long-range spatiotemporal dependencies. A particle swarm optimization based fusion module is used to adaptively balance the outputs of the two branches.

## Files

* `main.py`: training and evaluation script.
* `model.py`: implementation of CP-DFNet
* `attention.py`: self-attention layers used in CP-DFNet.
* `Importdata.py`: data loading and preprocessing

## Requirements

Install the required packages using:

```bash
pip install -r requirements.txt
```

Main dependencies include:

```text
torch
numpy
pandas
scikit-learn
netCDF4
haversine
matplotlib
```

## Data

The hourly air quality data used in this study were obtained from the China National Environmental Monitoring Centre: https://www.cnemc.cn/. 

The meteorological data used in this study were accessed through the European Centre for Medium-Range Weather Forecasts: https://cds.climate.copernicus.eu/.

Expected data directory structure:

```text
dataset/
└── stations/
    ├── meteroloy_data/
    │   ├── m202301.nc
    │   ├── ...
    │   └── m202412.nc
    ├── pollute_data/
    │   ├── AQI_stations_2023.csv
    │   ├── AQI_stations_2024.csv
    │   ├── ...
    │   └── PM2.5_stations_2024.csv
    └── stationcoordinates.csv
```

## Usage

Run the model with critical pollutant transport pathways:

```bash
python main.py --use_critical_paths --input_len 48 --out_len 24
```

For other forecasting horizons, change `--out_len`:

```bash
python main.py --use_critical_paths --input_len 48 --out_len 1
python main.py --use_critical_paths --input_len 48 --out_len 3
python main.py --use_critical_paths --input_len 48 --out_len 6
python main.py --use_critical_paths --input_len 48 --out_len 12
python main.py --use_critical_paths --input_len 48 --out_len 24
```
