# Employment Forecasting - Ontario

## Overview
LSTM-based employment forecasting model for Ontario, analyzing WoodGreen Community Services data combined with Statistics Canada employment statistics.

## Structure

```
employment_forecast/
├── data/                 # Data files (under 100MB)
├── jupyter_notebooks/    # Analysis notebooks
├── results/             # Visualization outputs
└── README.md
```

## Main Analysis
**[Employment_Forecasting_LSTM.ipynb](jupyter_notebooks/Employment_Forecasting_LSTM.ipynb)** - Primary LSTM forecasting model

## Data Files
- `employment_monthly.csv` - Monthly employment data
- `statcan_data.csv` - Statistics Canada data
- `employment_cleaned_DC.csv` - Cleaned employment data
- `employment_by_year_and_program.xlsx` - Employment by year/program
- `employment_registrations_by_year_and_program.xlsx` - Registration data
- `monthly_registrations_2024.csv` - 2024 monthly registrations
- `employment_status_counts.csv` - Employment status counts

**Note:** Large data files (>100MB) excluded:
- WoodGreen_All_Systems.csv (444MB)
- WoodGreen_Unlocked.xlsx (116MB)

## Notebooks
- `Employment_Forecasting_LSTM.ipynb` - Main LSTM forecasting analysis
- `Employment_WoodGreen.ipynb` - WoodGreen-specific analysis
- `eda_mix_woodgreen_stats_canada.ipynb` - Exploratory data analysis

## Results
- `november_2025_prediction.png` - November 2025 prediction
- `slide1_model_overview.png` - Model overview
- `slide2_results_forecast.png` - Results & forecast
- `woodgreen_finetuning_results.png` - Fine-tuning results
- `woodgreen_full_timeline.png` - Full timeline

## Requirements
Python 3.x, Jupyter, pandas, numpy, matplotlib, scikit-learn, PyTorch

## Usage
Open `jupyter_notebooks/Employment_Forecasting_LSTM.ipynb` and run the cells.
