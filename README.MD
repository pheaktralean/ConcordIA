# Let's Solve it (Fall 2025) with RBC Borealis & WoodGreen Community Services

## Project Statement

Across housing, mental health, and financial support programs, resource allocation remains predominantly reactive. Intake surges often strain staff, delay service delivery, and disrupt continuity of care. Current planning processes rely on historical averages or anecdotal predictions, lacking the ability to anticipate fluctuations driven by seasonal, economic, or policy-related shifts. This results in misaligned staffing, under- or over-utilization of resources, and service bottlenecks during periods of high demand.

Anticipating program-level demand supports proactive staffing, inventory planning, and funder alignment. Forecasting intake volumes allows organizations to adjust resource levels ahead of time, reducing wait times, improving service equity, and enhancing readiness across critical support areas such as shelters, rent assistance, employment services, and food security. This improves both operational efficiency and client outcomes.

## Project Overview

This project focuses on forecasting demand for various social service programs at **WoodGreen Community Services**. By integrating internal program intake data with external **macro-economic indicators** and **text search indices** (e.g., Google Trends), the goal is to build predictive models that help anticipate service needs across different domains such as Mental Health, Shelter, Employment, and Food Security.

The project leverages advanced time-series forecasting techniques, including **Lag-Llama** (a foundation model for time series) and **LSTM** networks, alongside traditional regression baselines.

## Problem Statement & Goals

Social service providers often face fluctuating demand that can be difficult to predict using only historical intake data. External factors—such as inflation, unemployment rates, and online search behaviors—can serve as leading indicators for these trends.

**Key Goals:**
- **Data Integration:** Create a unified weekly dataset merging internal intake numbers with external macro and search-based features.
- **Forecasting:** Predict weekly program intake to specific services (e.g., Tax Clinic, Family Newcomer Services).
- **Insight:** Understand the correlation between economic stressors (inflation, housing costs) and the demand for social services.

## Key Features & Pipeline

1.  **Data Cleaning & Standardization**
    - Pipelines to parse inconsistent dates and aggregate daily intake records into **weekly** time-series.
    - Specific robust cleaning scripts (e.g., for Shelter occupancy) to handle multiple CSV formats.

2.  **Feature Engineering**
    - **Macro-Financial:** Integration of CPI, unemployment rates, and other economic indices.
    - **Search Indices:** Utilization of search momentum (text indices) as exogenous variables.
    - **Lags:** Generation of lagged features to prevent data leakage and capture temporal dependencies.

3.  **Modeling Approaches**
    - **Baseline:** Linear/Ridge Regression and Naive (persistence) models.
    - **Deep Learning:** LSTM (Long Short-Term Memory) networks for sequential pattern recognition.
    - **Foundation Models:** Fine-tuning **Lag-Llama** for zero-shot or few-shot probabilistic forecasting.

## Tech Stack

- **Language:** Python 3.x
- **Data Manipulation:** Pandas, NumPy
- **Analysis & Visualization:** Jupyter Notebooks, Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn, PyTorch (for Deep Learning models)
- **Time Series:** Lag-Llama (GluonTS/HuggingFace), Statsmodels
- **Web/API (Potential):** Flask, SQLAlchemy (referenced in requirements for potential dashboarding/API layers)

## Repository Structure

```text
├── economy_n_text_indices_forcasting/   # Macro data processing & Lag-Llama models
│   ├── data_preprocessing_n_model_ready/ # Prepared datasets
│   ├── data_preprocessing_notebook/      # Notebooks for macro/index EDA
│   ├── lag_llama_model/                  # Application of Lag-Llama to specific programs
│   └── feedback_and_guidelines.md        # Project guidelines & methodology notes
│
├── employment_forecast/                 # Employment servcies forecasting
│   └── jupyter_notebooks/                # LSTM & EDA notebooks for employment data
│
├── ethnicity_forecasting/               # Analysis of demographics & ethnicity trends
│
├── food_security_forecasting/           # Food security program analysis
│
├── mental_health_forecasting/           # Mental health services analysis
│   ├── notebooks/                        # Analysis notebooks
│   └── data/                             # Raw/processed data
│
├── shelter_forecasting/                 # Shelter occupancy forecasting
│   ├── clean_shelter_data_csvs.py        # Robust script for cleaning shelter CSVs
│   └── notebook/                         # EDA and cleaning notebooks
│
├── woodgreen-data/                      # General data folder (WoodGreen specific)
│
├── requirements.txt                     # Python dependencies
└── README.md                            # Project documentation
```

## Data Notes

- **WoodGreen Data:** Internal private data containing program intake numbers. Should be placed in the respective `data/` folders or `woodgreen-data/`.
- **Macro/Index Data:** Publicly available economic indices and search trend data, preprocessed in the `economy_n_text_indices_forcasting` module.
- **Frequency:** The primary analysis unit is **Weekly**. Ensure all data is resampled to a weekly frequency (e.g., `W-MON`) before merging.

## Experiments & Results

Results are generated per-module within Jupyter Notebooks. Key findings can be found in:
- **Lag-Llama Performance:** Check `economy_n_text_indices_forcasting/lag_llama_model/` for comparisons of Lag-Llama vs. baselines on specific programs (Tax Clinic, Newcomer Services).
- **Employment Trends:** See `employment_forecast/jupyter_notebooks/Employment_Forecasting_LSTM.ipynb`.
- **Data Validation:** `shelter_forecasting/clean_shelter_data_csvs.py` generates validation reports and charts in `shelter_forecasting/data/curated/`.

## Acknowledgement
Team ConcordIA is very pleased and honoured to participate in RBC Borealis Let's Solve it Fall 2025. Thank you Antonio for helping us and guiding us through this project. Thank you WoodGreen organization for providing us your project proposal and data. 
