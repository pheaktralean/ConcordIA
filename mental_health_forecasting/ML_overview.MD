## ML Problem Statement
We aim to forecast the number of individuals receiving social assistance in Ontario using historical caseload data from the Ontario Works (OW) and Ontario Disability Support Program (ODSP).
By analyzing monthly caseload trends, we seek to build a time series forecasting model that predicts future demand for social assistance over the next 3 to 6 months. This can help policymakers and service providers anticipate program needs, allocate resources efficiently, and identify potential social or economic shifts that impact community well-being.


### Problem framed in ML terms:
<b>“Given historical monthly caseload data (features: month, year, and past caseload values), predict the total number of caseloads for upcoming months.”</b>



Forecast horizon: 3–6 months \
Granularity: Monthly \
Model type: Time series regression / forecasting (e.g., ARIMA, Prophet, LSTM) \
Target variable: Total caseload count \
Evaluation metrics: MAE, RMSE, or MAPE to measure forecasting accuracy. \
Success criteria: Forecasting future caseloads with an average error margin below 5–10% for short-term predictions. 