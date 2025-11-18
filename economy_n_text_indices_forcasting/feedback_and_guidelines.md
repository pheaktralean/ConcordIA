# Project Guidelines – Macro, Search Index, and Program Intake Forecasting

## Nice Stuff Already Implemented 

- Each notebook explores a different piece of the problem (program intake, macro-financial data, search index data).
- Data cleaning steps are mostly sensible: parsing dates, identifying missingness, and distinguishing numerical vs categorical columns. That's great stuff!
- Weekly resampling is being used consistently across macro and index datasets.
- Lagged correlations and preliminary EDA show good intuition about how past values relate to current intake.

## Potential Bugs and Issues

- In the index preprocessing notebook, categorical checks on `df_index` use `df[col]` perhaps by mistake.
- Program intake data moves between daily and weekly formats without clearly fixing a single frequency.
- There’s no consistent “canonical” cleaned table for:
  - weekly program intake,
  - weekly macro data,
  - weekly index data.
- Merging rules (e.g., how dates align across datasets) aren’t clearly defined.
- Risk of data leakage if lagged features aren’t handled carefully.

## Overall Goal

Build a single, coherent dataset where each row represents **one program in one week**, along with:
- the **weekly intake** (target),
- the **weekly macro-financial features**,
- the **weekly search/text index features**,
- optional program-level features.

Then, train a simple, time-aware model to see whether macro and index signals help predict program intake.  
The goal is not long-term forecasting—just showing that a model can reproduce and explain recent patterns.

## Structure Suggestion for a Consolidated Notebook

### **1. Define the problem**
Explain briefly what you’re trying to do:
> Create a weekly forecasting dataset that merges program intake with macro and search index features, and evaluate whether these features help predict week-to-week changes in intake.

Example:
- target = weekly `number_of_people`, per program  
- frequency = **weekly** (?)
- evaluation = last few weeks as a hold-out test (this means that training and evaluation do not see this data)

### **2. Build a clean weekly intake dataset** from Woodgreen's data:
- Parse dates
- Decide on a weekly definition (`resample("W")`)
- Aggregate intake per program per week
- Output a clean table with:  
  `date, program_name, number_of_people, (optional program stats)`

This becomes your **primary target dataset**.

### **3. Build weekly macro and search index datasets**
From the macro and index notebooks:
- Parse dates
- Resample both datasets to **the same weekly frequency**
- Ensure ranges match the intake dataset
- Keep only meaningful numeric features
- Output:  
  `df_macro_weekly` and `df_index_weekly`

Fix the bug in the index notebook when inspecting columns.

### **4. Merge everything**
- Merge intake with macro on `date`
- Merge the result with index features on `date`
- Keep only weeks where all needed features exist
- You can filter by a specific program or keep all programs, up to you

The resulting table should look like:

`date | program_name | number_of_people | macro_features... | index_features...`


### **5. Feature engineering**
- Add simple **lag features**, e.g.:

```python
df = df.sort_values(["program_name", "date"])
df["number_of_people_lag1"] = df.groupby("program_name")["number_of_people"].shift(1)
```

- Use lags for macro/index features only if justified.
- Avoid leakage: all lags must use shifted past values, never the future.

### **6. Split Data into Train and Test **

train = df[df["date"] <= cutoff_date]
test  = df[df["date"] >  cutoff_date]



### **7. Build a simple model**

Start simple:

- Linear Regression (or Ridge)
- Compare against a naive baseline: predict this week = last week
- Evaluate with MAE or RMSE, and plot actual vs predicted intake
- Goal: show whether macro/index features help over the naive baseline.

Here's a code snippet that can help you with some of the steps. Disclaimer: I haven't tested this code, and it's AI-generated haha so you might need to adjust. 
I just wanted a quick example, and this is sound enough as a structure and starting point.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# -----------------------------
# 1. Time-based train/test split
# -----------------------------
# Choose a cutoff date for the test period
cutoff_date = pd.to_datetime("2023-01-01")  # adjust to your data range

df = df.sort_values(["program_name", "date"])

train = df[df["date"] < cutoff_date].copy()
test  = df[df["date"] >= cutoff_date].copy()

# -----------------------------
# 2. Define features and target
# -----------------------------
feature_cols = [
    "cpi_change",              # example macro feature
    "unemployment_rate",       # example macro feature
    "tax_search_index",        # example search/index feature
    "foodbank_search_index",   # example search/index feature
    "number_of_people_lag1",   # lag of the target
]

target_col = "number_of_people"

X_train = train[feature_cols]
y_train = train[target_col]
X_test  = test[feature_cols]
y_test  = test[target_col]

# -----------------------------
# 3. Train a simple model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# -----------------------------
# 4. Naive baseline (last week)
# -----------------------------
# Here the naive prediction for each week is just last week's intake
y_naive = test["number_of_people_lag1"].to_numpy()

# -----------------------------
# 5. Compute errors (MAE, RMSE)
# -----------------------------
def mae_rmse(y_true, y_hat):
    mae = mean_absolute_error(y_true, y_hat)
    rmse = np.sqrt(mean_squared_error(y_true, y_hat))
    return mae, rmse

mae_model, rmse_model = mae_rmse(y_test, y_pred)
mae_naive, rmse_naive = mae_rmse(y_test, y_naive)

print(f"Model  - MAE: {mae_model:.2f}, RMSE: {rmse_model:.2f}")
print(f"Naive  - MAE: {mae_naive:.2f}, RMSE: {rmse_naive:.2f}")

# -----------------------------
# 6. Plot actual vs predicted over time for one program
# -----------------------------
program = "Tax Clinic"  # pick a program that exists in your data

test_prog = test[test["program_name"] == program].copy()
test_prog = test_prog.sort_values("date")

X_test_prog = test_prog[feature_cols]
y_test_prog = test_prog[target_col]
y_pred_prog = model.predict(X_test_prog)

plt.figure(figsize=(10, 5))
plt.plot(test_prog["date"], y_test_prog, marker="o", label="Actual")
plt.plot(test_prog["date"], y_pred_prog, marker="x", linestyle="--", label="Predicted")

plt.title(f"Actual vs Predicted Weekly Intake – {program}")
plt.xlabel("Week")
plt.ylabel("Number of people")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# 7. Optional: overall scatter
# -----------------------------
plt.figure(figsize=(5, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
plt.xlabel("Actual intake")
plt.ylabel("Predicted intake")
plt.title("Actual vs Predicted (all programs, test period)")
plt.grid(True)
plt.tight_layout()
plt.show()
```


