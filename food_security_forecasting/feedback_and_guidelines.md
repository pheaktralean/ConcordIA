# Project Directions – Integrating Food Insecurity, Demographics, and Nutritional Program Data

### Current Analysis

We have three separate notebooks:
- **Food insecurity by family type** (`economic_family_type`)
- **Demographics** (`selected_demographic`)
- **Nutritional program intake** (`nutritional_programming`)

Each one analyzes part of the story, but they’re disconnected.  
Right now:
- Datasets use different column names and formats.
- Years and regions don’t always line up.
- Some variables (like `Year`) are text instead of numbers.
- Forecasting is happening without checking how good the predictions actually are.
- There’s no clear, consistent structure across the notebooks.

---

### What We’re Trying to Do

The goal is to connect all three datasets and use them to **explain and predict changes in nutritional program intake** over time.  

In other words:  
> We want to see how socio-economic and demographic factors impact the demand for nutritional programs, and use that to forecast future intake.

Each dataset plays a role:
| Dataset | What it represents | Why it matters |
|----------|--------------------|----------------|
| `economic_family_type` | Food insecurity levels | Reflects social and economic pressure |
| `selected_demographic` | Population breakdowns | Gives context about who lives where |
| `nutritional_programming` | Program participation | The outcome we’re trying to forecast |

---

### General Plan

Below is the structure you could follow to make it easier to tackle this problem. Of course, you can adjust details as needed, but aim for this overall flow.

---

#### **1. Define the problem**
Start your notebook with a short paragraph summarizing what you’re doing — the goal, the target variable, and what you expect to find.  
Example:
> We’re building a simple model to forecast nutritional program intake using food insecurity and demographic data. The goal is to understand how population and economic context relate to participation over time.

---

#### **2. Load the datasets**
Load all three datasets in the same notebook.  
Check that columns are named consistently (`Year`, `Region`, etc.) and that `Year` is numeric.  
Quickly inspect them with `.head()` and `.info()` before doing any heavy cleaning.

Use names like `df_econ`, `df_demo`, `df_intake` to load the datasets so that you know how to refer to them in different parts of the notebook.

---

#### **3. Clean and align**
Make the datasets compatible so they can be merged later.  
Keep only what’s relevant (e.g., year, region, and any columns that you feel are relevant).  
Fix small issues:
- Rename inconsistent columns (so that it's easier to use them for merging)
- Convert year to numeric
- Strip extra spaces from string columns
- Handle missing values reasonably

You should end up with three clean DataFrames ready to merge. Here, in the intake data, you should see what you're trying to predict in terms of granularity. If you're looking
into predicting for a year or for a month, you should make sure that your dataset is at that granularity. Let's say, if you have intake per day, you can group by month or year to achieve 
the granularity you want to predict. I'll leave this up to you to decide. Same goes for the other data.

---

#### **4. Explore the data**
Before modeling, spend a bit of time exploring trends:
- Food insecurity vs. year (or month, if available)
- Demographics vs. year (or month, if available)
- Intake vs. year (or month, if available)
Look for obvious patterns, anomalies, or gaps.  
The goal here is to understand the data — not to overanalyze it.
You've done some of this already in your notebook, but it's worth quickly looking into this again with the clean data.
You're trying to see here if you can spot any patterns, etc.

---

#### **5. Merge the datasets**
Merge everything by year (or month) and region (and/or family type if available, or any other common columns that you feel is a common factor across all three datasets).  
After merging, check that the number of rows and regions make sense (just to make sure that you're not getting duplicates). The merging should be done in a way that you think of it as expanding/enhancing the intake data.
If there are missing values after merging, note where they come from.

Example:
```python
df_all = (
    df_intake
      .merge(df_food[["Year","Region","FoodInsecurityRate"]], on=["Year","Region"], how="left")
      .merge(df_demo[["Year","Region","Population","MedianAge"]], on=["Year","Region"], how="left")
)
```

Again, this is just an example, feel free to include as many columns as you find necessary, but if you want to start small that's ok too. The overall structure of the notebook
will give you flexibility to change this later.

#### **6. Prepare for modeling**

Decide what you’ll predict — in this case, `Intake` — and choose which variables will be used as predictors.  
These might include things like `FoodInsecurityRate`, `Population`, `MedianAge`, and `Year`.  

It can also help to include **lag features**, which represent the value of a variable in a previous year (for example, last year’s intake).  
Lags capture short-term memory in the data — they let the model “see” how past values influence the present.  
This is often useful when the outcome changes gradually over time or when there’s persistence from one year to the next.

Example of adding a lag feature:

```python
# Create a 1-year lag of intake for each region
df_all = df_all.sort_values(["Region", "Year"])
df_all["Intake_lag1"] = df_all.groupby("Region")["Intake"].shift(1)

Example:

```python
train = df_all[df_all["Year"] <= 2019]
test  = df_all[df_all["Year"] >  2019]

feature_cols = ["Year","FoodInsecurityRate","Population","MedianAge","Intake_lag1"]
X_train, y_train = train[feature_cols], train["Intake"]
X_test,  y_test  = test[feature_cols],  test["Intake"]
```

##### **Data Transformations Summary**

Before modeling, we need to make sure all variables are in a form the model can understand and learn from.  
Here’s a quick summary of the key transformations and why they matter:

| Transformation | Goal | When / Why to Use | Example |
|----------------|------|------------------|----------|
| **Lag features** | Capture the influence of past values on the present | When the target depends on its own history (common in time series) | `df["Intake_lag1"] = df.groupby("Region")["Intake"].shift(1)` |
| **One-hot encoding** | Convert categorical data into numeric form | When you have columns like `Region` or `FamilyType` | `df = pd.get_dummies(df, columns=["Region"], drop_first=True)` |
| **Scaling** | Put all numeric features on a similar range so no variable dominates | Needed for models like Linear Regression, Ridge, or Lasso | `scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)` |
| **Log transform** | Reduce the impact of very large values and skewed distributions | Useful for population counts or income-like variables | `df["Population_log"] = np.log1p(df["Population"])` |
| **Time-based split** | Respect the order of time (no leakage from the future) | Always, when doing forecasting | `train = df[df["Year"] <= 2019]; test = df[df["Year"] > 2019]` |

These steps help the model learn more reliably, make fair comparisons between variables, and prevent common forecasting mistakes such as using future data in training or letting one large-scale feature dominate the model.




#### **7. Build a simple model**

Start with a straightforward baseline like **Linear Regression**.  
The idea is to check whether features such as food insecurity and demographics can explain variations in program intake over time.

Compare your model against a simple baseline — for example, predicting that this year’s value is the same as last year’s (the “naive” approach).  
This helps confirm whether the model actually learns something meaningful.

Use a time-based train/test split (for example, train on data up to 2018 or 2019, test on the remaining years).  
Evaluate results using metrics like **Mean Absolute Error (MAE)** or **Root Mean Square Error (RMSE)**.

Focus on understanding and reasoning:
- Are predictions following the general trend?  
- Does the model capture the ups and downs reasonably well?  
- Which features seem to have the strongest relationship with intake?

Example of building a simple model:
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train_scaled, y_train)
```

Computing errors:
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
```

---

#### **8. Evaluate and visualize results**

Once you have predictions for the test period, plot **actual vs. predicted** intake values over time.  
This helps visually confirm whether the model is capturing patterns or lagging behind.

You don’t need to forecast years into the future — the focus is on demonstrating that the model generalizes well within the historical data that’s already available.


Plotting to visualize predictions
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8,4))
plt.plot(test["Year"], y_test, label="Actual", marker="o")
plt.plot(test["Year"], y_pred, label="Predicted", marker="x")
plt.title("Actual vs Predicted Nutritional Program Intake")
plt.xlabel("Year")
plt.ylabel("Intake")
plt.legend()
plt.grid(True)
plt.show()
```

---

#### **9. Wrap up**

As you look at the results, try to answer the following questions:
- Which variables helped explain intake the most?  
- How well did the model perform compared to the naive baseline?  
- What are the main limitations of the data or approach?

The goal here is not to predict far ahead, but to show that the model can **reproduce and explain past trends reliably** and provide insights about what drives changes in program intake.

