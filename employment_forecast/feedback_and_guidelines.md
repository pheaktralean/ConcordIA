# Project Guidance – Employment Indicators & Intake Forecasting

## Goal of the Work

The purpose of this project is to use **historical program intake** and **external indicators** (such as employment-related data) to understand whether these signals help explain or predict changes in intake over time.  
Your end goal is to build a **simple, well-structured forecasting model** that:

1. Takes in historical intake numbers  
2. Combines them with relevant employment indicators  
3. Learns how these features relate to intake  
4. Produces reasonable predictions on a hold-out test period  

This doesn't have to be a perfect forecasting — it’s about a clear, transparent analysis and a sound modelling workflow.

---

## What Has Been Attempted So Far

You loaded the employment tables, inspected their structure, and explored some of the fields.  
This is a good start — understanding the data before modelling is essential.

However, the notebook currently:
- Does not define a target variable clearly  
- Does not connect employment indicators to intake data  
- Does not build or evaluate any predictive model  

This means the work stops at exploration and doesn’t yet move toward the forecasting objective. That's ok, I'm adding some guidance below to help you achieve
the remaining steps.

---

## Areas to Improve

### **1. Connecting the datasets**
Create a single dataframe containing:
- `date`  
- `number_of_people` (target you want to predict)  
- employment indicators  
- optional lag features  

### **2. Standardizing the time dimension**
Pick **one frequency** (monthly?), convert everything to it.

### **3. Adding lag features**
Example:
```python
df = df.sort_values("date")
df["intake_lag1"] = df["number_of_people"].shift(1)
df["unemployment_rate_lag1"] = df["unemployment_rate"].shift(1)
```

### **4. Building a baseline model**
Start with a simple linear regression.

### **5. Creating a time-based train/test split**
Train on older data, test on newer data.

---

## Step-by-Step Plan

### **1. Define the problem**
State what you're predicting and why.

### **2. Load and clean the data**
Parse dates, keep relevant columns, resample to weekly.

### **3. Load intake data**
Align to the same frequency.

### **4. Merge datasets**
```python
df_all = df_intake.merge(df_emp, on="date", how="left")
```

### **5. Create lag features**
```python
df_all["intake_lag1"] = df_all["number_of_people"].shift(1)
df_all["employment_lag1"] = df_all["employment_rate"].shift(1)
```

### **6. Drop missing rows after lagging**
```python
df_all = df_all.dropna()
```

### **7. Train/test split**
The dates below are just examples, make sure you adjust to the date in the data.
```python
train = df_all[df_all["date"] < "2022-01-01"]
test  = df_all[df_all["date"] >= "2022-01-01"]
```

### **8. Build a simple model**
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```

### **9. Evaluate**
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
```

### **10. Plot actual vs predicted**
Use a time-based line plot.

---

## Final Notes

You don’t need a complex model.  
A clean dataset, consistent frequency, lag features, and a simple baseline model are enough to demonstrate predictive relationships.
We can try more complex models in another iteration. For now, this structure should give you the backbone needed for iteration.
