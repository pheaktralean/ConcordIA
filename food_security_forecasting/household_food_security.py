import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('./data/Household_food_security_living_arrangement.csv')

# Drop unneeded columns
cols_to_drop = [
    "DGUID", "VECTOR", "COORDINATE", "STATUS", "SYMBOL",
    "TERMINATED", "UOM_ID", "SCALAR_FACTOR", "SCALAR_ID"
]
df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

# Rename columns
rename_map = {
    "REF_DATE": "Year",
    "GEO": "Region",
    "Living arrangement": "LivingArrangement",
    "Household food security status": "FoodSecurityStatus",
    "Statistics": "StatisticType",
    "UOM": "Unit",
    "VALUE": "Value",
    "DECIMALS": "Decimals"
}
df.rename(columns=rename_map, inplace=True)

# Convert Value to numeric and drop missing rows
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
df = df.dropna(subset=["Value"])

# Extract first 4 digits of Year and convert to int
df["Year"] = df["Year"].astype(str).str[:4].astype(int)

# Drop constant or redundant columns if present
for col in ["Unit", "StatisticType", "Decimals"]:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

future_years = np.arange(2024, 2029).reshape(-1, 1)
categories = df.groupby(["LivingArrangement", "FoodSecurityStatus"])

print(df.groupby(["LivingArrangement", "FoodSecurityStatus"])["Year"].nunique())
for (living_arr, food_status), group in categories:
    # Train model
    X = group[["Year"]]
    y = group["Value"]

    # Skip if not enough data
    if len(X) < 2:
        # Plot only historical data if only one year
        print(f"Only one data point for {living_arr} - {food_status}")
        plt.figure(figsize=(8, 5))
        plt.plot(X, y, "o-", label="Historical")
        plt.title(f"Only one year available: '{living_arr}' - '{food_status}'")
        plt.xlabel("Year")
        plt.ylabel("Food insecurity (%)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        continue

    model = LinearRegression()
    model.fit(X, y)

    # Predict
    future_pred = model.predict(future_years)

    plt.figure(figsize=(8, 5))
    plt.plot(X, y, "o-", label="Historical")
    plt.plot(future_years, future_pred, "x--", label="Predicted", color="orange")
    plt.title(f"Food insecurity prediction for '{living_arr}' - '{food_status}'")
    plt.xlabel("Year")
    plt.ylabel("Food insecurity (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()