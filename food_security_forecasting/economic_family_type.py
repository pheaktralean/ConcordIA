import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('./data/Food_insecurity_economic_family_type.csv')

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
    "Economic family type": "FamilyType",
    "Household food security status": "FoodSecurityStatus",
    "Statistics": "StatisticType",
    "UOM": "Unit",
    "VALUE": "Value",
    "DECIMALS": "Decimals"
}
df.rename(columns=rename_map, inplace=True)

df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
df = df.dropna(subset=["Value"])

for col in ["Year", "Region", "FamilyType", "FoodSecurityStatus"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

future_years = np.arange(2024, 2029).reshape(-1, 1)
categories = df.groupby(["FamilyType", "FoodSecurityStatus"])

for (family_type, food_status), group in categories:
    # Skip if not enough data points
    if len(group["Year"].unique()) < 3:
        continue

    # Train model
    X = group[["Year"]]
    y = group["Value"]
    model = LinearRegression()
    model.fit(X, y)

    # Predict
    future_pred = model.predict(future_years)

    plt.figure(figsize=(8, 5))
    plt.plot(group["Year"], group["Value"], "o-", label="Historical")
    plt.plot(future_years, future_pred, "x--", label="Predicted")
    plt.title(f"Food insecurity prediction for '{family_type}' - '{food_status}'")
    plt.xlabel("Year")
    plt.ylabel("Food insecurity (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
