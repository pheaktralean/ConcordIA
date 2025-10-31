import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('./data/Food_insecurity_selected_demographic_characteristics.csv')

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
    "Household food security status": "FoodSecurityStatus",
    "Statistics": "StatisticType",
    "UOM": "Unit",
    "VALUE": "Value",
    "DECIMALS": "Decimals"
}
df.rename(columns=rename_map, inplace=True)

# Convert numeric
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
df["Year"] = pd.to_numeric(df["Year"].astype(str).str[:4], errors="coerce")

# Drop missing values
df.dropna(subset=["Value", "Year"], inplace=True)

# Clean whitespace
for col in ["Region", "FoodSecurityStatus", "Demographic characteristics"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# Example: Focus only on "Canada" to avoid repetition
df_canada = df[df["Region"].str.contains("Canada", case=False, na=False)]

# Keep only food insecurity (exclude "Food secure")
df_insecure = df_canada[df_canada["FoodSecurityStatus"].str.contains("Food insecure", case=False, na=False)]

future_years = np.arange(2024, 2029).reshape(-1, 1)  # Predict next 5 years

for demo_group, group in df_insecure.groupby("Demographic characteristics"):
    X = group[["Year"]]
    y = group["Value"]

    # Skip if insufficient data
    if len(X) < 2:
        continue

    # Train linear regression
    model = LinearRegression()
    model.fit(X, y)

    # Predict future values
    future_pred = model.predict(future_years)

    # Plot historical data
    plt.figure(figsize=(8, 5))
    plt.plot(X, y, "o-", label="Historical", color="blue")
    plt.plot(future_years, future_pred, "x--", label="Predicted (2024–2028)", color="orange")

    plt.title(f"Predicted Food Insecurity for '{demo_group}'")
    plt.xlabel("Year")
    plt.ylabel("Percentage (%)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()