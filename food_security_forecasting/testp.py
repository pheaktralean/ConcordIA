import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

demographic_df = pd.read_csv('./data/Food_insecurity_selected_demographic_characteristics.csv')
economic_df = pd.read_csv('./data/Food_insecurity_economic_family_type.csv')

scaler = StandardScaler()

demographic_cols_drop = [
    'DGUID', 'VECTOR', 'COORDINATE', 'STATUS', 'SYMBOL',
    'TERMINATED', 'UOM_ID', 'SCALAR_FACTOR', 'SCALAR_ID',
    'Statistics', 'UOM', 'DECIMALS'
]
demographic_df.drop(columns=[c for c in demographic_cols_drop if c in demographic_df.columns], inplace=True)

rename_demographic = {
    'REF_DATE': 'Year',
    'GEO': 'Region',
    'Demographic characteristics': 'Demographic',
    'Household food security status': 'FoodSecurityStatus',
    'VALUE': 'Value',
}
demographic_df.rename(columns=rename_demographic, inplace=True)

demographic_df['Year'] = pd.to_numeric(demographic_df['Year'], errors='coerce')
demographic_df = pd.get_dummies(demographic_df, columns=['Demographic', 'FoodSecurityStatus'], drop_first=True)


economic_cols_drop = [
    'DGUID', 'VECTOR', 'COORDINATE', 'STATUS', 'SYMBOL',
    'TERMINATED', 'UOM_ID', 'SCALAR_FACTOR', 'SCALAR_ID',
    'Statistics', 'UOM', 'DECIMALS'
]
economic_df.drop(columns=[c for c in economic_cols_drop if c in economic_df.columns], inplace=True)

rename_economic = {
    'REF_DATE': 'Year',
    'GEO': 'Region',
    'Economic family type': 'EconomicType',
    'Household food security status': 'FoodSecurityStatus',
    'VALUE': 'Value',
}
economic_df.rename(columns=rename_economic, inplace=True)

economic_df['Year'] = pd.to_numeric(economic_df['Year'], errors='coerce')
economic_df = pd.get_dummies(economic_df, columns=['EconomicType', 'FoodSecurityStatus'], drop_first=True)

merged_df = pd.merge(
    demographic_df, economic_df,
    on=['Region', 'Year'],
    how='left'
)
merged_df.info()

merged_df = merged_df.sort_values(['Year'])

merged_df['Intake_lag1'] = merged_df.groupby('Region')['Demographic_Persons 18 to 24 years'].shift(1)

# Fit only on training numeric columns
feature_cols = ['Year', 'Value_x', 'Value_y']

train = merged_df[merged_df['Year'] <= 2019]
test  = merged_df[merged_df['Year'] >  2019]

feature_cols = ['Year', 'Demographic_Persons 18 to 24 years', 'Value_x', 'Intake_lag1']
X_train, y_train = train[feature_cols], train['Demographic_Persons 18 to 24 years']
X_test,  y_test  = test[feature_cols], test['Demographic_Persons 18 to 24 years']

# Scale numeric features
scaler.fit(X_train[feature_cols])


X_train[feature_cols] = scaler.transform(X_train[feature_cols])
X_test[feature_cols]  = scaler.transform(X_test[feature_cols])