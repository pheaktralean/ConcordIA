import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

df_intake = pd.read_csv('./data/Nutritional_Programming_West.csv')
demographic_df = pd.read_csv('./data/Food_insecurity_selected_demographic_characteristics.csv')
economic_df = pd.read_csv('./data/Food_insecurity_economic_family_type.csv')

# Drop rows that are not associated with Nutritional Programming - West
df_intake = df_intake[df_intake['Program_CU'] == 'Nutritional Programming - West']
# Convert Creation Date to a year
df_intake['Creation Date'] = pd.to_datetime(df_intake['Creation Date'], errors='coerce')
# Convert year into numerical
df_intake['Year'] = df_intake['Creation Date'].dt.year.astype('Int64')
# Convert year into numerical
df_intake['Year of Birth'] = df_intake['Year of Birth'].astype('Int64')
# All users come from canada
df_intake['Region'] = 'Canada'
nutritional_cols_drop = [
    'Creation Date', 'Last Modified'
]
df_intake.drop(columns=[c for c in nutritional_cols_drop if c in df_intake.columns], inplace=True)
# Replace Nan values
df_intake = df_intake.fillna('Unknown')

grouped_sex_citizen = df_intake.groupby(['Citizenship_CU', 'Sex']).size().reset_index(name='Count')
print(grouped_sex_citizen)

plt.figure(figsize=(12, 6))
sns.barplot(x='Citizenship_CU', y='Count', hue='Sex', data=grouped_sex_citizen)
plt.title('Nutritional Programming by Citizenship and Sex')
plt.show()

grouped_sex_marital = df_intake.groupby(['Marital Status', 'Sex']).size().reset_index(name='Count')
print(grouped_sex_marital)

plt.figure(figsize=(12, 6))
sns.barplot(x='Marital Status', y='Count', hue='Sex', data=grouped_sex_marital)
plt.title('Nutritional Programming by Marital Status and Sex')
plt.show()

grouped_sex_culture = df_intake.groupby(['Culture_CU', 'Sex']).size().reset_index(name='Count')
print(grouped_sex_culture)

plt.figure(figsize=(12, 6))
sns.barplot(x='Culture_CU', y='Count', hue='Sex', data=grouped_sex_culture)
plt.title('Nutritional Programming by Culture and Sex')
plt.show()

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
demographic_df['Value'] = pd.to_numeric(demographic_df['Value'], errors='coerce')
# demographic_df = pd.get_dummies(demographic_df, columns=['Demographic', 'FoodSecurityStatus'], drop_first=False)

# Pivot table so each demographic is a column
demographic_pivot = demographic_df.pivot_table(
    index='Year',
    columns='Demographic',
    values='Value',
    aggfunc='sum'
).reset_index()

age_order = [
    'Persons under 18 years',
    'Persons 18 to 24 years',
    'Persons 25 to 34 years',
    'Persons 35 to 44 years',
    'Persons 45 to 54 years',
    'Persons 55 to 64 years',
    'Persons 65 years and over'
]

# Sort columns: age groups first in order, then any other demographics
demographic_cols = [col for col in age_order if col in demographic_pivot.columns] + \
                   [col for col in demographic_pivot.columns if col not in age_order + ['Year']]

plt.figure(figsize=(12,6))
for col in demographic_cols:
    plt.plot(demographic_pivot['Year'], demographic_pivot[col], marker='o', label=col)

plt.xlabel('Year')
plt.ylabel('Population')
plt.title('Demographics vs Year')
plt.legend()
plt.grid(True)
plt.show()

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
economic_df['Value'] = pd.to_numeric(economic_df['Value'], errors='coerce')

# Pivot table so each demographic is a column
economic_pivot = economic_df.pivot_table(
    index='Year',
    columns='EconomicType',
    values='Value',
    aggfunc='sum'
).reset_index()

economic_order = [
    'EconomicType_Non-seniors not in an economic family',
    'EconomicType_Persons in couple families with children',
    'EconomicType_Persons in economic families',
    'EconomicType_Persons in lone-parent families',
    'EconomicType_Persons in non-senior families',
    'EconomicType_Persons in senior families',
    'EconomicType_Persons not in an economic family',
    'EconomicType_Seniors not in an economic family'
]

# Sort columns: age groups first in order, then any other economics
economic_cols = [col for col in economic_order if col in economic_pivot.columns] + \
                [col for col in economic_pivot.columns if col not in economic_order + ['Year']]

plt.figure(figsize=(12,6))
for col in economic_cols:
    plt.plot(economic_pivot['Year'], economic_pivot[col], marker='o', label=col)

plt.xlabel('Year')
plt.ylabel('Population')
plt.title('Economics vs Year')
plt.legend()
plt.grid(True)
plt.show()

# Pivot table so each demographic is a column
insecurity_pivot = economic_df.pivot_table(
    index='Year',
    columns='FoodSecurityStatus',
    values='Value',
    aggfunc='sum'
).reset_index()

insecurity_order = [
    'FoodSecurityStatus_Food insecure',
    'FoodSecurityStatus_Food insecure, moderate or severe'
]

# Sort columns: age groups first in order, then any other economics
insecurity_cols = [col for col in insecurity_order if col in insecurity_pivot.columns] + \
                  [col for col in insecurity_pivot.columns if col not in insecurity_order + ['Year']]

plt.figure(figsize=(12,6))
for col in insecurity_cols:
    plt.plot(insecurity_pivot['Year'], insecurity_pivot[col], marker='o', label=col)

plt.xlabel('Year')
plt.ylabel('Population')
plt.title('Food Insecurity vs Year')
plt.legend()
plt.grid(True)
plt.show()

intake_agg = (
    df_intake.groupby(['Year','Region'])
    .size()
    .reset_index(name='Intake')
)

# Merge economic data
df_all = intake_agg.merge(
    economic_df, 
    on=['Year','Region'], 
    how='left'
)

# Merge demographic data
df_all = df_all.merge(
    demographic_df, 
    on=['Year','Region'], 
    how='left'
)

# Replace Nan values
df_all = df_all.fillna('Unknown')

# Transform values into numeric
df_all['Value_x'] = pd.to_numeric(df_all['Value_x'], errors='coerce')
df_all['Value_y'] = pd.to_numeric(df_all['Value_y'], errors='coerce')

# One-Hot Encode categorical columns
df_all = pd.get_dummies(
    df_all, 
    columns=['EconomicType', 'Demographic', 'FoodSecurityStatus_x', 'FoodSecurityStatus_y'], 
    drop_first=True
)

# Sort by year and region
df_all = df_all.sort_values(['Region','Year'])

# 1-year lag of Intake
df_all['Intake_lag1'] = df_all.groupby('Region')['Intake'].shift(1)

# Handle missing values
df_all['Intake_lag1'].fillna(method='bfill', inplace=True)

train = df_all[df_all['Year'] <= 2019]
test  = df_all[df_all['Year'] >  2019]

feature_cols = ['Year', 'Intake_lag1']
X_train, y_train = train[feature_cols], train['Intake']
X_test, y_test   = test[feature_cols], test['Intake']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')

plt.figure(figsize=(8,4))
plt.plot(test['Year'], y_test, label='Actual', marker='o')
plt.plot(test['Year'], y_pred, label='Predicted', marker='x')
plt.title('Actual vs Predicted Nutritional Program Intake')
plt.xlabel('Year')
plt.ylabel('Intake')
plt.legend()
plt.grid(True)
plt.show()