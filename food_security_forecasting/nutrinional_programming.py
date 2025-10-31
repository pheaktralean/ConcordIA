import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('./data/Nutritional_Programming_West.csv')

# Drop hashed columns
hashed_col = [
    'Surname', 'First Given Name', 'Birth Date (YYMMDD)', 'Month of Birth',
    'Day of Birth', 'Social Insurance Number', 'Provincial Health Insurance Number',
    'Home Address', 'Home Address Postal Code', 'emailaddress1_DC', 'mir_primary_phone_DC',
    'Full Name_AC', 'Client Preferred Name_AC', 'Client Supervisor Name_AC', 'Client Address Line 2_AC',
    'Client Email_AC', 'Client Phone (Main)_AC', 'Client Phone (Other)_AC', 'SADDR2_YD', 'Member Name_YD',
    'License Plate Number_YD', 'Property Code_YD', 'MemberFN_YD', 'MemberLN_YD', 'C_Prefername_HB',
    'C_Unit_HB', 'C_mPhoneType_ID_HB', 'C_mPhone_HB', 'C_sPhoneType_ID_HB', 'C_sPhone_HB', 'C_email_HB',
    'hbaddress_HB', 'e_FirstName_HB', 'e_LastName_HB', 'e_Phone_HB', 'e_email_HB', 'RG_Location_HB',
    'UpdateBy_HB', 'Associated Contact_CC', 'Location_CC', 'Current Service Location_CC', 'ClientName_CU',
    'Phone_CU'
]

df.drop(columns=hashed_col, axis=1, inplace=True, errors='ignore') # axis=1 refers to columns

# Drop columns where all values are NaN
df.dropna(axis=1, how='all', inplace=True)

print(df.columns)