# British Columbia Mental Health Dataset

The dataset:
[link](https://catalogue.data.gov.bc.ca/dataset/2e469ff2-dadb-45ea-af9d-f5683a4b9465/resource/de425d68-a87f-44ce-ab4d-fcb514c04975/download/mental-health.csv)


This dataset includes:
- MHSU_GUID : A unique ID for each mental health or substance use service record. Think of it like a "service number" to keep every entry distinct.
- GPCE_TAXONOMY_CLASSIFICATION: The official category or type of service (e.g., “Counselling”, “Addiction Support”, “Crisis Intervention”).
- AUDIENCE: The target group for the service. For example: adults, youth, seniors, families, or general public.
- STUDENTS: Indicates if the service is intended for or includes students (Yes/No or True/False).
- TAXONOMY_NAME: A more specific name of the service type — e.g., “Mental Health Counselling”, “Substance Use Assessment”. It’s like a label that classifies what kind of help is offered.
- RG_NAME: Regional Health Authority name (e.g., “Vancouver Coastal Health”, “Fraser Health”). This shows which region the service belongs to.
- SV_NAME: The actual name of the service provider — e.g., “WoodGreen Community Services” or “CMHA Vancouver Branch”.
- SV_DESCRIPTION: A short text describing what the service does — e.g., “Provides walk-in counselling for individuals experiencing stress or anxiety.”
- PHONE_NUMBER: Contact phone number for the service.
- WEBSITE: The service’s website URL for more info or online appointments.
- EMAIL_ADDRESS: Email address for inquiries or contact.
- WHEELCHAIR_ACCESSIBLE: Indicate whether the service location is wheelchair assessible (yes/no)
- LANGUAGE: Languages in which the service is offered (e.g., English, French, Mandarin).
- HOURS: The open hours or service schedule — e.g., “Mon–Fri: 9am–5pm”.
- STREET_NUMBER:Street address number.
- CITY: The city where the service is located — e.g., “Vancouver”, “Burnaby”.
- PROVINCE: The province — here it should always be “BC” (British Columbia).
- POSTAL_CODE: Postal code of the service’s location.
- LATITUDE: Latitude coordinate for mapping (north-south position).
- LONGITUDE: Longitude coordinate for mapping (east-west position).
- MHSU_SPECIFIC_TERMS_ENHANCED_SEARCH: A list of extra mental health/substance use–related keywords for improved searching (e.g., “anxiety, depression, trauma”).
- KEYWORD_SEARCH_TEXT: General searchable keywords for the service (like tags). Helps users find the service when searching online.


Where and Who? : CITY, PROVINCE, POSTAL_CODE, RG_NAME, AUDIENCE, STUDENTS
\
What service? : SV_NAME, TAXONOMY_NAME, SV_DESCRIPTION, GPCE_TAXONOMY_CLASSIFICATION
\
How to reach? : PHONE_NUMBER, EMAIL_ADDRESS, WEBSITE, HOURS, LANGUAGE, WHEELCHAIR_ACCESSIBLE
\
Mapping? : LATITUDE, LONGITUDE, KEYWORD_SEARCH_TEXT, MHSU_GUID
\

The BC Mental Health dataset is a directory of available services:
It lists where mental health resources exist like Who they serve, What kind of support they provide, and How accessible they are. 
but it doesn’t tell you how many clients come in daily or weekly, so it cannot directly be used to forecast demand trends.


***

# Ontario Disability Support Program – monthly caseload time series
The dataset:
[link](https://ouvert.canada.ca/data/dataset/8b2d095e-1099-44aa-a51c-85d5f74ec1de/resource/d7bbf5e9-983b-41f1-b363-75a84ff76b9a)

# Ontario Works – monthly caseload time series
The dataset:
[link](https://ouvert.canada.ca/data/dataset/8b2d095e-1099-44aa-a51c-85d5f74ec1de/resource/f0297116-6813-4253-ac20-99d1ce7134be)

To prepare the dataset (cleaning), the tasks are:
- Convert Month column to a real date: so models like ARIMA/Prophet can read it as a time index.
- Sort by date: ensures correct chronological order.
- Check for missing or duplicate months: missing months break forecasting models.
- Rename columns to simpler names: easier to use and remember.
- Drop unnecessary columns: if we only need total cases/beneficiaries.
- Set date as index: so we can run .plot() and model time series easily.

