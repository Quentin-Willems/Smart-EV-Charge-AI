# feature_engineering.py
# Author: Quentin Willems
# Project: Smart EV Charge recommendations leveraging AI and Elia open data
# Purposes:    1. Handle missing values
#              2. Engineer features to enrich master data for time series predictive modeling

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

###################################
### 0. Load input data          ###
###################################

# Import the `df_input` variable from the postgres_data_loader file (keeping everything in-memory)
try:
    from postgres_data_loader import df_input
    print("The df_input DataFrame was successfully imported from postgres_data_loader.py.")

    # Preview imported DataFrame
    print("\nPreview of the DataFrame imported into feature_engineering.py:")
    print(df_input.head())

except ImportError:
    print("Error: Could not import the DataFrame.")
    print("Please ensure that postgres_data_loader.py exists in the same directory.")
except Exception as e:
    print(f"An error occurred: {e}")


###################################
### 1. Handle missing values    ###
###################################

# Function to identify columns with missing values and impute them using a group-by mean (using mean for two reasons 1. Number of data point by group and 2. better capture time variability )
def fill_missing_values_by_group(df, group_cols):
    """
    Identifies columns with missing values and fills them with the mean of their respective groups.
    
    Args:
        df (pd.DataFrame): The DataFrame to process.
        group_cols (list): A list of column names to use for grouping.
    
    Returns:
        pd.DataFrame: The DataFrame with imputed values.
    """
    df_imputed = df.copy()  # Work on a copy to avoid modifying the original DataFrame directly
    
    # Identify columns with missing values
    columns_with_na = df_imputed.columns[df_imputed.isnull().any()].tolist()
    
    if not columns_with_na:
        print("No columns with missing values found. Imputation skipped.")
        return df_imputed
        
    print("Columns with missing values identified:", columns_with_na)
    
    # Impute missing values for each identified column
    for col in columns_with_na:
        print(f"  > Imputing missing values in '{col}'...")
        grouped_means = df_imputed.groupby(group_cols)[col].transform('mean')
        df_imputed[col] = df_imputed[col].fillna(grouped_means)
        
    return df_imputed


# Use the new function to perform a group-based mean imputation for all columns with NA.
# The `month`, `weekday`, and `hour` columns are used for grouping to capture temporal patterns.
df_input = fill_missing_values_by_group(df_input, ['month', 'weekday', 'hour'])

# Final check to see if all NA values have been filled
final_na_count = df_input.isnull().sum().sum()
print(f"\nTotal NA count in the final DataFrame after imputation: {final_na_count}")

if final_na_count == 0:
    print("Successfully imputed all identified missing values.")
else:
    print("Warning: Some missing values may still remain. This can happen if a group has no valid data to calculate a mean.")

# Display the final DataFrame to show the imputed values
print(f"\nThe final DataFrame looks like this (with imputed values):\n{df_input.head()}")



###################################
### 2. Feature engineering      ###
###################################

###########################################
#### 2.1 Electricity Market Price proxy ###
###########################################

# Create Electricity Market Price proxy leveraging Imbalance observed on the grid (Surplus or Shortage) and Positive Imbalance Price / Negative Imbalance Price
df_input['MarketPriceProxy'] = np.where(
    df_input['avg_systemimbalance'] > 0,
    df_input['avg_positive_price'],
    df_input['avg_negative_price']
)

print("\nSuccessfully created 'MarketPriceProxy' column.")
print("\nDataFrame with the new 'MarketPriceProxy' column:")
print(df_input[['avg_systemimbalance', 'avg_positive_price', 'avg_negative_price', 'MarketPriceProxy']].head())

#########################################
#### 2.2 Photovoltaic (PV) production ###
#########################################

# Drop unecessary aggregated view to avoid multicolinearity (i.e. Keep the PV production by province)
aggregated_regions = ['belgium', 'flanders', 'wallonia']

cols_to_drop = []
for region in aggregated_regions:
    cols_to_drop.append(f'area_{region}_pv_production')
    cols_to_drop.append(f'area_{region}_pv_capacity')
    
df_input.drop(columns=cols_to_drop, inplace=True, errors='ignore')
print(f"Dropped aggregated PV columns: {cols_to_drop}")

# Normalize the feature by looking at utilization rate (i.e. (PV production in an area) / (PV monitored capacity in this area))
provinces = ['antwerp', 'eastflanders', 'flemishbrabant', 'hainaut', 'limburg', 'liège', 'luxembourg', 'namur', 'walloonbrabant', 'westflanders', 'brussels']
cols_to_drop_pv = []
for p in provinces:
    prod_col = f'area_{p}_pv_production'
    cap_col = f'area_{p}_pv_capacity'
    utilization_rate_col = prod_col.replace('_production', '_utilization_rate')
    
    # Calculate the new utilization rate column.
    df_input[utilization_rate_col] = np.where(
        df_input[cap_col] > 0,
        df_input[prod_col] / df_input[cap_col],
        0
    )
    
    # Add original columns to the list to be dropped later.
    cols_to_drop_pv.append(prod_col)
    cols_to_drop_pv.append(cap_col)

# Drop the original production and capacity columns to avoid multicollinearity.
df_input.drop(columns=cols_to_drop_pv, inplace=True, errors='ignore')

print("\nNormalized PV production features created as utilization rates and original columns dropped.")
print("Preview of the new PV utilization rate columns:")
new_pv_columns = [f'area_{p}_pv_utilization_rate' for p in provinces]
print(df_input[new_pv_columns].head())

# Create lagged values to capture temporal effects for the new utilization rate features.

lag_periods = [1, 6, 24] # 1 hour, 6 hours, and 1 day ago.

for col in new_pv_columns:
    for lag in lag_periods:
        new_lag_col = f'{col}_lag_{lag}h'
        df_input[new_lag_col] = df_input[col].shift(periods=lag)

print("\nSuccessfully created lagged features for PV utilization rates.")
print(f"Preview of a lagged feature (e.g., '{new_pv_columns[0]}_lag_1h'):")
print(df_input[[f'{new_pv_columns[0]}', f'{new_pv_columns[0]}_lag_1h']].head())


############################
#### 2.3 Wind production ###
############################

wind_federal = ['federal']
wind_regions = ['flanders', 'wallonia']

# Drop unecessary aggregated view to avoid multicolinearity (i.e. Keep the wind production by main regions)

agg_cols_to_drop_wind = []
for region in wind_federal:
    agg_cols_to_drop_wind.append(f'area_{region}_pv_production')
    agg_cols_to_drop_wind.append(f'area_{region}_pv_capacity')
    
df_input.drop(columns=agg_cols_to_drop_wind, inplace=True, errors='ignore')
print(f"Dropped aggregated PV columns: {agg_cols_to_drop_wind}")

# Normalize the wind production by looking at utilization rate.
cols_to_drop_wind = []
for r in wind_regions:
    # Column names in the DataFrame are in lowercase and match the SQL aliases.
    prod_col = f'area_{r}_wind_production'
    cap_col = f'area_{r}_wind_capacity'
    utilization_rate_col = prod_col.replace('_production', '_utilization_rate')
    
    # Calculate the new utilization rate column.
    df_input[utilization_rate_col] = np.where(
        df_input[cap_col] > 0,
        df_input[prod_col] / df_input[cap_col],
        0
    )
    
    # Add original columns to the list to be dropped later.
    cols_to_drop_wind.append(prod_col)
    cols_to_drop_wind.append(cap_col)

# Drop the original production and capacity columns, ignoring errors if a column is not found.
df_input.drop(columns=cols_to_drop_wind, inplace=True, errors='ignore')

print("\nNormalized wind production features created as utilization rates and original columns dropped.")
print("Preview of the new wind utilization rate columns:")
new_wind_columns = [f'area_{r}_wind_utilization_rate' for r in wind_regions]
print(df_input[new_wind_columns].head())

# Create lagged values to capture temporal effects for the new utilization rate features.
lag_periods = [1, 6, 24] # 1 hour, 6 hours, and 1 day ago.

for col in new_wind_columns:
    for lag in lag_periods:
        new_lag_col = f'{col}_lag_{lag}h'
        df_input[new_lag_col] = df_input[col].shift(periods=lag)

print("\nSuccessfully created lagged features for wind utilization rates.")
print(f"Preview of a lagged feature (e.g., '{new_wind_columns[0]}_lag_1h'):")
print(df_input[[f'{new_wind_columns[0]}', f'{new_wind_columns[0]}_lag_1h']].head())


##################################
#### 2.4 Time Series  features ###
##################################

# Ensure 'hour' and 'weekday' columns are numeric for calculations
df_input['hour'] = pd.to_numeric(df_input['hour'], errors='coerce').astype('Int64')
df_input['weekday'] = pd.to_numeric(df_input['weekday'], errors='coerce').astype('Int64')
df_input['calendar_week'] = pd.to_numeric(df_input['calendar_week'], errors='coerce').astype('Int64')


# Create cyclical features for 'hour'
df_input['hour_sin'] = np.sin(2 * np.pi * df_input['hour'] / 24)
df_input['hour_cos'] = np.cos(2 * np.pi * df_input['hour'] / 24)
print("Created 'hour_sin' and 'hour_cos' features.")

# Create cyclical features for 'weekday'
df_input['weekday_sin'] = np.sin(2 * np.pi * df_input['weekday'] / 7)
df_input['weekday_cos'] = np.cos(2 * np.pi * df_input['weekday'] / 7)
print("Created 'weekday_sin' and 'weekday_cos' features.")

# Create 'is_weekend' binary feature
# Weekday 5 is Saturday, 6 is Sunday (0 is for Monday)
df_input['is_weekend'] = ((df_input['weekday'] == 5) | (df_input['weekday'] == 6)).astype(int)
print("Created 'is_weekend' feature.")