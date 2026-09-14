#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 14:19:18 2025

@author: iris

Camarlinghi Shonchalay
CS332 Fall 2025
Module: 7
Description:  Individual Project Assignment - Unsupervised Learning with KMeans Clustering 
Data source 1: Kaggle dataset manual download:  https://www.kaggle.com/datasets/shubhamcodez/s-and-p-500-daily-options-data-2010-2023 
               Kaggle dataset bash CLI download
                    pip install kaggle   # Install Kaggle API
                    kaggle datasets download -d shubhamcodez/s-and-p-500-daily-options-data-2010-2023   # Download dataset:
                    unzip s-and-p-500-daily-options-data-2010-2023.zip
Data source 2: CBOE SPX options data, downloaded manually from   https://www.cboe.com/delayed_quotes/spx/quote_table  
Ref: https://gatesboltonanalytics.com/?page_id=262 
    
"""

import os
import pandas as pd
import numpy as np
import statistics as stat
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
from scipy.stats import norm
from datetime import datetime, timedelta, date
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

pd.options.display.float_format = '{:,.4f}'.format

CS332_dir = "./"
HW_dir = "archive"
input_file = "spx_quotedata_raw.csv"
output_file1 = "spx_quotedata_cleaned.csv"
output_file2 = "spx_quotedata_reformatted.csv"
output_file3 = "spx_quotedata_reformatted_GEX.csv"


raw_filename = os.path.join(CS332_dir, HW_dir, input_file)
clean_filename_full = os.path.join(CS332_dir, HW_dir, output_file1)
clean_filename_reformatted = os.path.join(CS332_dir, HW_dir, output_file2)
clean_filename_reformatted_3 = os.path.join(CS332_dir, HW_dir, output_file3)
 
"""
#############################  RAW DATA EXPLORATION ###################################################
# A)   Read in the dataset as a dataframe.
print("Ingesting raw file into raw dataframe: ", raw_filename)  
df = pd.read_csv(raw_filename, low_memory=False)   # load large data into pandas dataframe object
#pd.set_option("display.max_columns", 40)  # set option for dataframe to show all column

print("Type of the object df")
print(type(df))
#   Use ndarray.shape: a tuple representing the dimensionality of the DataFrame.
print("\nShape (rows, columns) of dataframe: ", df.shape)

# B)  Print the dataframe
pd.set_option("display.max_columns", 22) # to print all 32 columns.
print("\nDisplaying first 10 rows: ")
print(df.head(10)) 


# D) Datatypes exploration and identifying duplicate, abnormal, missing datatypes.
 
# print technical summarywith datatypes.
 
print("\n\nGetting general info about dataset (including data types):  ")
print(df.info())
print(df.dtypes, df.head(1))




# Create a DataFrame to display both information                        
column_info_raw = pd.DataFrame({
    'Data Type': df.dtypes,
    'NaN Count': df.isna().sum(),
    'Example values': [df[c].unique()[:1].tolist() for c in df.columns]
})
print("\nRaw data column info: Data Type and NaN Count")
print(column_info_raw)

# Ref: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.round.html 
#print("\nColumn names: Total missing values")
# List total number of missing variables in each column:
#for name in df.columns.values: print(name, ": ",  df[name].isna().sum())


### Create and print a list of columns with  non-numeric data
    # and print a list of columns with numeric data only
    # Ref: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.select_dtypes.html 
print("\nA list of columns with non-numeric data: ")
non_numeric_columns = df.select_dtypes(exclude=["number"]).columns.tolist()
print(non_numeric_columns)

print("\nA list of columns with only numeric data: ")
numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
print(numeric_columns) 

  
# Drop rows or columns that have NaN values.
#Ref: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html 
nan_columns_mask = df.isnull().any()
columns_with_nan = nan_columns_mask[nan_columns_mask].index.tolist() # Filter the column names based on the mask
print("\nA list of columns with NaN values: \n", columns_with_nan)



#############################  END OF RAW DATA EXPLORATION #############################################


#############################  RAW DATA CLEANING ########################################################### 
print("\n\n\n###############################################################################################")
print("############################    RAW DATA CLEANING  ##################################################")
'''
#  1.  Drop columns with redundant Unix time 
drop_unix_time_dates = ['QUOTE_READTIME', 'QUOTE_UNIXTIME', 'EXPIRE_UNIX', 'QUOTE_TIME_HOURS']
for name in drop_unix_time_dates:
    df.drop(name, inplace=True, axis=1)

# 2.  Convert Date and Time columns (QUOTE_DATE,EXPIRE_DATE, QUOTE_TIME_HOURS) to Datetime format
date_col = ['QUOTE_DATE', 'EXPIRE_DATE']
time_col = ['QUOTE_TIME_HOURS']
for col in date_col:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        
# 3. transform DTE (Days to Expiration) and volume columns are integers
#int_col = ['DTE', 'C_VOLUME', 'P_VOLUME']
#for col in int_col:
#    if col in df.columns:
#        df[col] = pd.to_numeric(df[col], errors='coerce'.fillna(0).astype(int))
        
# 4. Convert categorical descriptors to strings.
categor_col = ['C_SIZE', 'P_SIZE']
for col in categor_col:
    if col in df.columns:
        df[col] = df[col].astype(str)
        

# 5. Convert all numeric columns to float for uniformity
float_columns = [
    'UNDERLYING_LAST', 'STRIKE', 'STRIKE_DISTANCE', 'STRIKE_DISTANCE_PCT',
    'C_DELTA', 'C_GAMMA', 'C_VEGA', 'C_THETA', 'C_RHO', 'C_IV',
    'C_LAST', 'C_BID', 'C_ASK',
    'P_DELTA', 'P_GAMMA', 'P_VEGA', 'P_THETA', 'P_RHO', 'P_IV',
    'P_LAST', 'P_BID', 'P_ASK']
for col in float_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        
# 6. Handle missing values
    # Get a boolean Series indicating which columns have at least one NaN
    # and put median value of the column for each NaN value: https://gatesboltonanalytics.com/?page_id=403 
nan_columns_mask = df.isnull().any()
columns_with_nan = nan_columns_mask[nan_columns_mask].index.tolist() # Filter the column names based on the mask
fill_dict = {}

for col in columns_with_nan:
    if col in df.columns:
        if col in columns_with_nan:
            fill_dict[col] = df[col].median()
      #  else:
            # For other columns, fill with NaN explicitly to keep consistent style,
            # or fill with 0 (or another value) if you prefer
      #      fill_dict[col] = np.zero

df.fillna(fill_dict, inplace=True)


 
# 7. Remove any duplicates
df.drop_duplicates(inplace=True)

 '''
 

#############################  END OF RAW DATA CLEANING ###########################################################


#############################  PRINT AND SAVE CLEAN DATA  ###########################################################
## F) Print cleaned dataframe

column_info_clean = pd.DataFrame({
    'Data Type': df.dtypes,
    'NaN Count': df.isna().sum(),
    'Example values': [df[c].unique()[:1].tolist() for c in df.columns]
})
print("\nCleaned Column Information:")
print(column_info_clean)
print("\n\nShape of cleaned dataframe: ", df.shape)
print(df.head(10)) 


##  Save cleaned dataframe to csv output file.

df.to_csv(clean_filename_full, index=False)
print("Saved cleaned data in: ", clean_filename_full )


#############################  END OF PRINT AND SAVE CLEAN DATA   ########################################################### 
"""

############################# Visualization and PLOTS ##################################################


# Inputs and Parameters
filename = 'spx_quotedata_raw.csv'

# Black-Scholes European-Options Gamma
def calcGammaEx(S, K, vol, T, r, q, optType, OI):
    if T == 0 or vol == 0:
        return 0

    dp = (np.log(S/K) + (r - q + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    dm = dp - vol*np.sqrt(T) 

    if optType == 'call':
        gamma = np.exp(-q*T) * norm.pdf(dp) / (S * vol * np.sqrt(T))
        return OI * 100 * S * S * 0.01 * gamma 
    else: # Gamma is same for calls and puts. This is just to cross-check
        gamma = K * np.exp(-r*T) * norm.pdf(dm) / (S * S * vol * np.sqrt(T))
        return OI * 100 * S * S * 0.01 * gamma 

def isThirdFriday(d):
    return d.weekday() == 4 and 15 <= d.day <= 21

# This assumes the CBOE file format hasn't been edited, i.e. table beginds at line 4
optionsFile = open(filename)
optionsFileData = optionsFile.readlines()
optionsFile.close()

# Get SPX Spot
spotLine = optionsFileData[1]
spotPrice = float(spotLine.split('Last:')[1].split(',')[0])
fromStrike = 0.8 * spotPrice
toStrike = 1.2 * spotPrice

# Get Today's Date
"""
dateLine = optionsFileData[2]
todayDate = dateLine.split('Date: ')[1].split(' ')
monthDay = todayDate[0].split(' ')
print(todayDate[2])
print(len(todayDate))

# Handling of US/EU date formats
if len(monthDay) == 2:
    year = int(todayDate[1])
    month = monthDay[0]
    day = int(monthDay[1])
else:
    year = int(monthDay[2])
    month = monthDay[1]
    day = int(monthDay[0])
"""
todayDate = datetime.strptime("September",'%B')
todayDate = todayDate.replace(day=11, year=2025)

# Get SPX Options Data
df = pd.read_csv(filename, sep=",", header=None, skiprows=4)
print("Raw data before processing: ")
column_info_clean = pd.DataFrame({
    'Data Type': df.dtypes,
    'NaN Count': df.isna().sum(),
    'Example values': [df[c].unique()[:1].tolist() for c in df.columns]
})
print("\nCleaned Column Information:")
print(column_info_clean)
print("\n\nShape of cleaned dataframe: ", df.shape)
print(df.head(10)) 


df.columns = ['ExpirationDate','Calls','CallLastSale','CallNet','CallBid','CallAsk','CallVol',
              'CallIV','CallDelta','CallGamma','CallOpenInt','StrikePrice','Puts','PutLastSale',
              'PutNet','PutBid','PutAsk','PutVol','PutIV','PutDelta','PutGamma','PutOpenInt']

df['ExpirationDate'] = pd.to_datetime(df['ExpirationDate'], format='%a %b %d %Y')
df['ExpirationDate'] = df['ExpirationDate'] + timedelta(hours=16)
df['StrikePrice'] = df['StrikePrice'].astype(float)
df['CallIV'] = df['CallIV'].astype(float)
df['PutIV'] = df['PutIV'].astype(float)
df['CallGamma'] = df['CallGamma'].astype(float)
df['PutGamma'] = df['PutGamma'].astype(float)
df['CallOpenInt'] = df['CallOpenInt'].astype(float)
df['PutOpenInt'] = df['PutOpenInt'].astype(float)


## F) Print cleaned dataframe

column_info_clean = pd.DataFrame({
    'Data Type': df.dtypes,
    'NaN Count': df.isna().sum(),
    'Example values': [df[c].unique()[:1].tolist() for c in df.columns]
})
print("\nCleaned Column Information:")
print(column_info_clean)
print("\n\nShape of cleaned dataframe: ", df.shape)
print(df.head(10)) 

df.to_csv(clean_filename_reformatted_3, index=False)
print("Saved cleaned data in: ", clean_filename_reformatted_3 )


# ---=== CALCULATE SPOT GAMMA ===---
# Gamma Exposure = Unit Gamma * Open Interest * Contract Size * Spot Price 
# To further convert into 'per 1% move' quantity, multiply by 1% of spotPrice
df['CallGEX'] = df['CallGamma'] * df['CallOpenInt'] * 100 * spotPrice * spotPrice * 0.01
df['PutGEX'] = df['PutGamma'] * df['PutOpenInt'] * 100 * spotPrice * spotPrice * 0.01 * -1

df['TotalGamma'] = (df.CallGEX + df.PutGEX) / 10**9
dfAgg = df.groupby(['StrikePrice']).sum(numeric_only=True)
strikes = dfAgg.index.values

# Chart 1: Absolute Gamma Exposure
plt.grid()
plt.bar(strikes, dfAgg['TotalGamma'].to_numpy(), width=6, linewidth=0.1, edgecolor='k', label="Gamma Exposure")
plt.xlim([fromStrike, toStrike])
chartTitle = "Total Gamma: $" + str("{:.2f}".format(df['TotalGamma'].sum())) + " Bn per 1% SPX Move"
plt.title(chartTitle, fontweight="normal", fontsize=12)
plt.xlabel('Strike', fontweight="normal")
plt.ylabel('Spot Gamma Exposure ($ billions/1% move)', fontweight="normal")
plt.axvline(x=spotPrice, color='r', lw=1, label="SPX Spot: " + str("{:,.0f}".format(spotPrice)))
plt.legend()
#plt.show()
plt.savefig("images/AbsGammaExp.png")
plt.close()

# Chart 2: Absolute Gamma Exposure by Calls and Puts
plt.grid()
plt.bar(strikes, dfAgg['CallGEX'].to_numpy() / 10**9, width=6, linewidth=0.1, edgecolor='k', label="Call Gamma")
plt.bar(strikes, dfAgg['PutGEX'].to_numpy() / 10**9, width=6, linewidth=0.1, edgecolor='k', label="Put Gamma")
plt.xlim([fromStrike, toStrike])
chartTitle = "Total Gamma: $" + str("{:.2f}".format(df['TotalGamma'].sum())) + " Bn per 1% SPX Move"
plt.title(chartTitle, fontweight="normal", fontsize=12)
plt.xlabel('Strike', fontweight="normal")
plt.ylabel('Spot Gamma Exposure ($ billions/1% move)', fontweight="normal")
plt.axvline(x=spotPrice, color='r', lw=1, label="SPX Spot:" + str("{:,.0f}".format(spotPrice)))
plt.legend()
#plt.show()
plt.savefig("images/GammaExpCallPuts.png")
plt.close()


###### CALCULATE GAMMA PROFILE ##########
levels = np.linspace(fromStrike, toStrike, 60)

# For 0DTE options, I'm setting DTE = 1 day, otherwise they get excluded
df['daysTillExp'] = [1/262 if (np.busday_count(todayDate.date(), x.date())) == 0 \
                           else np.busday_count(todayDate.date(), x.date())/262 for x in df.ExpirationDate]

nextExpiry = df['ExpirationDate'].min()

df['IsThirdFriday'] = [isThirdFriday(x) for x in df.ExpirationDate]
thirdFridays = df.loc[df['IsThirdFriday'] == True]
nextMonthlyExp = thirdFridays['ExpirationDate'].min()

totalGamma = []
totalGammaExNext = []
totalGammaExFri = []

# For each spot level, calc gamma exposure at that point
for level in levels:
    df['callGammaEx'] = df.apply(lambda row : calcGammaEx(level, row['StrikePrice'], row['CallIV'], 
                                                          row['daysTillExp'], 0, 0, "call", row['CallOpenInt']), axis = 1)

    df['putGammaEx'] = df.apply(lambda row : calcGammaEx(level, row['StrikePrice'], row['PutIV'], 
                                                         row['daysTillExp'], 0, 0, "put", row['PutOpenInt']), axis = 1)    

    totalGamma.append(df['callGammaEx'].sum() - df['putGammaEx'].sum())

    exNxt = df.loc[df['ExpirationDate'] != nextExpiry]
    totalGammaExNext.append(exNxt['callGammaEx'].sum() - exNxt['putGammaEx'].sum())

    exFri = df.loc[df['ExpirationDate'] != nextMonthlyExp]
    totalGammaExFri.append(exFri['callGammaEx'].sum() - exFri['putGammaEx'].sum())

totalGamma = np.array(totalGamma) / 10**9
totalGammaExNext = np.array(totalGammaExNext) / 10**9
totalGammaExFri = np.array(totalGammaExFri) / 10**9

# Find Gamma Flip Point
zeroCrossIdx = np.where(np.diff(np.sign(totalGamma)))[0]

negGamma = totalGamma[zeroCrossIdx]
posGamma = totalGamma[zeroCrossIdx+1]
negStrike = levels[zeroCrossIdx]
posStrike = levels[zeroCrossIdx+1]

zeroGamma = posStrike - ((posStrike - negStrike) * posGamma/(posGamma-negGamma))
zeroGamma = zeroGamma[0]

# Chart 3: Gamma Exposure Profile
fig, ax = plt.subplots()
plt.grid()
plt.plot(levels, totalGamma, label="All Expiries")
plt.plot(levels, totalGammaExNext, label="Ex-Next Expiry")
plt.plot(levels, totalGammaExFri, label="Ex-Next Monthly Expiry")
chartTitle = "Gamma Exposure Profile, SPX, " + todayDate.strftime('%d %b %Y')
plt.title(chartTitle, fontweight="bold", fontsize=20)
plt.xlabel('Index Price', fontweight="bold")
plt.ylabel('Gamma Exposure ($ billions/1% move)', fontweight="bold")
plt.axvline(x=spotPrice, color='r', lw=1, label="SPX Spot: " + str("{:,.0f}".format(spotPrice)))
plt.axvline(x=zeroGamma, color='g', lw=1, label="Gamma Flip: " + str("{:,.0f}".format(zeroGamma)))
plt.axhline(y=0, color='grey', lw=1)
plt.xlim([fromStrike, toStrike])
trans = ax.get_xaxis_transform()
plt.fill_between([fromStrike, zeroGamma], min(totalGamma), max(totalGamma), facecolor='red', alpha=0.1, transform=trans)
plt.fill_between([zeroGamma, toStrike], min(totalGamma), max(totalGamma), facecolor='green', alpha=0.1, transform=trans)
plt.legend()
#plt.show()
plt.savefig("images/GammaExpProfile.png", bbox_inches='tight')
plt.close()




print("\n\n\n###############################################################################################")
print("############################    UNLABELED QUANTITATIVE DATA FOR K-MEANS  ##################################")

# Select only quantitative (numerical) columns for K-means clustering
# Remove identifier columns and keep only feature columns
quantitative_columns = [
    'StrikePrice', 'CallIV', 'PutIV', 'CallDelta', 'PutDelta', 
    'CallGamma', 'PutGamma', 'CallOpenInt', 'PutOpenInt', 'daysTillExp'
]

# Create the unlabeled quantitative dataframe
df_kmeans = df[quantitative_columns].copy()

# Remove any rows with missing values that would break K-means
print("Original shape before cleaning: ", df_kmeans.shape)
df_kmeans = df_kmeans.dropna()
print("Shape after removing missing values: ", df_kmeans.shape)

# Remove any infinite or extreme values that could distort K-means
df_kmeans = df_kmeans[np.isfinite(df_kmeans).all(axis=1)]


# Check for and remove any constant columns (zero variance) which break K-means
numeric_df = df_kmeans.select_dtypes(include=[np.number])
constant_columns = [col for col in numeric_df.columns if numeric_df[col].nunique() <= 1]
if constant_columns:
    print("Removing constant columns: ", constant_columns)
    df_kmeans = df_kmeans.drop(columns=constant_columns)

print("Final shape for K-means dataset: ", df_kmeans.shape)

# Display basic statistics of the quantitative data
print("\nBasic statistics of quantitative data for K-means:")
print(df_kmeans.describe())

# Display the first 10 rows of the quantitative dataframe
print("\nFirst 10 rows of unlabeled quantitative data for K-means:")
print(df_kmeans.head(10))

# Save the quantitative dataframe to CSV for later use in K-means
kmeans_filename = os.path.join(CS332_dir, HW_dir, "spx_quantitative_kmeans_data.csv")
df_kmeans.to_csv(kmeans_filename, index=False)
print(f"\nSaved quantitative data for K-means to: {kmeans_filename}")

# Create a visualization of the quantitative data distributions
plt.figure(figsize=(15, 10))

# Plot distributions of all quantitative variables
for i, column in enumerate(df_kmeans.columns, 1):
    plt.subplot(3, 4, i)
    plt.hist(df_kmeans[column], bins=30, alpha=0.7, edgecolor='black')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.suptitle('Distributions of Quantitative Variables for K-means Clustering', 
             fontsize=16, fontweight='bold', y=1.02)
#plt.show()
plt.savefig("images/distributions_of_quant_variables_for_Kmeans_clustering.png", bbox_inches='tight')
plt.close()

# Additional: Create pairplot to visualize relationships (use sample if dataset is large)
if len(df_kmeans) > 1000:
    df_sample = df_kmeans.sample(n=1000, random_state=42)
    print("Sampling 1000 points for pairplot visualization...")
else:
    df_sample = df_kmeans

# Create pairplot (commented out if too many variables, can be slow)
 
try:
    sns.pairplot(df_sample, diag_kind='hist', plot_kws={'alpha': 0.6})
    plt.suptitle('Pairwise Relationships in Quantitative Data (Sample)', 
                 fontsize=16, fontweight='bold', y=1.02)
    #plt.show()
    plt.savefig("images/pairplot", bbox_inches='tight')
    plt.close()

except Exception as e:
    print(f"Pairplot could not be generated: {e}")
    print("This is normal for datasets with many variables or memory constraints")

## F) Print cleaned dataframe

column_info_clean = pd.DataFrame({
    'Data Type': df_kmeans.dtypes,
    'NaN Count': df_kmeans.isna().sum(),
    'Example values': [df_kmeans[c].unique()[:1].tolist() for c in df_kmeans.columns]
})
print("\nCleaned Column Information:")
print(column_info_clean)
print("\n\nShape of cleaned dataframe: ", df_kmeans.shape)
print(df_kmeans.head(10)) 


print("\nQuantitative data preparation for K-means completed successfully!")
print("This dataset contains only numerical features suitable for clustering analysis.")

#########################################  Data preperation for KMeans ##############################


####################### 

# Load data
df = pd.read_csv(kmeans_filename) # 'spx_quantitative_kmeans_data.csv'


# Create correlation heatmap to understand relationships between variables, and remove those with high correlation.
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Quantitative Variables for K-means', 
          fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()      
#plt.show()
plt.savefig("images/correlation_heatmap.png")
plt.close()




# Preprocess: Select features & normalize
# Standardize the features
features = ['StrikePrice', 'PutOpenInt', 'PutGamma']
SPX = df[features].copy()

scaler = StandardScaler()
SPX_scaled = scaler.fit_transform(SPX)


# Find optimal k using Elbow Method
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(SPX_scaled)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss); plt.title('Elbow Method');
#plt.show()
plt.savefig("images/find_optimal_K_means_Elbow_method.png", bbox_inches='tight')
plt.close()


My_k=4  
# Apply K-Means clustering with k=3
kmeans = KMeans(n_clusters=My_k)
cluster_labels = kmeans.fit_predict(SPX_scaled)

# Add cluster labels to the original dataframe for 3D plotting
df_clean = df.loc[SPX.index].copy()
df_clean['Cluster'] = cluster_labels

# Create 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Define colors for each cluster
colors = ['black', 'green', 'red', 'blue']

# Plot each cluster with different color
for cluster_id in range(My_k):
    cluster_data = df_clean[df_clean['Cluster'] == cluster_id]
    ax.scatter(
        cluster_data['StrikePrice'],
        cluster_data['PutOpenInt'], 
        cluster_data['PutGamma'],
        c=colors[cluster_id],
        label=f'Cluster {cluster_id}',
        alpha=0.7,
        s=50
    )



# Print cluster statistics
print("Cluster Centers (Original Scale):")
cluster_centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_stats = pd.DataFrame(
    cluster_centers_original,
    columns=features,
    index=[f'Cluster {i}' for i in range(My_k)]
)
print(cluster_stats)

print("\nCluster Sizes:")
print(df_clean['Cluster'].value_counts().sort_index())
 


# Set labels and title
ax.set_xlabel('Strike Price', fontsize=12, labelpad=10)
ax.set_ylabel('PutOpenInt', fontsize=12, labelpad=10)
ax.set_zlabel('Put Gamma', fontsize=12, labelpad=10)
ax.set_title(f'K-Means Clustering (k={My_k}) - SPX Options Data', fontsize=14, pad=20)

# Add legend
ax.legend(loc='upper left', bbox_to_anchor=(0, 1))

# Adjust viewing angle for better visualization
ax.view_init(elev=20, azim=45)

plt.tight_layout()
#plt.show()
plt.savefig("images/K_means_clustering_statistics.png", bbox_inches='tight')
plt.close()



