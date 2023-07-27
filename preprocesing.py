import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImuter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
!gdown --id '1GiT4CozIKyh8dOcab6txpg5qZmxYSv4Y'
!gdown --id '1IzWnUoZjyvktnaTeBgSs7Re92e4hmEzS'
train_df = pd.read_csv('/content/Train_bigmart.csv',header=None)
test_df = pd.read_csv('/content/Test_bigmart.csv',header= None)
#Handling the columns name:
my_cols = ['Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility',
           'Item_Type', 'Item_MRP', 'Outlet_Identifier', 'Outlet_Establishment_Year',
           'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Outlet_Sales' ] 
cols = dict(zip(train_df.columns, my_cols))
train_df = train_df.rename(columns=cols)
cols = dict(zip(test_df.columns, my_cols))
test_df = test_df.rename(columns=cols)
#Preprocesing
#categorical variables handling
temp_X = pd.concat([train_df, test_df], axis=0)
temp_X_new = pd.get_dummies(temp_X)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(temp_X_new)
#Identify missing values
missing_values = train_df.isnull().sum()
print("Missing Values:\n", missing_values)
df_dropna = train_df.dropna()  # Drop rows with any missing values
df_dropna.to_csv('without_missing_values.csv',index=False)
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(temp_X_new), columns=temp_X_new.columns)
df_imputed.to_csv('dataset_imputed.csv', index=False)
df_fillna = train_df.fillna(value=0)  # Fill missing values with 0
df_fillna.to_csv('dataset_filled_with_zeros.csv', index=False)
# Handling Outliers
# Remove outliers using z-score
z_scores = (train_df - train_df.mean()) / train_df.std()
df_no_outliers = train_df[(z_scores < 3).all(axis=1)]
df_no_outliers.to_csv('dataset without outliers.csv', index=False)
#train test split
X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]
X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]
