import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
df=pd.read_csv('wine.csv',header=None,usecols=[0,1,2],skiprows=1)
df.columns=['classlabel','Alcohol','Malic Acid']
print("Original Data Frame:")
print(df)
scaling=MinMaxScaler()
scaled_value=scaling.fit_transform(df[['Alcohol','Malic Acid']])
df[['Alcohol','Malic Acid']]=scaled_value
print("\nData Frame after Min-Max Scaling:")
print(df)
scaling=StandardScaler()
scaled_standard_value=scaling.fit_transform(df[['Alcohol','Malic Acid']])
df[['Alcohol','Malic Acid']]=scaled_standard_value
print("\nData Frame after Standard Scaling:")
print(df)
