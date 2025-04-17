import pandas as pd
df = pd.read_csv('titanic.csv', na_values='\\N')
print(df)
df.head(10)
print("Dataset after dropping NA value:")
df.dropna(inplace=True)
print(df)
