import pandas as pd
df = pd.read_csv('titanic.csv')
print(df)
df.head(10)
print("Dataset after filling NA value with 0:")
df2 = df.fillna(value=0)
print(df2)
