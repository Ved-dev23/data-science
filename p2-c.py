import pandas as pd
iris=pd.read_csv('Iris.csv')
setosa=iris[iris['Species']=='setosa']
print("Setosa sample:")
print(setosa.head())
sorted_iris=iris.sort_values(by='SepalLengthCm', ascending=False)
print("\nSorted iris dataset:")
print(sorted_iris.head())
grouped_species=iris.groupby('Species').mean()
print("\nMean measurement of each  species:")
print(grouped_species)
