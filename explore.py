import pandas as pd

df = pd.read_csv("diabetes.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset shape (rows, columns):")
print(df.shape)

print("\nBasic statistics:")
print(df.describe())

print("\nAny missing values?")
print(df.isnull().sum())