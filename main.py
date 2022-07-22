# First line
import pandas as pd

df = pd.read_csv('data\\wine_data.csv', index_col=0)

print(df)
