import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # linear algebra


productCSV = pd.read_csv(
    'E:/Master-Thesis/OptimizingBusinessSales/dataset/ProductDetails-MT.csv')

lowRatings = productCSV.loc[productCSV['Avg'] < 3].sort_values('Avg').head(3)

highRatings = productCSV.loc[productCSV['Avg'] > 4].sort_values('Avg').head(3)

neutralRatings = productCSV.loc[((productCSV['Avg'] > 3) & (
    productCSV['Avg'] < 4))].sort_values('Avg').head(3)
