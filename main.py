# Imports
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# Load data to dataframe
df = pd.read_csv('data\\wine_data.csv', index_col=0)
print(df.columns)

df = df[['country', 'points', 'price', 'province', 'variety']]

# Preprocessing
df = df.dropna()

# Define features
features = ['points']

# Set variables
y = df.price
X = df[features]

# Split into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Shape of training data (num_rows, num_columns)
print(train_X.shape)

# Number of missing values in each column of training data
missing_val_count_by_column = (train_X.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])


# Model
# Define the model and set random_state to 1
rf_model = RandomForestRegressor(random_state=1)

# Fit model and predict
rf_model.fit(train_X, train_y)
rf_pred = rf_model.predict(val_X)

# Calculate mean absolute error of rf predictions
rf_val_mae = mean_absolute_error(rf_pred, val_y)
print('Mean Absolute Error: ', rf_val_mae)
