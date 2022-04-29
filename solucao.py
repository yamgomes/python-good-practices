"""KNN model application on predicting car prices
"""
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

pd.options.display.max_columns = 99

cols = ['symboling', 'normalized-losses', 'make',
        'fuel-type', 'aspiration', 'num-of-doors',
        'body-style', 'drive-wheels', 'engine-location',
        'wheel-base', 'length', 'width', 'height',
        'curb-weight', 'engine-type', 'num-of-cylinders',
        'engine-size', 'fuel-system', 'bore', 'stroke',
        'compression-rate', 'horsepower', 'peak-rpm',
        'city-mpg', 'highway-mpg', 'price']

cars = pd.read_csv('imports-85.data', names=cols)

print(cars.head())

# Select only the columns with continuous values from
# https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.names
continuous_values_cols = ['normalized-losses', 'wheel-base', 'length', 'width',
                          'height', 'curb-weight', 'engine-size', 'bore', 'stroke',
                          'compression-rate', 'horsepower', 'peak-rpm',
                          'city-mpg', 'highway-mpg', 'price']

numeric_cars = cars[continuous_values_cols]

# Cleaning the dataset

numeric_cars = numeric_cars.replace('?', np.nan)
numeric_cars = numeric_cars.astype('float')

# Replace missing values in other columns using column means.
numeric_cars = numeric_cars.fillna(numeric_cars.mean())

# Normalize all columnns to range from 0 to 1 except the target column.
price_col = numeric_cars['price']
numeric_cars = (numeric_cars - numeric_cars.min()) / \
    (numeric_cars.max() - numeric_cars.min())
numeric_cars['price'] = price_col

# Univariate model


def knn_train_test(train_column, target_column, dataframe):
    """Train the model using the data
    Args:
        train_column: Columns to train the data
        target_column: Column targeted by the training
        dataframe: Dataframe used in the training

    Returns:
        rmse: The root-mean-square error
    """

    knn = KNeighborsRegressor()
    np.random.seed(1)

    # Randomize order of rows in data frame.
    shuffled_index = np.random.permutation(dataframe.index)
    rand_dataframe = dataframe.reindex(shuffled_index)

    # Divide number of rows in half and round.
    last_train_row = int(len(rand_dataframe) / 2)

    # Select the first half and set as training set.
    # Select the second half and set as test set.
    train_dataframe = rand_dataframe.iloc[0:last_train_row]
    test_dataframe = rand_dataframe.iloc[last_train_row:]

    # Fit a KNN model using default k value.
    knn.fit(train_dataframe[[train_column]], train_dataframe[target_column])

    # Make predictions using model.
    predicted_labels = knn.predict(test_dataframe[[train_column]])

    # Calculate and return RMSE.
    mse = mean_squared_error(test_dataframe[target_column], predicted_labels)
    rmse = np.sqrt(mse)
    return rmse


rmse_results = {}
train_cols = numeric_cars.columns.drop('price')

# For each column (minus `price`), train a model, return RMSE value
# and add to the dictionary `rmse_results`.
for col in train_cols:
    rmse_val = knn_train_test(col, 'price', numeric_cars)
    rmse_results[col] = rmse_val

# Create a Series object from the dictionary so
# we can easily view the results, sort, etc
rmse_results_series = pd.Series(rmse_results)
rmse_results_series.sort_values()


def knn_train_test_k_neighbors(train_col, target_col, dataframe):
    """Train the model using the data
    Args:
        train_col: Columns to train the data
        target_col: Column targeted by the training
        dataframe: Dataframe used in the training

    Returns:
        k_rmses: The root-mean-square errors
    """

    np.random.seed(1)

    # Randomize order of rows in data frame.
    shuffled_index = np.random.permutation(dataframe.index)
    rand_dataframe = dataframe.reindex(shuffled_index)

    # Divide number of rows in half and round.
    last_train_row = int(len(rand_dataframe) / 2)

    # Select the first half and set as training set.
    # Select the second half and set as test set.
    train_dataframe = rand_dataframe.iloc[0:last_train_row]
    test_dataframe = rand_dataframe.iloc[last_train_row:]

    k_values = [1, 3, 5, 7, 9]
    k_rmses = {}

    for value in k_values:
        # Fit model using k nearest neighbors.
        knn = KNeighborsRegressor(n_neighbors=value)
        knn.fit(train_dataframe[[train_col]], train_dataframe[target_col])

        # Make predictions using model.
        predicted_labels = knn.predict(test_dataframe[[train_col]])

        # Calculate and return RMSE.
        mse = mean_squared_error(test_dataframe[target_col], predicted_labels)
        rmse = np.sqrt(mse)

        k_rmses[value] = rmse
    return k_rmses


k_rmse_results = {}

# For each column (minus `price`), train a model, return RMSE value
# and add to the dictionary `rmse_results`.
train_cols = numeric_cars.columns.drop('price')
for col in train_cols:
    rmse_val = knn_train_test_k_neighbors(col, 'price', numeric_cars)
    k_rmse_results[col] = rmse_val

print(k_rmse_results)


for k, v in k_rmse_results.items():
    x = list(v.keys())
    y = list(v.values())

    plt.plot(x, y)
    plt.xlabel('k value')
    plt.ylabel('RMSE')

# Multivariate model

# Compute average RMSE across different `k` values for each feature.
feature_avg_rmse = {}
for k, v in k_rmse_results.items():
    avg_rmse = np.mean(list(v.values()))
    feature_avg_rmse[k] = avg_rmse
series_avg_rmse = pd.Series(feature_avg_rmse)
sorted_series_avg_rmse = series_avg_rmse.sort_values()
print(sorted_series_avg_rmse)

sorted_features = sorted_series_avg_rmse.index


def knn_train_test_multivariate(training_columns, target_col, dataframe):
    """Train the model using the data
    Args:
        train_columns: Columns to train the data
        target_col: Column targeted by the training
        dataframe: Dataframe used in the training

    Returns:
        k_rmses: The root-mean-square errors
    """

    np.random.seed(1)

    # Randomize order of rows in data frame.
    shuffled_index = np.random.permutation(dataframe.index)
    rand_dataframe = dataframe.reindex(shuffled_index)

    # Divide number of rows in half and round.
    last_train_row = int(len(rand_dataframe) / 2)

    # Select the first half and set as training set.
    # Select the second half and set as test set.
    train_dataframe = rand_dataframe.iloc[0:last_train_row]
    test_dataframe = rand_dataframe.iloc[last_train_row:]

    k_values = [5]
    k_rmses = {}

    for value in k_values:
        # Fit model using k nearest neighbors.
        knn = KNeighborsRegressor(n_neighbors=value)
        knn.fit(train_dataframe[training_columns], train_dataframe[target_col])

        # Make predictions using model.
        predicted_labels = knn.predict(test_dataframe[training_columns])

        # Calculate and return RMSE.
        mse = mean_squared_error(test_dataframe[target_col], predicted_labels)
        rmse = np.sqrt(mse)

        k_rmses[value] = rmse
    return k_rmses


k_rmse_results = {}

for nr_best_feats in range(2, 7):
    k_rmse_results['{nr_best_feats} best features'] = knn_train_test_multivariate(
        sorted_features[:nr_best_feats],
        'price',
        numeric_cars
    )

print(k_rmse_results)

# Hyperparameter tuning

def knn_train_test_hyperparameter_tuning(training_columns, target_col, dataframe):
    """Train the model using the data
    Args:
        training_columns: Columns to train the data
        target_col: Column targeted by the training
        dataframe: Dataframe used in the training

    Returns:
        k_rmses: The root-mean-square errors
    """

    np.random.seed(1)

    # Randomize order of rows in data frame.
    shuffled_index = np.random.permutation(dataframe.index)
    rand_dataframe = dataframe.reindex(shuffled_index)

    # Divide number of rows in half and round.
    last_train_row = int(len(rand_dataframe) / 2)

    # Select the first half and set as training set.
    # Select the second half and set as test set.
    train_dataframe = rand_dataframe.iloc[0:last_train_row]
    test_dataframe = rand_dataframe.iloc[last_train_row:]

    k_values = list(range(1, 25))
    k_rmses = {}

    for value in k_values:
        # Fit model using k nearest neighbors.
        knn = KNeighborsRegressor(n_neighbors=value)
        knn.fit(train_dataframe[training_columns], train_dataframe[target_col])

        # Make predictions using model.
        predicted_labels = knn.predict(test_dataframe[training_columns])

        # Calculate and return RMSE.
        mse = mean_squared_error(test_dataframe[target_col], predicted_labels)
        rmse = np.sqrt(mse)

        k_rmses[value] = rmse
    return k_rmses


k_rmse_results = {}

for nr_best_feats in range(2, 6):
    k_rmse_results['{nr_best_feats} best features'] = knn_train_test(
        sorted_features[:nr_best_feats],
        'price',
        numeric_cars
    )

print(k_rmse_results)

for k, v in k_rmse_results.items():
    x = list(v.keys())
    y = list(v.values())
    plt.plot(x, y, label="{k}")

plt.xlabel('k value')
plt.ylabel('RMSE')
