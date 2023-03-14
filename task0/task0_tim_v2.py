# imports 
import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# read csv files
train_df = pd.read_csv("train.csv")
train_df = train_df.set_index("Id")
test_df = pd.read_csv("test.csv")
test_df = test_df.set_index("Id")
# display(train_df.head(3))
# display(test_df.head(3))


# extract values from train set:
train_array = train_df.to_numpy()
y_train = train_array[:, 0]
X_train = train_array[:, 1:]


# split X_train into training and test batches
X_train_batch, X_test_batch, y_train_batch, y_test_batch = train_test_split(X_train, y_train, test_size=0.33, random_state=0)

# linear regression on training batch of X_train
regr = LinearRegression().fit(X_train_batch, y_train_batch)

# predict output of test batch of X_train
y_test_batch_predict = regr.predict(X_test_batch)
prediction_error = mean_squared_error(y_test_batch, y_test_batch_predict)
if prediction_error > 0.001: print("large prediction error")

# predict output of X_test using regression parameters
X_test = test_df.to_numpy()
y_test = regr.predict(X_test)
# display(y_test)

result_df = pd.DataFrame(columns=["Id", "y"])
result_df.Id = test_df.index
result_df.y = y_test
result_df = result_df.set_index("Id")


# check results vs. mean of all x values
mean_vals = test_df.mean(axis=1)
mean_df = pd.DataFrame(mean_vals, columns=["y"])

max_err = max(abs(mean_df.y - result_df.y))
if max_err > 0.001: print("Error somewhere (deviation from mean value)")

# write to csv
result_df.to_csv("output_2_tim.csv")
# print("done")