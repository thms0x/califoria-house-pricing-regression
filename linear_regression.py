import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# Fetch datasets
X, y = fetch_california_housing(as_frame=True, data_home="data/", return_X_y=True)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, Y_train)

predict = model.predict(X_test)
# print(X.head())
# print(y.head())
results = pd.DataFrame({"Actual Price": Y_test, "Predicted Price": predict})
mae = mean_absolute_error(Y_test, predict)
print(results.head())
print(mae)