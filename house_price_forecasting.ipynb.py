# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the California Housing dataset
california = fetch_california_housing(as_frame=True)
df = california.frame

# Display first few rows
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Define features and target
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal'] * 100000  # Scaling target for better interpretability

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    print(f"{name} - RMSE: {rmse:.2f}, R2 Score: {r2:.2f}")

# Visualization for XGBoost predictions
xgb_model = models["XGBoost"]
xgb_predictions = xgb_model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, xgb_predictions, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("XGBoost: Actual vs Predicted House Prices")
plt.show()