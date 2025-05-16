# Forecasting House Prices Accurately Using Smart Regression Techniques in Data Science

This project aims to predict California house prices using smart regression techniques. By analyzing various housing features and applying advanced machine learning models, we forecast the *median house value* with high accuracy. This project serves as part of our academic curriculum and showcases applied data science methodologies.

---

##  Team Information

Team Name: Team SmartPredictors  
Team Members:
- Member 1 – Abdul Razith E
- Member 2 – Dhamodharan K
- Member 3 – Dineshkumar M
- Member 4 – Kishore E
- Member 5 – Muhammed Umer S

---

## Dataset Information

- *Dataset*: California Housing Dataset  
- *Source*: Built-in dataset from sklearn.datasets  
- *Target Variable*: MedHouseVal (Median House Value)

### Features:
| Feature     | Description                        |
|-------------|------------------------------------|
| MedInc      | Median income in block             |
| HouseAge    | Median house age                   |
| AveRooms    | Average number of rooms            |
| AveBedrms   | Average number of bedrooms         |
| Population  | Block population                   |
| AveOccup    | Average occupancy                  |
| Latitude    | Block latitude                     |
| Longitude   | Block longitude                    |

---

##  Techniques and Tools Used

- *Programming Language*: Python
- *Libraries*:
  - pandas, numpy for data handling
  - matplotlib, seaborn for EDA & visualization
  - scikit-learn for model building and evaluation
  - xgboost for advanced regression modeling

- *Machine Learning Models*:
  - Linear Regression
  - Random Forest Regressor
  - XGBoost Regressor

- *Evaluation Metrics*:
  - Root Mean Squared Error (RMSE)
  - R² Score

---

## Model Performance Summary

| Model              | RMSE (↓)   | R² Score (↑) |
|-------------------|------------|--------------|
| Linear Regression | ~74,558    | 0.58         |
| Random Forest     | ~50,710    | 0.80         |
| XGBoost           | *~47,672| **0.83*     |


XGBoost delivered the best predictive performance and is selected as the final model.

---

##Project Structure