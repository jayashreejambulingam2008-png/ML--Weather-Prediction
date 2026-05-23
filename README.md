# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the weather dataset.
2. Select features and target variables, then clean and split the dataset into training and testing sets.
3. Train the Random Forest Regression models and make predictions for temperature, PM2.5, and energy.
4. Evaluate the models using R², MAE, RMSE, perform cross-validation, and visualize actual vs predicted results using plots.

## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: JAYASHREE J
RegisterNumber:  212225040145

# Ex 10 - Random Forest Algorithm for Weather Prediction

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ------------------------------
# Step 1: Load Dataset
# ------------------------------
df = pd.read_csv("weather_data.csv")

# ------------------------------
# Step 2: Select Required Columns
# ------------------------------
df = df[['hum', 'pressure', 'wind_speed', 'tem', 'pm2_5', 'tsr']]

# Remove missing values
df = df.dropna()

# Convert to numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Remove invalid rows
df = df.dropna()

# ------------------------------
# Step 3: Features and Targets
# ------------------------------
X = df[['hum', 'pressure', 'wind_speed']]

y_temp = df['tem']
y_pm25 = df['pm2_5']
y_energy = df['tsr']

# ------------------------------
# Step 4: Train-Test Split
# ------------------------------
X_train, X_test, y_temp_train, y_temp_test = train_test_split(
    X, y_temp, test_size=0.2, random_state=42)

_, _, y_pm25_train, y_pm25_test = train_test_split(
    X, y_pm25, test_size=0.2, random_state=42)

_, _, y_energy_train, y_energy_test = train_test_split(
    X, y_energy, test_size=0.2, random_state=42)

# ------------------------------
# Step 5: Create Models
# ------------------------------
rf_temp = RandomForestRegressor(n_estimators=100, random_state=42)
rf_pm25 = RandomForestRegressor(n_estimators=100, random_state=42)
rf_energy = RandomForestRegressor(n_estimators=100, random_state=42)

# Train models
rf_temp.fit(X_train, y_temp_train)
rf_pm25.fit(X_train, y_pm25_train)
rf_energy.fit(X_train, y_energy_train)

# ------------------------------
# Step 6: Predictions
# ------------------------------
temp_pred = rf_temp.predict(X_test)
pm25_pred = rf_pm25.predict(X_test)
energy_pred = rf_energy.predict(X_test)

# ------------------------------
# Step 7: Evaluation Function
# ------------------------------
def evaluate_model(y_test, y_pred, model, X, y, name):

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    cv = cross_val_score(model, X, y, cv=3, scoring='r2')

    print(f"\n{name} Prediction Results")
    print("R² Score:", r2)
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("Cross Validation Score:", cv.mean())

# ------------------------------
# Step 8: Evaluate Models
# ------------------------------
evaluate_model(y_temp_test, temp_pred, rf_temp, X, y_temp, "Temperature")

evaluate_model(y_pm25_test, pm25_pred, rf_pm25, X, y_pm25, "PM2.5")

evaluate_model(y_energy_test, energy_pred, rf_energy, X, y_energy, "Energy")

# ------------------------------
# Step 9: Plot Actual vs Predicted
# ------------------------------
plt.figure(figsize=(15,5))

# Temperature Plot
plt.subplot(1,3,1)
plt.scatter(y_temp_test, temp_pred)
plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.title("Temperature Prediction")

# PM2.5 Plot
plt.subplot(1,3,2)
plt.scatter(y_pm25_test, pm25_pred)
plt.xlabel("Actual PM2.5")
plt.ylabel("Predicted PM2.5")
plt.title("PM2.5 Prediction")

# Energy Plot
plt.subplot(1,3,3)
plt.scatter(y_energy_test, energy_pred)
plt.xlabel("Actual Energy")
plt.ylabel("Predicted Energy")
plt.title("Energy Prediction")

plt.tight_layout()
plt.show()

# ------------------------------
# Step 10: Predict New Data
# ------------------------------
new_data = np.array([[95.5, 999.9, 1.2]])

temp_new = rf_temp.predict(new_data)
pm25_new = rf_pm25.predict(new_data)
energy_new = rf_energy.predict(new_data)

print("\nPredictions for New Environmental Data:")
print("Predicted Temperature:", temp_new[0])
print("Predicted PM2.5:", pm25_new[0])
print("Predicted Energy:", energy_new[0])
*/
```

## Output:
<img width="790" height="506" alt="image" src="https://github.com/user-attachments/assets/ea8db79d-945e-4196-9363-cbdd647316df" />
<img width="1434" height="660" alt="image" src="https://github.com/user-attachments/assets/741d1f41-cbd1-4ac0-b79e-cec396ff0ca5" />


## Result:
Hence the program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm has been executed successfully.
