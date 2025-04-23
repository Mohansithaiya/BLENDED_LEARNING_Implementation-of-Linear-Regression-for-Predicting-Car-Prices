# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Libraries:
Import necessary libraries such as pandas, numpy, matplotlib, and sklearn.
2. Load Dataset:
Load the dataset containing car prices and relevant features.
3. Data Preprocessing:
Handle missing values and perform feature selection if necessary.
4. Split Data:
Split the dataset into training and testing sets.
5. Train Model:
Create a linear regression model and fit it to the training data.
6. Make Predictions:
Use the model to make predictions on the test set.
7. Evaluate Model:
Assess model performance using metrics like R² score, Mean Absolute Error (MAE), etc.
8. Check Assumptions:
Plot residuals to check for homoscedasticity, normality, and linearity.
9. Output Results:
Display the predictions and evaluation metrics.
Program to implement linear regression model for predicting car prices and test assumptions.


## Program:

```PYTHON
Program to implement linear regression model for predicting car prices and test assumptions.
Developed by: MOHAN S
RegisterNumber:  212223240094
```

```PYTHON
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
```

```PYTHON
df = pd.read_csv("CarPrice_Assignment.csv")
df
```

```python
print(df.head())
```

```python
df = df.dropna()
```

```python
X = df[['horsepower', 'curbweight', 'enginesize', 'highwaympg']]
y = df['price']
```

## Standardize the features:
```python
SS = StandardScaler()

X = SS.fit_transform(X)
y = SS.fit_transform(y.values.reshape(-1, 1))
```

## Split the data into training and testing sets:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Train the linear regression model:
```python
model = LinearRegression()
model.fit(X_train, y_train)
```

## Make predictions:
```python
y_pred = model.predict(X_test)
```

## Evaluate the model:
```python
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
```

## Check model assumptions:
## 1. Linearity Assumption:
```python
plt.figure(figsize=(10, 6))
for i, col in enumerate(['horsepower', 'curbweight', 'enginesize', 'highwaympg']):
    plt.subplot(2, 2, i+1)
    plt.scatter(df[col], df['price'])
    plt.xlabel(col)
    plt.ylabel('Price')
    plt.title(f'Price vs {col}')
plt.tight_layout()
plt.show()
```

## 2. Homoscedasticity:
```python
plt.scatter(y_pred, y_test - y_pred)
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Prices')
plt.axhline(0, color='red', linestyle='--')
plt.show()
```

## 3. Normality:
```python
plt.figure(figsize=(10, 6))
plt.hist(y_test - y_pred, bins=30)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Histogram of Residuals')
plt.show()
```

## 4. Multicollinearity:
```python
corr_matrix = df[['horsepower', 'curbweight', 'enginesize', 'highwaympg']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
```

## Output:
![Screenshot 2025-04-23 212701](https://github.com/user-attachments/assets/5481dd89-ee07-4213-8393-8069326ab482)

![Screenshot 2025-04-23 212714](https://github.com/user-attachments/assets/99712916-9d40-40e5-a0d7-fcf1b9f583f6)

![Screenshot 2025-04-23 212726](https://github.com/user-attachments/assets/19532b39-a143-490e-b5bf-c1816131ce7e)

![Screenshot 2025-04-23 212734](https://github.com/user-attachments/assets/f0de0ea8-80a6-4bf7-b5e2-4f875d4454b6)

![Screenshot 2025-04-23 212741](https://github.com/user-attachments/assets/7d3a569e-1b9b-4028-a73c-aba63a48fbf3)

![Screenshot 2025-04-23 212749](https://github.com/user-attachments/assets/fa31c609-2b7b-4cb4-b456-1ee421469918)


## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
