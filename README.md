# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Start the program

Begin the execution.

Step 2: Import required libraries

Import necessary Python libraries such as:

NumPy
Pandas
Scikit-learn
Step 3: Load the dataset

Read the dataset containing multiple independent variables and one dependent variable.

Step 4: Separate input and output variables
Assign independent variables to X
Assign dependent variable to y
Step 5: Split the dataset

Divide the dataset into:

Training data
Testing data

Usually using 80% training and 20% testing.

Step 6: Perform feature scaling

Standardize the input features using StandardScaler to improve SGD performance.

Step 7: Create the SGD Regressor model

Initialize the SGD Regressor with suitable parameters such as:

learning rate
number of iterations
random state
Step 8: Train the model

Fit the SGD Regressor model using the training dataset.

Step 9: Predict the output

Use the trained model to predict values for the testing dataset.

Step 10: Evaluate the model

Calculate performance metrics such as:

Mean Squared Error (MSE)
R² Score
Step 11: Display the results

Print predicted values and evaluation metrics.

Step 12: Stop the program

End the execution.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: SUJITH MANO M
RegisterNumber:  212225220109
*/
```
```
# Ex:No 4

import numpy as np

X = np.array([
    [2, 80, 50],
    [3, 60, 40],
    [5, 90, 70],
    [7, 85, 80],
    [9, 95, 90]
], dtype=float)

y = np.array([50, 45, 70, 80, 95], dtype=float)

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

X = np.c_[np.ones(X.shape[0]), X]  # shape becomes (n_samples, n_features + 1)

n_features = X.shape[1]
weights = np.zeros(n_features)

learning_rate = 0.01
epochs = 1000
for epoch in range(epochs):
    for i in range(X.shape[0]):
        xi = X[i]
        yi = y[i]
        y_pred = np.dot(xi, weights)
        error = y_pred - yi
        # Update weights
        weights -= learning_rate * error * xi

print("Trained Weights (including intercept):", weights)

y_pred_all = np.dot(X, weights)
print("Predicted values:", y_pred_all)
```
## Output:

<img width="988" height="92" alt="image" src="https://github.com/user-attachments/assets/0218a1ab-16e3-447f-a411-dcb079c9bdb7" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
