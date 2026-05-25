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
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor

data = {
    'Size': [1000, 1200, 1500, 1800, 2000],
    'Bedrooms': [2, 2, 3, 3, 4],
    'Price': [300000, 350000, 400000, 450000, 500000],
    'Occupants': [2, 3, 4, 5, 6]
}

df = pd.DataFrame(data)

X = df[['Size', 'Bedrooms']]

y = df[['Price', 'Occupants']]

model = MultiOutputRegressor(SGDRegressor())

model.fit(X, y)

prediction = model.predict([[1600, 3]])

print("Predicted Price:", prediction[0][0])
print("Predicted Occupants:", prediction[0][1])

plt.scatter(df['Size'], df['Price'])

plt.plot(df['Size'], model.predict(X)[:,0])

plt.xlabel("House Size")
plt.ylabel("House Price")
plt.title("House Price Prediction")

plt.show()

plt.scatter(df['Size'], df['Occupants'])

plt.plot(df['Size'], model.predict(X)[:,1])

plt.xlabel("House Size")
plt.ylabel("Occupants")
plt.title("Occupants Prediction")

plt.show()
```
## Output:

<img width="425" height="62" alt="image" src="https://github.com/user-attachments/assets/087819af-b94e-4386-a807-de4651ba18fd" />
<img width="793" height="569" alt="image" src="https://github.com/user-attachments/assets/ac55405d-1b0e-463b-b5cf-58c5f78dde19" />
<img width="770" height="584" alt="image" src="https://github.com/user-attachments/assets/c7ed8ed4-0d8b-486e-acaf-e4e613ebcab7" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
