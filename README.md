# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required packages and print the present data.
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices. 
```
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: THARANI RAMESHBABU
RegisterNumber: 212225240170
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import metrics

df = pd.read_csv('D:\DATA\Placement_Data.csv')
df1 = df.copy()

df1 = df1.drop(['sl_no', 'salary'], axis=1)

le = LabelEncoder()
categorical_cols = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 
                    'degree_t', 'workex', 'specialisation', 'status']

for col in categorical_cols:
    df1[col] = le.fit_transform(df1[col])


X = df1.iloc[:, :-1]
y = df1['status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print("Accuracy Score:", accuracy)
print("Confusion Matrix:\n", confusion)
print("\nClassification Report:\n", cr)


disp = metrics.ConfusionMatrixDisplay(
    confusion_matrix=confusion, display_labels=['Placed', 'Not Placed']
)
disp.plot()
*/
```

## Output:
<img width="676" height="309" alt="Screenshot_21-5-2026_82213_localhost" src="https://github.com/user-attachments/assets/f712b05f-a4a8-4ad7-bad9-e8e4c9adf1cf" />

<img width="1203" height="607" alt="Screenshot_21-5-2026_82228_localhost" src="https://github.com/user-attachments/assets/1ee5608e-ac70-4494-b096-df03fd6a1063" />

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
