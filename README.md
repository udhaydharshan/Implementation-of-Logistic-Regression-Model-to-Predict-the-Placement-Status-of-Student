# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: UDHAYDHARSHAN S
RegisterNumber: 212225230286 
*/
```
```
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
data = pd.read_csv(r"C:/Users/acer/Downloads/Placement_Data.csv")
data.head()
datal = data.copy()
datal = datal.drop(["sl_no", "salary"], axis=1)
datal.head()
print("Missing values:\n", datal.isnull().sum())
print("Duplicate rows:", datal.duplicated().sum())
le = LabelEncoder()
datal["gender"] = le.fit_transform(datal["gender"])
datal["ssc_b"] = le.fit_transform(datal["ssc_b"])
datal["hsc_b"] = le.fit_transform(datal["hsc_b"])
datal["hsc_s"] = le.fit_transform(datal["hsc_s"])
datal["degree_t"] = le.fit_transform(datal["degree_t"])
datal["workex"] = le.fit_transform(datal["workex"])
datal["specialisation"] = le.fit_transform(datal["specialisation"])
datal["status"] = le.fit_transform(datal["status"])
datal
x = datal.iloc[:, :-1]
y = datal["status"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)
y_pred

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion)
classification_report_output = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report_output)
```

## Output:
HEAD
<img width="961" height="213" alt="Screenshot 2026-02-04 092912" src="https://github.com/user-attachments/assets/23bff4f1-5f83-4677-bf7f-69353468d47d" />
COPY
<img width="905" height="219" alt="Screenshot 2026-02-04 092935" src="https://github.com/user-attachments/assets/3325c76b-cba9-46b1-b6da-ec242fe1e950" />
FIT TRANSFORM
<img width="902" height="487" alt="Screenshot 2026-02-04 092953" src="https://github.com/user-attachments/assets/2d6cb44e-124a-4e17-add5-8c1a04950298" />
LOGISTIC REGRESSION
<img width="889" height="248" alt="Screenshot 2026-02-04 093837" src="https://github.com/user-attachments/assets/b2934383-fabd-4309-b215-29b32023b60c" />
ACCURACY SCORE
<img width="900" height="67" alt="Screenshot 2026-02-04 093010" src="https://github.com/user-attachments/assets/48b3f10a-03d1-45ea-bfca-255f09722262" />
CONFUSION MATRIX
<img width="895" height="105" alt="Screenshot 2026-02-04 093022" src="https://github.com/user-attachments/assets/5a49f3fe-b84b-4ed5-99bb-3771b5e06a28" />
CLASSIFICATION REPORT & PREDICTION
<img width="892" height="197" alt="Screenshot 2026-02-04 093039" src="https://github.com/user-attachments/assets/d541133d-d33e-4ad3-a6b7-23cb6b612975" />






## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
