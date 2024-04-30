import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

data = pd.read_csv('customer_data.csv')

le = LabelEncoder()
data['gender'] = le.fit_transform(data['gender'])
data['education'] = le.fit_transform(data['education'])
data['region'] = le.fit_transform(data['region'])
data['loyalty_status'] = le.fit_transform(data['loyalty_status'])
data['purchase_frequency'] = le.fit_transform(data['purchase_frequency'])
data['product_category'] = le.fit_transform(data['product_category'])

print(data)

from sklearn.model_selection import train_test_split

X = data.drop('purchase_frequency', axis=1)  # Features
y = data['purchase_frequency']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression() 
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



