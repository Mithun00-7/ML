from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
from sklearn import metrics
import numpy as np
df = pd.read_csv('tennisdata.csv')
df
len(df)
df.shape
df.head()
df.tail()
df.describe()
string_to_int = preprocessing.LabelEncoder()
df = df.apply(string_to_int.fit_transform)
df
f_col = ['Outlook', 'Temperature', 'Humidity', 'Windy']
x = df[f_col]
y = df.PlayTennis
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
classifier = DecisionTreeClassifier(criterion="gini", random_state=100)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print("accuracy:", metrics.accuracy_score(y_test, y_pred))
data_p = pd.DataFrame({'Actual': y_test, 'Predicited': y_pred})
data_p
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
