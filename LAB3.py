from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_text
import pandas as pd
!pip install scikit-learn


data = pd.read_csv('tennisdata.csv')

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

le_outlook = LabelEncoder()
X['Outlook'] = le_outlook.fit_transform(X['Outlook'])

le_temperature = LabelEncoder()
X['Temperature'] = le_temperature.fit_transform(X['Temperature'])

le_humidity = LabelEncoder()
X['Humidity'] = le_humidity.fit_transform(X['Humidity'])

le_windy = LabelEncoder()
X['Windy'] = le_windy.fit_transform(X['Windy'])

le_playTennis = LabelEncoder()
y = le_playTennis.fit_transform(y)

classifier = DecisionTreeClassifier()
classifier.fit(X, y)

tree_rules = export_text(classifier, feature_names=X.columns.tolist())
print("Decision Tree Structure:\n")
print(tree_rules)

inp = ["Rainy", "Mild", "High", "False"]

inp_encoded = [
    le_outlook.transform([inp[0]])[0],
    le_temperature.transform([inp[1]])[0],
    le_humidity.transform([inp[2]])[0],
    le_windy.transform([inp[3]])[0]
]

inp_encoded_df = pd.DataFrame([inp_encoded], columns=X.columns)

prediction = classifier.predict(inp_encoded_df)

print(
    f"\nFor input {inp}, we predict: {le_playTennis.inverse_transform(prediction)[0]}")
