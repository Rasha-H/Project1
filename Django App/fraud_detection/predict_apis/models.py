import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

data = pd.read_csv('fraud_data.csv')

label_encoder = LabelEncoder()
data['Customer Email Domain'] = label_encoder.fit_transform(data['Customer Email Domain'])

X = data[['Order Amount', 'Customer Email Domain', 'Past Fraud History']]
y = data['Fraud Probability (%)']

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, 'fraud_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')