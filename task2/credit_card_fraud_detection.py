import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib


train_df = pd.read_csv('archive (1)/fraudTrain.csv')
test_df = pd.read_csv('archive (1)/fraudTest.csv')




train_df = train_df.select_dtypes(include=['number'])
test_df = test_df.select_dtypes(include=['number'])


X_train = train_df.drop('is_fraud', axis=1)
y_train = train_df['is_fraud']
X_test = test_df.drop('is_fraud', axis=1)
y_test = test_df['is_fraud']


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)
print("\n Classification Report:\n", classification_report(y_test, y_pred))
print(" Accuracy:", accuracy_score(y_test, y_pred))


joblib.dump(model, 'fraud_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\n Model and Scaler saved successfully.")


def predict_transaction(input_list):
    model = joblib.load('fraud_model.pkl')
    scaler = joblib.load('scaler.pkl')
    scaled_input = scaler.transform([input_list])
    prediction = model.predict(scaled_input)
    return " Fraudulent" if prediction[0] == 1 else " Legitimate"


print("\n Enter transaction details:")
user_input = []
columns = X_train.columns.tolist()

for col in columns:
    while True:
        try:
            value = float(input(f"{col}: "))
            user_input.append(value)
            break
        except ValueError:
            print("⚠ Please enter a valid number.")

result = predict_transaction(user_input)
print("\n This transaction is predicted to be:", result) 