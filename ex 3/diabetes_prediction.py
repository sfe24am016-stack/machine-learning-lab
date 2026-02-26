import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
data = {
    'Glucose': [80, 120, 90, 140, 150, 85, 160, 95, 170, 130],
    'BloodPressure': [70, 80, 75, 85, 90, 65, 95, 72, 100, 88],
    'BMI': [25, 28, 26, 30, 32, 23, 35, 27, 36, 31],
    'Age': [25, 35, 29, 40, 45, 22, 50, 30, 55, 42],
    'Diabetes': [0, 1, 0, 1, 1, 0, 1, 0, 1, 1]  
}
df = pd.DataFrame(data)
X = df[['Glucose', 'BloodPressure', 'BMI', 'Age']]
y = df['Diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {acc*100:.2f}%")
print("\n------ Diabetes Prediction ------")
glucose = float(input("Enter Glucose level (mg/dL): "))
bp = float(input("Enter Blood Pressure (mm Hg): "))
bmi = float(input("Enter BMI value: "))
age = int(input("Enter Age: "))
pred = model.predict([[glucose, bp, bmi, age]])[0]
prob = model.predict_proba([[glucose, bp, bmi, age]])[0][1]
if pred == 1:
    print(f"\nThe person is likely to have diabetes. (Confidence: {prob*100:.2f}%)")
else:
    print(f"\nThe person is unlikely to have diabetes. (Confidence: {(1-prob)*100:.2f}%)")
labels = ['No Diabetes', 'Diabetes']
probabilities = [1 - prob, prob]
plt.figure(figsize=(6, 4))
plt.bar(labels, probabilities)
plt.ylim(0, 1)
plt.ylabel("Probability")
plt.title("Diabetes Prediction Result")
for i, v in enumerate(probabilities):
    plt.text(i, v + 0.03, f"{v*100:.1f}%", ha='center', fontweight='bold')
plt.show()
