import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
df = pd.read_excel("house_data.xlsx")
print(df.columns)
X = df[['Area_sqft']]        
y = df['Price_lakhs']        
model = LinearRegression()
model.fit(X, y)
plt.scatter(X, y, label='Actual Data')
plt.plot(X, model.predict(X), label='Regression Line')
plt.xlabel("Area (sqft)")
plt.ylabel("Price (in Lakhs)")
plt.title("Simple Linear Regression - House Price Prediction")
plt.legend()
plt.show()
print("------ House Price Prediction ------")
area = float(input("Enter the area of the house in sqft: "))
predicted_price = model.predict([[area]])[0]
print(f"\nEstimated House Price for {area:.0f} sqft = ₹{predicted_price:.2f} Lakhs")
plt.scatter(X, y, label='Training Data')
plt.plot(X, model.predict(X), label='Regression Line')
plt.scatter(area, predicted_price, s=120, color='red', label='Predicted Point')
plt.xlabel("Area (sqft)")
plt.ylabel("Price (in Lakhs)")
plt.title("Predicted Price Visualization")
plt.legend()
plt.show()