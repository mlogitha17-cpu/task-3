import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Task 3 - Car Price Prediction - CodeAlpha
print("--- TASK 3 - Car Price Prediction ---")

# Sample Data
data = {
    'Age': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'KMs_Driven': [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000],
    'Price': [10.5, 9.2, 8.0, 6.8, 5.5, 4.5, 3.8, 3.0, 2.2, 1.5] # in Lakhs
}
df = pd.DataFrame(data)
print(df)

# Model
X = df[['Age', 'KMs_Driven']]
y = df['Price']
model = LinearRegression()
model.fit(X, y)

# Prediction Example
new_car = [[3, 35000]] # 3 years, 35k KMs
predicted_price = model.predict(new_car)
print(f"\nPredicted Price for 3 year old car (35k KMs): {predicted_price[0]:.2f} Lakhs")

# Graph
plt.scatter(df['Age'], df['Price'], color='blue', label='Actual Price')
plt.plot(df['Age'], model.predict(X), color='red', linewidth=2, label='Predicted Line')
plt.title('Car Price vs Age - CodeAlpha Task 3')
plt.xlabel('Car Age (Years)')
plt.ylabel('Price (Lakhs)')
plt.legend()
plt.grid(True)
plt.show()

print("\nTask 3 Completed Successfully!")