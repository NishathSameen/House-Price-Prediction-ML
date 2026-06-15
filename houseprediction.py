# House Price Prediction using Linear Regression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load Kaggle dataset
df = pd.read_csv("train.csv")

# Select features
X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = df['SalePrice']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Accuracy
print("R2 Score:", round(r2_score(y_test, y_pred), 2))

# ---------------- User Input ----------------
area = float(input("Enter Living Area (sq ft): "))
bedrooms = int(input("Enter Number of Bedrooms: "))
bathrooms = int(input("Enter Number of Bathrooms: "))

# Create input DataFrame
new_house = pd.DataFrame({
    'GrLivArea': [area],
    'BedroomAbvGr': [bedrooms],
    'FullBath': [bathrooms]
})

# Predict price
price = model.predict(new_house)

print("\nPredicted House Price: $", round(price[0], 2))

# ---------------- Graph 1: Your Input Values ----------------
plt.figure(figsize=(7, 5))

bars = plt.bar(
    ['Area', 'Bedrooms', 'Bathrooms'],
    [area, bedrooms, bathrooms]
)

# Add values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height,
        f'{height}',
        ha='center',
        va='bottom'
    )

plt.title('Your Input Values')
plt.ylabel('Values')
plt.show()
# ---------------- Graph 2: Price Comparison ----------------
avg_price = df['SalePrice'].mean()

plt.figure(figsize=(6, 5))
plt.bar(
    ['Average House Price', 'Your Predicted Price'],
    [avg_price, price[0]]
)
plt.title('Price Comparison')
plt.ylabel('Price ($)')
plt.show()

# ---------------- Graph 3: Your House on Dataset ----------------
plt.figure(figsize=(8, 5))

# Dataset points
plt.scatter(df['GrLivArea'], df['SalePrice'], label='Dataset Houses')

# User's house
plt.scatter(
    area,
    price[0],
    marker='*',
    s=300,
    label='Your House'
)

plt.xlabel('Living Area (sq ft)')
plt.ylabel('Sale Price ($)')
plt.title('Your House on the Dataset')
plt.legend()
plt.show()










