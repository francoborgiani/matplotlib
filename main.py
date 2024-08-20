import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# House sizes in sqft
house_sizes = np.array([750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200])
house_sizes = house_sizes.reshape(-1, 1)

# House prices in dollars
house_prices = np.array([150000, 155000, 160000, 165000, 170000, 175000, 180000, 185000, 190000, 195000])
house_prices = house_prices.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(house_sizes, house_prices, test_size=0.2, random_state=42)


# Model creation

model = LinearRegression()

# Model training
model.fit(X_train, y_train)

# Model testing

predictions = model.predict(X_test)

# Plotting the results
plt.scatter(house_sizes, house_prices, color='blue', label='Actual Data')  # original data
plt.plot(X_test, predictions, color='red', label='Regression Line')  # predictions
plt.title('House Size vs Price')
plt.xlabel('Size (Square Feet)')
plt.ylabel('Price (Dollars)')
plt.legend()
plt.show()
