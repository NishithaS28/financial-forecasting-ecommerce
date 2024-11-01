import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt  # Add this line to import matplotlib

# Sample data (use your actual sales data here)
df = pd.DataFrame({
    'ds': pd.date_range(start='2020-01-01', periods=100, freq='D'),
    'y': [x + (x * 0.05) for x in range(100)]  # Simulating some sales data
})

# Initialize and fit Prophet model
model = Prophet()
model.fit(df)

# Make future predictions for the next 30 days
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Print the forecast
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

# Plot the forecast
fig = model.plot(forecast)

# Save the plot as an image file
fig.savefig('forecast_visualization.png')  # Add this line to save the plot

# Display the plot
plt.show()  # Optionally display the plot if you're running it locally
