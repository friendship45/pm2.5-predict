import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_squared_error, mean_absolute_error 
import numpy as np
import matplotlib.pyplot as plt
import random

dt = pd.read_csv('last_amb_hour_time.csv')
dt1 = dt.iloc[::-1]
dt2 = dt1.dropna(axis=0)

def remove_outliers(df, column): 
    Q1 = df[column].quantile(0.25)  
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1 
    lower_bound = Q1 - 1.5 * IQR 
    upper_bound = Q3 + 1.5 * IQR 
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

columns_to_check = ['PM2.5', '오존', '이산화질소', '일산화탄소', '아황산가스'] 
for col in columns_to_check: 
    dt2 = remove_outliers(dt2, col)

input_data = dt2[['오존', '이산화질소', '일산화탄소', '아황산가스']] 
target_data = dt2['PM2.5']

train_input, test_input, train_target, test_target = train_test_split(input_data, target_data, random_state=30)

rf = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=5, random_state=30) 
rf.fit(train_input, train_target)

print("Training R²:", rf.score(train_input, train_target)) 
print("Testing R²:", rf.score(test_input, test_target))

test_predictions = rf.predict(test_input)

plt.figure(figsize=(6,6))
plt.scatter(test_target, test_predictions, alpha=0.6)
plt.plot([min(test_target), max(test_target)], [min(test_target), max(test_target)], color='red', linestyle='--')

r2_test = rf.score(test_input, test_target)
plt.text(0.05, 0.95, f'R² = {r2_test:.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

plt.xlabel('Actual PM2.5') 
plt.ylabel('Predicted PM2.5') 
plt.title('Actual vs Predicted PM2.5')
plt.grid(True)
plt.tight_layout()
plt.show()

future_steps = 30 * 24  
last_input = input_data.iloc[-1].copy()
future_predictions = []
random.seed(42)

for _ in range(future_steps):
    input_array = last_input.values.reshape(1, -1)
    next_prediction = rf.predict(input_array)[0]
    future_predictions.append(next_prediction)

    last_input['오존'] *= random.uniform(0.99, 1.01)
    last_input['이산화질소'] *= random.uniform(0.99, 1.01)
    last_input['일산화탄소'] *= random.uniform(0.99, 1.01)
    last_input['아황산가스'] *= random.uniform(0.99, 1.01)

plt.figure(figsize=(12, 5))
plt.plot(range(len(future_predictions)), future_predictions, color='green', label='Predicted PM2.5 (Next 30 days)')
plt.xlabel('Future Hours')
plt.ylabel('PM2.5')
plt.title('Predicted PM2.5 Over Next 30 Days')
plt.legend()
plt.show()

