import pandas as pd
import numpy as np

### Question 1 - version of pandas
version = (pd.__version__)
## 2.3.1
print(f'Question 1 answer is:', version)

### Question 2 - total number of rows
df = pd.read_csv('car_fuel_efficiency.csv')
##make sure df loaded
# print(df.tail())
# ## 9703 + 1
# pd.options.display.max_rows = 99999
# print(df)
# # 9704
print(f'Question 2 answer is:', )

### Question 3 - Fuel types
# #check df cols
# print(df.head())
# # print all cols
# pd.set_option('display.width', None)
# #print 1st row
# print(df.iloc[[0]])
# return unique 'fuel_types'
fuels = df['fuel_type'].unique()
fuel_count = len(fuels)
print(f'Question 3 answer is:', fuel_count)

###Question 4 - number of cols missing values 
missing_values = df.isnull()
# missing_vals_per_col = df.isnull().sum()
has_missing = df.isnull().any().sum()
print(f'Question 4 answer is:', has_missing)

## Question 5 - What's the maximum fuel efficiency of cars from Asia?
# find col specifiying asian cars - in 'origin'

pd.set_option('display.width', None)
# print(df.head())
# check overall max - 25.96
# max_efficiency = df['fuel_efficiency_mpg'].max()
# print(max_efficiency)
# make sure values aren't null
filtered_df = df[df['origin'].notnull() & (df['fuel_efficiency_mpg'].notnull())]
# mask to only cars originated in Asia
asia_cars = df[df['origin'] == 'Asia']
# look at fuel efficiency col - in 'fuel_efficiency_mpg' and find max
max_eff = asia_cars['fuel_efficiency_mpg'].max()
print(f'Question 5 answer is:', max_eff)

## Question 6 - Median value of horsepower
# print(df.head())
# median_value = df['horsepower'].median()
# print(median_value) # 149
mode_value = df['horsepower'].mode()
print(mode_value) # 152
mode_value = df['horsepower'].mode().iloc[0]
# fill na values with mode
df['horsepower'] = df['horsepower'].fillna(mode_value)
print(df)
median_value = df['horsepower'].median() # 152
print(f'Question 6 answer is:', median_value)

## Question 7 Sum of weights
# select all cars from Asia - filter
# print(asia_cars)
# select first 7 rows
slice = asia_cars[['vehicle_weight','model_year']].iloc[0:7]
# turn slice in to numpy array
x = slice.to_numpy()
print(x)
# transpose with numpy
transposed = x.T
# now compute matrix-matrix multiplication
xtx = np.dot(x.T, x)
print(f"this is xtx:", xtx)
#invert xtx
inv = np.linalg.inv(xtx)
print(f"this is inverted:", inv)
# create an array with values
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
print(f"This is y:", y)
# print(type(y))
# Multiply the inverse of XTX with the transpose of X, and then multiply the result by y
a = np.dot(inv, transposed)
w = np.dot(a, y)
print(f'Question 7 answer is:', w)