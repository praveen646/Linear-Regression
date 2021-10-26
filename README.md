# Linear-Regression
predicting stock price

Linear Regression
Our data contains only one independent variable ($X$)</strong> which represents the <em>date</em> and the <strong>dependent variable ($Y$)</strong> we are trying to predict is the <em>Stock Price</em>. To fit a line to the data points, which then represents an estimated relationship between $X$ and $Y$, we can use a Simple Linear Regression.

The best fit line can be described with
$$
Y = \beta_0 + \beta_1 X
$$

where

$Y$ is the predicted value of the dependent variable
$\beta_0$ is the y-intercept
$\beta_1$ is the slope
$X$ is the value of the independent variable
The goal is to find such coefficients $\beta_0$ and $\beta_1$ that the Sum of Squared Errors, which represents the difference between each point in the dataset with it’s corresponding predicted value outputted by the model, is minimal.

mport Libraries
In [251]:
import numpy as np
import pandas as pd
import quandl
import datetime

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-darkgrid')
plt.rc('figure', figsize=(16,10))
plt.rc('lines', markersize=4)
Configure Quandl
In [2]:
# Import API key from file
import API_config
In [3]:
# Quandl API Auth
quandl.ApiConfig.api_key = API_config.API_KEY
Get the Data
In [4]:
# Set start and end date for stock prices
start_date = datetime.date(2009, 3,8)
end_date = datetime.date.today()
# Load data from Quandl
data = quandl.get('FSE/SAP_X', start_date=start_date, end_date=end_date)
# Save data to CSV file
data.to_csv('data/sap_stock.csv')
In [182]:
data.head()
Out[182]:
Open	High	Low	Close	Change	Traded Volume	Turnover	Last Price of the Day	Daily Traded Units	Daily Turnover
Date										
2009-03-09	25.16	25.82	24.48	25.59	NaN	5749357.0	145200289.0	None	None	NaN
2009-03-10	25.68	26.95	25.68	26.87	NaN	7507770.0	198480965.0	None	None	NaN
2009-03-11	26.50	26.95	26.26	26.64	NaN	5855095.0	155815439.0	None	None	NaN
2009-03-12	26.15	26.47	25.82	26.18	NaN	6294955.0	164489409.0	None	None	NaN
2009-03-13	26.01	26.24	25.65	25.73	NaN	6814568.0	176228331.0	None	None	NaN
In [183]:
# Check data types in columns
data.info()
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 2550 entries, 2009-03-09 to 2019-03-08
Data columns (total 10 columns):
Open                     2242 non-null float64
High                     2543 non-null float64
Low                      2543 non-null float64
Close                    2550 non-null float64
Change                   11 non-null float64
Traded Volume            2504 non-null float64
Turnover                 2497 non-null float64
Last Price of the Day    0 non-null object
Daily Traded Units       0 non-null object
Daily Turnover           7 non-null float64
dtypes: float64(8), object(2)
memory usage: 219.1+ KB
In [184]:
# Get descriptive statistics summary of data set 
data.describe()
Out[184]:
Open	High	Low	Close	Change	Traded Volume	Turnover	Daily Turnover
count	2242.000000	2543.000000	2543.000000	2550.000000	11.000000	2.504000e+03	2.497000e+03	7.0
mean	56.686896	61.563225	60.535073	60.995955	-0.070000	3.296818e+06	1.828440e+08	0.0
std	18.320821	21.184135	20.934460	21.097480	0.709761	2.004323e+06	9.350710e+07	0.0
min	25.160000	25.820000	24.480000	25.590000	-0.740000	0.000000e+00	1.767350e+05	0.0
25%	41.500000	43.430000	42.590000	42.950000	-0.500000	2.131686e+06	1.300462e+08	0.0
50%	56.560000	58.480000	57.580000	58.015000	-0.290000	2.852772e+06	1.626544e+08	0.0
75%	67.732500	78.365000	77.085000	77.762500	0.085000	3.878528e+06	2.104511e+08	0.0
max	100.100000	108.520000	107.020000	107.800000	1.250000	3.645671e+07	1.369431e+09	0.0
In [399]:
# Display features in data set
data.columns
Out[399]:
Index(['Open', 'High', 'Low', 'Close', 'Change', 'Traded Volume', 'Turnover',
       'Last Price of the Day', 'Daily Traded Units', 'Daily Turnover'],
      dtype='object')
Select Subset with relevant features
We use the daily closing price Close as the value to predict, so we can discard the other features.

'Close' column has numerical data type
The 'Date' is the index column and contains datetime values
In [400]:
# Create a new DataFrame with only closing price and date
df = pd.DataFrame(data, columns=['Close'])

# Reset index column so that we have integers to represent time for later analysis
df = df.reset_index()
In [6]:
df.head()
Out[6]:
Date	Close
0	2009-03-09	25.59
1	2009-03-10	26.87
2	2009-03-11	26.64
3	2009-03-12	26.18
4	2009-03-13	25.73
In [7]:
# Check data types in columns
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2550 entries, 0 to 2549
Data columns (total 2 columns):
Date     2550 non-null datetime64[ns]
Close    2550 non-null float64
dtypes: datetime64[ns](1), float64(1)
memory usage: 39.9 KB
In [389]:
# Check for missing values in the columns
df.isna().values.any()
Out[389]:
False
Explore the Data
When we take a look at the price movement over time by simply plotting the Closing price vs Time, we can already see, that the price continously increases over time and we can also estimate that trend could be linear.

In [397]:
# Import matplotlib package for date plots
import matplotlib.dates as mdates

years = mdates.YearLocator() # Get every year
yearsFmt = mdates.DateFormatter('%Y') # Set year format

# Create subplots to plot graph and control axes
fig, ax = plt.subplots()
ax.plot(df['Date'], df['Close'])

# Format the ticks
ax.xaxis.set_major_locator(years)
ax.xaxis.set_major_formatter(yearsFmt)

# Set figure title
plt.title('Close Stock Price History [2009 - 2019]', fontsize=16)
# Set x label
plt.xlabel('Date', fontsize=14)
# Set y label
plt.ylabel('Closing Stock Price in $', fontsize=14)

# Rotate and align the x labels
fig.autofmt_xdate()

# Show plot
plt.show()

Linear Regression
Our data contains only one independent variable ($X$)</strong> which represents the <em>date</em> and the <strong>dependent variable ($Y$)</strong> we are trying to predict is the <em>Stock Price</em>. To fit a line to the data points, which then represents an estimated relationship between $X$ and $Y$, we can use a Simple Linear Regression.

The best fit line can be described with
$$
Y = \beta_0 + \beta_1 X
$$

where

$Y$ is the predicted value of the dependent variable
$\beta_0$ is the y-intercept
$\beta_1$ is the slope
$X$ is the value of the independent variable
The goal is to find such coefficients $\beta_0$ and $\beta_1$ that the Sum of Squared Errors, which represents the difference between each point in the dataset with it’s corresponding predicted value outputted by the model, is minimal.

Training a Linear Regression Model
Train Test Split
In [ ]:
# Import package for splitting data set
from sklearn.model_selection import train_test_split
In [9]:
# Split data into train and test set: 80% / 20%
train, test = train_test_split(df, test_size=0.20)
Create and Train the Model
In [ ]:
# Import package for linear model
from sklearn.linear_model import LinearRegression
In [12]:
# Reshape index column to 2D array for .fit() method
X_train = np.array(train.index).reshape(-1, 1)
y_train = train['Close']
In [242]:
# Create LinearRegression Object
model = LinearRegression()
# Fit linear model using the train data set
model.fit(X_train, y_train)
Out[242]:
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
         normalize=False)
Model Evaluation
In [428]:
# The coefficient
print('Slope: ', np.asscalar(np.squeeze(model.coef_)))
# The Intercept
print('Intercept: ', model.intercept_)
Slope:  0.02762424028961455
Intercept:  25.723432313963862
Interpreting the coefficients:

The slope coefficient tells us that with a 1 unit increase in date the closing price increases by 0.0276 $
The intercept coefficient is the price at wich the closing price measurement started, the stock price value at date zero
In [169]:
# Train set graph
plt.figure(1, figsize=(16,10))
plt.title('Linear Regression | Price vs Time')
plt.scatter(X_train, y_train, edgecolor='w', label='Actual Price')
plt.plot(X_train, model.predict(X_train), color='r', label='Predicted Price')
plt.xlabel('Integer Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

Prediction from our Model
In [39]:
# Create test arrays
X_test = np.array(test.index).reshape(-1, 1)
y_test = test['Close']
In [63]:
# Generate array with predicted values
y_pred = model.predict(X_test)
Regression Evaluation
Let's have a look at how the predicted values compare with the actual value on random sample from our data set.

In [417]:
# Get number of rows in data set for random sample
df.shape
Out[417]:
(2550, 3)
In [421]:
# Generate 25 random numbers
randints = np.random.randint(2550, size=25)

# Select row numbers == random numbers
df_sample = df[df.index.isin(randints)]
In [422]:
df_sample.head()
Out[422]:
Date	Close	Prediction
104	2009-07-31	33.00	28.596353
628	2011-08-12	37.24	43.071455
629	2011-08-15	36.99	43.099079
646	2011-09-07	37.24	43.568692
657	2011-09-22	36.28	43.872558
In [425]:
# Create subplots to plot graph and control axes
fig, ax = plt.subplots()
df_sample.plot(x='Date', y=['Close', 'Prediction'], kind='bar', ax=ax)

# Set figure title
plt.title('Comparison Predicted vs Actual Price in Sample data selection', fontsize=16)

# 

# Set x label
plt.xlabel('Date', fontsize=14)

# Set y label
plt.ylabel('Stock Price in $', fontsize=14)

# Show plot
plt.show()

We can see some larger variations between predicted and actual values in the random sample.
Let's see how the model performed over the whole test data set.

In [424]:
# Plot fitted line, y test
plt.figure(1, figsize=(16,10))
plt.title('Linear Regression | Price vs Time')
plt.plot(X_test, model.predict(X_test), color='r', label='Predicted Price')
plt.scatter(X_test, y_test, edgecolor='w', label='Actual Price')

plt.xlabel('Integer Date')
plt.ylabel('Stock Price in $')

plt.show()

In [434]:
# Plot predicted vs actual prices
plt.scatter(y_test, y_pred)

plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')

plt.title('Predicted vs Actual Price')

plt.show()

The data points are mostly close to a diagonal, which indicates, that the predicted values are close to the actual value and the model's performance is largerly quite good.
Yet there are some areas, around 55 to 65, the model seems to be quite random and shows no relationship between the predicted and actual value.
Also in the area around 85 - 110 the data point are spread out quite heavily and the predictions don't cover the values above 100.

Residual Histogram
The residuals are nearly normally distributed around zero, with a slight skewedness to the right.

In [388]:
# Import norm package to plot normal distribution
from scipy.stats import norm

# Fit a normal distribution to the data:
mu, std = norm.fit(y_test - y_pred)

ax = sns.distplot((y_test - y_pred), label='Residual Histogram & Distribution')

# Calculate the pdf over a range of values         
x = np.linspace(min(y_test - y_pred), max(y_test - y_pred), 100)
p = norm.pdf(x, mu, std)

# And plot on the same axes that seaborn put the histogram
ax.plot(x, p, 'r', lw=2, label='Normal Distribution') 

plt.legend()
plt.show()

In [415]:
# Add new column for predictions to df
df['Prediction'] = model.predict(np.array(df.index).reshape(-1, 1))
In [416]:
df.head()
Out[416]:
Date	Close	Prediction
0	2009-03-09	25.59	25.723432
1	2009-03-10	26.87	25.751057
2	2009-03-11	26.64	25.778681
3	2009-03-12	26.18	25.806305
4	2009-03-13	25.73	25.833929
Error Evaluation Metrics
Mean Absolute Error (MAE) is the mean of the absolute value of the errors:$$
\frac{1}{N} \sum_{i = 1}^{N} |y_i - \hat{y}_i|
$$

Mean Squared Error (MSE) is the mean of the squared errors:$$
\frac{1}{N} \sum_{i = 1}^{N} (y_i - \hat{y}_i)^2
$$

Root Mean Squared Error (RMSE) is the square root of the mean of the squared errors:$$
\sqrt{\frac{1}{N} \sum_{i = 1}^{N} (y_i - \hat{y}_i)^2}
$$

All of these are cost functions we want to minimize.

In [72]:
# Import metrics package from sklearn for statistical analysis
from sklearn import metrics
In [430]:
# Statistical summary of test data
df['Close'].describe()
Out[430]:
count    2550.000000
mean       60.995955
std        21.097480
min        25.590000
25%        42.950000
50%        58.015000
75%        77.762500
max       107.800000
Name: Close, dtype: float64
In [73]:
# Calculate and print values of MAE, MSE, RMSE
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
Mean Absolute Error: 4.172498876833842
Mean Squared Error: 28.76248048285447
Root Mean Squared Error: 5.3630663321326235
The MAE is 3% (of minimum) and 6% (of maximum) of the Closing Price.
The other two errors are larger, because the errors are squared and have therefore a greater influence on the result.
Accuracy Evaluation Metrics
To see how accurate our model is, we can calculate the Coefficient of determination, which describes the ratio between the total error and the error, that is explained by our model. It's value is between 0 and 1, with 1 meaning 100% of the error is acoounted for by the model.

Coefficient of determination$$
R^2 = 1 - \frac{RSS}{TSS}
$$

with

Residual Sum of Squares (RSS)$$
RSS = \sum_{i = 1}^{N} \epsilon_i^2 = \sum_{i = 1}^{N} (y_i - \hat{y}_i)^2
$$

Total Sum of Squares (TSS)$$
TSS = \sum_{i = 1}^{N} (y_i - \bar{y}_i)^2
$$

In [247]:
print('R2: ', metrics.r2_score(y_test, y_pred))
R2:  0.9395015859480149
In [433]:
from sklearn.metrics import explained_variance_score
explained_variance_score(y_test, y_pred)
Out[433]:
0.9397266931814002
The value of $R^2$ shows that are model accounts for nearly 94% of the differences between the actual stock prices and the predicted prices.
