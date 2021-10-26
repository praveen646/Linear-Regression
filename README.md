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
