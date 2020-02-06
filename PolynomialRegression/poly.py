# Firstly the program will be using polynomial regression to pefrom prediction based on the inputs
# The inputs or variables we will be anlysing to study is the relation between the price and the size of the pizza
# Lastly the data will displayed via a graph which indicates the "non-linear" relationship between the 2 features; in a user friednly manner


import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x_train = [[8], [18], [13], [17], [30]]
y_train = [[13], [22], [28], [30], [40]]

x_test = [[8], [18], [11], [16]] 
y_test = [[13], [22], [15], [18]] 

regresor = LinearRegression()
regresor.fit(x_train, y_train)
xx = np.linspace(0, 26, 100)
yy = regresor.predict(xx.reshape(xx.shape[0], 1))
plt.plot(xx, yy)

quadratic_featurizer = PolynomialFeatures(degree=2)

x_train_quadratic = quadratic_featurizer.fit_transform(x_train)
x_test_quadratic = quadratic_featurizer.transform(x_test)

regressor_quadratic = LinearRegression()
regressor_quadratic.fit(x_train_quadratic, y_train)
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))

plt.plot(xx, regressor_quadratic.predict(xx_quadratic), c='y', linestyle='--')
plt.title('Pizza price regressed on the diameter')
plt.xlabel('Diameter in Cm')
plt.ylabel('Price in Rands')
plt.axis([0, 30, 0, 30])
plt.grid(True)
plt.scatter(x_train, y_train)
plt.show()
print (x_train)
print ()
print (x_train_quadratic)
print ()
print (x_test)
print ()
print (x_test_quadratic)
