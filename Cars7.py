#In this case, we can ask for the coefficient value of weight against CO2,
#and for volume against CO2. The answer(s) we get tells us what would happen if we increase, or decrease, one of the independent values.
import pandas
from sklearn import linear_model

df = pandas.read_csv("cars1.csv")

X = df[['Weight', 'Volume']]
y = df['CO2']

regr = linear_model.LinearRegression()
regr.fit(X, y)

print(regr.coef_)


#These values tell us that if the weight increase by 1kg, the CO2 emission increases by 0.00755095g.

#And if the engine size (Volume) increases by 1 cm3, the CO2 emission increases by 0.00780526 g.