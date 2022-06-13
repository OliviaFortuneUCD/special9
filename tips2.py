# Analysis
import pandas as pd
import numpy as np

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Models
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Reporting
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
#Now that we have are happy there is at least some correlation between bill size and tips size, we can look to build a linear model off it.
tips = sns.load_dataset('tips')
X = tips[['total_bill']] # Pass in a list, expecting a 2D array
y = tips['tip']
#Making a Prediction
#We can use this regression line to make a
# prediction on future tips. To do this, we call the predict() method and pass in our 2D list.
# In this case, we want to know how much tip we should add to a €100 bill.
reg = LinearRegression().fit(X, y)
print(reg.predict([[100]]))
#We can start to add in other features that might yield a more accuracte model. Here we add in the party size as well
X = tips[['total_bill','size']]
y = tips['tip']
reg = LinearRegression().fit(X, y)
def tip(bill, party):
    bill_arr = np.array([[bill, party]])
    return round(reg.predict(bill_arr)[0],2)

#a bill of 30 with a party of 5
print(tip(30,5))
#a bill of 100 with a party of 5
print(tip(100,5))
#a bill of 50 with a party of 2
print(tip(75,2))