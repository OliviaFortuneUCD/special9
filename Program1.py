import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import normaltest, skew
from sklearn import metrics
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

df_red  = pd.read_csv('winequality.csv')
print(df_red.info(), df_red.head())

plt.figure(figsize=(10, 6))
sns.scatterplot(x='density', y='alcohol', data= df_red, hue='quality')