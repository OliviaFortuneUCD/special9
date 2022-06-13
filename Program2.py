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


plt.figure(figsize=(18, 8))
sns.heatmap(df_red.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
plt.title('Correlation Map Of Red Wine Quality', fontdict={'fontsize':12}, pad=12);