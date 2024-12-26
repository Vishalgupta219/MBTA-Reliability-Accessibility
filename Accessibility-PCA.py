import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

accessibility = pd.read_csv('Accessibility.csv')
accessibility['Service_Date'] = accessibility['Service_Date'].astype(str)
accessibility['Service_Date'] = accessibility['Service_Date'].str.replace('-', '')
accessibility['Service_Date'] = accessibility['Service_Date'].astype(int)

accessibility = accessibility[['Service_Date','Station_Name','Line', 'Mode', 
                               'Direction_ID', 'Day_Type', 'Rail_Time', 'Average_ONs']]
accessibility = accessibility.dropna(axis=0)

accessibility = pd.get_dummies(accessibility, drop_first=True)

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
scalar.fit(accessibility)
scalar_df = scalar.transform(accessibility)

from sklearn.decomposition import PCA
pca = PCA(n_components=10)

pca.fit(scalar_df)
pca_df = pca.transform(scalar_df)
pca.explained_variance_ratio_
print('Considering 10 PCA columns explains only ',np.round(np.sum(pca.explained_variance_ratio_)*100,0),
      '% of variance in data')
#These 10 columns explains  21.0 % of variance in data
pca_df = pd.DataFrame(pca_df, columns=["PC1","PC2","PC3","PC4","PC5","PC6","PC7","PC8","PC9","PC10"])

x = pca_df
y = accessibility['Average_ONs']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)

lm = LinearRegression()
lm.fit(x_train, y_train)

lm.intercept_
lm.coef_

y_pred = lm.predict(x_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
print('RMSE:', round(rmse,2))
#RMSE: 769.78
#RMSE is observed to be improved but we cannot go forward with it as currently it only explains 21% of variance in data
print('RMSE is observed to be improved but we cannot go forward with it, as it currently explains 21% of variance in data')
