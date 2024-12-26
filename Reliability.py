import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

reliability = pd.read_csv('Reliability.csv')
reliability['Service_Date'] = reliability['Service_Date'].astype(str)
reliability['Service_Date'] = reliability['Service_Date'].str.replace('-', '')
reliability['Service_Date'] = reliability['Service_Date'].astype(int)

reliability = reliability[['Service_Date','Station_Name','Line', 'Gate_Entries', 'Total_Trips', 'Serious?']]
reliability = reliability.dropna(axis=0)

sns.countplot(x='Serious?', data=reliability)

reliability = pd.get_dummies(reliability, drop_first=True)

x = reliability[['Service_Date', 'Gate_Entries', 'Total_Trips', 'Station_Name_Alewife',
       'Station_Name_Andrew', 'Station_Name_Aquarium',
       'Station_Name_Arlington', 'Station_Name_Ashmont',
       'Station_Name_Assembly', 'Station_Name_Back Bay',
       'Station_Name_Ball Square', 'Station_Name_Beachmont',
       'Station_Name_Bowdoin', 'Station_Name_Boylston',
       'Station_Name_Braintree', 'Station_Name_Broadway',
       'Station_Name_Central', 'Station_Name_Charles/MGH',
       'Station_Name_Chinatown', 'Station_Name_Community College',
       'Station_Name_Copley', 'Station_Name_Courthouse', 'Station_Name_Davis',
       'Station_Name_Downtown Crossing', 'Station_Name_East Somerville',
       'Station_Name_Fields Corner', 'Station_Name_Forest Hills',
       'Station_Name_Gilman Square', 'Station_Name_Government Center',
       'Station_Name_Green Street', 'Station_Name_Harvard',
       'Station_Name_Haymarket', 'Station_Name_Hynes Convention Center',
       'Station_Name_JFK/UMass', 'Station_Name_Jackson Square',
       'Station_Name_Kendall/MIT', 'Station_Name_Kenmore',
       'Station_Name_Lechmere', 'Station_Name_Magoun Square',
       'Station_Name_Malden Center', 'Station_Name_Massachusetts Avenue',
       'Station_Name_Mattapan Line', 'Station_Name_Maverick',
       'Station_Name_Medford/Tufts', 'Station_Name_North Quincy',
       'Station_Name_North Station', 'Station_Name_Oak Grove',
       'Station_Name_Orient Heights', 'Station_Name_Park Street',
       'Station_Name_Porter', 'Station_Name_Prudential',
       'Station_Name_Quincy Adams', 'Station_Name_Quincy Center',
       'Station_Name_Revere Beach', 'Station_Name_Riverside',
       'Station_Name_Roxbury Crossing', 'Station_Name_Ruggles',
       'Station_Name_Savin Hill', 'Station_Name_Science Park',
       'Station_Name_Shawmut', 'Station_Name_South Station',
       'Station_Name_State Street', 'Station_Name_Stony Brook',
       'Station_Name_Suffolk Downs', 'Station_Name_Sullivan Square',
       'Station_Name_Symphony', 'Station_Name_Tufts Medical Center',
       'Station_Name_Union Square', 'Station_Name_Wellington',
       'Station_Name_Wollaston', 'Station_Name_Wonderland',
       'Station_Name_Wood Island', 'Station_Name_World Trade Center',
       'Line_Green Line', 'Line_Mattapan Line', 'Line_Orange Line',
       'Line_Red Line', 'Line_Silver Line']]
y = reliability['Serious?_Yes']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)

lm = LinearRegression()
lm.fit(x_train, y_train)

lm.intercept_
lm.coef_

y_pred = lm.predict(x_test)
threshold = np.mean(y_pred)
y_pred = ['True' if x > threshold else 'False' for x in y_pred]
y_pred = pd.Series(y_pred)
y_pred = y_pred.map(lambda x: True if x == 'True' else False)

c_matrix = pd.DataFrame(confusion_matrix(y_test, y_pred, labels=[0,1]),
                                         index=["Actual:0", "Actual:1"],
                                         columns=["Pred:0", "Pred:1"])
#          Pred:0  Pred:1
#Actual:0    9415    3718
#Actual:1     499   10165
# Accuracy Score = 19850/23797 = 82%

f1_score(y_pred, y_test)
# F1 Score = 82%
precision_score(y_pred, y_test)
# Precision_score = 95%
recall_score(y_pred, y_test)
# Recall_score = 73%

#Predicts the seriousness of reliability in Ruggles station (Orange Line) of future date 10-Jan-2024
#with 1000 poeple in every 2 hours and 100 trips were estimated to be taken that day
print('Predicts the seriousness of reliability in Ruggles station (Orange Line) of future date 10-Jan-2024 with 1000 poeple in every 2 hours and 100 trips were estimated to be taken that day')
prediction  = lm.predict([[100124, 1000, 100,   
                           0,   0,   0,   0,   0, 
               0,   0,   0,   0,   0,   0,   0,   0, 
               0,   0,   0,   0,   0,   0,   0,   0, 
               0,   0,   0,   0,   0,   0,   0,   0, 
               0,   0,   0,   0,   0,   0,   0,   0, 
               0,   0,   0,   0,   0,   0,   0,   0, 
               0,   0,   0,   0,   0,   0,   0,   0, 
               1,   0,   0,   0,   0,   0,   0,   0, 
               0,   0,   0,   0,   0,   0,   0,   0,   0,
               0,   0,   1,   0,   0]])
prediction = ['True' if x > threshold else 'False' for x in abs(prediction)]
# In the prediction of model, Ruggles station is expected to have reliability issues and we
# are 82% confident in saying this.
print('In the prediction of model, Ruggles station is expected to have reliability issues and we are 82% confident in saying this.')
