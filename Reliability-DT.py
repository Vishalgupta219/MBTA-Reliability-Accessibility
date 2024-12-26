import pandas as pd
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt

reliability = pd.read_csv('Reliability.csv')
reliability['Service_Date'] = reliability['Service_Date'].astype(str)
reliability['Service_Date'] = reliability['Service_Date'].str.replace('-', '')
reliability['Service_Date'] = reliability['Service_Date'].astype(int)

reliability = reliability[['Service_Date','Station_Name','Line', 'Gate_Entries', 'Total_Trips', 'Serious?']]
reliability = reliability.dropna(axis=0)

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
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=10)

dt = DecisionTreeClassifier(random_state=1)
dt.fit(x_train, y_train)

y_pred = dt.predict(x_train)
confusion_matrix(y_train, y_pred)
f1_score(y_train, y_pred) #100

y_pred_test = dt.predict(x_test)
confusion_matrix(y_test, y_pred_test)
f1_score(y_test, y_pred_test) #100
print('The model is overfitted and it is important to tune the hyper parameters')

plt.figure(figsize=(10, 8))
plot_tree(dt)
plt.show()
dt.tree_.max_depth #24

parameter_grid = {'max_depth':range(1,24), 'min_samples_split':range(2,40)}
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(dt, parameter_grid, verbose=3, scoring='f1', cv=10)
grid.fit(x_train, y_train)
grid.best_params_
#{'max_depth': 23, 'min_samples_split': 2}

dt_new = DecisionTreeClassifier(max_depth=23,min_samples_split=2)
dt_new.fit(x_train, y_train)

y_pred = dt_new.predict(x_train)
confusion_matrix(y_train, y_pred)
f1_score(y_train, y_pred) #1.0 to #0.998

y_pred_test = dt_new.predict(x_test)
confusion_matrix(y_test, y_pred_test)
f1_score(y_test, y_pred_test) #1.0 to #0.999

plt.figure(figsize=(10, 8))
plot_tree(dt_new)
plt.show()

#Predicts the seriousness of reliability in Ruggles station (Orange Line) of future date 10-Jan-2024
#with 1000 poeple in every 2 hours and 100 trips were estimated to be taken that day
print('Predicts the seriousness of reliability in Ruggles station (Orange Line) of future date 10-Jan-2024 with 1000 poeple in every 2 hours and 100 trips were estimated to be taken that day')
prediction  = dt_new.predict([[100124, 1000, 100,   
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
# In the prediction of model, Ruggles station is expected to have reliability issues and we
# are 99% confident in saying this.
print('In the prediction of model, Ruggles station will not reliability issues and we are 99% confident in saying this.')
