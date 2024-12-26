import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

accessibility = pd.read_csv('Accessibility.csv')
accessibility['Service_Date'] = accessibility['Service_Date'].astype(str)
accessibility['Service_Date'] = accessibility['Service_Date'].str.replace('-', '')
accessibility['Service_Date'] = accessibility['Service_Date'].astype(int)

accessibility = accessibility[['Service_Date','Station_Name','Line', 'Mode', 
                               'Direction_ID', 'Day_Type', 'Rail_Time', 'Average_ONs']]
accessibility = accessibility.dropna(axis=0)

accessibility = pd.get_dummies(accessibility, drop_first=True)

x = accessibility[['Service_Date', 'Mode', 'Direction_ID',
       'Station_Name_Alewife', 'Station_Name_Andrew', 'Station_Name_Aquarium',
       'Station_Name_Arlington', 'Station_Name_Ashmont',
       'Station_Name_Assembly', 'Station_Name_Back Bay',
       'Station_Name_Beachmont', 'Station_Name_Bowdoin',
       'Station_Name_Boylston', 'Station_Name_Braintree',
       'Station_Name_Broadway', 'Station_Name_Central',
       'Station_Name_Charles/MGH', 'Station_Name_Chinatown',
       'Station_Name_Community College', 'Station_Name_Copley',
       'Station_Name_Davis', 'Station_Name_Downtown Crossing',
       'Station_Name_Fields Corner', 'Station_Name_Forest Hills',
       'Station_Name_Government Center', 'Station_Name_Green Street',
       'Station_Name_Harvard', 'Station_Name_Haymarket',
       'Station_Name_Hynes Convention Center', 'Station_Name_JFK/UMass',
       'Station_Name_Jackson Square', 'Station_Name_Kendall/MIT',
       'Station_Name_Kenmore', 'Station_Name_Lechmere',
       'Station_Name_Malden Center', 'Station_Name_Massachusetts Avenue',
       'Station_Name_Maverick', 'Station_Name_North Quincy',
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
       'Station_Name_Wellington', 'Station_Name_Wollaston',
       'Station_Name_Wonderland', 'Station_Name_Wood Island',
       'Line_Green Line', 'Line_Orange Line', 'Line_Red Line',
       'Day_Type_sunday', 'Day_Type_weekday', 'Rail_Time_11 PM - 1 AM',
       'Rail_Time_12 PM - 2 PM', 'Rail_Time_2 PM - 4 PM',
       'Rail_Time_4 PM - 6 PM', 'Rail_Time_5 AM - 6 AM',
       'Rail_Time_6 AM - 8 AM', 'Rail_Time_6 PM - 9 PM',
       'Rail_Time_8 AM - 11 AM', 'Rail_Time_9 PM - 11 PM']]
y = accessibility['Average_ONs']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)

lm = LinearRegression()
lm.fit(x_train, y_train)

lm.intercept_
lm.coef_

y_pred = lm.predict(x_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
print('RMSE:', round(rmse,2))

#Predicts the average number of passengers that are going to broad trains in Ruggles station (Orange Line) 
#of future date 10-Jan-2024 (weekday) to direction towards Forest Hills from 6 PM to 9 PM 
print('Predicts the average number of passengers that are going to board trains in Ruggles station (Orange Line) of future date 10-Jan-2024 (weekday) to direction towards Forest Hills from 6 PM to 9 PM ')
prediction  = lm.predict([[20240110, 1, 0,   
                           0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  
                           0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  
                           0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  
                           0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  
                           0,  0,  
                           0,  1,  0,  
                           0,  1,  
                           0,  0,  0,  0,  0,  0,  1,  0,  0]])
print('Average number of passengers to be boarded on 10-Jan-2024 (weekday) from Ruggles Station from 6PM to 9PM: ',np.round(prediction,0).astype(int))
