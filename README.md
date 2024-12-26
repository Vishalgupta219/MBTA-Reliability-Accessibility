# MBTA-Reliability-Accessibility
This project aims to analyze and predict the reliability and accessibility of the Massachusetts Bay Transportation Authority (MBTA) subway system using machine learning models. The goal is to enhance commuter experiences, optimize operations, and assist MBTA in better service planning.

*Project Overview*:
The MBTA operates the fourth busiest subway system in the United States, serving the Greater Boston area. This project focuses on predicting the reliability and accessibility of the system based on historical ridership data. By using predictive modeling, we can identify potential issues in advance and improve service quality.

Key Objectives:

Reliability Analysis: Predict serious reliability issues (e.g., delays or service interruptions) across different MBTA stations.
Accessibility Analysis: Estimate the number of passengers likely to board at various stations during different time periods.

*Dataset*:

The project utilizes data from MBTA’s open data platform, covering the period from 2021 to 2023. The dataset includes information on:

-Service Date: The date of service.

-Station Name: The name of the station in the MBTA system.

-The specific subway or train line (e.g., Green Line, Red Line).

-The number of people entering the gates at each station.

-The total number of trips taken during the service day.

-A binary indicator of whether a serious reliability issue occurred (1 = Yes, 0 = No).

-Includes mode of transportation, direction of travel, day type, rail time, and average number of passengers boarding.

Methodology:
1. Data Cleaning & Preprocessing
Removed missing values.
Converted categorical variables (e.g., station names, lines) into dummy variables for use in models.
Scaled and normalized the data as required.
2. Modeling
The following machine learning models were applied:

-Linear Regression: Used for predicting both reliability and accessibility metrics.

-Decision Tree Classifier: Applied to predict reliability issues and optimize predictions through hyperparameter tuning.

-Principal Component Analysis (PCA): Used for dimensionality reduction in accessibility predictions, although the model only explained 21% of the variance.

3. Performance Metrics

-Reliability Models: Evaluated using accuracy, F1 score, precision, and recall.

-Accessibility Models: Evaluated using Root Mean Squared Error (RMSE) to assess prediction accuracy for passenger boarding.

*Key Findings*:
1. Reliability Prediction: The linear regression model achieved an 82% accuracy in predicting serious reliability issues, with notable accuracy in forecasting issues at stations like Ruggles.
2. Decision Tree Model: Initially overfitted, with a 100% F1 score. After hyperparameter tuning, the model achieved 99% accuracy in predicting reliability.
3. Accessibility Prediction: The linear regression model with PCA achieved an RMSE of 770, although the explained variance was limited to 21%, indicating the need for further improvements.

*Conclusion*:
This project provides valuable insights into improving MBTA operations by predicting potential service disruptions and estimating passenger traffic. By forecasting reliability issues and accessibility patterns, the MBTA can better plan service schedules, optimize train frequencies, and improve overall commuter experience.

*Files Included*:

Reliability.py: Linear regression model for predicting reliability issues.

Reliability-DT.py: Decision tree model for reliability predictions.

Accessibility.py: Linear regression model for predicting passenger boarding.

Accessibility-PCA.py: PCA-based model for reducing dimensionality in accessibility prediction.

Reliability-2.csv: Data on service reliability, gate entries, and trips.

Accessibility.csv: Data on accessibility, including mode, direction, day type, and boarding patterns.

MBTA Reliability and Accessibility.pptx: Presentation summarizing the analysis, models, and insights.

*Future Work*:

Further improvements could include:
Incorporating more external factors, such as weather or events, to improve model predictions.
Expanding the use of machine learning models like Random Forests or Neural Networks for higher accuracy.
Enhancing the PCA model for better dimensionality reduction and improved predictions.
