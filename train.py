#%% This code snippet is to train and save the machine learning models
# The datset was extracted from https://archive.ics.uci.edu/ml/datasets/energy+efficiency

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import joblib
###############################################################################################################################
#%% load the data
df=pd.read_csv('./data/ENB2012_CSV_data.csv')

# Naming the columns based on the documentation from machine learning repository (link above)
for col in df.columns:
    if col=='Y1':
        df.rename(columns={col:'Heating_Load'}, inplace=True)
    if col=='Y2':
        df.rename(columns={col:'Cooling_Load'}, inplace=True)
    if col=='X1':
        df.rename(columns={col:'Relative_compactness'}, inplace=True)
    if col=='X2':
        df.rename(columns={col:'Surface_area'}, inplace=True)
    if col=='X3':
        df.rename(columns={col:'Wall_area'}, inplace=True)
    if col=='X4':
        df.rename(columns={col:'Roof_area'}, inplace=True)
    if col=='X5':
        df.rename(columns={col:'Overall_height'}, inplace=True)
    if col=='X6':
        df.rename(columns={col:'Orientation'}, inplace=True)
    if col=='X7':
        df.rename(columns={col:'Glazing_area'}, inplace=True)
    if col=='X8':
        df.rename(columns={col:'Glazing_area_distribution'}, inplace=True)

###############################################################################################################################
#%% train/test split and normlization of data
X=df.drop(["Heating_Load","Cooling_Load"],axis=1)
y1=df['Heating_Load']
y3=df['Cooling_Load'] # we want to use y3 to geenrate y2

# X_train will be used for model building and parametrs tunning and X_test is used for testing the model
X_train,X_test,y_train_1,y_test_1=train_test_split(X,y1,test_size=0.2,random_state=4)
y_train_2=y3[y_train_1.index]
y_test_2=y3[y_test_1.index]

# scaling the input data
scaler_x = MinMaxScaler()
scaler_x.fit(X_train)
scaled_X_train=scaler_x.transform(X_train)
scaled_X_test = scaler_x.transform(X_test)

# scaling the output data
scaler_y1=MinMaxScaler()
scaler_y1.fit(y_train_1.values.reshape(-1, 1))
scaled_y_train_1=scaler_y1.transform(y_train_1.values.reshape(-1, 1))
scaled_y_train_1=scaled_y_train_1.ravel()
scaled_y_test_1 = scaler_y1.transform(y_test_1.values.reshape(-1, 1))
scaled_y_test_1=scaled_y_test_1.ravel()

scaler_y2=MinMaxScaler()
scaler_y2.fit(y_train_2.values.reshape(-1, 1))
scaled_y_train_2=scaler_y2.transform(y_train_2.values.reshape(-1, 1))
scaled_y_train_2=scaled_y_train_2.ravel()
scaled_y_test_2 = scaler_y2.transform(y_test_2.values.reshape(-1, 1))
scaled_y_test_2=scaled_y_test_2.ravel()


###############################################################################################################################
#%% Training the random forest model using RandomizedSearchCV for optimization of parameters

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
           'max_features': max_features,
           'max_depth': max_depth,
           'min_samples_split': min_samples_split,
           'min_samples_leaf': min_samples_leaf,
           'bootstrap': bootstrap}

# Heating Load
rf_1 = RandomForestRegressor() #n_estimators = 1000, random_state = 42
rf_random_1 = RandomizedSearchCV(estimator = rf_1, param_distributions = random_grid, n_iter = 100, cv = 10, verbose=2, random_state=42, n_jobs = -1)
rf_random_1.fit(scaled_X_train, scaled_y_train_1)

# Cooling Load
rf_2 = RandomForestRegressor() #n_estimators = 1000, random_state = 42
rf_random_2 = RandomizedSearchCV(estimator = rf_2, param_distributions = random_grid, n_iter = 100, cv = 10, verbose=2, random_state=42, n_jobs = -1)
rf_random_2.fit(scaled_X_train, scaled_y_train_2)
###############################################################################################################################
#%% model evaluation

y_pred1=rf_random_1.predict(scaled_X_test)
y_pred1=scaler_y1.inverse_transform(y_pred1.reshape(-1, 1))
y_pred2=rf_random_2.predict(scaled_X_test)
y_pred2=scaler_y2.inverse_transform(y_pred2.reshape(-1, 1))
###############################################################################################################################
#%% print the results of evaluation        


print("#### Heating Load Models were trained and tested with average r2 score of {:0.2f} %".format(r2_score(y_test_1,y_pred1)*100))
print("")
print("#### Cooling Load Models were trained and tested with average r2 score of {:0.2f} %".format(r2_score(y_test_2,y_pred2)*100))


###############################################################################################################################
#%% save the models

# scalers
joblib.dump(scaler_x,"./models/scaler_x.pkl")
joblib.dump(scaler_y1,"./models/scaler_y1.pkl")
joblib.dump(scaler_y2,"./models/scaler_y2.pkl")
# models
joblib.dump(rf_random_1, "./models/rf_random_1.joblib")
joblib.dump(rf_random_2, "./models/rf_random_2.joblib")




