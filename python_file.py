import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv(r'E:\untitled2\datasets_33080_43333_car data.csv')
df.insert(1, value=2020 - df['Year'], column='no_year')
df.drop(['Year', 'Car_Name'], axis=1, inplace=True)

df = pd.get_dummies(df, drop_first=True)

# print(np.array(df.columns))

corr = df.corr()
top_corr = corr.index
plt.figure(figsize=(20, 20))
g = sns.heatmap(df[top_corr].corr(), annot=True, cmap="RdYlGn")

y_train = df['Selling_Price']
df.drop(['Selling_Price'], axis=1, inplace=True)

X_train = df

print(np.array(X_train.columns))

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, train_size=0.8, random_state=0)

from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 30, num=6)]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

rf = RandomForestRegressor()

rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, scoring='neg_mean_squared_error',
                               n_iter=10, cv=5, verbose=2, random_state=42, n_jobs=1)

rf_random.fit(X_train, y_train)


print(rf_random.best_params_)

predictions = rf_random.predict(X_test)

print('MAE:', sklearn.metrics.mean_absolute_error(y_test, predictions))
print('MSE:', sklearn.metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(sklearn.metrics.mean_squared_error(y_test, predictions)))


import pickle
file = open('random_forest_regression_model.pkl', 'wb')
pickle.dump(rf_random, file)
