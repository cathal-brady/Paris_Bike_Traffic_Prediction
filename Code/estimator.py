import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin  
from sklearn.pipeline import Pipeline
import holidays
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error



class ColumnSelector(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[['counter_id','date']]


class DateFormatter(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        X_copy['date'] = pd.to_datetime(X_copy['date'])
        X_copy['year'] = X_copy['date'].dt.year
        X_copy['month'] = X_copy['date'].dt.month
        X_copy['weekday'] = (X_copy['date'].dt.dayofweek + 1)
        X_copy['hour'] = X_copy['date'].dt.hour
        return X_copy


class HolidaysFR(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        is_holiday = lambda date: 1 if date in holidays.FR() else 0
        is_weekend = lambda day: 1 if day in (6,7) else 0
        X_copy = X.copy()
        X_copy['is_Holiday'] = X_copy['date'].apply(is_holiday)
        X_copy['is_Weekend'] = X_copy['weekday'].apply(is_weekend)
        #X_copy.drop(columns='date', inplace=True)
        return X_copy


class EncodeCounter(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        X_copy = pd.get_dummies(X_copy, columns=['counter_id'], dtype=int, drop_first=True)
        return X_copy

class MergeWeather(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = pd.read_csv("weather_data_cleaned.csv")
        data['date'] = pd.to_datetime(data['date']).astype('datetime64[us]')
        merged_data = pd.merge_asof(X, data, on='date')
        merged_data.drop(columns='date', inplace=True)
        return merged_data


preprocess = Pipeline([
    ("ColumnSelector", ColumnSelector()),
    ("DateFormatter", DateFormatter()),
    ("HolidaysFR", HolidaysFR()),
    ("EncodeCounter", EncodeCounter()),
    ("MergeWeather", MergeWeather())
])        

df = pd.read_parquet("train.parquet")
df = df.sort_values('date') # Sort by date 

X = preprocess.fit_transform(df)
y = df['log_bike_count']

model = RandomForestRegressor()

# Fit the model on the training data
model.fit(X,y)

# Import test set
df_test = pd.read_parquet("final_test.parquet")
df_test = df_test.sort_values('date') # Sort by date 
new_order = df_test.index.tolist() #Keep index order


df_test = preprocess.transform(df_test)
predictions = model.predict(df_test)
predictions_df = pd.DataFrame({'Id': new_order, 'log_bike_count': predictions})
predictions_df = predictions_df.sort_values('Id')

# Specify the file path
csv_file_path = 'submission.csv'

# Write the DataFrame to a CSV file
predictions_df.to_csv(csv_file_path, index=False)
