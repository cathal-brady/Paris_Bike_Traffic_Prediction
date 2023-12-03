import pandas as pd
import numpy as np
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import holidays
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


class ColumnSelector(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[['counter_id', 'date']]  # , 'coordinates'


class DateFormatter(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['date'] = pd.to_datetime(X_copy['date'])
        X_copy['year'] = X_copy['date'].dt.year
        X_copy['month'] = X_copy['date'].dt.month
        # X_copy['week'] = X_copy['date'].dt.isocalendar().week
        X_copy['weekday'] = (X_copy['date'].dt.dayofweek + 1)
        # X_copy['day'] = X_copy['date'].dt.day
        X_copy['hour'] = X_copy['date'].dt.hour
        # X_copy['minute'] = X_copy['date'].dt.minute  # Not relevant
        return X_copy


class HolidaysFR(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def is_holiday(date): return 1 if date in holidays.FR() else 0
        def is_weekend(day): return 1 if day in (6, 7) else 0
        X_copy = X.copy()
        X_copy['is_Holiday'] = X_copy['date'].apply(is_holiday)
        X_copy['is_Weekend'] = X_copy['weekday'].apply(is_weekend)
        return X_copy


class EncodeCounter(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy = pd.get_dummies(
            X_copy, columns=['counter_id'], dtype=int, drop_first=True)
        return X_copy


class MergeWeather(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        file_path_weather = "/kaggle/input/weather_data-cleaned.csv"
        data = pd.read_csv(file_path_weather)        
        data['date'] = pd.to_datetime(data['date']).astype('datetime64[us]')
        merged_data = pd.merge_asof(X, data, on='date')
        # merged_data.drop(columns='date', inplace=True)
        return merged_data


class MergeMultimodal(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        # Import Multimodal Data
        file_path_multimodal = "/kaggle/input/multimodal_data.csv"
        mult_df = pd.read_csv(file_path_multimodal)
        mult_df['date'] = pd.to_datetime(
            mult_df['date']).astype('datetime64[us]')
        # Averaging and scaling the count
        mult_df = pd.DataFrame(mult_df.groupby(['date'])[
                               'count'].sum()).reset_index()
        scaler = StandardScaler()
        numerical_columns = mult_df.select_dtypes(include='number').columns
        mult_df[numerical_columns] = scaler.fit_transform(
            mult_df[numerical_columns])
        # Merging data
        merged_data = pd.merge_asof(X_copy, mult_df, on='date')
        merged_data.rename(columns={'count': 'average_multimodal_count'})
        merged_data.drop(columns='date', inplace=True)
        return merged_data


preprocess = Pipeline([
    ("ColumnSelector", ColumnSelector()),
    ("DateFormatter", DateFormatter()),
    ("HolidaysFR", HolidaysFR()),
    ("EncodeCounter", EncodeCounter()),
    ("MergeWeather", MergeWeather()),
    ("MergeMultimodal", MergeMultimodal())
])


file_path_train = "/kaggle/input/mdsb-2023/train.parquet"

df = pd.read_parquet(file_path_train)
df = df.sort_values('date')  # Sort by date

X = preprocess.fit_transform(df)
y = df['log_bike_count']

model = RandomForestRegressor()

# Fit the model on the training data
model.fit(X, y)

# Import test set
file_path_test = "/kaggle/input/mdsb-2023/final_test.parquet"
df_test = pd.read_parquet(file_path_test)
df_test = df_test.sort_values('date')  # Sort by date
new_order = df_test.index.tolist()  # Keep index order


df_test = preprocess.transform(df_test)
predictions = model.predict(df_test)
predictions_df = pd.DataFrame({'Id': new_order, 'log_bike_count': predictions})
predictions_df = predictions_df.sort_values('Id')

# Specify the file path
csv_file_path = '/kaggle/output/working/submission.csv'

# Write the DataFrame to a CSV file
predictions_df.to_csv(csv_file_path, index=False)
