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
        return X[['counter_id', 'date']]


class DateFormatter(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['date'] = pd.to_datetime(X_copy['date'])
        X_copy['year'] = X_copy['date'].dt.year
        X_copy['mnth'] = X_copy['date'].dt.month
        #X_copy['week'] = X_copy['date'].dt.isocalendar().week
        X_copy['weekday'] = (X_copy['date'].dt.dayofweek + 1)
        #X_copy['day'] = X_copy['date'].dt.day
        X_copy['hr'] = X_copy['date'].dt.hour
        X_copy['hr_sin'] = np.sin(X_copy.hr*(2.*np.pi/24))
        X_copy['hr_cos'] = np.cos(X_copy.hr*(2.*np.pi/24))
        X_copy['mnth_sin'] = np.sin((X_copy.mnth-1)*(2.*np.pi/12))
        X_copy['mnth_cos'] = np.cos((X_copy.mnth-1)*(2.*np.pi/12))
        X_copy.drop(['mnth', 'hr'], axis=1, inplace=True)


class AddRestrictionLevel(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Define date ranges and corresponding restriction levels
        date_ranges = [
            ('16/10/2020', '17/10/2020'),
            ('17/10/2020', '28/11/2020'),
            ('28/11/2020', '15/12/2020'),
            ('15/12/2020', '16/01/2021'),
            ('16/01/2021', '19/03/2021'),
            ('19/03/2021', '03/05/2021'),
            ('03/05/2021', '09/06/2021'),
            ('09/06/2021', '20/06/2021'),
            ('20/06/2021', '30/06/2021')
        ]

        restriction_levels = [3, 5, 4, 2, 1, 5, 4, 2, 1, 0]

        # Convert date strings to datetime objects
        date_ranges = [(pd.to_datetime(start, dayfirst=True), pd.to_datetime(
            end, dayfirst=True)) for start, end in date_ranges]

        # Add restriction_level column based on date ranges
        X_copy['restriction_level'] = 0  # Default value
        for level, (start_date, end_date) in zip(restriction_levels, date_ranges):
            mask = (X_copy['date'] >= start_date) & (X_copy['date'] < end_date)
            X_copy.loc[mask, 'restriction_level'] = level

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
        # X_copy.drop(columns='date', inplace=True)
        return X_copy


class EncodeCounter(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy = pd.get_dummies(
            X_copy, columns=['counter_id'], dtype=int, drop_first=True)
        return X_copy


class MergeWeatherCovid(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = pd.read_csv(os.path.join(
            "..", "Datasets", "weather_data_cleaned.csv"))
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
        mult_df = pd.read_csv(os.path.join(
            "..", "Datasets", "multimodal_data.csv"))
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
    ("AddRestrictionLevel", AddRestrictionLevel()),
    ("HolidaysFR", HolidaysFR()),
    ("EncodeCounter", EncodeCounter()),
    ("MergeWeatherCovid", MergeWeatherCovid()),
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
csv_file_path = '/kaggle/working/submission.csv'

# Write the DataFrame to a CSV file
predictions_df.to_csv(csv_file_path, index=False)
