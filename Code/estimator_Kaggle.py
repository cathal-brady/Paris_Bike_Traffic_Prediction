# Imports
from datetime import datetime as dt
import os
import joblib
import numpy as np
import pandas as pd
import holidays
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline


# Classes Required
class ColumnSelector(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy.reset_index(drop=True)
        return X_copy[['counter_id', 'date', 'site_id', 'log_bike_count']]


class DateFormatter(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['date'] = pd.to_datetime(X_copy['date'])
        X_copy['month'] = X_copy['date'].dt.month
        X_copy['week'] = X_copy['date'].dt.isocalendar().week
        X_copy['weekday'] = (X_copy['date'].dt.dayofweek + 1)
        X_copy['hr'] = X_copy['date'].dt.hour
        X_copy['hr_sin'] = np.sin(X_copy.hr*(2.*np.pi/24))
        X_copy['hr_cos'] = np.cos(X_copy.hr*(2.*np.pi/24))
        X_copy.drop('week', axis=1, inplace=True)
        X_copy = X_copy.sort_values('date')
        X_copy['track_id'] = X_copy.index
        return X_copy


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
        def school_holiday(date, school_hols): return 1 if any(
            start <= date <= end for start, end in school_hols) else 0

        Autumn_20 = (dt(2020, 10, 18), dt(2023, 11, 1))
        Xmas_20 = (dt(2020, 12, 20), dt(2021, 1, 3))
        Winter_21 = (dt(2021, 2, 14), dt(2021, 2, 28))
        Spring_21 = (dt(2021, 4, 11), dt(2021, 4, 25))
        Summer_21 = (dt(2021, 7, 7), dt(2021, 9, 7))

        school_hols = [Autumn_20, Xmas_20, Winter_21, Spring_21, Summer_21]

        X_copy = X.copy()
        X_copy['is_Holiday'] = X_copy['date'].apply(is_holiday)
        X_copy['is_Weekend'] = X_copy['weekday'].apply(is_weekend)
        X_copy['is_School_Holiday'] = X_copy['date'].apply(
            lambda date: school_holiday(date, school_hols))
        # X_copy.drop(columns='date', inplace=True)
        return X_copy


class RushHour(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['rush_hour'] = ((X_copy['weekday'] <= 5) &
                               ((X_copy['hr'].between(7, 9)) | (X_copy['hr'].between(17, 20))) &
                               (X_copy['is_Holiday'] == 0)).astype(int)
        X_copy.drop('hr', axis=1, inplace=True)  # no longer needed
        return X_copy


class MergeWeatherCovid(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        weather_name = '/kaggle/input/weather_data_cleaned.csv'
        data = pd.read_csv(weather_name)
        data['date'] = pd.to_datetime(data['date']).astype('datetime64[us]')
        merged_data = pd.merge_asof(X, data, on='date')
        return merged_data


class SplitBySite(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        sub_dataframes = []

        unique_site_ids = X_copy['site_id'].unique()

        for site_id in unique_site_ids:
            sub_df = X_copy[X_copy['site_id'] == site_id].copy()
            sub_dataframes.append(sub_df)

        return sub_dataframes


class MergeMultiModalSites(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        encoded_dataframes = []

        # Import Multimodal Data
        multimodal_name = '/kaggle/input/multimodal_dummy_clean.csv'
        mult_df = pd.read_csv(multimodal_name)
        mult_df['date'] = pd.to_datetime(
            mult_df['date']).astype('datetime64[us]')

        unique_values_dict = dict(
            zip(mult_df['nearest site'].unique(), mult_df['minimum distance'].unique()))

        for i in range(len(X_copy)):
            if unique_values_dict[X_copy[i]['site_id'].iloc[0]] > 1:
                temp = mult_df.drop(
                    columns=['site_id', 'latitude', 'longitude', 'minimum distance', 'nearest site'])
                result_df = temp.groupby('date').mean().reset_index()
                X_copy[i] = pd.merge_asof(X_copy[i], result_df, on='date')
            else:
                mask = mult_df['nearest site'] == X_copy[i]['site_id'].iloc[0]
                temp = mult_df[mask].copy()
                temp.drop(columns=['site_id', 'latitude', 'longitude',
                          'minimum distance', 'nearest site'], inplace=True)
                X_copy[i] = pd.merge_asof(X_copy[i], temp, on='date')
                numeric_cols = X_copy[i].select_dtypes(
                    include=['number']).columns
                X_copy[i][numeric_cols] = X_copy[i][numeric_cols].fillna(
                    X_copy[i][numeric_cols].mean())
            encoded_dataframes.append(X_copy[i])
        return encoded_dataframes


class EncodeCounter(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        encoded_dataframes = []

        for i in range(len(X_copy)):
            X_copy[i]['counter_id'] = X_copy[i]['counter_id'].astype('object')
            encoded_df = pd.get_dummies(
                X_copy[i], columns=['counter_id'], dtype=int, drop_first=True)
            encoded_dataframes.append(encoded_df)
        return encoded_dataframes


class DropOutliers(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=3):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        cleaned_dataframes = []
        for i in range(len(X_copy)):
            mean_value = X_copy[i]['log_bike_count'].mean()
            std_dev = X_copy[i]['log_bike_count'].std()
            # #Define a threshold
            threshold = 3
            # # Identify outliers
            outliers = (X_copy[i]['log_bike_count'] -
                        mean_value).abs() > threshold * std_dev
            # # Drop outliers
            mask = outliers == False
            cleaned_dataframes.append(X_copy[i][mask])
        return cleaned_dataframes


class ModelGen(BaseEstimator, TransformerMixin):
    def __init__(self, model=CatBoostRegressor(loss_function='RMSE', depth=10, iterations=200, learning_rate=0.1, verbose=False),
                 random_state=42, save_path='/kaggle/working/'):

        # Checking for the existence of the target path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.model = model
        self.random_state = random_state
        self.best_models = []  # List to store the best model from each fold
        self.save_path = save_path

    def fit(self, X, y=None):
        self.best_models = []  # Clear previous best models
        kf = KFold(n_splits=10, shuffle=True, random_state=self.random_state)

        for idx, df in enumerate(X):
            # Extract DataFrames from the list
            X_train = df.drop(
                columns=['log_bike_count', 'site_id', 'date', 'track_id'], axis=1)
            y_train = df['log_bike_count']

            # Fit the model on the current fold
            self.model.fit(X_train, y_train)

            # Save the trained model to the desktop
            model_filename = f"site_ID_{df['site_id'].iloc[0]}_model_catboost.joblib"
            model_path = os.path.join(self.save_path, model_filename)
            joblib.dump(self.model, model_path)

            # Store the model for reference
            self.best_models.append(self.model)

        return self

    def predict(self, X):
        predictions = []
        for idx, df in enumerate(X):
            site_id_value = df['site_id'].iloc[0]
            model_path = os.path.join(
                self.save_path, f"site_ID_{site_id_value}_model_catboost.joblib")

            if os.path.exists(model_path):
                model = joblib.load(model_path)
                df['prediction'] = model.predict(
                    df.drop(columns=['log_bike_count', 'site_id', 'date', 'track_id'], axis=1))
                predictions.append(df)
            else:
                print(f"Model file not found for site_id {site_id_value}")

        return pd.concat(predictions, ignore_index=True)


def add_prediction_column(X):
    for df in X:
        # Extract the first value of the column 'site_id'
        site_id_value = df['site_id'].iloc[0]
        # df.drop(columns='log_bike_count', inplace=True)

        # Construct the path for the model file
        model_filename = f"site_ID_{site_id_value}_model_catboost.joblib"
        model_path = os.path.join('/kaggle/working/', model_filename)

        # Check if the model file exists
        if os.path.exists(model_path):
            # Load the model
            model = joblib.load(model_Fpath)
            # Add a column 'prediction' to the DataFrame with model predictions
            df['prediction'] = model.predict(
                df.drop(columns=['log_bike_count', 'site_id', 'date', 'track_id'], axis=1))
            # df.drop('log_bike_count', inplace=True)
        else:
            print(f"Model file not found for site_id {site_id_value}")
    out = pd.concat(X, ignore_index=True)
    out.drop(columns='log_bike_count', inplace=True)

    return out


# Pipelines
preprocessor = Pipeline([
    ('column_selector', ColumnSelector()),
    ('date_formatter', DateFormatter()),
    ('add_restriction_level', AddRestrictionLevel()),
    ('holidays_fr', HolidaysFR()),
    ('add_rush_hours', RushHour()),
    ('MergeWeatherCovid', MergeWeatherCovid()),
])

spliter = Pipeline([
    ('SplitBySite', SplitBySite()),
    ("MergeMultiModalSites", MergeMultiModalSites()),
    ('EncodeCounter', EncodeCounter()),
    ('DropOutliers', DropOutliers())
])

# Combined Pipeline
combined_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('spliter', spliter)
])

# Training and Generating Models:
train_name = '/kaggle/input/mdsb-2023/train.parquet'
df = pd.read_parquet(train_name)
train_data = combined_pipeline.fit_transform(df)
ModelGen().fit(train_data)

# Predicting
test_name = '/kaggle/input/mdsb-2023/final_test.parquet'
df_test = pd.read_parquet(test_name)
df_test['log_bike_count'] = 0

# Run Preprocessing Pipeline
df_test_preprocessed = combined_pipeline.fit_transform(df_test)
df_test_preprocessed = add_prediction_column(df_test_preprocessed)
df_sorted = df_test_preprocessed.sort_values(by='track_id')
df_sorted.rename(
    columns={'track_id': 'Id', 'prediction': 'log_bike_count'}, inplace=True)
# Extract the selected columns
selected_columns = ['Id', 'log_bike_count']
result_df = df_sorted[selected_columns]

# Specify the file path
csv_file_path = '/kaggle/working/submission.csv'

# Write the DataFrame to a CSV file
result_df.to_csv(csv_file_path, index=False)
