import pandas as pd
import logging 
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom transformer for categorical to numerical conversion
class CategoricalToNumerical(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if 'Id' in X.columns:
            X.drop(columns=['Id'], inplace=True)
        logger.info("Converting categorical columns to numerical columns")
        X.dropna(inplace=True) 
        for column in X.select_dtypes(include=[object]).columns:
            if column == 'StateHoliday':
                if 0 in X[column].values:
                    X[column] = X[column].replace(0, '0')
            X[column] = LabelEncoder().fit_transform(X[column])
        return X

# Custom transformer for feature extraction
class FeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.info("Extracting new features")

        # Ensure the 'Date' column is properly converted to datetime before transformations
        if 'Date' in X.columns:
            X['Date'] = pd.to_datetime(X['Date'], errors='coerce')  # Handle invalid dates

            # Calculate whether it's the weekend
            X['Weekend'] = X['DayOfWeek'].apply(lambda x: 1 if x >= 6 else 0)

            # Create a new feature 'MonthPosition'
            X['MonthPosition'] = X['Date'].dt.day.apply(lambda x: 'Start' if x <= 10 else ('Mid' if x <= 20 else 'End'))

            # Convert 'MonthPosition' into numerical values (label encoding)
            X['MonthPosition'] = LabelEncoder().fit_transform(X['MonthPosition'])

            # Convert 'Date' to the number of days since the start of the year for numerical purposes
            X['DaysSinceStartOfYear'] = X['Date'].apply(lambda x: (x - pd.Timestamp(f'{x.year}-01-01')).days if pd.notnull(x) else np.nan)

            # Extract all holiday dates
            all_holiday_dates = X[(X['StateHoliday'] != '0') & (X['StateHoliday'].notnull())]['Date'].unique()
            holidays = pd.to_datetime(all_holiday_dates)

            # Calculate days to the next holiday
            X['DaysToNextHoliday'] = X['Date'].apply(lambda x: self.days_to_next_holiday(x, holidays))

            # Calculate days after the most recent holiday
            X['DaysAfterHoliday'] = X['Date'].apply(
                lambda x: (x - holidays[holidays <= x].max()).days if len(holidays[holidays <= x]) > 0 else np.nan
            )

            # Drop the 'Date' column after extraction if it's not needed anymore
            X = X.drop(columns=['Date'])

        return X

    def days_to_next_holiday(self, date, holidays):
        holidays = pd.Series(holidays)
        future_holidays = holidays[holidays >= date]
        if len(future_holidays) > 0:
            return (future_holidays.min() - date).days
        next_year = date.year + 1
        next_year_holidays = holidays[holidays.dt.year == next_year]
        if len(next_year_holidays) > 0:
            return (next_year_holidays.min() - date).days
        return -1

# Load data function
def load_data(path):
    logger.info("Loading the dataset")
    try:
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        return None

def main(path):
    try:
        # Load dataset
        df = load_data(path)
        logger.info("Separating the target variable")

        # Drop unnecessary columns
        df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')  # Ignore if not present
        if 'Id' in df.columns:
            df.drop(columns=['Id'], inplace=True)

        # Train-test split
        X = df.drop(columns=['Sales', 'Customers'])
        y = df['Sales']
        logger.info("Target variable split complete")

        logger.info("Splitting the data into train and test sets")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info("Data split complete")

        logger.info("Creating the sklearn pipeline")
        # Create the pipeline
        pipeline = Pipeline([
            ('categorical_to_numerical', CategoricalToNumerical()),
            ('feature_extractor', FeatureExtractor()),
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(random_state=42))
        ])
        logger.info("Sklearn pipeline created")

        logger.info("Fitting the model")
        pipeline.fit(X_train, y_train)
        logger.info("Model fitting completed")

        # Predict on test data
        y_pred = pipeline.predict(X_test)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Log and print metrics
        logger.info(f"Mean Squared Error (MSE): {mse}")
        logger.info(f"Mean Absolute Error (MAE): {mae}")
        logger.info(f"R-squared (R²): {r2}")

        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"R-squared (R²): {r2}")

        return pipeline
    except Exception as e:
        logger.error(f"Error: {e}")

