import pandas as pd
import logging
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np



logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Format of the log messages
)
# Create a logger object
logger = logging.getLogger(__name__)

# define the path to the Logs directory one level up
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','logs')

# create the logs directory if it doesn't exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
# define file paths
log_file_info = os.path.join(log_dir, 'info.log')
log_file_error = os.path.join(log_dir, 'error.log')

# Create handlers
info_handler = logging.FileHandler(log_file_info)
info_handler.setLevel(logging.INFO)

error_handler = logging.FileHandler(log_file_error)
error_handler.setLevel(logging.ERROR)

# Create a formatter and set it for the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
info_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)

# Create a logger and set its level
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Capture all info and above
logger.addHandler(info_handler)
logger.addHandler(error_handler)


def load_data(path):
    logger.info("loadding the data set")
    try:
        train = pd.read_csv(path)
        logger.info("the data sucessfully uploaded")
        return train
    except Exception  as e:
        logger.error(f"error happend : {e}")
# step one convert catagorical column to numerical column
def catagorical_to_numerical(data,column):
    logger.info("convert catagorical column to numerical column")
    try:
        if column == 'StateHoliday':
            # Step 1: Replace string '0' with integer 0
            data['StateHoliday'] = data['StateHoliday'].replace(0,'0')
        data[column] = LabelEncoder().fit_transform(data[column])
    except Exception as e:
        logger.error(f"error happened : {e}")
def extract_new_features(train):
    logger.info("extracting new featuires")
    try:
        logger.info("date changing to datetime")
        train['Date'] = pd.to_datetime(train['Date'])
        logger.info("date changinged to datetime succesfully")

        logger.info("extrating weekend")
        train['Weekend'] = train['DayOfWeek'].apply(lambda x: 1 if x >= 6 else 0)
        logger.info("extrating weekend succesfull")
        
        logger.info("month postioning")
        # MonthPosition (Start = 1-10, Mid = 11-20, End = 21+)
        train['MonthPosition'] = train['Date'].dt.day.apply(lambda x: 'Start' if x <= 10 else ('Mid' if x <= 20 else 'End'))
        logger.info("month postioning finished")

        logger.info("calcuating the days to next holiday and after holidat")

        logger.info("days are calculated sucsesfully")
        holiday_dates(train)
        logger.info("calculating sales growth")

        # Step 4: Calculate sales growth rate (percentage change of sales over time)
        train['SalesGrowthRate'] = train['Sales'].pct_change()
        train['SalesGrowthRate'].fillna(0)

        logger.info("calculating sales growth complete")
    except Exception as e:
        logger.error(f"error while {e}")
def holiday_dates(train):
    logger.info("finding the holiday days in the data set")
    try:
        all_holiday_dates= train[(train['StateHoliday'] != 0)]['Date'].unique()
        logger.info("all holidays found")
        holidays = pd.to_datetime(all_holiday_dates)  
        # Apply the function to calculate DaysToNextHoliday
        train['DaysToNextHoliday'] = train['Date'].apply(lambda x: days_to_next_holiday(x, holidays))

        # Calculate number of days since the most recent holiday
        train['DaysAfterHoliday'] = train['Date'].apply(lambda x: (x - holidays[holidays <= x].max()).days if any(holidays <= x) else np.nan)

        return train
    except Exception as e:
        logger.error(f"error occured : {e} ")
# Calculate number of days until the next holiday
def days_to_next_holiday(date, holidays):
    holidays = pd.Series(holidays)
    # Filter holidays for the same year or later
    future_holidays = holidays[holidays >= date]
    
    # If there's a future holiday this year, calculate the days
    if len(future_holidays) > 0:
        return (future_holidays.min() - date).days
    else:
        #  # If no future holiday exists, find the earliest holiday from the next year
        # next_year = date.year + 1
        # next_year_holidays = holidays[holidays.dt.year == next_year]
        
        # if len(next_year_holidays) > 0:
        #     next_year_first_holiday = next_year_holidays.min()
        #     return (next_year_first_holiday - date).days
        # else:
        return -1  # Handle the case where there are no holidays even in the next year

