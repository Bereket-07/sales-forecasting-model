import pandas as pd
import os 
import logging



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


def create_a_holiday_period(train):
    logger.info("Create a new column to mark the time periods: before , during , after holidays")
    train['Holidayperiod'] = 'Before Holiday'
    train.loc[ (train['StateHoliday'] == 1) | (train['SchoolHoliday'] == 1) , 'Holidayperiod'] = 'During Holiday'

    # Assuming that days after holidays are the next two day
    train['Holidayperiod'].shift(1).fillna('After Holiday')

    return train

