import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Format of the log messages
)
# Create a logger object
logger = logging.getLogger(__name__)

def load_data(path , name):
    logger.info('data loading started')
    try:
        data = []
        for p, n in zip(path,name):
            n = pd.read_csv(p)
            logger.debug(f"the  {n} data is sucessfully loaded")
            data.append(n)
        return data
    except Exception as e:
        logger.error(f"Error loading the data {e}")

def indivdual_histogram(data , name_of_the_data_set):
    if name_of_the_data_set == 'train':
        data['StateHoliday'] = data['StateHoliday'].astype(str)
        data['SchoolHoliday'] = data['SchoolHoliday'].astype(str)
    # Individual Histograms
    num_columns = len(data.columns)
    num_rows = (num_columns + 2) // 2  # Number of rows needed if 3 columns per row
    plt.figure(figsize=(15, num_rows * 5))

    for i, column in enumerate(data.columns):
        plt.subplot(num_rows, 3, i + 1)
        plt.hist(data[column].dropna(), bins=20, edgecolor='black')
        plt.title(f'Histogram of {column}')
        plt.xlabel('Value')
        plt.ylabel('Frequency')

    plt.suptitle(f'Histograms for {name_of_the_data_set}')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit suptitle
    plt.show()
def see_and_handle_missing_values_and_outliers(data,name_of_the_data_set):
    logger.info("the missing values are going to be processed")
    try:
        missing_values = data.isnull().sum()
        logger.debug(f"the missing values {missing_values}")
        logger.debug(f"here is the outliers boxplot for the data set")
        indivdual_histogram(data,name_of_the_data_set)
    except Exception as e:
        logger.error(f" error while performing to visualize the data cleanes {e}")
def edit_cstate_holiday_column(train):
    logger.info("making changing the train stateholiday columns from this ['0', 'a', 'b', 'c', 0] to this ['0', 'a', 'b', 'c', ]")
    try:
        train.loc[(train['StateHoliday'] == '0'),'StateHoliday'] = 0
        return train
    except Exception as e:
        logger.error(f"error while changing column : {e}")

      




