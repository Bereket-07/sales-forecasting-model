import logging
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose 
from statsmodels.tsa.stattools import acf , pacf


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


def load_data(file_path):
    logger.info('data loading started')
    try:
        data = []
        for p in file_path:
            n = pd.read_csv(p)
            logger.debug(f"the  {n} data is sucessfully loaded")
            data.append(n)
        return data
    except Exception as e:
        logger.error(f"Error loading the data {e}")
        return None
def set_date_index(data):
    try:
        logger.info('changing the date column to standaed date time format')
        data['Date'] = pd.to_datetime(data['Date'])
        # set data as index 
        logger.info("making the date column the index of the data frame")
        data.set_index('Date' , inplace=True)
        return data
    except Exception as e:
        logger.error(f' error while making date and index {e}')

def plot_weekly_sales(data):
    logger.info("Plotting weekly sales ...")
    try:
        weekly_sales =data['Sales'].resample('W').sum()
        plt.figure(figsize=(15,7))
        plt.plot(weekly_sales.index , weekly_sales)
        plt.title("Weekly sales over Time")
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.show()
    except Exception as e:
        logger.error(f"error in ploting weekly sales: {e}")
def plot_monthly_sales(data):
    logger.info("plotting the monthly sales ...")
    try:
        monthly_sales = data['Sales'].resample('M').sum()
        plt.figure(figsize=(15,7))
        plt.plot(monthly_sales.index , monthly_sales)
        plt.title("monthly sales over Time")
        plt.xlabel("Date")
        plt.ylabel("Sales")
        plt.show()
    except Exception as e:
        logger.error("error while trying to make monthly plot {e}")
def plot_seasonal_decomposition(data):
    logger.info("Performing seasonal decompotation ....")
    try:
        monthly_sales = data['Sales'].resample('M').sum()
        result = seasonal_decompose(monthly_sales,model='additive')
        result.plot()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.error(f"Error during seasonal decomposition : {e}")
def plot_acf_pacf(data):
    logger.info("Plotting ACF and PACF ...")
    try:
        monthly_sales = data['Sales'].resample('M').sum()
        n_lags = len(monthly_sales) // 3
        acf_values = acf(monthly_sales.dropna(),nlags=n_lags)   
        pacf_values = pacf(monthly_sales.dropna() , nlags=n_lags)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        ax1.stem(range(len(acf_values)), acf_values, use_line_collection=True)
        ax1.axhline(y=0, linestyle='--', color='gray')
        ax1.axhline(y=-1.96/np.sqrt(len(monthly_sales)), linestyle='--', color='gray')
        ax1.axhline(y=1.96/np.sqrt(len(monthly_sales)), linestyle='--', color='gray')
        ax1.set_title('Autocorrelation Function')
        ax1.set_xlabel('Lag')
        ax1.set_ylabel('Correlation')

        ax2.stem(range(len(pacf_values)), pacf_values, use_line_collection=True)
        ax2.axhline(y=0, linestyle='--', color='gray')
        ax2.axhline(y=-1.96/np.sqrt(len(monthly_sales)), linestyle='--', color='gray')
        ax2.axhline(y=1.96/np.sqrt(len(monthly_sales)), linestyle='--', color='gray')
        ax2.set_title('Partial Autocorrelation Function')
        ax2.set_xlabel('Lag')
        ax2.set_ylabel('Correlation')

        plt.tight_layout()
        plt.show()    
    except Exception as e:
        logger.error(f"Error in ploting ACF and PACF: {e}")
def plot_rolling_statistics(data):
    logger.info("Plotting rolling statistics ..")
    try:
        monthly_sales = data['Sales'].resample('M').sum()
        rolling_mean = monthly_sales.rolling(window=12).mean()
        rolling_std = monthly_sales.rolling(window=12).std()

        plt.figure(figsize=(15,7))
        plt.plot(monthly_sales.index , monthly_sales , label='Monthly sales')
        plt.plot(rolling_mean.index , rolling_mean , label='12 - month Rolling mean')
        plt.plot(rolling_std.index , rolling_std , label = "12-month Rolling std")
        plt.legend()
        plt.title('Monthly Sales - Rolling Mean & Standard Deviation')
        plt.show()
    except Exception as e:
        logger.error(f"Error in ploting rolling statistics:{e}")
def plot_day_of_week_sales(data):
    logger.info("PLotting average sales by day of week ..") 
    try:
        day_of_week_sales = data.groupby('DayOfWeek')['Sales'].mean()

        plt.figure(figsize=(10,6))
        day_of_week_sales.plot(kind='bar')
        plt.title('Average sales by Day of week')
        plt.xlabel('Day of errk (0=Monday , 6=Sunday)')
        plt.ylabel('Avergae Sales')
        plt.show()
    except Exception as e:
        logger.error(f"Error in plotting day of errk sales: {e}")
def plot__StateHoliday_sales_distribution(data):
    logger.info("Plotting sales distribution: StateHoliday vs Non Holiday ....")
    try:
        plt.figure(figsize=(10,6))
        sns.boxplot(x = 'StateHoliday' , y='Sales' , data=data)
        plt.title('Sales Distribution: StateHoliday vs Non-Holiday')
        plt.xticks([0,1] , ['Non-Holiday' , 'StateHoliday'])
        plt.show()
    except Exception as e:
        logger.error(f"Error in ploting holiday sales distribution : {e}")
def plot__SchoolHoliday_sales_distribution(data):
    logger.info("Plotting sales distribution: SchoolHoliday vs Non Holiday ....")
    try:
        plt.figure(figsize=(10,6))
        sns.boxplot(x = 'SchoolHoliday' , y='Sales' , data=data)
        plt.title('Sales Distribution: Holiday vs Non-Holiday')
        plt.xticks([0,1] , ['Non-Holiday' , 'SchoolHoliday'])
        plt.show()
    except Exception as e:
        logger.error(f"Error in ploting holiday sales distribution : {e}")
def plot_general_holiday_as_or_fun(data):
    logger.info("the distribution of sales between holiday either school or state holiday ...")
    try:
        data['General_holiday'] = data['StateHoliday'] | data['SchoolHoliday']
        plt.figure(figsize=(10,6))
        sns.boxplot(x = 'school holiday + state Holiday' , y='Sales' , data=data)
        plt.title('Sales Distribution: Holiday vs Non-Holiday')
        plt.xticks([0,1] , ['Non-Holiday' , 'either SchoolHoliday or state Holiday'])
        plt.show()
    except Exception as e:
        logger.error(f"Error in ploting holiday or state holiday sales distribution : {e}")
def print_statistics(data):
    logger.ingfo("Prining summary statistics ")
    try:
        print("\n Day of the week sales:")
        print(data.groupby('DayOfWeek')['Sales'].describe())
        print("\n StateHoliday vs Non-Holiday Sales:")
        print(data.groupby("StateHoliday")['Sales'].describe())
        print("\n schoolHoliday vs Non-Holiday Sales:")
        print(data.groupby("SchoolHoliday")['Sales'].describe())
        print("\n either State or School Holiday vs Non-Holiday Sales:")
        print(data.groupby("General_holiday")['Sales'].describe())
    except Exception as e:
        logger.info(f"error while priniting statistics: {e}")
def plot_promo_effect(data):
    logger.info("Plotting promo effect over time .....")
    try:
        monthly_promo_sales = df.groupby([df.index.to_period('M'), 'Promo'])['Sales'].mean().unstack()
        monthly_promo_sales.columns = ['No Promo', 'Promo']

        monthly_promo_sales[['No Promo', 'Promo']].plot(figsize=(15, 7))
        plt.title('Monthly Average Sales: Promo vs No Promo')
        plt.xlabel('Date')
        plt.ylabel('Average Sales')
        plt.legend(['No Promo', 'Promo'])
        plt.show()
    except Exception as e:
        logger.error(f"Error in plotting promo effect: {e}")
def plot_store_type_performance(df):
    logger.info("Plotting store type performance over time...")
    try:
        store_type_sales = df.groupby([df.index.to_period('M'), 'Store_Type'])['Sales'].mean().unstack()
        store_type_sales.plot(figsize=(15, 7))
        plt.title('Monthly Average Sales by Store Type')
        plt.xlabel('Date')
        plt.ylabel('Average Sales')
        plt.legend(title='Store Type')
        plt.show()
    except Exception as e:
        logger.error(f"Error in plotting store type performance: {e}")

def plot_sales_correlation(df):
    logger.info("Plotting correlation between sales and customers...")
    try:
        correlation = df[['Sales', 'Customers']].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Correlation between Sales and Customers')
        plt.show()
    except Exception as e:
        logger.error(f"Error in plotting sales correlation: {e}")
def plot_cumulative_sales(df):
    logger.info("Plotting cumulative sales over time...")
    try:
        df['CumulativeSales'] = df['Sales'].cumsum()
        plt.figure(figsize=(15, 7))
        plt.plot(df.index, df['CumulativeSales'])
        plt.title('Cumulative Sales Over Time')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Sales')
        plt.show()
    except Exception as e:
        logger.error(f"Error in plotting cumulative sales: {e}")
def plot_sales_growth_rate(df):
    logger.info("Plotting daily sales growth rate...")
    try:
        df['SalesGrowthRate'] = df['Sales'].pct_change()
        plt.figure(figsize=(15, 7))
        plt.plot(df.index, df['SalesGrowthRate'])
        plt.title('Daily Sales Growth Rate')
        plt.xlabel('Date')
        plt.ylabel('Growth Rate')
        plt.show()
    except Exception as e:
        logger.error(f"Error in plotting sales growth rate: {e}")
