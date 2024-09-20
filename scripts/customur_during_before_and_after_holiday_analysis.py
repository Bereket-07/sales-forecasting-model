import pandas as pd
import os 
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.stats import f_oneway

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
    logger.info("Create a new column to mark the time periods: before, during, after holidays")
    try:
        # Initialize the 'Holidayperiod' column
        train['Holidayperiod'] = 'Before Holiday'
        
        # Mark 'During Holiday' periods
        train.loc[((train['StateHoliday'] != '0') & (train['StateHoliday'] != 0)) | (train['SchoolHoliday'] == 1), 'Holidayperiod'] = 'During Holiday'
        
        # Correctly assign 'After Holiday' periods
        train['Holidayperiod_shift'] = train['Holidayperiod'].shift(1)
        
        # Identify 'During Holiday' periods to extend for 'After Holiday'
        train['Holidayperiod'] = np.where(
            (train['Holidayperiod'] == 'Before Holiday') & 
            (train['Holidayperiod_shift'] == 'During Holiday'), 
            'After Holiday', 
            train['Holidayperiod']
        )
        
        # Ensure no 'After Holiday' periods are marked where they shouldn't be
        train.loc[train['Holidayperiod'] == 'Before Holiday', 'Holidayperiod'] = 'Before Holiday'
        
        # Clean up temporary columns
        train.drop(['Holidayperiod_shift'], axis=1, inplace=True)
        # aggergate sales based on the holiday period
        sales_behavior = train.groupby("Holidayperiod")['Sales'].sum().reset_index()
        # plot sales beahviot 
        plt.figure(figsize=(10,6))
        sns.barplot(x='Holidayperiod' , y='Sales' , data=sales_behavior , palette='coolwarm')
        plt.title('Sales Behavior Before, During, and After Holidays')
        plt.xlabel('Holiday Period')
        plt.ylabel('Total Sales')
        plt.show()
    except Exception as e:
        logger.error(f"error while creating the holidayperiod column : {e}")
def ab_hypotesesi(train):
    logger.info("performing the ab hypotesis test over the before during and after holiday sales distribution")
    try:    
        # extract sales for each period
        sales_before = train[train['Holidayperiod'] == 'Before Holiday']['Sales']
        sales_during = train[train['Holidayperiod'] == 'During Holiday']['Sales']
        sales_after = train[train['Holidayperiod'] == 'After Holiday']['Sales']

        f_stat , p_value = f_oneway(sales_before,sales_during,sales_after)
        print(f"ANOVA Test Statistic: {f_stat}")
        print(f"P-Value: {p_value}")

        # Interpretation of P-Value
        if p_value < 0.05:
            print("There is a significant difference in sales behavior across the holiday periods. which in before , During and after Holiday ")
        else:
            print("There is no significant difference in sales behavior across the holiday periods. which in before , During and after Holiday ")
    except Exception as e:
        logger.error(f"error while performing ab hypotesesi test")
def seasonal_holiday_behaviour(train):
    ester_sales_behaviour = train[(train['Holidayperiod'] == 'During Holiday') & (train['StateHoliday'] == 'b') ]
    x_mass_sales_behaviour = train[(train['Holidayperiod'] == 'During Holiday') & (train['StateHoliday'] == 'c') ]
    public_sales_behaviour = train[(train['Holidayperiod'] == 'During Holiday') & (train['StateHoliday'] == 'a') ] 
    # Aggregate sales data
    sales_summary = {
        'Easter': ester_sales_behaviour['Sales'].sum(),
        'Christmas': x_mass_sales_behaviour['Sales'].sum(),
        'Public Holiday': public_sales_behaviour['Sales'].sum()
    }    
    sales_df = pd.DataFrame(sales_summary.items(), columns=['Holiday', 'Total Sales'])

    plt.figure(figsize=(8, 5))
    sns.barplot(x='Holiday', y='Total Sales', data=sales_df, palette='coolwarm')
    plt.title('Total Sales During Seasonal Holidays')
    plt.xlabel('Holiday')
    plt.ylabel('Total Sales')
    plt.show()   

    f_stat , p_value = f_oneway(ester_sales_behaviour['Sales'],x_mass_sales_behaviour['Sales'],public_sales_behaviour['Sales'])
    print(f"ANOVA Test Statistic: {f_stat}")
    print(f"P-Value: {p_value}")

    # Interpretation of P-Value
    if p_value < 0.05:
        print("There is a significant difference in sales behavior across the ester christmas and public holidays ")
    else:
        print("There is no significant difference in sales behavior across the ester christmas and public holidays")
def corelation_sales_vs_no_of_customers(train):
    logger.info("performing the correlation between the number of customers and sales")
    try:
        corelation = train['Customers'].corr(train['Sales'])
        plt.scatter(train['Customers'], train['Sales'])
        plt.title("Corelation between the sales and number of customers")
        plt.xlabel("Number of customers")
        plt.ylabel("Sales")
        plt.grid(True)
        plt.show()
    except Exception as e:
        logger.error(f"Error while performing correlation analysis : {e}")
def impact_of_promotion_on_number_of_customers_and_sales(train):
    logger.info("The impact of promotion on the number of customers and sales")
    try:
        avg_customer_promo = train[train['Promo'] == 1]['Customers'].mean()
        avg_customer_no_promo = train[train['Promo'] == 0]['Customers'].mean()

        print(f"Average customers during promotion days: {avg_customer_promo}")
        print(f"Average customers during non-promotion days: {avg_customer_no_promo}")

        avg_sale_promo = train[train['Promo'] == 1]['Sales'].mean()
        avg_sale_no_promo = train[train['Promo'] == 0]['Sales'].mean()

        print(f"Average sales during promotion days: {avg_sale_promo}")
        print(f"Average sales during non-promotion days: {avg_sale_no_promo}")

        avg_customers = [avg_customer_promo, avg_customer_no_promo]
        avg_sales = [avg_sale_promo, avg_sale_no_promo]
        visualize_promo_effect(avg_customers,avg_sales)
        promo_impact_on_existing_customers(train)
    except Exception as e:
        logger.error(f"Error while tring to analyze the effect of promo on the number of customers and sales")
def visualize_promo_effect(avg_customers,avg_sales):
    logger.info("Visualizing the impact of promotion on number of customers and sales")
    try:
        plt.figure(figsize=(10, 5))

        # Bar chart for customers
        plt.subplot(1, 2, 1)
        plt.bar(['Promo', 'No Promo'], avg_customers, color=['blue', 'green'])
        plt.title('Average Number of Customers')
        plt.ylabel('Number of Customers')

        # Bar chart for sales
        plt.subplot(1, 2, 2)
        plt.bar(['Promo', 'No Promo'], avg_sales, color=['blue', 'green'])
        plt.title('Average Sales ($)')
        plt.ylabel('Sales ($)')

        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.error("Error while plotting the promo impact: {e}")
def promo_impact_on_existing_customers(train):
    logger.info("Impact of promotion on existing customers")
    try:
        sales_per_customer_promotion = train[train['Promo'] == 1]['Sales'].mean() / train[train['Promo'] == 1]['Customers'].mean()
        sales_per_customer_no_promotion = train[train['Promo'] == 0]['Sales'].mean() / train[train['Promo'] == 0]['Customers'].mean()
        print(f"Sales per customer during promotion days: {sales_per_customer_promotion}")
        print(f"Sales per customer during non-promotion days: {sales_per_customer_no_promotion}")

        avg_sales_on_existing_customers_on_promo_and_no_promo = [sales_per_customer_promotion,sales_per_customer_no_promotion]
        plt.figure(figsize=(10, 5))

        # Bar chart for customers
        plt.bar(['Promo', 'No Promo'], avg_sales_on_existing_customers_on_promo_and_no_promo, color=['blue', 'green'])
        plt.title('Average sales per customer')
        plt.ylabel('Number of sales')
    except Exception as e:
        logger.error(f"Error while performing impact of promo on existing customers: {e}")









