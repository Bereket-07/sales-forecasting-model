import pandas as pd
import os 
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.stats import f_oneway
from scipy.stats import chisquare

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
def impact_of_promotion_by_store(train):
    logger.info("Anlayzing the better promotion on specific stores")
    try:
        grouped = train.groupby(['Store' , 'Promo']).agg({
            'Customers':'mean',
            'Sales':'mean'
        }).reset_index()
        return grouped
    except Exception as e:
        logger.error(f"error while aggregating: {e}")
def calculat_customer_and_sales_uplift(grouped):
    logger.info("calcuclating customer and sales uplift")
    try:
        # Create a pivot table to compare promo vs. non-promo days
        pivot_table = grouped.pivot(index='Store', columns='Promo', values=['Customers', 'Sales'])

        # Calculate the uplift in customers and sales 
        pivot_table['Customer_Uplift'] = pivot_table['Customers'][1] - pivot_table['Customers'][0]
        pivot_table['Sales_Uplift'] = pivot_table['Sales'][1] - pivot_table['Sales'][0]

        visualize_uplift(pivot_table)
        
        return pivot_table
    except Exception as e:
        logger.error(f"Error while calculating uplift : {e}")
def visualize_uplift(pivot_table):
    logger.info("visualizing the uplif")
    try:
        # Plot customer uplift by store
        plt.bar(pivot_table.index, pivot_table['Customer_Uplift'], color='blue')
        plt.title('Customer Uplift by Store during Promotions')
        plt.xlabel('Store ID')
        plt.ylabel('Customer Uplift')
        plt.show()

        # Plot sales uplift by store
        plt.bar(pivot_table.index, pivot_table['Sales_Uplift'], color='green')
        plt.title('Sales Uplift by Store during Promotions')
        plt.xlabel('Store ID')
        plt.ylabel('Sales Uplift ($)')
        plt.show()
    except Exception as e:
        logger.error(f"Error while trying to visualize uplif: {e}")
def impact_of_opening_all_week_days(train):
    logger.info("Showing the influence of opening all week days on weekends")
    try:
        # Step 1: Identify stores open on all weekdays
        weekday_stores = train[(train['DayOfWeek'] < 6) & (train['Open'] == 1)]
        stores_open_weekdays = weekday_stores['Store'].unique()

        # Step 2: Count unique open weekdays for each store
        store_weekday_count = train[train['DayOfWeek'] < 6].groupby('Store')['Open'].sum().reset_index()
        store_weekday_count = store_weekday_count.rename(columns={'Open': 'OpenDays'})

        # Step 3: Identify stores that are not open all weekdays
        stores_not_open_weekdays = store_weekday_count[store_weekday_count['OpenDays'] < 6]['Store'].unique()

        # Step 3: Calculate weekend sales for both groups
        weekend_sales_open = train[(train['Store'].isin(stores_open_weekdays)) & (train['DayOfWeek'] >= 6)]
        weekend_sales_summary_open = weekend_sales_open.groupby('Store')['Sales'].sum().reset_index()

        weekend_sales_not_open = train[(train['Store'].isin(stores_not_open_weekdays)) & (train['DayOfWeek'] >= 6)]
        weekend_sales_summary_not_open = weekend_sales_not_open.groupby('Store')['Sales'].sum().reset_index()

        # Handle empty dataframe case
        if weekend_sales_summary_not_open.empty:
            logger.info("No stores found that close during any weekdays.")
            print("All stores are open on all weekdays. No comparison data for stores closed on weekdays.")
            combined_sales = weekend_sales_summary_open.assign(Group='Open All Weekdays')
        else:
            combined_sales = pd.concat([
                weekend_sales_summary_open.assign(Group='Open All Weekdays'),
                weekend_sales_summary_not_open.assign(Group='Not Open All Weekdays')
            ])

        # Print summaries
        print("  open all weekdays weekend sales \n")
        print(weekend_sales_open)
        print(f"  open all weekdays weekend sales SUMMARY \n ") 
        print(weekend_sales_summary_open)

        print(f" not open all weekdays weekend sales \n ")
        print(weekend_sales_not_open)
        print(f" not open all weekdays weekend sales SUMMARY \n")
        print(weekend_sales_summary_not_open)

        # Step 4: Visualization with grouped bar chart
        plt.figure(figsize=(10, 6))
        bar_width = 0.35
        index = range(len(combined_sales))

        # Plot bars for each group
        for i, group in enumerate(combined_sales['Group'].unique()):
            subset = combined_sales[combined_sales['Group'] == group]
            plt.bar([x + i * bar_width for x in index[:len(subset)]], subset['Sales'], width=bar_width, label=group)

        # Formatting the plot
        plt.title('Weekend Sales Comparison: Stores Open vs Not Open on All Weekdays')
        plt.xlabel('Store')
        plt.ylabel('Total Weekend Sales ($)')
        plt.xticks(index, combined_sales['Store'].unique(), rotation=0)
        plt.legend()
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logger.error(f" Error while showing influence of opening all weekdays : {e}")
def open_all_week_days(train):
    # Step 1: Filter for weekdays and only include stores that are open
    weekday_stores = train[(train['DayOfWeek'] < 6) & (train['Open'] == 1)]

    # Step 2: Count unique weekdays for each store
    weekday_count = weekday_stores.groupby('Store')['DayOfWeek'].unique().reset_index()
    
    return weekday_count
    # Step 3: Identify stores open on all weekdays (which is 5 unique weekdays)
    stores_open_all_weekdays = weekday_count[weekday_count['DayOfWeek'] == 5]['Store'].unique()

    # Display results
    print("Stores open on all weekdays:")
    print(stores_open_all_weekdays)
def analyze_weekday_closures_and_weekend_sales(train):
    try:
        df = train.copy()
        # Step 1: Create week identifiers (Year and Week)
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.isocalendar().year
        df['Week'] = df['Date'].dt.isocalendar().week
        
        # Step 2: Count open days during weekdays (Monday to Friday: DayOfWeek < 6)
        weekday_open_counts = df[(df['DayOfWeek'] < 6) & (df['Open'] == 1)] \
                              .groupby(['Store', 'Year', 'Week'])['Open'] \
                              .sum().reset_index()
        weekday_open_counts = weekday_open_counts.rename(columns={'Open': 'OpenWeekdays'})

        # Step 3: Identify stores that close at least one weekday (OpenWeekdays < 5)
        stores_close_at_least_one_weekday = weekday_open_counts[weekday_open_counts['OpenWeekdays'] < 5]['Store'].unique()

        # Step 4: Identify stores that are always open on all weekdays (OpenWeekdays == 5)
        stores_open_all_weekdays = weekday_open_counts[weekday_open_counts['OpenWeekdays'] == 5]['Store'].unique()

        # Step 5: Calculate weekend sales for both groups (Saturday and Sunday: DayOfWeek >= 6)
        weekend_sales_open = df[(df['Store'].isin(stores_open_all_weekdays)) & (df['DayOfWeek'] >= 6)]
        weekend_sales_summary_open = weekend_sales_open.groupby('Store')['Sales'].sum().reset_index()

        weekend_sales_closed = df[(df['Store'].isin(stores_close_at_least_one_weekday)) & (df['DayOfWeek'] >= 6)]
        weekend_sales_summary_closed = weekend_sales_closed.groupby('Store')['Sales'].sum().reset_index()

        print(weekend_sales_summary_closed)

        # Step 6: Combine results for visualization
        combined_sales = pd.concat([
            weekend_sales_summary_open.assign(Group='Open All Weekdays'),
            weekend_sales_summary_closed.assign(Group='Closed at least 1 Weekday')
        ])

        # Step 7: Visualization
        plt.figure(figsize=(12, 8))
        sns.barplot(data=combined_sales, x='Store', y='Sales', hue='Group', palette='muted')

        # Formatting the plot
        plt.title('Weekend Sales Comparison: Stores Open All Weekdays vs Closed at Least 1 Weekday')
        plt.xlabel('Store')
        plt.ylabel('Total Weekend Sales ($)')
        plt.xticks(rotation=90)
        plt.legend()
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()


    except Exception as e:
        print(f"Error occurred: {e}")





