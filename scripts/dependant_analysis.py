import pandas as pd
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns

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


def load_and_merge_data(*args):
    logger.info("load and merge data")
    try:
        logger.info("loadding data")
        store = pd.read_csv(args[0])
        train = pd.read_csv(args[1])
        logger.info("data loaded sucessfully")

        # step one merge data
        merged_data = pd.merge(train,store, on='Store' , how='left')
        return merged_data
    except Exception as e:
        logger.error(f" error occured : {e}")
        return None
def assortmnet_analysis(merged_data):
    logger.info("Assortment analysis")
    try:
        assortmnet_sales = merged_data.groupby('Assortment')['Sales'].mean().reset_index()

        # Visualizing the impact of Assortment type on sales
        plt.figure(figsize=(8,5))
        plt.bar(assortmnet_sales['Assortment'],assortmnet_sales['Sales'],color=['Blue','Green','orange'])
        plt.title('Average Sales by Assortment Type')
        plt.xlabel('Assortment Type')
        plt.ylabel('Avergae Type')
        plt.grid('True')
        plt.show()

        logger.info(f"Assortment sales summary: \n{assortmnet_sales}")
        return assortmnet_sales
    except Exception as e:
        logger.error(f"Error while analyzing assortment impact on sales: {e}")
def impote_CompetitionDistance(merged_data):
    logger.info("imputeing the compitition distance with the very big number 100000")
    try:
        large_values = 100000
        merged_data['CompetitionDistance'].fillna(large_values, inplace=True)
        # verify the missing values are imputed 
        missing_values = merged_data['CompetitionDistance'].isnull().sum()
        logger.info(f"the missing value of CompetitionDistance columns : {missing_values}")
        return merged_data
    except Exception as e:
        logger.error(f" error happende : {e}")

def impact_of_compution_distance(merged_data):
    logger.info("Anlayzing the impact of compitetter distance")
    try:
        # Binning CompetitionDistance
        merged_data['CompetitionDistance_bins'] = pd.cut(merged_data['CompetitionDistance'], bins=[0, 500, 1000, 2000, 5000, 10000, float('inf')], 
                                           labels=['<500m', '500-1km', '1-2km', '2-5km', '5-10km', '>10km'])
        # Plot sales vs competition distance bins
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='CompetitionDistance_bins', y='Sales', data=merged_data)
        plt.title('Sales Distribution by Competition Distance')
        plt.xlabel('Competition Distance')
        plt.ylabel('Sales')
        plt.show()

        # plt.figure(figsize=(10, 6))
        # sns.swarmplot(x='CompetitionDistance_bins', y='Sales', data=merged_data)
        # plt.title('Sales Distribution by Competition Distance (Swarm Plot)')
        # plt.xlabel('Competition Distance')
        # plt.ylabel('Sales')
        # plt.show()


        # plt.figure(figsize=(10, 6))
        # sns.regplot(x='CompetitionDistance', y='Sales', data=merged_data, scatter_kws={'alpha':0.3})
        # plt.title('Sales vs. Competition Distance (Scatter with Regression Line)')
        # plt.xlabel('Competition Distance')
        # plt.ylabel('Sales')
        # plt.show()

    except Exception as e:
        logger.error(f"error while : {e}")
def city_center_impact(merged_data):
    logger.info("Analyzing the imapact of compitetor distance when the store is in center city")
    try:
        # Assuming StoreType 'a' represents city centers (adjust based on your dataset knowledge)
        city_center_stores = merged_data[merged_data['StoreType'] == 'a']

        # Compare city center vs non-city center stores
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='CompetitionDistance_bins', y='Sales', data=city_center_stores)
        plt.title('Sales Distribution in City Centers by Competition Distance')
        plt.xlabel('Competition Distance (City Centers)')
        plt.ylabel('Sales')
        plt.show()

    except Exception as e:
        logger.error(f"error happened while: {e}")

def effect_of_opening_and_reopening_of_competitors_impact(*args):
    store = pd.read_csv(args[0])
    train = pd.read_csv(args[1])
    merged_df = pd.merge(train, store, on='Store', how='left')

    print("Total NA in CompetitionDistance:", merged_df['CompetitionDistance'].isna().sum())

    na_competition = merged_df[merged_df['CompetitionDistance'].isna()]
    valid_competition = merged_df[merged_df['CompetitionDistance'].notna()]
    affected_stores = na_competition['Store'].unique()

    print("Affected Stores:", affected_stores)

    stores_with_updates = valid_competition[valid_competition['Store'].isin(affected_stores)]
    print("Stores with valid CompetitionDistance updates:", stores_with_updates['Store'].unique())

    merged_df['Date'] = pd.to_datetime(merged_df['Date'])
    before_update = merged_df[merged_df['Store'].isin(affected_stores) & 
                                (merged_df['CompetitionDistance'].isna())]
    after_update = merged_df[merged_df['Store'].isin(affected_stores) & 
                                (merged_df['CompetitionDistance'].notna())]

    print("Sales Before Update:", before_update[['Store', 'Sales']].head())
    print("Sales After Update:", after_update[['Store', 'Sales']].head())

    before_sales_summary = before_update.groupby('Store')['Sales'].sum().reset_index()
    after_sales_summary = after_update.groupby('Store')['Sales'].sum().reset_index()

    sales_comparison = pd.merge(before_sales_summary, after_sales_summary, on='Store', suffixes=('_before', '_after'))

    print("Sales Comparison:", sales_comparison)

    # Assuming you call the function and get sales_comparison

    # Melt the DataFrame for easier plotting
    sales_melted = sales_comparison.melt(id_vars='Store', value_vars=['Sales_before', 'Sales_after'],
                                        var_name='Period', value_name='Sales')

    # Set up the plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=sales_melted, x='Store', y='Sales', hue='Period')
    plt.title('Sales Comparison Before and After Competitor Distance Update')
    plt.xlabel('Store')
    plt.ylabel('Total Sales ($)')
    plt.xticks(rotation=45)
    plt.legend(title='Period', labels=['Before Update', 'After Update'])
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()  
    
    return sales_comparison



