import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
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




def check_for_distribution_and_plot(train,test):
    logger.info("check for distribution and plot")
    try:
        train_promo_dist = train['Promo'].value_counts(normalize=True)
        test_promo_dist = test['Promo'].value_counts(normalize=True)
        print(f"train promo distribution : {train_promo_dist}")
        print(f"test promo distribution: {test_promo_dist}")
        logger.info("distribution calcualted sucessfully")
        # Plot the distributions for visual comparison
        plt.figure(figsize=(10, 5))

        # Plot for training set
        plt.subplot(1, 2, 1)
        sns.barplot(x=train_promo_dist.index, y=train_promo_dist.values, palette='Blues')
        plt.title('Promo Distribution - Training Set')
        plt.xlabel('Promo')
        plt.ylabel('Frequency')

        # Plot for test set
        plt.subplot(1, 2, 2)
        sns.barplot(x=test_promo_dist.index, y=test_promo_dist.values, palette='Greens')
        plt.title('Promo Distribution - Test Set')
        plt.xlabel('Promo')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()
    
    except Exception as e:
        logger.error(f" error while checking for distribution of promo on train and test data set {e}")
def chi_square_test(train,test):
    logger.info("working on Statistical Test - Chi-Squared Test")
    try:
        train_promo_count = train['Promo'].value_counts(normalize=True)
        test_promo_count = test['Promo'].value_counts(normalize=True)

        # Align both series to ensure they have the same indices (Promo = 0 and Promo = 1)
        train_promo_count = train_promo_count.reindex([0, 1], fill_value=0)
        test_promo_count = test_promo_count.reindex([0, 1], fill_value=0)
        chi_2 , p_value = chisquare(f_obs=train_promo_count, f_exp=test_promo_count)
        print(f"Chi-Squared Test Statistic: {chi_2}")
        print(f"P-Value: {p_value}")
        # Interpretation of P-Value
        if p_value < 0.05:
            print("The Promo distributions are significantly different.")
        else:
            print("The Promo distributions are similar.")
    except Exception as e:
        logger.error(f"error while performing statistical analysis {e}")



