# 🛡️ Sales Prediction Model 

## Table of Contents

- [Overview](#overview)
- [Technologies](#technologies)
- [Folder Organization](#folder-organization)
- [Setup](#setup)
- [Notes](#notes)
- [Contributing](#contributing)
- [License](#license)

## Overview: Key Functionalities

### Task 1 - Exploration of customer purchasing behavior

### Data Summarization Overview


# Customer Purchasing Behavior Analysis

## 1. Project Overview
This project involves the exploration and analysis of customer purchasing behavior across various stores using machine learning and statistical techniques. The main goal is to understand how factors like promotions, store types, holidays, and competitor distances affect sales and customer behavior.

## 2. Data Cleaning & Preparation
- **Data Cleaning Pipeline**
- **Outlier Detection**
- **Logging**

## 3. Exploratory Data Analysis (EDA)
- **Promo Distribution**
- **Holiday Sales Analysis**
- **Correlation Analysis**
- **Promotion Effectiveness**
- **Store Weekday Sales**
- **Assortment Type Analysis**
- **Competitor Distance Impact**  
## 4. Logging & Traceability
- All steps, from data loading and cleaning to analysis, were logged using the Python `logger` module.
- This ensures reproducibility and traceability for future use and collaboration.

## 6. Visualization & Results
- A wide range of visual tools (scatter plots, histograms, line charts) were used to clearly communicate the results and insights derived from the analysis.

## 7. Conclusion
The project provides actionable insights into customer behavior, the impact of promotions, and the factors affecting sales. These insights can be used for future business strategies, marketing campaigns, and store management decisions.

# Model Development 
 
 - **Project Overview**: This project focuses on predicting store sales using both machine learning and deep learning techniques.
  
- **Machine Learning Approach**:
  - Utilizes a Random Forest Regressor with sklearn pipelines for modular and reproducible modeling.
  - Chooses an interpretable loss function to evaluate model performance.
  - Analyzes feature importance and estimates confidence intervals for predictions.
  - Serializes models with timestamps for tracking daily predictions.

- **Deep Learning Approach**:
  - Builds a Long Short-Term Memory (LSTM) model using TensorFlow and Keras.
  - Transforms time series data into a supervised learning format to predict future sales.
  - Ensures efficient execution in Google Colab.

# Tools & Libraries Used

1. **Programming Language**: [![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=yellow)](https://www.python.org/)
2. **Data Manipulation**: [![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
3. **Numerical Computing**: [![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
4. **Data Visualization**: [![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=plotly&logoColor=white)](https://matplotlib.org/) [![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat&logo=python&logoColor=white)](https://seaborn.pydata.org/)
5. **Logging**: [![Logger](https://img.shields.io/badge/Logging-4B8BBE?style=flat&logo=python&logoColor=yellow)](https://docs.python.org/3/howto/logging.html)
6. **Statistical Analysis**: [![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
7. **Machine Learning Framework**: [![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
8. **Deep Learning Framework**: [![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)](https://keras.io/) | [![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F20?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
9. **Jupyter Notebooks**: [![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org/)
10. **Version Control**: [![Git](https://img.shields.io/badge/Git-F05032?style=flat&logo=git&logoColor=white)](https://git-scm.com/)
11. **Code Formatting & Linting**: [![Black](https://img.shields.io/badge/Black-000000?style=flat&logo=python&logoColor=white)](https://github.com/psf/black)
12. **Continuous Integration (CI)**: [![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-2088FF?style=flat&logo=github-actions&logoColor=white)](https://github.com/features/actions)
## Folder Organization

```

📁.dvc
└──
    └── 📁cache
    └── 📁tmp
    └── 📜.gitignore
    └── 📃config
    └── 📃config.local

📁.github
└──
    └── 📁workflows
         └── 📃unittests.yml
└── 📁notebooks
         └── 📓analysis_with_store_data_set.ipynb
         └── 📓deeplearningmodel.ipynb
         └── 📓eda.ipynb
         └── 📓model_building.ipynb
         └── 📓model_without_grid.ipynb
         └── 📓model.ipynb
         └── 📓salesAnalysis.ipynb
└── 📁scripts
         └── 📁model_training
                  └── 📃__init__.py
                  └── 📃pipeline_withotgrid.py
                  └── 📃pipeline.py
                  └── 📃preprocessing.py
         └── 📃__init__.py
         └── 📃chi_squaredAnlaysis.py
         └── 📃customur_during_before_and_after_holiday_analysis.py
         └── 📃dependant_analysis.py
         └── 📃exploration_customer_beahvior.py
         └── 📃sales_analysis.py
└── 💻src
    └── 📁dashboard-div
                    └── 📁static
                            └── 📝styles.css
                    └── 📁templates
                            └── 📝index.html
                    └── 📝app.py
└── ⌛tests
         └── 📃__init__.py

└── 📜.gitignore
└── 📰README.md
└── 🔋requirements.txt
└── 📇templates.py

```

### Folder Structure: A Deep Dive

- **📁.github**: This folder contains GitHub-related configurations, including CI/CD workflows.

  - **📁workflows**: Contains the CI/CD pipeline definitions.
    - **📃blank.yml**: Configuration for Continuous Integration.
    - **📃unittests.yml**: Configuration for running unit tests.

- ## 📁notebooks: This directory holds Jupyter notebooks and related Python files.

### **📓analysis_with_store_data_set.ipynb**

**Overview**: This notebook performs an in-depth analysis by merging the store dataset with the train dataset to uncover key insights, with a particular focus on evaluating the influence of competitor distance on store sales.

### **📓deeplearningmodel.ipynb**

**Overview**: This notebook focuses on building a deep learning model to predict store sales. It employs Long Short-Term Memory (LSTM) networks to leverage sequential data, enhancing the accuracy of sales forecasts through advanced machine learning techniques.

### **📓eda.ipynb**

**Overview**: This notebook is dedicated to Exploratory Data Analysis (EDA), with a primary focus on data cleaning and outlier detection/handling for the project.

### **📓 model_building.ipynb**

**Overview**: This notebook focuses on building a logistic regression model utilizing GridSearchCV for hyperparameter tuning. It systematically explores various parameter combinations to optimize model performance, ensuring the best possible predictions for store sales.


### **📓 model_without_grid.ipynb**

**Overview**: This notebook focuses on building a model without using GridSearchCV for hyperparameter tuning. It directly implements a logistic regression model, providing insights into model performance and evaluation without the additional complexity of parameter optimization.



### **📓salesAnalysis.ipynb**

**Overview**: This notebook is focused on performing an in-depth sales analysis of the dataset, with the goal of uncovering trends, insights, and factors that influence sales performance.


### **Expected Outcomes**

- **Data Summary**: An overview of dataset structure and basic statistics.
- **Data Quality**: Insights into missing values and outliers with appropriate handling.
- **Visual Insights**: A set of visualizations to understand data distributions, relationships, and trends.

- **📁scripts**: Contains Python scripts used throughout the project.

  - ## Modules Overview

This directory contains essential Python modules for analyzing and processing customer engagement data. Each module serves a specific purpose in the data analysis pipeline.

### **Modules**

- **📃 `__init__.py`**: Initializes the package and allows importing of modules.

- **📃 `eda.py`**: a module for a exploratory data analysis

### **Usage**

These modules are designed to be used in conjunction with each other to streamline the data analysis process, from data preparation and cleaning to in-depth analysis and model creation.

- **💻src**: The main source code of the project, including the Streamlit dashboard and other related files.

  - **📁dashboard-div**: Holds the code for the dashboard.
    - **📝app.py**: Main application file for the dashboard.
    - **📝README.md**: Documentation specific to the dashboard component.

- **⌛tests**: Contains test files, including unit and integration tests.

  - \***\*init**.py\*\*: Initialization file for the test module.

- **📜.gitignore**: Specifies files and directories to be ignored by Git.

- **📰README.md**: The main documentation for the entire project.

- **🔋requirements.txt**: Lists the Python dependencies required to run the project.

- **📇templates.py**: Contains templates used within the project, possibly for generating or processing data.

## Setup

1. Clone the repo

```bash
git clone https://github.com/Bereket-07/User_Analysis_and_Engagement.git
```

2. Change directory

```bash
cd User_Analysis_and_Engagement
```

3. Install all dependencies

```bash
pip install -r requirements.txt
```

4. change directory to run the Flask app locally.

```bash
cd src\dashboard-div
```

5. Start the Flask app

```bash
Python app.py
```

## Contributing

We welcome contributions to this project! To get started, please follow these guidelines:

### How to Contribute

1. **Fork the repository**: Click the "Fork" button at the top right of this page to create your own copy of the repository.
2. **Clone your fork**: Clone the forked repository to your local machine.
   ```bash
   git clone https://github.com/your-username/your-repository.git
   ```
3. **Create a new branch**: Create a new branch for your feature or bugfix.
   ```bash
   git checkout -b feature/your-feature
   ```
4. **Make your changes**: Implement your feature or fix the bug. Ensure your code adheres to the project's coding standards and style.
5. **Commit your changes**: Commit your changes with a descriptive message.
   ```bash
   git add .
   git commit -m 'Add new feature or fix bug'
   ```
6. **Push your branch**: Push your branch to your forked repository.
   ```bash
   git push origin feature/your-feature
   ```
7. **Create a Pull Request**: Go to the repository on GitHub, switch to your branch, and click the `New Pull Request` button. Provide a detailed description of your changes and submit the pull request.

## Additional Information

- **Bug Reports**: If you find a bug, please open an issue in the repository with details about the problem.

- **Feature Requests**: If you have ideas for new features, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License

### Summary

The MIT License is a permissive free software license originating at the Massachusetts Institute of Technology (MIT). It is a simple and easy-to-understand license that places very few restrictions on reuse, making it a popular choice for open source projects.

By using this project, you agree to include the original copyright notice and permission notice in any copies or substantial portions of the software.
