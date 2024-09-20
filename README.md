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
- **Data Cleaning Pipeline**: Missing values were handled and outliers were detected to ensure the dataset's integrity. Specifically, we imputed missing values in key columns like `CompetitionDistance` and removed unnecessary columns.
- **Outlier Detection**: Potential outliers in sales and customer data were examined and handled appropriately.
- **Logging**: All key steps in the data cleaning and analysis process were logged using the Python `logger` library for traceability and reproducibility.

## 3. Exploratory Data Analysis (EDA)
- **Promo Distribution**: The distribution of promotions between the training and test sets was compared.
- **Holiday Sales Analysis**: Sales behavior was compared before, during, and after public holidays such as Christmas, Easter, and other state holidays.
- **Correlation Analysis**: A detailed correlation analysis between the number of customers and sales was performed using visual tools like scatter plots and heatmaps.
- **Promotion Effectiveness**: We examined the impact of promotions on sales and customer count to assess whether promotions attracted more customers or boosted spending among existing customers.
- **Store Weekday Sales**: The effect of stores being open on all weekdays on their weekend sales was analyzed to see if there was any significant difference.
- **Assortment Type Analysis**: We explored how the store assortment type affects sales performance across stores.
- **Competitor Distance Impact**: The effect of competitor proximity on store sales was analyzed, especially in cases where competitor distances were initially missing but later became available.

## 4. Key Findings
- Promotions significantly increased both the number of customers and sales.
- Holiday sales showed a marked difference, with public holidays generating higher sales.
- Store sales were generally higher on weekdays than weekends, and certain store types saw less impact from competitor proximity.
  
## 5. Logging & Traceability
- All steps, from data loading and cleaning to analysis, were logged using the Python `logger` module.
- This ensures reproducibility and traceability for future use and collaboration.

## 6. Visualization & Results
- A wide range of visual tools (scatter plots, histograms, line charts) were used to clearly communicate the results and insights derived from the analysis.

## 7. Conclusion
The project provides actionable insights into customer behavior, the impact of promotions, and the factors affecting sales. These insights can be used for future business strategies, marketing campaigns, and store management decisions.
 


# Tools & Libraries Used

1. **Programming Language**: [![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=yellow)](https://www.python.org/)

2. **Data Manipulation**: [![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
   - Used for data cleaning, filtering, and manipulation of large datasets.

3. **Numerical Computing**: [![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
   - Essential for numerical computations and array manipulations.

4. **Data Visualization**: 
   - [![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat&logo=plotly&logoColor=white)](https://matplotlib.org/) 
     - Used for creating various plots such as scatter plots, histograms, and line graphs.
   - [![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat&logo=python&logoColor=white)](https://seaborn.pydata.org/)
     - Used for enhanced statistical visualizations and heatmaps.

5. **Logging**: [![Logger](https://img.shields.io/badge/Logging-4B8BBE?style=flat&logo=python&logoColor=yellow)](https://docs.python.org/3/howto/logging.html)
   - Python's `logger` library used for tracking the steps and errors during the analysis for traceability and reproducibility.

6. **Statistical Analysis**: [![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
   - Conducted ANOVA and other statistical tests to understand patterns in the data.

7. **Machine Learning Framework**: 
   - [![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
     - For predictive modeling and evaluation (if relevant models are applied).

8. **Jupyter Notebooks**: [![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)](https://jupyter.org/)
   - Used as the primary development environment for interactive data analysis and visualization.

9. **Version Control**: [![Git](https://img.shields.io/badge/Git-F05032?style=flat&logo=git&logoColor=white)](https://git-scm.com/)
   - For versioning and collaboration across the project repository.

10. **Code Formatting & Linting**: [![Black](https://img.shields.io/badge/Black-000000?style=flat&logo=python&logoColor=white)](https://github.com/psf/black)
    - Automated code formatting for maintaining consistent style throughout the project.

11. **Continuous Integration (CI)**: [![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-2088FF?style=flat&logo=github-actions&logoColor=white)](https://github.com/features/actions)
    - Implemented CI pipelines to ensure quality and robustness in the development workflow.



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
         └── 📓eda.ipynb
         └── 📓salesAnalysis.ipynb
└── 📁scripts
         └── 📃__init__.py
         └── 📃chi_squaredAnlaysis.py
         └── 📃customur_during_before_and_after_holiday_analysis.py
         └── 📃dependant_analysis.py
         └── 📃exploration_customer_beahvior.py
         └── 📃sales_analysis.py
└── 💻src
    └── 📁dashboard-div
                    └── 📝app.py
└── ⌛tests
         └── 📃__init__.py

└── 📜.gitignore
└── 📰README.md
└── 🔋requirements.txt
└── 📇setup.py.py
└── 📇TA_Lib-0.4.29-cp312-cp312-win_amd64.whl
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

### **📓eda.ipynb**

**Overview**: This notebook is dedicated to Exploratory Data Analysis (EDA), with a primary focus on data cleaning and outlier detection/handling for the project.

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

4. change directory to run the streamlit app locally.

```bash
cd src\dashboard-div
```

5. Start the streamlit app

```bash
streamlit run app.py
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
