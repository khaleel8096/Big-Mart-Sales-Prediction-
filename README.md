# Big-Mart-Sales-Prediction-
The BigMart Sales Prediction project involves analyzing historical sales data to build various regression models, including Linear Regression, Decision Trees, Random Forest, and others, to forecast product sales in different outlets, providing actionable insights for improving sales strategies.
### Big Mart Sales Prediction Project Overview

The Big Mart Sales Prediction project aims to analyze and predict the sales of products in various outlets based on multiple attributes. The dataset consists of information about 1559 products sold across 10 different stores in different cities during the year 2013. The task is to build a predictive model that can accurately forecast the sales of each product in a specific outlet. This predictive analysis will help BigMart understand the factors influencing sales and make strategic decisions to increase revenue.

### Data Dictionary

#### Train Data (8523 entries)
- **Item_Identifier:** Unique ID for each product
- **Item_Weight:** Weight of the product
- **Item_Fat_Content:** Indicates whether the product is low fat or not
- **Item_Visibility:** The percentage of total display area allocated to the particular product in the store
- **Item_Type:** Category to which the product belongs
- **Item_MRP:** Maximum Retail Price (list price) of the product
- **Outlet_Identifier:** Unique ID for each store
- **Outlet_Establishment_Year:** The year in which the store was established
- **Outlet_Size:** Size of the store in terms of ground area covered
- **Outlet_Location_Type:** Type of city in which the store is located
- **Outlet_Type:** Indicates whether the outlet is a grocery store or a supermarket
- **Item_Outlet_Sales:** Sales of the product in the particular store (Target variable)

#### Test Data (5681 entries)
- **Item_Identifier:** Unique ID for each product
- **Item_Weight:** Weight of the product
- **Item_Fat_Content:** Indicates whether the product is low fat or not
- **Item_Visibility:** The percentage of total display area allocated to the particular product in the store
- **Item_Type:** Category to which the product belongs
- **Item_MRP:** Maximum Retail Price (list price) of the product
- **Outlet_Identifier:** Unique ID for each store
- **Outlet_Establishment_Year:** The year in which the store was established
- **Outlet_Size:** Size of the store in terms of ground area covered
- **Outlet_Location_Type:** Type of city in which the store is located
- **Outlet_Type:** Indicates whether the outlet is a grocery store or a supermarket

### Problem Statement

BigMart has provided a dataset with historical sales data for various products in their outlets. The objective is to build a predictive model based on this dataset to forecast the sales of products in the test dataset. The model needs to be accurate and reliable to help BigMart understand the product and outlet attributes that significantly impact sales.

### Approach

1. **Data Exploration and Cleaning:**
   - Analyze the dataset to identify missing values, outliers, and data inconsistencies.
   - Handle missing values by imputing appropriate values or removing rows/columns with missing data.
   - Address outliers and ensure consistency in categorical variables.

2. **Feature Engineering:**
   - Explore relationships between variables and create new features if necessary.
   - Transform categorical variables into numerical representations using techniques like one-hot encoding or label encoding.
   - Standardize or normalize numerical features for consistent scaling.

3. **Model Building:**
   - Choose appropriate regression algorithms such as Linear Regression, Decision Trees, Random Forest, Support Vector Regression, etc.
   - Split the training data into features (X) and target variable (y) for model training.
   - Train multiple models and evaluate their performance using metrics like R-squared, Mean Squared Error (MSE), or Root Mean Squared Error (RMSE).

4. **Model Evaluation and Selection:**
   - Compare the performance of different models and select the one with the best accuracy and generalization.
   - Fine-tune hyperparameters if necessary to improve the model's predictive power.

5. **Sales Prediction:**
   - Use the selected model to predict sales for the test dataset.
   - Generate the final output with predicted sales for each product in each outlet.

6. **Insights and Recommendations:**
   - Analyze the model's predictions to identify key factors influencing sales.
   - Provide actionable insights and recommendations to BigMart for improving sales based on the model's findings.

### Repository Contents

- **data_preprocessing.py:** Python script for data cleaning, transformation, and preprocessing.containing different regression models for sales prediction.
- **requirements.txt:** File listing all the Python packages required to run the code.

### How to Use

1. **Clone the Repository:**
   ```
   git clone https://github.com/khaleel8096/BigMart-Sales-Prediction.git
   ```

2. **Install Dependencies:**
   ```
   cd BigMart-Sales-Prediction
   pip install -r requirements.txt
   ```

3. **Run the Code:**
   - Modify the file paths in the code to point to your dataset files.
   - Execute the Python scripts for data preprocessing and sales prediction.

### Dataset

The dataset used in this project contains historical sales data from BigMart outlets. It includes information about products, outlets, and corresponding sales figures.

### Models Used

- Linear Regression
- Decision Tree Regression
- Random Forest Regression
- Support Vector Regression
- K-Nearest Neighbors Regression
- Gradient Boosting Regression
- XGBoost Regression
- LightGBM Regression
- CatBoost Regression

### Results
The model performance is evaluated using the R-squared (coefficient of determination) scores, which measure the proportion of the variance in the dependent variable that is predictable from the independent variables in a regression model. Higher R-squared values indicate a better fit of the model to the data.

- **Random Forest Regressor (0.656):** This model explained approximately 65.6% of the variance in the sales data, indicating a relatively strong predictive capability.

- **Decision Tree Regressor (0.636):** The decision tree model accounted for about 63.6% of the variance in sales, demonstrating a good fit to the data.

- **Linear Regression (0.564):** The linear regression model explained around 56.4% of the variance in sales, providing a moderate level of prediction accuracy.

- **Support Vector Machine (SVR) (0.207):** The SVR model showed a limited ability to predict sales, explaining only 20.7% of the variance in the data.

- **K-Nearest Neighbors (KNN) (0.658):** KNN performed well, explaining 65.8% of the variance in sales, indicating strong predictive power.

- **Gradient Boosting Regressor (0.628):** This model explained approximately 62.8% of the variance in sales data, showing a good fit to the observed data.

- **XGBoost Regressor (0.628):** Similar to Gradient Boosting, XGBoost demonstrated a predictive capability of 62.8%, suggesting robust modeling.

- **LightGBM Regressor (0.623):** LightGBM explained about 62.3% of the variance in sales, indicating a reasonable fit to the data.

- **CatBoost Regressor (0.613):** CatBoost showed a predictive capability of 61.3%, indicating a moderate fit to the sales data.

These R-squared scores provide insights into the effectiveness of each regression model in capturing the underlying patterns in the sales data. Higher scores imply better predictive performance, helping to identify the most suitable model for making sales predictions.

In the competition with 6266 participants, my model achieved an impressive accuracy, ranking at 424th place. This accomplishment showcases the effectiveness of the implemented approach in accurately predicting the sales data, demonstrating strong performance among a large pool of competitors.

<img src="https://github.com/khaleel8096/Big-Mart-Sales-Prediction-/assets/87635567/ea5d302f-8ecc-469f-993a-36170d4317d0" alt="Image Description">

