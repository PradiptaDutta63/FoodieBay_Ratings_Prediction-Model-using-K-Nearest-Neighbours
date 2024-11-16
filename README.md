# FoodieBay Ratings Prediction and Analysis

This project uses machine learning to predict restaurant ratings based on various features. The primary goal is to determine the factors that significantly impact restaurant ratings, enabling business stakeholders to make data-driven decisions to improve their offerings.

## Project Overview

- **Data Source**: The data for this project contains restaurant information, including average cost, types of cuisine, restaurant type, availability of online order, and table booking options.
- **Objective**: Predict restaurant ratings and provide insights on how various features influence ratings.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/pradiptadutta63/restaurant-ratings-prediction.git
   cd restaurant-ratings-prediction
   ```
   
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Preparation

**1. Data Loading:** The dataset (```FoodieBay.csv```) is loaded, and unnecessary columns such as ```url```, ```address```, ```name```, ```phone```, ```location```,
```menu_item```, and ```dish_liked``` are dropped.

**2. Exploratory Data Analysis (EDA):**
   - Created a word cloud for cuisines to visualize popular cuisines.
   - Analyzed restaurant types with services like ```online_order``` and ```book_table```.
   - Visualized the relationship between restaurant type, average cost for two, and rating.

**3. Feature Engineering:**
   - Converted categorical features into numerical using one-hot encoding.
   - Selected features impacting ratings: average cost for two, votes, average review ranking, and other restaurant/service types.
   - Split the data into training and test sets (70% train, 30% test).
   - Standardized numerical features using ```StandardScaler```.

## Model Development

- **K-Nearest Neighbors (KNN) Regressor:**

- First trained a KNN model with ```k=3``` to predict ratings.
- Performance metrics:

   - **RMSE:** Root Mean Squared Error
   - **R-Squared:** Explained Variance
   - **MAE:** Mean Absolute Error

- Optimized `k` by evaluating RMSE for values between 1 and 30. The optimal `k` was found to be `2`.
- Final model with `k=2` showed improved metrics.

## Results
- **Best Model** (KNN with `k=2`):
   - **RMSE:** 0.253
   - **R-Squared:** 0.692
   - **MAE:** 0.137

These results indicate that the optimized KNN model effectively predicts restaurant ratings with relatively low error.

## Future Improvements

- Experiment with other models like Random Forest and Gradient Boosting for potentially better accuracy.
- Implement cross-validation and hyperparameter tuning to further improve performance.
- Add more features or additional data sources to capture other factors influencing restaurant ratings.

## Contributors
Pradipta Dutta - Data Scientist
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/pradiptadutta63)
