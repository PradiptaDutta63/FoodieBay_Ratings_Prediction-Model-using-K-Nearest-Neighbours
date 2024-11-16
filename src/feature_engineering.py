#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def feature_engineering(df):

    print("Performing feature engineering...")
    
    # Convert categorical variables
    df = pd.get_dummies(df, columns=['rest_type', 'listed_in_type', 'online_order', 'book_table'], drop_first=True)
    X = df.drop(['cuisines', 'listed_in_city', 'rate'], axis=1)
    y = df['rate']
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2023)
    
    # Scale numerical features
    scaler = StandardScaler()
    features_to_scale = ['ave_cost_for_two', 'votes', 'ave_review_ranking']
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train[features_to_scale]), columns=features_to_scale)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test[features_to_scale]), columns=features_to_scale)
    
    # Add categorical features
    features_to_keep = [col for col in X_train.columns if col not in features_to_scale]
    X_train_scaled = pd.concat([X_train_scaled, X_train[features_to_keep].reset_index(drop=True)], axis=1)
    X_test_scaled = pd.concat([X_test_scaled, X_test[features_to_keep].reset_index(drop=True)], axis=1)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

