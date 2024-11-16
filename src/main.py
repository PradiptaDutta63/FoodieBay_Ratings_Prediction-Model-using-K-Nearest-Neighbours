#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from src.data_preparation import load_and_prepare_data
from src.eda import perform_eda
from src.feature_engineering import feature_engineering
from src.model_training import train_knn_model
from src.model_evaluation import evaluate_model
from src.model_optimization import optimize_model

def main():
    print("Starting the pipeline...\n")
    
    # Step 1: Data Preparation
    df = load_and_prepare_data("data/FoodieBay.csv")
    
    # Step 2: EDA
    perform_eda(df)
    
    # Step 3: Feature Engineering
    X_train_scaled, X_test_scaled, y_train, y_test = feature_engineering(df)
    
    # Step 4: Model Training
    knn_model = train_knn_model(X_train_scaled, y_train)
    
    # Step 5: Model Evaluation
    evaluate_model(knn_model, X_test_scaled, y_test)
    
    # Step 6: Model Optimization
    best_k, _ = optimize_model(X_train_scaled, y_train, X_test_scaled, y_test)
    
    print("\nPipeline complete!")

if __name__ == "__main__":
    main()

