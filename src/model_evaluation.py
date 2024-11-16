#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

def evaluate_model(model, X_test, y_test):

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"RMSE: {rmse:.3f}, R2: {r2:.3f}, MAE: {mae:.3f}")

