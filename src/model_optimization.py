#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

def optimize_model(X_train, y_train, X_test, y_test):

    print("Optimizing model...")
    best_k, best_rmse = None, float('inf')
    for k in range(1, 31):
        knn = KNeighborsRegressor(k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        if rmse < best_rmse:
            best_k, best_rmse = k, rmse
    print(f"Best k: {best_k}, Best RMSE: {best_rmse:.3f}")
    return best_k, best_rmse

