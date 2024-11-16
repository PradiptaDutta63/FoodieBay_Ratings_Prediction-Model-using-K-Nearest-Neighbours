#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.neighbors import KNeighborsRegressor

def train_knn_model(X_train, y_train, k=3):

    print(f"Training KNN model with k={k}...")
    knn = KNeighborsRegressor(k)
    knn.fit(X_train, y_train)
    return knn

