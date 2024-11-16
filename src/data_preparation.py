#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

def load_and_prepare_data(filepath):
    print("Loading and preparing data...")
    df = pd.read_csv(filepath)
    df = df.drop(["url", "address", "name", "phone", "location", "menu_item", "dish_liked"], axis=1)
    return df

