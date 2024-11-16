#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def perform_eda(df):

    print("Performing Exploratory Data Analysis (EDA)...")
    
    # Wordcloud for cuisines
    cuisines = " ".join(cuisine for cuisine in df['cuisines'].astype(str))
    wordcloud = WordCloud(background_color="white").generate(cuisines)
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Wordcloud for Cuisine")
    plt.show()
    
    # Crosstab and scatter plots
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, y='ave_cost_for_two', x='rate', hue='rest_type')
    plt.title("Rating vs Average Cost by Restaurant Type")
    plt.show()

