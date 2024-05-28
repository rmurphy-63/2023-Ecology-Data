# -*- coding: utf-8 -*-
"""
Created on Wed May 22 11:49:02 2024

@author: profo
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import KNNImputer
import streamlit as st
filepath = 'Global Ecological Footprint 2023.csv'

data = pd.read_csv(filepath, encoding='latin-1')

# Rename the columns 
data.rename(columns={'Total biocapacity ': 'Total biocapacity'}, inplace=True)
data.rename(columns={'Life Exectancy': 'Life Expectancy'}, inplace=True)

# Removing '$' and ',' to convert'Per Capita GDP' to a numeric column 
data['Per Capita GDP'] = data['Per Capita GDP'].str.replace('$', '')
data['Per Capita GDP'] = data['Per Capita GDP'].str.replace(',', '')

# Convert non-numeric columns to numeric
columns_to_convert = ['SDGi', 'Life Expectancy', 'HDI', 'Population (millions)', 'Per Capita GDP']

for col in columns_to_convert:
    data[col] = pd.to_numeric(data[col], errors='coerce')
    
missing_values_per_row = data.isnull().sum(axis=1)

missing_values_df = pd.DataFrame({'Row': missing_values_per_row.index, 'Missing Values': missing_values_per_row.values})

missing_values_df['Country'] = data['Country']
missing_values_df= missing_values_df[missing_values_df['Missing Values']>0]

sorted_missing_values_df = missing_values_df.sort_values(by='Missing Values', ascending=False)

sorted_missing_values_df = sorted_missing_values_df.loc[sorted_missing_values_df['Missing Values']>=11]

dropped_countries = sorted_missing_values_df['Country'].values.tolist()

culled_data = data.copy()

rows_to_drop = culled_data.apply(lambda row: row.isin(dropped_countries).any(), axis=1)

culled_data = culled_data.drop(index=culled_data[rows_to_drop].index)

# Let us imput the rest of the columns 
columns_to_impute = culled_data.columns[culled_data.isnull().any()].tolist()
# Initialize the KNN imputer with k=5 (5 nearest neighbors)
imputer = KNNImputer(n_neighbors=5)
# Fit and transform the data
culled_data[columns_to_impute] = imputer.fit_transform(culled_data[columns_to_impute])

numerical_df = culled_data.select_dtypes(include='number')

correlation_matrix = numerical_df.corr()

top_footprint_df = culled_data.sort_values(by='Total Ecological Footprint (Consumption)', ascending=False)

top_footprint_df = top_footprint_df[['Country','Total Ecological Footprint (Consumption)']]


#Build Dashboard
st.set_page_config(layout='wide')
add_sidebar = st.sidebar.selectbox('Global or Individual Country', ('Global Correlations', 'Country Statistics'))
st.title('Global Ecological Footprints 2023')
##Global
if add_sidebar == 'Global Correlations':
    st.write('Average Ecological Footprints for Different Activities (global hectacres per person)')
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric(label='Cropland Footprint',
                  value=round(culled_data['Cropland Footprint'].mean(),3))
    with col2:
        st.metric(label='Grazing Footprint',
                  value=round(culled_data['Grazing Footprint'].mean(),3))
        
    with col3:
        st.metric(label='Forest Product Footprint',
                  value=round(culled_data['Forest Product Footprint'].mean(),3))
        
    with col4:
        st.metric(label='Carbon Footprint',
                  value=round(culled_data['Carbon Footprint'].mean(),3))
        
    with col5:
        st.metric(label='Fish Footprint',
                  value=round(culled_data['Fish Footprint'].mean(),3))
        
    with col6:
        st.metric(label='Total Ecological Footprint',
                  value=round(culled_data['Total Ecological Footprint (Consumption)'].mean(),3))
    st.write(correlation_matrix.style.background_gradient(cmap='coolwarm'))
    fig, ax = plt.subplots()
    sns.barplot(y='Country', x='Total Ecological Footprint (Consumption)', 
                data=top_footprint_df.head(10), ax=ax, palette='muted')
    ax.set_title('Top 10 Countries With Highest Ecological Footprint')
    st.pyplot(fig)
##Individual Country
if add_sidebar == 'Country Statistics':
    st.write('This page shows the footprint statistics for each country included in the analysis. Some countries were not included due to them missing too much data.')
    country = st.selectbox('Select a country', culled_data['Country'])
    country_df = culled_data.loc[culled_data['Country']==country]
    st.write('This dataframe shows the various parameters associated with {country}.'.format(country=country))
    st.dataframe(country_df.reset_index(drop=True))
    st.write('Below are footprint values with a percent difference from the global average. Each value has also been given a rank in relation to the total number of countries studied.')
    ranked_df = culled_data.rank(axis=0, ascending=False)
    total = len(culled_data)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        value = country_df['Cropland Footprint'].values[0]
        average = round(culled_data['Cropland Footprint'].mean(),3)
        percent_difference = (value-average)/average*100
        st.metric(label='Cropland Footprint',
                  value=value,
                  delta=f'{percent_difference:.2f}%', delta_color='inverse')
        rank = int(ranked_df.at[culled_data.index[culled_data['Country']==country].tolist()[0] , 'Cropland Footprint'])
        st.write('Ranked:')
        st.write(f'{rank}/{total}')
        
    with col2:
        value = country_df['Grazing Footprint'].values[0]
        average = round(culled_data['Grazing Footprint'].mean(),3)
        percent_difference = (value-average)/average*100
        st.metric(label='Grazing Footprint',
                  value=value,
                  delta=f'{percent_difference:.2f}%', delta_color='inverse')
        rank = int(ranked_df.at[culled_data.index[culled_data['Country']==country].tolist()[0] , 'Grazing Footprint'])
        st.write('Ranked:')
        st.write(f'{rank}/{total}')
        
    with col3:
        value = country_df['Forest Product Footprint'].values[0]
        average = round(culled_data['Forest Product Footprint'].mean(),3)
        percent_difference = (value-average)/average*100
        st.metric(label='Forest Product Footprint',
                  value=value,
                  delta=f'{percent_difference:.2f}%', delta_color='inverse')
        rank = int(ranked_df.at[culled_data.index[culled_data['Country']==country].tolist()[0] , 'Forest Product Footprint'])
        st.write('Ranked:')
        st.write(f'{rank}/{total}')
        
    with col4:
        value = country_df['Carbon Footprint'].values[0]
        average = round(culled_data['Carbon Footprint'].mean(),3)
        percent_difference = (value-average)/average*100
        st.metric(label='Carbon Footprint',
                  value=value,
                  delta=f'{percent_difference:.2f}%', delta_color='inverse')
        rank = int(ranked_df.at[culled_data.index[culled_data['Country']==country].tolist()[0] , 'Carbon Footprint'])
        st.write('Ranked:')
        st.write(f'{rank}/{total}')
        
    with col5:
        value = country_df['Fish Footprint'].values[0]
        average = round(culled_data['Fish Footprint'].mean(),3)
        percent_difference = (value-average)/average*100
        st.metric(label='Fish Footprint',
                  value=value,
                  delta=f'{percent_difference:.2f}%', delta_color='inverse')
        rank = int(ranked_df.at[culled_data.index[culled_data['Country']==country].tolist()[0] , 'Fish Footprint'])
        st.write('Ranked:')
        st.write(f'{rank}/{total}')
        
    with col6:
        value = country_df['Total Ecological Footprint (Consumption)'].values[0]
        average = round(culled_data['Total Ecological Footprint (Consumption)'].mean(),3)
        percent_difference = (value-average)/average*100
        st.metric(label='Total Ecological Footprint (Consumption)',
                  value=value,
                  delta=f'{percent_difference:.2f}%', delta_color='inverse')
        rank = int(ranked_df.at[culled_data.index[culled_data['Country']==country].tolist()[0] , 'Total Ecological Footprint (Consumption)'])
        st.write('Ranked:')
        st.write(f'{rank}/{total}')

    
country = 'United States of America'
country_df = culled_data.loc[culled_data['Country']==country]
value = country_df['Carbon Footprint'].values[0]
ranked_df = culled_data.rank(axis=0, ascending=False)
rank = int(ranked_df.at[culled_data.index[culled_data['Country']==country].tolist()[0] , 'Carbon Footprint'])
