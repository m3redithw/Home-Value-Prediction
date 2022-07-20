# import essential libraries
import pandas as pd
import numpy as np

# import splitting and imputing functions
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# import data acquisition
import acquire

def prep_zillow(df):
    # Drop outliers
    df = df.dropna()
    
    # Change data type
    df['fips_code'] = df['fips_code'].astype(int)
    df['year_built'] = df['year_built'].astype(int)
    df['county_id'] = df['county_id'].astype(int)
    
    # Handle Outliers:
    # The general rule for outliers are:
    ## Upper bond: Q3 + 1.5*IQR
    ## Lower bund: Q1 - 1.5*IQR
    # Bonds are manually adjusted for each feature
    df = df[df.bedrooms <= 6.5]
    df = df[df.bedrooms >= 1]

    df = df[df.bathrooms <= 6.5]
    df = df[df.bathrooms >= 0.5]

    df = df[df.square_feet <= 7982]
    df = df[df.square_feet >= 493]

    df = df[df.lot_size <= 152597]
    df = df[df.lot_size >= 787]

    df = df[df.assessed_value <= 2520956]
    df = df[df.assessed_value >= 45366]

def split(df):
    '''
    This function drops the customer_id column and then splits a dataframe into 
    train, validate, and test in order to explore the data and to create and validate models. 
    It takes in a dataframe and contains an integer for setting a seed for replication. 
    Test is 20% of the original dataset. The remaining 80% of the dataset is 
    divided between valiidate and train, with validate being .30*.80= 24% of 
    the original dataset, and train being .70*.80= 56% of the original dataset. 
    The function returns, train, validate and test dataframes. 
    '''
    train, test = train_test_split(df, test_size = .2, random_state=123)   
    train, validate = train_test_split(train, test_size=.3, random_state=123)
    
    return train, validate, test