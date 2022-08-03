# import essential libraries
import pandas as pd
import numpy as np

# import splitting and imputing functions
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# import data acquisition
import acquire

def prep_zillow(df):

    # Handle null values for pool
    df.has_pool = df.has_pool.replace(np.nan, 0)

    # Drop missing values
    df = df.dropna()
    
    # Change data type
    # df['fips_code'] = df['fips_code'].astype('object')
    df['age'] = df['age'].astype(int)
    
    # Data mapping
    df['county'] = df.fips_code.map({'06037': 'Los Angeles', '06059': 'Orange', '06111': 'Ventura'})
    
    # Categorize bedrooms, bathrooms
    df['bedrooms_size'] = pd.cut(df.bedrooms, bins = [0,2,4,6],
                            labels = ['small', 'medium', 'large'])
    df['bathrooms_size'] = pd.cut(df.bathrooms, bins = [0,2.5,4.5,6.5],
                            labels = ['small', 'medium', 'large'])

    
    # Make dummy columns and concatenate on original dataframe 
    dummy_df = pd.get_dummies(df[['county', 'bedrooms_size', 'bathrooms_size']], dummy_na=False, drop_first=False)
    df = pd.concat([df, dummy_df], axis=1)
    
    # Handle Outliers:
    # The general rule for outliers are:
    ## Upper bond: Q3 + 1.5*IQR
    ## Lower bund: Q1 - 1.5*IQR
    # Bonds are manually adjusted for each feature
    df = df[df.bedrooms <= 6]
    df = df[df.bedrooms >= 1]

    df = df[df.bathrooms <= 6.5]
    df = df[df.bathrooms >= 0.5]

    df = df[df.square_feet <= 4800]
    df = df[df.square_feet >= 500]

    df = df[df.lot_size <= 100000]
    df = df[df.lot_size >= 900]

    df = df[df.assessed_value <= 1030000]
    df = df[df.assessed_value >= 45500]
    return df

def split(df):
    '''
    This function splits a dataframe into 
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