import pandas as pd
import os
import env
import requests

# Getting conncection to mySQL database, and acquiring data
def get_connection(db, user=env.user, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

# Loading raw data from Zillow database
def new_zillow_data():
    '''
    This function reads the Zillow data from the mySQL database into a df.
    '''
    # Create SQL query.
    sql_query = '''
    SELECT latitude, longitude, bedroomcnt as bedrooms, bathroomcnt as bathrooms, calculatedfinishedsquarefeet as square_feet, lotsizesquarefeet as lot_size, fips as fips_code, yearbuilt as year_built, taxvaluedollarcnt as assessed_value, taxamount as tax_amount
FROM properties_2017 as p JOIN predictions_2017 as pred USING(parcelid) JOIN
propertylandusetype as ptype using (propertylandusetypeid)
WHERE ptype.propertylandusedesc LIKE '%%Single%%' and pred.transactiondate LIKE '2017%%';
    '''
    # Read in DataFrame from Codeup db.
    df = pd.read_sql(sql_query, get_connection('zillow'))
    
    return df

def get_zillow_data():
    '''
    This function reads in zillow data from Zillow database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow.csv'):
        
        # If csv file exists, read in data from csv file.
        df = pd.read_csv('zillow.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame.
        df = new_zillow_data()
        
        # Write DataFrame to a csv file.
        df.to_csv('zillow.csv')
        
    return df