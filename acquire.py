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
    SELECT 
    CONCAT(SUBSTRING(longitude, 1, 4),
                    ',',
                    SUBSTRING(longitude, 5, 10)) as longitude,
	CONCAT(SUBSTRING(latitude, 1, 2),
                    ',',
                    SUBSTRING(latitude, 3, 10)) as latitude,
    bedroomcnt AS bedrooms,
    bathroomcnt AS bathrooms,
    calculatedfinishedsquarefeet AS square_feet,
    lotsizesquarefeet AS lot_size,
    CONCAT ('0',fips) AS fips_code,
    (2017 - yearbuilt) AS age,
    taxvaluedollarcnt AS assessed_value,
    taxamount AS tax_amount
FROM
    properties_2017 AS p
        JOIN
    predictions_2017 AS pred USING (parcelid)
        JOIN
    propertylandusetype AS ptype USING (propertylandusetypeid)
WHERE
    ptype.propertylandusedesc LIKE '%%Single%%'
        AND pred.transactiondate LIKE '2017%%';
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