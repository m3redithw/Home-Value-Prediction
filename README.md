# Zillow Home Value Prediction by Meredith Wang
<a href="#"><img alt="Python" src="https://img.shields.io/badge/Python-013243.svg?logo=python&logoColor=white"></a>
<a href="#"><img alt="Pandas" src="https://img.shields.io/badge/Pandas-150458.svg?logo=pandas&logoColor=white"></a>
<a href="#"><img alt="NumPy" src="https://img.shields.io/badge/Numpy-2a4d69.svg?logo=numpy&logoColor=white"></a>
<a href="#"><img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-8DF9C1.svg?logo=matplotlib&logoColor=white"></a>
<a href="#"><img alt="seaborn" src="https://img.shields.io/badge/seaborn-65A9A8.svg?logo=pandas&logoColor=white"></a>
<a href="#"><img alt="plotly" src="https://img.shields.io/badge/plotly-adcbe3.svg?logo=plotly&logoColor=white"></a>
<a href="#"><img alt="sklearn" src="https://img.shields.io/badge/sklearn-4b86b4.svg?logo=scikitlearn&logoColor=white"></a>
<a href="#"><img alt="SciPy" src="https://img.shields.io/badge/SciPy-1560bd.svg?logo=scipy&logoColor=white"></a>

![logo](https://user-images.githubusercontent.com/105242871/180632056-229e205f-d3a5-4e04-a26d-d403f50a585f.jpg)


The current property buying or selling is hectic and expensive. Efficiently predicting the property pricing for real estate customers with respect to their budgets and priorities is essential.

In this project, we will use statistical analysis to analyze the key drivers of perperty value for single family properties, develop a ML regression model to predict property tax assessed value, and provide recommendations on making better homes' values prediction.
## :house:   Project Goals
▪️ Find the key drivers of property value for **single family properties** in 2017.

▪️ Construct an ML Regression model that predict **propery tax assessed values** ('assessed_value') of Single Family Properties using attributes of the properties.

▪️ Deliver a report that the data science team can read through and replicate, understand what steps were taken, why and what the outcome was.

▪️ Make recommendations on what works or doesn't work in prediction these homes' values.

## :memo:   Initial Questions
▪️ What associated with `assessed_value` the most?

▪️ Is having pool positively associated with `assessed value`?

▪️ Is number of bedrooms and bathrooms associated with `assessed value`?

▪️ Is `assessed_value` significantly different across 3 different counties?
## :open_file_folder:   Data Dictionary
**Variable** |    **Value**    | **Meaning**
---|---|---
*Latitude* | Float number | Latitude of the middle of the parcel
*Longitude* | Float number | Longitude of the middle of the parcel
*Bedrooms* | Integer ranging from 1-6 | Number of bedrooms in home 
*Bathrooms* | Float ranging from 0.5-6.5| Number of bathrooms in home including fractional bathrooms
*Square Feet* | Float number | Calculated total finished living area of the home 
*Lot Size* | Float number | Area of the lot in square feet
*Age* | Integer |  This indicate the age of the property in 2017, calculated using the year the principal residence was built 
*Assessed Value* | Float number | The total tax assessed value of the parcel
*Tax Amount*| Float number | The total property tax assessed for that assessment year
*County* | 1) Ventura 2) Los Angeles 3) Orange | County in which the property is located

## :placard:    Process
#### :one:   Data Acquisition

<details>
<summary> Gather data from mySQL database</summary>

- Create env.py file to establish connection to mySQL server

- Use **zillow** database in the mySQL server

- Read data dictionary and extract meaningful columns 

- Write query to join useful tables to gather all data about the houses in the region:  <u>properties_2017, predictions_2017, propertylandusetype </u>
     ```sh
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
    poolcnt AS has_pool,
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
     ```
</details>

<details>
<summary> acqure.py</summary>

- Create acquire.py and user-defined function `get_zillow_data()` to gather data from mySQL
     ```sh
     def get_zillow_data():
     
     if os.path.isfile('zillow.csv'):
        df = pd.read_csv('zillow.csv', index_col=0)
    else:
        df = new_zillow_data()
        df.to_csv('zillow.csv')
        
    return df
    ```
- Import [acquire.py](acquire.py)

- Test acquire function

- Calling the function, and store the table in the form of dataframe
    ```sh
    df = acquire.get_zillow_data()
    ```
</details>

#### :two:   Data Preparation

<details>
<summary> Data Cleaning</summary>

- **Missing values:**
    - Null values for `has_pool` column is replaced with 0
        ```sh
        df.has_pool = df.has_pool.replace(np.nan, 0)
        ``` 
    - Other null values are dropped
         ```sh
        df = df.dropna()
        ```
- **Data types: float is converted to `int` datatype**
     ```sh
     df['fips_code'] = df['fips_code'].astype(int)
     df['age'] = df['age'].astype(int)
     ```
- **Data mapping**
    - created new `county` column with county name corresponding to **fips_code**
    - created new bins `bedrooms_size` and `bathrooms_size` for `bedrooms` and `bathrooms`
             
             df['bedrooms_size'] = pd.cut(df.bedrooms, bins = [0,2,4,6],
                            labels = ['small', 'medium', 'large'])
             df['bathrooms_size'] = pd.cut(df.bathrooms, bins = [0,2.5,4.5,6.5],
                            labels = ['small', 'medium', 'large'])
             
- **Dummy variables:**
    - Created dummy variables for categorical feature `county`, `bedrooms_size`, `bathrooms_size`
    - Concatenated all dummy variables onto original dataframe

- **Outliers**
    - General rull for handling outliers:
        - Upper bond: Q3 + 1.5 * IQR
        - Lower bond: Q1 - 1.5 * IQR
    
        **Note:** each feature has minor adjustment based on data distribution
    - Outliers for each feature are dropped
        ```sh
        df = df[df.bedrooms <= 6]
        df = df[df.bedrooms >= 1]

        df = df[df.bathrooms <= 6.5]
        df = df[df.bathrooms >= 0.5]

        df = df[df.square_feet <= 7982]
        df = df[df.square_feet >= 493]

        df = df[df.lot_size <= 152597]
        df = df[df.lot_size >= 787]

        df = df[df.assessed_value <= 2520956]
        df = df[df.assessed_value >= 45366]
        ```
- Create function `prep_zillow` to clean and prepare data with steps above

- Import [prepare.py](prepare.py)

- Test prepare function

- Call the function, and store the cleaned data in the form of dataframe
</details>

<details>
<summary> Data Splitting</summary>

- Create function `split()` to split data into **train, validate, test**

- Test split function

- Check the size of each dataset
     ```sh
     train.shape, validate.shape, test.shape
     ```
- Call the function, and store the 3 data samples separately in the form of dataframe
     ```sh
     train, validate, test = prepare.split(df)
     ```
</details>

#### :three:   Exploratory Analysis
- Ask questions to find what are the key features that are associated with property assessed value

- Explore each feature's correlation with assessed value

- Using visualizations to better understand the relationship between features

#### :four:    Statistical Testing & Modeling
- Conduct T-Test for categorical variable vs. numerical variable

- Conduct Chi^2 Test for categorical variable vs. categorical variable

- Conclude hypothesis and address the initial questions
#### :five:    Modeling Evaluation
- Create multiple regression model and use Recursive Feature Elimination (RFE) to select features

- Find the amount of features that can gerenate the highest performance (evaluated using Root Mean Squared Error)

- Generate polynomial model, fit and tranform the train dataset into feature

- Find the degree that generates the best performing model (evaluated using RMSE)

- Create lasso-lars model object, fit the model to our training data, and use the model to make predictions

- Create generalized linear model `(TweedieRegressor)` and fit train dataset

- Pick the top 3 models among all the models and evaluate performance on validate dataset

- Pick the model with highest accuracy and evaluate on test dataset

## :repeat:   Steps to Reproduce
- [x] You will need an **env.py** file that contains the hostname, username and password of the mySQL database that contains the telco table. Store that env file locally in the repository.
- [x] Clone my repo (including the **imports.py**, **acquire.py**, **prepare.py**) 
- [x] Confirm **.gitignore** is hiding your env.py file
- [x] Libraries used are pandas, matplotlib, seaborn, plotly, sklearn, scipy
- [x] Follow instructions in [zillow_eda](zillow_eda.ipynb) workbook and README file
- [x] Good to run final report :smile_cat:

## :key:    Key Findings

▪️ The top 5 drivers of property assessed value are:

 - square feet
 
 - size of bedrooms and bathrooms
 
 - lot size
 
 - county
 
 - has pool or not
 
▪️ **Square feet** has a strong positive relationship with property tax assessed value

▪️ Property tax assessed value is dependent on the number of **bedrooms** and **bathrooms**.

▪️ Mean of property tax assessed value of Los Angeles, Orange, and Ventura County are not all equal.
- Los Angeles county has the lowest property tax assessed value on average
- Orange county has the highest property tax assessed value on average

▪️ Properties with **pool** has a higher property tax assessed value on average than properties without pool.

▪️ The age of the property (2017 - year_built) has a negative medium correlation with property tax assessed value.

▪️ The meachine learning model: polynomial features degree 3 is expected to predict housing prices within variance of **$202015** on average on future unseen data

## :high_brightness:    Recommendations
▪️ To better understand the relationship between features and the target vairable, we need more information on the properties' **location**

▪️ To improve model's accuracy, we need more accurate latitude and longtitude data to pinpoint the property.


## 🔜  Next Steps
▪️ Collect more **geographic** data on the property(e.g. local school, surrounding properties, distance from downtown, city population, etc.)

▪️ Develop machine learning models with higher accuracy (lower RMSE) with these additonal data and make better predictions.

▪️ Collect data on previous years to analyze the general trend of each area, and determine what features drive the housing prices the most.
