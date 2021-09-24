# README for  Zillow Clustering Project
# Zillow: What is driving the errors in the Zestimates?

GOALS:
-----
- Uncover what the drivers of the error in the zestimate using Zillow data using clustering methodologies.
- Define and use single unit / single family homes based on transaction data from May 1, 2017 through August 31, 2017,


KEY FINDINGS & TAKEAWAYS:
-------------------------
 Modeling Takeaways:

- Polynomial Regression Model performed better than the baseline by having a l Ir RMSE than the baseline.
- The model explained 35.6% of the variance on out of sample data very similar to our validate dataset. 

 Location & Tax Rates Takeaways:
- All properties are located in the state of California
- All properties are located in one of 3 counties:
    - Los Angeles County
    - Orange County
    - Ventura County
    
- The max tax rate was 0.01885 was located in Orange County
- The mix tax rate was 0.00622 was located in Orange County
- The highest average tax rate was in Los Angeles County with an average of 0.013054
- The l Ist average tax rate was Ventura County with an average of  0.011513
- The highest number of properties I located in Los Angeles County with 21,469 properties
- The l Ist number of properties I located in Ventura County with 3,189 properties


DATA DICTIONARY:
----------------

|Target|Datatype|Definition|
|:-------|:--------|:----------|
| tax_value                  | |35078 non-null:: float| value of the property in dollars  |              |

|Feature|Datatype|Definition|
|:-------|:--------|:----------|
|   bedrooms               |35078 non-null:  float64| number of bedrooms |              |
|   bathrooms              |35078 non-null:  float64| number of bathrooms  |              |
|   sq_footage             |35078 non-null:  float64| square footage of the building  |              |            |
|   yr_built               |35078 non-null:  float64| year property was built
|   taxamount              |35078 non-null:  float64| tax amount due for that property  |              ||              |
|   fips                   |35078 non-null:  float64| unique ID for location of county   |              |
|   propertylandusetypeid  |35078 non-null:  float64| unique ID use to identify property type (i.e. 261 - Single Family Residential)|              |
|   propertylandusedesc    |35078 non-null:  object | Description of property type (i.e. Condominium, Townhome, Single Family Residential)|              |
|   tax_rate               |35078 non-null:  float64| Tax rate for that property|              |
|  county                 |35078 non-null:  object | County the property is located in|              |
|  state                  |35078 non-null:  object | State the property is located in|              |




DATA SCIENCE PIPELINE
----------------------

PLAN:
-----

PLAN -> Acquire -> Prepare -> Explore -> Model & Evaluate -> Deliver

See my Trello board here --> https://trello.com/b/XcUxYoW0/zillow-clustering-project

Working through the data science pipeline, I will acquire data using an acquire.py file which pulls data from the Zillow database using SQL and joins 3 tables. I will prepare the data using the wrangle.py file which will get rid of unneeded columns, rename columns and create columns for county, state and tax rates.
Then I will explore the data by looking for possible relationships between features and look at how they are distribute by creating plots and looking at the data.
Next I will create hypothesis and models. I will then compare the models that I ran on training data to validate data before running our model on the test data.   I will then present the findings in a verbal presentation using slides.  



ACQUIRE:
--------
Call the acquire.py file within the wrangle.py to run the functions to obtain Zillow data using a SQL query from the Codeup Data Science Database: zillow
It returns a pandas dataframe.  The SQL query joins two tables and filters on single family properties that I listed between May 1, 2017 to August 31, 2017.
The data is filtering to return single unit properties defined within the propertylandusetypeid of codes: 261- Single Family Residential, 262 - Rural Residence, 
263 - Mobile Home, 264 - Townhouse, 265 - Cluster Home, 266 - Condominium, and 279 - Inferred Single Family Residential.  


    The SQL query that is run is shown below:
                '''
                 SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips, propertylandusetypeid, propertylandusedesc
                FROM properties_2017
                LEFT JOIN predictions_2017 USING (parcelid)
                LEFT JOIN propertylandusetype USING (propertylandusetypeid)
                WHERE propertylandusetypeid IN ('261','262','263','264','265','266','279') 
                AND transactiondate BE IEN '2017-05-01' AND '2017-08-31';  
                '''


PREPARE:
--------
Prepped and cleaned the acquired data in the wrangle.py file.  Used functions within the wrangle.py to accomplish the following:

- Nulls & duplicates were removed
- Columns were renamed for easier use & readability
- Outliers were removed per function using Interquartile Range (IQR) to find outliers for
- A column named tax_rate was created by dividing taxamount and taxvaluedollarcnt
- A column named county was created to name the county where the property is located (derived from fips code)
- A column named state was created to name the state where the property is located (derived from fips code)
- A prepped dataframe containing 35,078 entries and 12 columns
- Split data into train, validate and test datasets.

EXPLORE:
--------
Visualize combinations of variables to compare relationships between variables.
    - Price seems to have a correlation to square footage, number of bedrooms, number of bathrooms, fips, and year built.

Hypothesis 1:

Null Hypothesis: Mean price for single family residential <= Mean of all single unit properties
Alternative Hypothesis: Mean price for single family residential > Mean of monthly charges of all customers

Finding: 

    - We reject the null hypothesis that the mean price for single family residential <= Mean of all single unit properties

Hypothesis 2:

Null Hypothesis: There is no correlation between the square footage and price.
Alternative Hypothesis: Square footage and price are correlated

Finding: 
    - We reject the null hypothesis that there is no correlation between the square footage and price.

Hypothesis 3:

Null Hypothesis: There is no correlation between number of bedrooms and location (fips).
Alternative Hypothesis: Number of bedrooms and location (fips) are correlated.

Finding: 
    - We reject the null hypothesis that there is no correlation between number of bedrooms and location (fips).

Hypothesis 4:

Null Hypothesis: There is no correlation between number of bathrooms and price .
Alternative Hypothesis: Number of bedrooms and price.

Finding: 
    - We reject the null hypothesis that there is no correlation between number of bathrooms and price .

MODEL:
------

The goal is to develop a regression model that performs better than the baseline.

Predicted all prices to be $418,560.92, which is equal to the mean of tax_value for the training sample.
Predicted all final prices to be $359,073.00, which is equal to the median of tax_value for the training sample. 
Computed the RMSE comparing actual final price (tax_value) to price_pred_mean.
Computed the RMSE comparing actual final grade (tax_value) to price_pred_median


Polynomial Regression Model performed better than the baseline by having a l Ir RMSE than the baseline
The model explained 35.6% of the variance on out of sample data very similar to our validate dataset.

DELIVER:
-------

- Present a report in the form of verbally supported slides.

    The report/presentation slides should summarize findings about the drivers of the single unit property values. 


- A github repository containing your work with any .py files required to acquire and prepare the data and a clearly labeled final Jupyter Notebook that walks through the pipeline.


HOW TO RECREATE THIS PROJECT:
----------------------------

To recreate this project you will need the following files from this repository:
- README.md
- acquire.py
- prepare.py
- explore.py
- final_zillow_regression_project.ipynb

You will also need your own env file with your data base credentials to connect to the SQL database containing Zillow data 

Instructions:
- Read the README.md
- Download the aquire.py, prepare.py, explore.py and final_zillow_regression_project.ipynb files into your working directory, or clone this repository
- Add your own env file to your directory (user, password, host)  
- Run the final_zillow_regression_project.ipynb 






