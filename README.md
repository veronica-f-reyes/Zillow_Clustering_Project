## README for  Zillow Clustering Project
***
## Zillow: What is driving the errors in the Zestimates?

## Background:
Zillow, a real estate website in the United States, uses a Zestimate to estimate a property's market value.
"The Zestimate® home valuation model is Zillow’s estimate of a home’s market value. A Zestimate incorporates public, MLS and user-submitted data into Zillow’s proprietary formula, also taking into account home facts, location and market trends. It is not an appraisal and can’t be used in place of an appraisal." - zillow.com
For this project, we will look into finding drivers of error in the Zestimate. Using clustering methodologies, feature identification and comparison, visualizations, statistical testing, and regression models, we are to find drivers of error to predict log error.

## Project Goals:
Create at least 4 regression models to predict log error
Use clustering methodologies to help identify drivers of log error
Deliver findings in a final Jupyter Notebook presentation

## Executive Summary:

### What Did We Find Out About Error in the Zestimate?
We were successfully able to create regression models to predict log error but only one model showed slight improvement from the baseline.
Our model was able to reduce log error by 3% in out of sample data.
Features chosen for the model that may be drivers of error include bedrooms, bathrooms, square footage, latitude and longitude.
Even though the model performed slightly better than the baseline, all models seem to indicate that there may not be many significant drivers of log error in this data.
### Next Steps:
With more time, I would like to try running models on the clusters since they showed dependency on log error
I would also like to keep exploring other features and clusters in the models.
I would try different hyper parameters and explore other models available for predicting log error.

### Technical Findings:
- I was successfully able to create multiple regression models to predict log error with one model beating the baseline.
- Using clustering, I was able to confirm that latitude, longitude, number of bedrooms, and square footage may be drivers of error.
- The Ordinary Least Squares (OLS) Linear Regression model performed best on validate and was chosen to be used on unseen, out of sample data.
- The Linear Regression model was able to beat the baseline thus possibly reduce log error in actual predictions.


DATA DICTIONARY:
----------------

|Target|Datatype|Definition|
|:-------|:--------|:----------|
| logerror                    |67490 non-null  float64|  logarithmic error of the estimate of value |             |

|Feature|Datatype|Definition|
|:-------|:--------|:----------|
|   bathrooms                   |67490 non-null  float64|  number of bedrooms |                |    
|   bedrooms                    |67490 non-null  float64|  number of bathrooms  |               |    
|   buildingqualitytypeid       |67490 non-null  float64|  unique identifier classifying quality of building |                |   
|   sq_footage                  |67490 non-null  float64|  square footage of the building  |                 |   
|   fips                        |67490 non-null  float64|  unique ID for location of county   |                 |  
|   latitude                    |67490 non-null  float64|  geographical latitude of the property |                |   
|   longitude                   |67490 non-null  float64|  geographical longitude of the property |               |     
|   lotsizesquarefeet           |67490 non-null  float64|  square footage of the property lot |                  |  
|   propertylandusetypeid       |67490 non-null  float64|  unique ID use to identify property type (i.e. 261 - Single Family Residential)|                 |   
|   rawcensustractandblock      |67490 non-null  float64|  property tract and block as indicated on census |              |    
|   regionidcity                |67490 non-null  float64|  identifier for the city |                  | 
|   regionidcounty              |67490 non-null  float64|  identifier for the county |               |    
|   regionidzip                 |67490 non-null  float64|  identifier the zip code |                  | 
|   roomcnt                     |67490 non-null  float64|  number of rooms in property |                |  
|   unitcnt                     |67490 non-null  float64|  number of units on property |                |   
|   yr_built                    |67490 non-null  float64|  year property was built |                 |  
|   structuretaxvaluedollarcnt  |67490 non-null  float64|  dollar value for the structure only |                |    
|   tax_value                   |67490 non-null  float64|  value of the property in dollars  |               |      
|   assessmentyear              |67490 non-null  float64|  year the property was assessed |                |    
|   landtaxvaluedollarcnt       |67490 non-null  float64|  dollar value of the land only   |                |
|   taxamount                   |67490 non-null  float64|  tax amount due for that property  |                 |  
|   logerror                    |67490 non-null  float64|  logarithmic error of the estimate of value |                 |  
|   transactiondate             |67490 non-null  datetime64[ns]|  date of the sale of the property |              |
|   heatingorsystemdesc         |67490 non-null  object  |  description of the heating system |                | 
|   propertylandusedesc         |67490 non-null  object  |  description of property type (i.e. Condominium, Townhome, Single Family Residential)|                |   
|   county                      |67490 non-null  object  |  county the property is located in|                |   
|   sq_footage_bins             |67490 non-null  category  |  square footage range of property (i.e. 1000 - 2000) |              |
|   month                       |67490 non-null  int64  |    month the property was sold |                 |
|   month_bins                  |67490 non-null  category |  range of months the property was sold between|              |


DATA SCIENCE PIPELINE
----------------------

PLAN:
-----

PLAN -> Acquire -> Prepare -> Explore -> Model & Evaluate -> Deliver

See my Trello board here --> https://trello.com/b/XcUxYoW0/zillow-clustering-project

Working through the data science pipeline, I will acquire data using an acquire.py file which pulls data from the Zillow database using SQL and joins 8 tables. I will prepare the data using the wrangle_excs.py file which will get rid of unneeded columns and rename columns.
Then I will explore the data by looking for possible relationships between features and look at how they are distributed by creating plots and looking at the data.
Next I will create hypothesis and models to find drivers of log error. I will then compare the models that I ran on training data to validate data before running our model on the test data.   I will then present the findings in a review of the final Jupyter Notebook.  



ACQUIRE:
--------
Call the wrangle_excs.py to run the functions to obtain Zillow data using a SQL query from the Codeup Data Science Database: zillow
It returns a pandas dataframe.  The SQL query joins eight tables and filters on single family properties that I listed between May 1, 2017 to August 31, 2017.
The data is filtering to return single unit properties defined within the propertylandusetypeid of codes: 261- Single Family Residential, 262 - Rural Residence, 
263 - Mobile Home, 264 - Townhouse, 265 - Cluster Home, 266 - Condominium, and 279 - Inferred Single Family Residential.  It is also only selecting properties with latitude and longitude not null and only the latest transaction from each property, eliminating possible duplicate property sales in the same year.  


    The SQL query that is run is shown below:
                '''
                SELECT prop.*,
                pred.logerror,
                pred.transactiondate,
                air.airconditioningdesc,
                arch.architecturalstyledesc,
                build.buildingclassdesc,
                heat.heatingorsystemdesc,
                landuse.propertylandusedesc,
                story.storydesc,
                construct.typeconstructiondesc,
                prop.regionidcity,
                prop.regionidcounty,
                prop.regionidneighborhood,
                prop.regionidzip
                FROM   properties_2017 prop
                INNER JOIN (SELECT parcelid,
                   Max(transactiondate) transactiondate
                   FROM   predictions_2017

                   GROUP  BY parcelid) pred
                USING (parcelid)
               			JOIN predictions_2017 as pred USING (parcelid, transactiondate)
                LEFT JOIN airconditioningtype air USING (airconditioningtypeid)
                LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid)
                LEFT JOIN buildingclasstype build USING (buildingclasstypeid)
                LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid)
                LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid)
                LEFT JOIN storytype story USING (storytypeid)
                LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid)
                WHERE  prop.latitude IS NOT NULL
                AND prop.longitude IS NOT NULL;
                '''


PREPARE:
--------
Prepped and cleaned the acquired data in the wrangle_excs.py file.  Used functions within the wrangle_excs.py to accomplish the following:

- Created and used functions in wrangle.py to acquire and prep data
- Used a SQL query to join 8 tables
- Selected only single family properties filtered by:
- Properties with propertylandusetypeid = [261, 262, 263, 264, 265, 266, 279]
- Only retrieving latest transaction date, eliminating duplicates sold more than once in same year
- Only selecting properties with at least 1 bath & bed and 350 sqft area
- Only properties with a latitude and longitude that is not null
- Dropping columns that are less than 70% populated, mostly empty columns
- Adding a 'county' column based on FIPS unique county identifier feature
- Dropping columns that are not useful features or redundant
- Filling null values in unitcnt column with 1 since all are single unit properties
- Replacing nulls with median values for select columns:
- 7313 for lotsizesquarefeet
- 6.0 for buildingqualitytypeid
- Since this is Southern CA, filling null with 'None' for heatingorsystemdesc because most likely don't have one
- Rename columns for easier readability
-  Removed outliers based on Inter Quartile Rule for properties with outliers in square footage, price, bedrooms, and bathrooms


EXPLORE:
--------
Visualize combinations of variables to compare relationships between variables.
    - Log error seems to have a correlation to square footage, number of bedrooms, number of bathrooms, latitude and longitude .

Hypothesis 1:

Null Hypothesis: There is no correlation between latitude and log error .
Alternative Hypothesis: Latitude and log error are correlated.

Finding: 
    - We reject the null hypothesis that there is no correlation between latitude and log error .

Hypothesis 2:

Null Hypothesis: There is no correlation between longitude and log error .
Alternative Hypothesis: Longitude and log error are correlated.

Finding: 
    - We reject the null hypothesis that there is no correlation between longitude and log error .

Hypothesis 3:

Null Hypothesis: There is no correlation between square footage and log error .
Alternative Hypothesis: Square footage and log error are correlated.

Finding: 
    - We reject the null hypothesis that there is no correlation between square footage and log error .

Hypothesis 4:

Null Hypothesis: There is no correlation between number of bedrooms and log error .
Alternative Hypothesis: Number of bedrooms and log error are correlated.

Finding: 
    - We reject the null hypothesis that there is no correlation between number of bedrooms and log error .

Clustering Hypothesis:
    - All 5 clusters created using a combination of latitude, longitude, and square footage were found to be correlated to log error.     

MODEL:
------

The goal is to develop a regression model that performs better than the baseline by using features discoverd in exploration thru clustering or hypothesis testing.

Using square footage, number of bedrooms, number of bathrooms, latitude, and longitude, I created 5 different models including:
    - Linear Regression Model
    - Lasso Lars
    - Tweedie Regressor
    - Polynomial Linear Regression with Degree 2
    - Polynomial Linear Regression with Degree 3

The baseline log error was computed to be 0.164983, which is equal to the mean of log error for the training sample.

After trying 5 different models, only the Linear Regression model showed any slight improvement over the baseline.  

The Linear Regression model predicted a value of 0.160489 for the log error which is lower than the baseline with a variance (R^2) of 0.001860.

DELIVER:
-------

- Present findings in a review of a final Jupyter Notebook.
- The report/presentation should summarize findings about the drivers of log error. 
- A github repository containing your work with any .py files required to acquire and prepare the data and a clearly labeled final Jupyter Notebook that walks through the pipeline.


HOW TO RECREATE THIS PROJECT:
----------------------------

To recreate this project you will need the following files from this repository:
- README.md
- env.py*
- wrangle_excs.py
- Final_Clustering_Project_Notebook.ipynb

*Your personal env file requires  your data base credentials to connect to the SQL database containing Zillow data 

### Instructions:
- Read the README.md
- Download the wrangle_excs.py and Final_Clustering_Project_Notebook.ipynb files into your working directory, or clone this repository
- Add your own env file to your directory (user, password, host)  
- Run the Final_Clustering_Project_Notebook.ipynb 






