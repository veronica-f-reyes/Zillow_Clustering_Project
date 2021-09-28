#Zillow Clustering Project

# 1. Acquire data from mySQL using the python module to connect and query. You will want to end with a single dataframe. Make sure to include: the logerror, all fields related to the properties that are available. You will end up using all the tables in the database.
# Be sure to do the correct join (inner, outer, etc.). We do not want to eliminate properties purely because they may have a null value for airconditioningtypeid.
# Only include properties with a transaction in 2017, and include only the last transaction for each property (so no duplicate property ID's), along with zestimate error and date of transaction.
# Only include properties that include a latitude and longitude value.

# Functions to obtain Zillow data from the Codeup Data Science Database: zillow
#It returns a pandas dataframe.
#--------------------------------

# regular imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import env

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score

# SQL Query to pull data from Zillow Database that joins all tables and returns all features for properties sold during 2017 with only latest transaction date
# and excluding data that has null latitude and longitude
sql = """
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
       AND prop.longitude IS NOT NULL
"""

def get_db_url(database):
    from env import host, user, password
    url = f'mysql+pymysql://{user}:{password}@{host}/{database}'
    return url


# acquire zillow data using the query
def get_zillow(sql):
    url = get_db_url('zillow')
    zillow_df = pd.read_sql(sql, url, index_col='id')
    return zillow_df


def handle_missing_values(df, prop_required_column = .5, prop_required_row = .70):
	#function that will drop rows or columns based on the percent of values that are missing:\
	#handle_missing_values(df, prop_required_column, prop_required_row
    threshold = int(round(prop_required_column*len(df.index),0))
    df = df.dropna(axis=1, thresh=threshold)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    return df


def remove_columns(df, cols_to_remove):
	#remove columns not needed
    df = df.drop(columns=cols_to_remove)
    return df

def wrangle_zillow():
    if os.path.isfile('zillow_cached.csv') == False:
        df = get_zillow(sql)
        df.to_csv('zillow_cached.csv',index = False)
    else:
        df = pd.read_csv('zillow_cached.csv')

    # Restrict df to only properties that meet single use criteria
    single_use = [261, 262, 263, 264, 265, 266, 279]
    df = df[df.propertylandusetypeid.isin(single_use)]

    # Restrict df to only those properties with at least 1 bath & bed and 350 sqft area
    df = df[(df.bedroomcnt > 0) & (df.bathroomcnt > 0) & ((df.unitcnt<=1)|df.unitcnt.isnull())\
            & (df.calculatedfinishedsquarefeet>350)]

    # Handle missing values i.e. drop columns and rows based on a threshold
    df = handle_missing_values(df)

    # Add column for counties
    df['county'] = df['fips'].apply(
        lambda x: 'Los Angeles' if x == 6037\
        else 'Orange' if x == 6059\
        else 'Ventura')

    # drop unnecessary columns
    dropcols = ['parcelid',
         'calculatedbathnbr',
         'finishedsquarefeet12',
         'fullbathcnt',
         'heatingorsystemtypeid',
         'propertycountylandusecode',
         'propertyzoningdesc',
         'censustractandblock'
         ]

    df = remove_columns(df, dropcols)

    # replace nulls in unitcnt with 1
    df.unitcnt.fillna(1, inplace = True)

    # assume that since this is Southern CA, null means 'None' for heating system
    df.heatingorsystemdesc.fillna('None', inplace = True)

    # replace nulls with median values for select columns
    df.lotsizesquarefeet.fillna(7313, inplace = True)
    df.buildingqualitytypeid.fillna(6.0, inplace = True)

    # Columns to look for outliers
    #df = df[df.taxvaluedollarcnt < 5_000_000]
    #df = df[df.calculatedfinishedsquarefeet < 8000]

    #Remove outliers
    df = remove_outliers(df, 2.99, ['calculatedfinishedsquarefeet',
       'taxvaluedollarcnt','bedroomcnt', 'bathroomcnt'] )

    # Drop any Nulls left
    df = df.dropna()

    # Rename
    df = df.rename(columns = { 'bedroomcnt':'bedrooms', 
                              'bathroomcnt':'bathrooms', 
                              'calculatedfinishedsquarefeet':'sq_footage',
                              'taxvaluedollarcnt':'tax_value', 
                              'yearbuilt':'yr_built'
        })  

    return df

def min_max_scaler(train, valid, test):
    '''
    Uses the train & test datasets created by the split_my_data function
    Returns 3 items: mm_scaler, train_scaled_mm, test_scaled_mm
    This is a linear transformation. Values will lie between 0 and 1
    '''
    num_vars = list(train.select_dtypes('number').columns)
    scaler = MinMaxScaler()
    train[num_vars] = scaler.fit_transform(train[num_vars])
    valid[num_vars] = scaler.transform(valid[num_vars])
    test[num_vars] = scaler.transform(test[num_vars])
    return scaler, train, valid, test

def outlier_function(df, cols, k):
	#function to detect and handle oulier using IQR rule
    for col in df[cols]:
        q1 = df.annual_income.quantile(0.25)
        q3 = df.annual_income.quantile(0.75)
        iqr = q3 - q1
        upper_bound =  q3 + k * iqr
        lower_bound =  q1 - k * iqr
        df = df[(df[col] < upper_bound) & (df[col] > lower_bound)]
    return df

def train_validate_test_split(df):
    train_and_validate, test = train_test_split(df, train_size=0.8, random_state=123)
    train, validate = train_test_split(train_and_validate, train_size=0.75, random_state=123)
    return train, validate, test


def nulls_by_col(df):
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    prcnt_miss = num_missing / rows * 100
    cols_missing = pd.DataFrame({'num_rows_missing': num_missing, 'percent_rows_missing': prcnt_miss})
    return cols_missing

def nulls_by_row(df):
    num_missing = df.isnull().sum(axis=1)
    prcnt_miss = num_missing / df.shape[1] * 100
    rows_missing = pd.DataFrame({'num_cols_missing': num_missing, 'percent_cols_missing': prcnt_miss})\
    .reset_index()\
    .groupby(['num_cols_missing', 'percent_cols_missing']).count()\
    .rename(index=str, columns={'customer_id': 'num_rows'}).reset_index()
    return rows_missing

def summarize(df):
    '''
    summarize will take in a single argument (a pandas dataframe)
    and output to console various statistics on said dataframe, including:
    # .head()
    # .info()
    # .describe()
    # value_counts()
    # observation of nulls in the dataframe
    '''
    print('=====================================================\n\n')
    print('Dataframe head: ')
    print(df.head(3).to_markdown())
    print('=====================================================\n\n')
    print('Dataframe info: ')
    print(df.info())
    print('=====================================================\n\n')
    print('Dataframe Description: ')
    print(df.describe().to_markdown())
    num_cols = [col for col in df.columns if df[col].dtype != 'O']
    cat_cols = [col for col in df.columns if col not in num_cols]
    print('=====================================================')
    print('DataFrame value counts: ')
    for col in df.columns:
        if col in cat_cols:
            print(df[col].value_counts())
        else:
            print(df[col].value_counts(bins=10, sort=False))
    print('=====================================================')
    print('nulls in dataframe by column: ')
    print(nulls_by_col(df))
    print('=====================================================')
    print('nulls in dataframe by row: ')
    print(nulls_by_row(df))
    print('============================================')

# FUNCTION to remove outliers per John Salas
# ------------------------------------------
#   The value for k is a constant that sets the threshold. 
#   Usually, you’ll see k start at 1.5, or 3 or less, depending on how many outliers you want to keep. 
#   The higher the k, the more outliers you keep. Recommend not going beneath 1.5, but this is worth using, 
#   especially with data w/ extreme high/low values

def remove_outliers(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df


#FUNCTION to get min/max from columns in a dataframe
# --------------------------------------------------
# To call this function use:   df.apply(minMax)  (Outside will have to call as:  df.apply(wrangle.minMax))
# x is a list dataframe or 
def minMax(x):
    return pd.Series(index=['min','max'],data=[x.min(),x.max()])

# functions to create clusters and scatter-plot:


def create_cluster(df, X, k):
    
    """ Takes in df, X (dataframe with variables you want to cluster on) and k
    # It scales the X, calcuates the clusters and return train (with clusters), the Scaled dataframe,
    #the scaler and kmeans object and unscaled centroids as a dataframe"""
    
    scaler = MinMaxScaler().fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns.values).set_index([X.index.values])

    ## sklearn implementation of KMeans

    #define the thing
    kmeans = KMeans(n_clusters = k, random_state = 321)

    # fit the thing
    kmeans.fit(X_scaled)

    # Use (predict using) the thing
    kmeans.predict(X_scaled)
    df['cluster'] = kmeans.predict(X_scaled)
    df['cluster'] = 'cluster_' + df.cluster.astype(str)

    #Create centroids of clusters
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X_scaled.columns)

    return df, X_scaled, scaler, kmeans, centroids


def create_scatter_plot(x,y,df,kmeans, X_scaled, scaler):
    
    """ Takes in x and y (variable names as strings, along with returned objects from previous
    function create_cluster and creates a plot"""
    
    plt.figure(figsize=(14, 9))
    sns.scatterplot(x = x, y = y, data = df, hue = 'cluster')
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X_scaled.columns)
    centroids.plot.scatter(y=y, x= x, ax=plt.gca(), alpha=.30, s=500, c='black')
    plt.title('Visualizing Clusters')


def make_metric_df(y, y_pred, model_name, metric_df):
    if metric_df.size ==0:
        metric_df = pd.DataFrame(data=[
            {
            'model': model_name, 
            'RMSE_validate': mean_squared_error(
                y,
                y_pred) ** .5,
            'r^2_validate': explained_variance_score(
                y,
                y_pred)
            }])
        return metric_df
    else:
        return metric_df.append(
        {
            'model': model_name, 
            'RMSE_validate': mean_squared_error(
                y,
                y_pred) ** .5,
            'r^2_validate': explained_variance_score(
                y,
                y_pred)
        }, ignore_index=True)

    

def inertia_plot(X):
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(9, 6))
        pd.Series({k: KMeans(k).fit(X).inertia_ for k in range(2, 12)}).plot(marker='x')
        plt.xticks(range(2, 12))
        plt.xlabel('k')
        plt.ylabel('inertia')
        plt.title('Change in inertia as k increases')          

def plot_actual(train, grp_by_var, x, y):
    plt.figure(figsize=(14, 9))
    sns.scatterplot(x = x, y = y, data = train, hue = grp_by_var)

    #for cluster, subset in train.groupby(grp_by_var):
       # x = x
        #y = y
        #print(x,y)
       # plt.scatter(subset.x,subset.y, label=str(cluster), alpha=.6)
     #centroids.plot.scatter(y=y_var, x=x_var, c='black', marker='x', s=1000, ax=plt.gca(), label='centroid')

    plt.legend()
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title('Visualizing Actual Data - Not Clusters')
    plt.show()      



def calc_cluster_mean(train, cluster):
# Looping through all the regions in the cluster to calculate it's mean and compare it
# to the overall log error mean to see if it may be of significance


    count = train.cluster.max()

    print (count)

    logerror_mean =  train.logerror.mean()

    i = 0
    while i <= count:

            region_mean = train[train.cluster== i].logerror.mean()
        
            difference = abs(logerror_mean - region_mean)
    
            region_mean_df = pd.DataFrame(data=[
                {
                    'region': i,
                    'mean':region_mean,
                    'difference': difference
                }
                ])
        
            print(region_mean_df)
            i += 1