import pandas as pd
import datetime as dt
import numpy as np
import requests
from pymongo import MongoClient
from sklearn.metrics import r2_score, mean_squared_error

# Fire data cleaning

def drop_offshore_points(df):
    '''
    Removes datapoints located in the water or in US. Locations found through mapping data. Removed through a series of lat/long filters and individual indices.
    '''
    idx1 = df[(df['LONGITUDE'] < -134.5) & (df['LATITUDE'] < 58.5)].index
    df.drop(idx1, inplace=True)
    idx2 = df[(df['LONGITUDE'] < -133.08) & (df['LATITUDE'] < 55)].index
    df.drop(idx2, inplace=True)
    idx3 = df[(df['LONGITUDE'] < -126.89) & (df['LATITUDE'] < 49.8)].index
    df.drop(idx3, inplace=True)
    idx4 = df[(df['LONGITUDE'] > -123.04) & (df['LONGITUDE'] < -95) & (df['LATITUDE'] < 48.95)].index
    df.drop(idx4, inplace=True)
    idx5 = df[(df['LONGITUDE'] > -130.99) & (df['LONGITUDE'] < -130) & (df['LATITUDE'] > 51.87) & (df['LATITUDE'] < 53.62)].index
    df.drop(idx5, inplace=True)
    
    df.drop(index=[423718, 423531, 375825, 375826, 375827, 375828, 375823, 375830, 375829, 419691, 146985], inplace=True)
    
    return df


def impute_ecozones(df):
    '''
    Fills in missing ECOZ_NAME values using KNNImputer with lat/long.
    '''
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import LabelEncoder
    imputer = KNNImputer(missing_values=0)
    encoder = LabelEncoder()
    
    # encode ecozone names
    df['ECOZ_ENCODED'] = encoder.fit_transform(df['ECOZ_NAME'])
    
    # impute missing values
    df_filled = imputer.fit_transform(df[['LATITUDE', 'LONGITUDE', 'ECOZ_ENCODED']])
    df['ECOZ_IMPUTE']= df_filled[:, 2]
    df['ECOZ_IMPUTE'] = df['ECOZ_IMPUTE'].apply(int)
    
    # Convert from labels back to names and replace column 
    df['ECOZ_NAME'] = encoder.inverse_transform(df['ECOZ_IMPUTE'])
    df.drop(columns=['ECOZ_IMPUTE', 'ECOZ_ENCODED'], inplace=True)

    return df


def clean_data(filepath):
    '''
    Imports CANADA_WILDFIRES.csv and addresses null values.
    '''
    df = pd.read_csv('data/CANADA_WILDFIRES.csv')

    # remove missing dates (3713 rows)
    df.dropna(subset=['REP_DATE'], inplace=True)
    
    # fill missing causes with unknown (241 rows)
    df['CAUSE'] = df['CAUSE'].fillna('U')
    
    # drop rows 0, 0 lat, long (146 rows)
    idx = df[df['LATITUDE'] == 0].index
    df.drop(idx, inplace=True)

    # Correct longitudes that should be negative (6 rows).
    df['LONGITUDE'] = -df['LONGITUDE'].abs()
    
    # Drop points outside of Canada's borders
    df = drop_offshore_points(df)
    
    # Drop row from 1930 (only one from this year, the rest of the data starts at 1946)
    df.drop(115345, inplace=True)

    # Impute missing ECOZ_NAME values
    df = impute_ecozones(df)

    # Adjust case of column names
    df.columns = df.columns.str.title()
    df.rename(columns={'Fid': 'FID'}, inplace=True)    

    return df



# Feature Adding

def split_dates(df):
    df['Rep_Date'] = pd.to_datetime(df['Rep_Date'])
    df['Year'] = df['Rep_Date'].dt.year
    df['Month'] = df['Rep_Date'].dt.month
    df['Date'] = df['Rep_Date'].dt.day
    df['Day_of_Week'] = df['Rep_Date'].dt.dayofweek

    return df

def decade_groups(df):
    '''
    Adds column with fires binned by decade.
    '''
    bins = [1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020, 2030]
    labels=['40s', '50s', '60s', '70s', '80s', '90s', '00s','10s', '20s']
    df['Decade'] = pd.cut(df['Year'], bins=bins, labels=labels)
    
    return df

def add_province(df):
    prov_dict = {'BC': 'BC', 'AB': 'AB', 'SK': 'SK', 'MB': 'MB', 'ON': 'ON', 'QC': 'QC', 'NL': 'NL', 'NB': 'NB', 'NS': 'NS', 'YT': 'YT', 'NT': 'NT',
       'PC-NA': 'NT', 'PC-WB': 'AB', 'PC-VU': 'YT', 'PC-BA': 'AB', 'PC-EI': 'AB', 'PC-WP': 'MB', 'PC-JA': 'AB',
       'PC-PA': 'SK', 'PC-GL': 'BC', 'PC-KO': 'BC', 'PC-RE': 'BC', 'PC-BT': 'SK', 'PC-YO': 'BC', 'PC-RM': 'MB',
       'PC-GF': 'BC' , 'PC-GR': 'SK', 'PC-WL': 'AB', 'PC-FR': 'BC', 'PC-PU': 'ON', 'PC-KG': 'NB', 'PC-LM': 'QC',
       'PC-CB': 'NS', 'PC-PE': 'PE', 'PC-BP': 'ON', 'PC-TI': 'ON', 'PC-SL': 'ON', 'PC-KE': 'NS', 'PC-PP': 'ON',
       'PC-SY': 'NT', 'PC-SE': 'NT' , 'PC-NC': 'YT', 'PC-KL': 'YT', 'PC-RE-GL': 'BC', 'PC-GM': 'NL', 'PC-PR': 'BC',
       'PC-TN': 'NL', 'PC-GI': 'QC', 'PC-FW': 'SK', 'PC-FO': 'QC', 'PC-GB': 'ON', 'PC-LO': 'NS', 'PC-FU': 'NB',
       'PC-MM': 'NL', 'PC-TH': 'NT'}
    df['Province'] = df['Src_Agency'].map(prov_dict)

    return df

def size_bins(df):
    """
    Takes fire_df and assigns categories to fires of different sizes.
    """
    bins = [0, 15, 5000, df['Size_Ha'].max()]
    labels = ['small', 'medium', 'large']
    df['Size_Bin'] = pd.cut(x=df['Size_Ha'], bins=bins, labels=labels, include_lowest=True)

    return df

def add_fire_features(df):
    df = split_dates(df)
    df = add_province(df)
    df = size_bins(df)
    df = decade_groups(df)

    return df

# Repeat Locations
def calculate_avg_period(df):
    '''
    Takes the fire_df and finds locations with multiple fires.
    Returns a dictionary of structure {location_coordinates: (number of fires, avg time between fires)}
    '''
    # Make sure dates are in datetime format
    df['Rep_Date'] =  pd.to_datetime(df['Rep_Date'])
    # Sort the DataFrame by location and date
    df.sort_values(by=['Latitude', 'Longitude', 'Rep_Date'], inplace=True)

    # Initialize an empty dictionary to store average periods
    avg_periods = {}

    # Iterate over unique locations
    locations = df.groupby(['Latitude', 'Longitude']).groups
    for location, indices in locations.items():
        if len(indices) > 1:
            # Store number of fires
            fire_count = len(indices)

            # Get the dates for the location
            dates = df.loc[indices, 'Rep_Date']

            # Calculate the differences between consecutive dates
            date_diffs = dates.diff()

            # Calculate the average period by taking the mean of the differences
            avg_period = date_diffs.mean()
            
            # Store the average period in the dictionary
            avg_periods[location] = (fire_count, avg_period)

    return avg_periods

def multiple_fire_locations(df):
    '''
    Takes the fire_df, finds locations with multiple fires, calculates the average time between fires and returns a new dataframe.
    '''
    avg_periods = calculate_avg_period(df)
    avg_periods_df = pd.DataFrame(columns=['Lat', 'Long', 'Fire_Count', 'Avg_Period'])
    lats = [list(avg_periods.keys())[i][0] for i in range(len(avg_periods))]
    longs = [list(avg_periods.keys())[i][1] for i in range(len(avg_periods))]
    count = [list(avg_periods.values())[i][0] for i in range(len(avg_periods))]
    period = [list(avg_periods.values())[i][1] for i in range(len(avg_periods))]
    avg_periods_df['Lat'] = lats
    avg_periods_df['Long'] = longs
    avg_periods_df['Fire_Count'] = count
    avg_periods_df['Avg_Period'] = period
    avg_periods_df['Avg_Period'] = avg_periods_df['Avg_Period'].dt.days
    
    return avg_periods_df

# Weather API Pulls


def get_API_params(df):
    '''
    Given the fire_df dataframe, pull the fire ID, location coordinates, and date. 
    Find date 2 weeks prior and format variables for API request.
    '''
    lat = df['Latitude']
    long = df['Longitude']
    end_date = df['Rep_Date']
    start_date = end_date - dt.timedelta(days=14)
    FID = df['FID']

    # take just the date part of dates as a string
    end_date = str(end_date).split()[0]
    start_date = str(start_date).split()[0]

    return lat, long, start_date, end_date, FID



def api_request(lat, long, start_date, end_date):
    '''
    Given location and dates, sends API request to Open Meteo for daily weather info
    for each day during time period. Returns JSON of API response.
    '''
    url = f'https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={long}&start_date={start_date}&end_date={end_date}&daily=temperature_2m_max,temperature_2m_mean,precipitation_sum,windspeed_10m_max,winddirection_10m_dominant&timezone=auto'
    response = requests.get(url)
    print(end_date, response)
    if response.status_code == 200:
        data = response.json()
    else:
        data = None
    return data


def save_to_db(data, FID):
    '''
    Saves API response JSON to the weather collection in final_project_data Database.
    '''
    if data != None:
        client = MongoClient()
        db = client.final_project_data
        weather = db.weather
        data['FID'] = float(FID)
        weather.insert_one(data)

        print(f'FID {FID} weather data added to database.')
    else:
        print(f'No weather data available for {FID}')


def weather_api_pipeline(df):
    '''
    Given fire_df, pulls data to send an API request for each row. Makes API request and saves
    responses to MongoDB
    '''
    
    lat, long, start_date, end_date, FID = get_API_params(df)
    data = api_request(lat, long, start_date, end_date)
    save_to_db(data, FID)


# Extract weather data from MongoDB

def expand_dictionary_column(df, column_name):
    '''
    Given df from MongoDB data, expands columns that contain dictionaries.
    '''
    expanded_data = pd.DataFrame(df[column_name].tolist())
    expanded_df = pd.concat([df.drop(column_name, axis=1), expanded_data], axis=1)
    
    return expanded_df

def weather_to_df():
    # retrieve from MongoDB
    client = MongoClient()
    db = client.final_project_data
    weather = db.weather
    df = pd.DataFrame(list(weather.find()))
    df = df[['FID', 'latitude', 'longitude', 'elevation', 'daily']]
    client.close()
    # expand daily column
    df = expand_dictionary_column(df, 'daily')

    return df

# Weather DF Cleaning and Feature Engineering

def clean_weather(df):
    # Total Rainfall
    df['2_Week_Rainfall'] = df['precipitation_sum'].apply(sum)
    # Average high for 2 weeks prior
    df['Avg_High'] = df['temperature_2m_max'].apply(np.mean)
    # High temp on the start date of the fire
    df['Start_Date_High'] = df['temperature_2m_max'].apply(lambda x: x[-1])
    # Average of daily average temps for 2 weeks prior
    df['Avg_Temp'] = df['temperature_2m_mean'].apply(np.mean)
    # Windspeed max on day of fire
    df['Start_Date_Wind_Speed'] = df['windspeed_10m_max'].apply(lambda x: x[-1])
    # Wind Direction on day of fire
    df['Wind_Direction'] = df['winddirection_10m_dominant'].apply(lambda x: x[-1])
    # Capitalize elevation
    df.rename(columns={'elevation': 'Elevation'}, inplace=True)
    # Convert FID to integer
    df['FID'] = df['FID'].astype('int')
    
    # remove unused columns
    df.drop(columns=['precipitation_sum','temperature_2m_max', 'temperature_2m_mean', 'time', 'latitude', 'longitude', 'windspeed_10m_max', 'winddirection_10m_dominant'], inplace=True)
    return df


# ML Modeling

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    rsquared = r2_score(y_test, y_pred)

    print(f'RMSE of the model: {rmse:.3f}')
    print(f'R2 of the model: {rsquared:.3f}')