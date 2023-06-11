import pandas as pd
import datetime as dt
import numpy as np
import requests
from pymongo import MongoClient

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

    return df


def split_dates(df):
    df['REP_DATE'] = pd.to_datetime(df['REP_DATE'])
    df['YEAR'] = df['REP_DATE'].dt.year
    df['MONTH'] = df['REP_DATE'].dt.month
    df['DATE'] = df['REP_DATE'].dt.day
    df['DAY_OF_WEEK'] = df['REP_DATE'].dt.dayofweek

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
    df['Province'] = df['SRC_AGENCY'].map(prov_dict)

    return df



# Weather API Pulls


def get_API_params(df):
    '''
    Given the fire_df dataframe, pull the fire ID, location coordinates, and date. 
    Find date 2 weeks prior and format variables for API request.
    '''
    lat = df['LATITUDE']
    long = df['LONGITUDE']
    end_date = df['REP_DATE']
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


# extract weather data from MongoDB

def expand_dictionary_column(df, column_name):
    '''
    Given df from MongoDB data, expands columns that contain dictionaries.
    '''
    expanded_data = pd.DataFrame(df[column_name].tolist())
    expanded_df = pd.concat([df.drop(column_name, axis=1), expanded_data], axis=1)
    
    return expanded_df

def weather_to_df(collection):
    # retrieve from MongoDB
    df = pd.DataFrame(list(collection.find()))
    df = df[['FID', 'latitude', 'longitude', 'elevation', 'daily']]

    # expand daily column
    df = expand_dictionary_column(df, 'daily')

    return df