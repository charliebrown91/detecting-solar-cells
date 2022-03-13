# -*- coding: utf-8 -*-
"""Libraries"""
#!pip install fiona


"""Packages"""
import os
import zipfile
import csv
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import datetime
import glob
import requests
import pytz
#import fiona


"""Methods"""
def log(msg='...', path=''):
    tz_ZH = pytz.timezone('Europe/Zurich') 
    now = datetime.datetime.now(tz_ZH)
    now_string = now.strftime("%H:%M:%S")
    print('log: {} {:<20s} {:>45}'.format(now_string, msg, path))

def download_url(url, save_path, chunk_size=128):
    # source https://stackoverflow.com/questions/9419162/download-returned-zip-file-from-url
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

def download_file_from_google_drive(id, destination):
    # source https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url/60132855#60132855
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
             for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def cleanup_bfe(directory, filename):
    df=pd.DataFrame()
    path=os.path.join(directory, filename)
    if os.path.exists(path):
        log('loading '+filename.split('.')[0]+' data:', path)
        df=pd.read_csv(path)
        # filter
        df = df[(df['Canton'] == "LU") & (df['SubCategory'] == "subcat_2")]
        # Delete columns "Canton", "xtf_id", "Municipality"
        df = df.drop(columns=["Canton", "xtf_id", "Municipality", "MainCategory", "InitialPower", "_x", "_y"])
        # Renaming columns
        df= df.rename(columns={"TotalPower":"Leistung",
                               "SubCategory":"Anlagetyp",
                               "PlantCategory":"Kollektorart",
                               "BeginningOfOperation":"Inbetriebnahmedatum",
                               "Address":"Adresse"
                               })
        # Rename values
        df["Anlagetyp"] = df["Anlagetyp"].str.replace("subcat_2", "pv")
        df["Kollektorart"] = df["Kollektorart"].str.replace("plantcat_8", "Angebaut")
        df["Kollektorart"] = df["Kollektorart"].str.replace("plantcat_9", "Integriert")
        df["Kollektorart"] = df["Kollektorart"].str.replace("plantcat_10", "Freistehend")
        # change to datetime
        df["Inbetriebnahmedatum"] = pd.to_datetime(df["Inbetriebnahmedatum"])
        # merge address + postcode to address
        df["Adresse"] = df["Adresse"] + ' ' + df["PostCode"].astype(str)
        # Delete column "PostCode"
        df = df.drop(columns=["PostCode"])
        # save
        path = r"C:/Users/Thomas/PycharmProjects/pythonProject/data/03_cleaned"
        df.to_csv(os.path.join(path,r"bfe_cleaned.csv"), index=False)
    else:
        log('file not found:',path)
    return df

def cleanup_fb_2009(directory, filename):
    df=pd.DataFrame()
    path=os.path.join(directory, filename)
    if os.path.exists(path):
        log('loading '+filename.split('.')[0]+' data:', path)
        df=pd.read_csv(path, encoding = 'utf-8', sep=";")
        # filter rows without date of beginning of operation
        df=df.dropna(subset=['Inbetriebnahme'])
        # fill NaN-values with spaces
        df = df.replace(np.nan, '', regex=True)
        # Renaming columns
        df= df.rename(columns={"Address":"Adresse",
                               "Inbetriebnahme":"Inbetriebnahmedatum",
                               "Kollektortyp":"Kollektorart"})
        # Rename values
        df["Kollektorart"] = df["Kollektorart"].replace("selektive, verglaste Kollektoren (Faktor 1.0)", "Flachkollektoren")
        df["Kollektorart"] = df["Kollektorart"].replace("Vakuumröhrenkollektoren (Faktor 1.3)", "Röhrenkollektoren")
        # Delete several columns
        df = df.drop(columns=["Geschäftsnummer", "Kollektorfabrikat", "Aperturfläche pro Kollektor", "Anzahl Kollektoren", "Gesamt-Aperturfläche", "Realisierte Aperturfläche"])
        
        # Change datetime format
        df["Inbetriebnahmedatum"] = pd.to_datetime(df["Inbetriebnahmedatum"])

        # save
        path = r"C:/Users/Thomas/PycharmProjects/pythonProject/data/03_cleaned"
        df.to_csv(os.path.join(path,r"fb_cleaned_2009.csv"), index=False)
    else:
        log('file not found:',path)
    return df

def cleanup_fb_2017(directory, filename):
    df=pd.DataFrame()
    path=os.path.join(directory, filename)
    if os.path.exists(path):
        log('loading '+filename.split('.')[0]+' data:', path)
        df=pd.read_csv(path, encoding = 'utf-8', sep=";")
        # Delete several columns
        df = df.drop(columns=["Administration\nGesuch-ID",
                              "Strasse",
                              "LS-Nr",
                              "PLZ",
                              "Ort",
                              "Auszahlungen\nBetrag in CHF",
                              "Auszahlungen\nStatus"])
        # Renaming columns
        df= df.rename(columns={"Baufertigstellung":"Inbetriebnahmedatum",
                               "Thermische Kollektor-Nennleistung der Anlage [kw]":"Leistung"})
        # 
        # Rename values
        df["Kollektorart"] = df["Kollektorart"].replace("Flachkollektoren verglast", "Flachkollektoren")
        df["Kollektorart"] = df["Kollektorart"].replace("Flachkollektoren unverglast", "Flachkollektoren")
        df["Kollektorart"] = df["Kollektorart"].replace("Flachkollektor", "Flachkollektoren")
        df["Kollektorart"] = df["Kollektorart"].replace("Röhrenkollektor", "Röhrenkollektoren")
        # save
        path = r"C:/Users/Thomas/PycharmProjects/pythonProject/data/03_cleaned"
        df.to_csv(os.path.join(path,r"fb_cleaned_2017.csv"), index=False)
    else:
        log('file not found:',path)
    return df

def cleanup_addresses(directory, filename):
    df=pd.DataFrame()
    path=os.path.join(directory, filename)
    if os.path.exists(path):
        log('loading '+filename.split('.')[0]+' data:', path)
        df=pd.read_csv(path, encoding = 'utf-8', sep=";")
        # merge address + postcode to address
        df["Adresse"] = df["LOK_NAME"] + ' ' + df["HAUSNUMMER"].astype(str) + ' ' + df["PLZ"].astype(str)
        # Delete several columns 
        df = df.drop(df.columns[0:9], axis=1)
        df = df.drop(df.columns[1:17], axis=1)
        # Renaming columns
        df= df.rename(columns={"Coord_E":"E_Koordinaten",
                               "Coord_N":"N_Koordinaten",
                               "GWR_EGID":"EGID"})
        # save
        path = r"C:/Users/Thomas/PycharmProjects/pythonProject/data/03_cleaned"
        df.to_csv(os.path.join(path,r"addresses_cleaned.csv"), index=False)
    else:
        log('file not found:',path)
    return df
    df.to_csv()


"""Creating file structure"""
parent_dir='data'
if not os.path.exists(parent_dir):    
    log('creating directory:',parent_dir)
    os.makedirs(parent_dir)
    
directories=['01_downloads','02_extracts','03_cleaned']
for dir in directories: 
    path = os.path.join(parent_dir, dir) 
    if not os.path.exists(path):    
        os.makedirs(path) 
        log('creating directory:',path)


"""BFE data containing pv"""
# Downloading data from BFE (pv)
url="https://data.geo.admin.ch/ch.bfe.elektrizitaetsproduktionsanlagen/csv/2056/ch.bfe.elektrizitaetsproduktionsanlagen.zip"
zip_file_path=os.path.join(parent_dir,os.path.join(directories[0], 'elektrizitaetsproduktionsanlagen.zip'))
log('downloading:', url)
download_url(url, zip_file_path)

# Unzipping ElectricityProductionPlant data
log('extracting:','ElectricityProductionPlant.csv <---- '+zip_file_path)
zip = zipfile.ZipFile(zip_file_path , 'r')
zip.extract('ElectricityProductionPlant.csv',os.path.join(parent_dir,directories[1]))
zip.close()

dir=os.path.join('data','02_extracts')
cleanup_bfe(dir, 'ElectricityProductionPlant.csv')


"""Address data"""
# Downloading adress data from canton Lucerne - Google Drive
downloading = {"AV_Gebaeudeeingaenge": "1nLY3rL3TgBPoVBQPFQCjEjqubRWE7DnA"} # file name : file id
download_dir = os.path.join(parent_dir, directories[0]) # path to directory "01_downloads"

for file_name, file_id in downloading.items():
  path=os.path.join(download_dir, file_name+".csv")
  log("downloading:", path)
  download_file_from_google_drive(file_id, path) # download and save

dir=os.path.join('data','01_downloads')
cleanup_addresses(dir, 'AV_Gebaeudeeingaenge.csv')


"""JOIN BFE data + Address data"""
def import_csv(directory, filename):
    df=pd.DataFrame()
    path=os.path.join(directory, filename)
    if os.path.exists(path):
        log('loading '+filename.split(';')[0]+' data:', path)
        df=pd.read_csv(path)
    else:
        log('file not found:',path)
    return df

bfe=import_csv(os.path.join(parent_dir,directories[2]), "bfe_cleaned.csv")
address=import_csv(os.path.join(parent_dir,directories[2]), "addresses_cleaned.csv")

df_bfe=bfe.merge(address, how="inner", on="Adresse")
df_bfe.drop(columns="Adresse", inplace=True)


"""Foerderbeitraege 2009-16 data"""
# Downloading data from Foerderbeitraege (thermal) - Google Drive
downloading = {"Solarthermie_2009-16": "17O0sQZypnK13aF1PvZBXBgx0guTkEkGA"} # file name : file id
download_dir = os.path.join(parent_dir, directories[0]) # path to directory "01_downloads"

for file_name, file_id in downloading.items():
  path=os.path.join(download_dir, file_name+".csv")
  log("downloading:", path)
  download_file_from_google_drive(file_id, path) # download and save

dir=os.path.join('data','01_downloads')
cleanup_fb_2009(dir, 'Solarthermie_2009-16.csv')


"""Join Foerderbeitraege 2009-16 + Address data"""
fb_2009=import_csv(os.path.join(parent_dir,directories[2]), "fb_cleaned_2009.csv")
address=import_csv(os.path.join(parent_dir,directories[2]), "addresses_cleaned.csv")

fb_2009=fb_2009.merge(address, how="inner", on="Adresse") 
fb_2009.drop(columns="Adresse", inplace=True)


"""Foerderbeitraege 2017-21 data"""
# Downloading adress data from canton Lucerne - Google Drive
downloading = {"Solarthermie_2017-21": "139v8q2rP4kDWlQbWqKRNrXNiUlOPYpVW"} # file name : file id
download_dir = os.path.join(parent_dir, directories[0]) # path to directory "01_downloads"

for file_name, file_id in downloading.items():
  path=os.path.join(download_dir, file_name+".csv")
  log("downloading:", path)
  download_file_from_google_drive(file_id, path) # download and save

dir=os.path.join('data','01_downloads')
cleanup_fb_2017(dir, 'Solarthermie_2017-21.csv')

"""Join Foerderbeitraege 2017-21 + Address data"""
fb_2017=import_csv(os.path.join(parent_dir,directories[2]), "fb_cleaned_2017.csv")
address=import_csv(os.path.join(parent_dir,directories[2]), "addresses_cleaned.csv")

fb_2017=fb_2017.merge(address, how="inner", on="EGID") 
fb_2017.drop(columns="Adresse", inplace=True)


"""Append 2009-16 + 2017-21"""
fb=fb_2017.append(fb_2009)


"""Append thermal on pv"""
ground_truth = fb.append(df_bfe)
ground_truth["Anlagetyp"] = ground_truth["Anlagetyp"].replace(np.nan, 'thermal', regex=True)


"""Zeitspanne vor 2014 (erster Flug)"""
mask = (ground_truth["Inbetriebnahmedatum"] < "2014-06-08")
gt_2014 = ground_truth.loc[mask]

"""Save as shape file"""
schema = {
    'geometry':'Point',
    'properties':[('EGID','str'),
                  ('Inbetriebnahmedatum', 'str'),
                  ('Kollektorart', 'str'),
                  ('Leistung', 'float'),
                  ('Anlagetyp', 'str')]
}
pointShp = fiona.open("gt_2014.shp", mode="w", driver="ESRI Shapefile", schema = schema, crs = "EPSG:2056")

#iterate over each row in the dataframe and save record
for index, row in gt_2014.iterrows():
    rowDict = {
        'geometry' : {'type':'Point',
                     'coordinates': (row.E_Koordinaten,row.N_Koordinaten)},
        'properties': {'EGID' : row.EGID,
                       'Inbetriebnahmedatum': row.Inbetriebnahmedatum,
                       'Kollektorart': row.Kollektorart,
                       'Leistung': row.Leistung,
                       'Anlagetyp': row.Anlagetyp},
    }
    pointShp.write(rowDict)
#close fiona object
pointShp.close()


"""Zeitspanne vor 2017 (erster Flug)"""
mask = (ground_truth["Inbetriebnahmedatum"] < "2017-05-26")
gt_2017 = ground_truth.loc[mask]

"""Save as shape file"""
schema = {
    'geometry':'Point',
    'properties':[('EGID','str'),
                  ('Inbetriebnahmedatum', 'str'),
                  ('Kollektorart', 'str'),
                  ('Leistung', 'float'),
                  ('Anlagetyp', 'str')]
}
pointShp = fiona.open("gt_2017.shp", mode="w", driver="ESRI Shapefile", schema = schema, crs = "EPSG:2056")

#iterate over each row in the dataframe and save record
for index, row in gt_2017.iterrows():
    rowDict = {
        'geometry' : {'type':'Point',
                     'coordinates': (row.E_Koordinaten,row.N_Koordinaten)},
        'properties': {'EGID' : row.EGID,
                       'Inbetriebnahmedatum': row.Inbetriebnahmedatum,
                       'Kollektorart': row.Kollektorart,
                       'Leistung': row.Leistung,
                       'Anlagetyp': row.Anlagetyp},
    }
    pointShp.write(rowDict)
#close fiona object
pointShp.close()


"""Zeitspanne vor 2020 (erster Flug)"""
mask = (ground_truth["Inbetriebnahmedatum"] < "2020-05-07")
gt_2020 = ground_truth.loc[mask]

"""Save as shape file"""
schema = {
    'geometry':'Point',
    'properties':[('EGID','str'),
                  ('Inbetriebnahmedatum', 'str'),
                  ('Kollektorart', 'str'),
                  ('Leistung', 'float'),
                  ('Anlagetyp', 'str')]
}
pointShp = fiona.open("gt_2020.shp", mode="w", driver="ESRI Shapefile", schema = schema, crs = "EPSG:2056")

#iterate over each row in the dataframe and save record
for index, row in gt_2020.iterrows():
    rowDict = {
        'geometry' : {'type':'Point',
                     'coordinates': (row.E_Koordinaten,row.N_Koordinaten)},
        'properties': {'EGID' : row.EGID,
                       'Inbetriebnahmedatum': row.Inbetriebnahmedatum,
                       'Kollektorart': row.Kollektorart,
                       'Leistung': row.Leistung,
                       'Anlagetyp': row.Anlagetyp},
    }
    pointShp.write(rowDict)
#close fiona object
pointShp.close()
