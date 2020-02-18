# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 19:57:24 2020

@author: ASUS
"""

import networkx as nx
from nxviz import CircosPlot
import seaborn as sns
from bokeh.plotting import figure, show
from bokeh.io import output_file
from bokeh.io import output_notebook
from bokeh.models import HoverTool
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import skew
from datetime import datetime
from datetime import timedelta
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter 
from tqdm import tqdm
import geopandas as gpd 
from shapely.geometry import Point
import folium
import geojson
from operator import itemgetter
import os
os.environ['PROJ_LIB'] = 'C:/Users/ASUS/Anaconda3/pkgs/proj4-5.2.0-ha925a31_1/Library/share'
from mpl_toolkits.basemap import Basemap
import cartopy.crs as crs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# Import modules for ML
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion
from warnings import warn
import sklearn.metrics as sklm
import math
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from scipy.cluster.vq import whiten
from scipy.cluster.vq import kmeans, vq
import seaborn as sns

#%%
#df=pd.read_excel('Base_de_Datos_Pineda_15.xlsx',dtype={'Dia': str, 'Mes':str})
df_1=pd.read_excel('RM_435_valentina.xlsx',dtype={'Dia': str, 'Mes':str})
df_2=pd.read_excel('RM_436_santiago.xlsx',dtype={'Dia': str, 'Mes':str})
df_3=pd.read_excel('RM_437_valentina.xlsx',dtype={'Dia': str, 'Mes':str})
df_4=pd.read_excel('RM_438_valentina.xlsx',dtype={'Dia': str, 'Mes':str})
df_5=pd.read_excel('RM_439_valentina.xlsx',dtype={'Dia': str, 'Mes':str})
df_6=pd.read_excel('RM_440_valentina.xlsx',dtype={'Dia': str, 'Mes':str})
df_7=pd.read_excel('RM_441_valentina.xlsx',dtype={'Dia': str, 'Mes':str})
df_8=pd.read_excel('RM_443_santiago.xlsx',dtype={'Dia': str, 'Mes':str})
df_9=pd.read_excel('RM_444_santiago.xlsx',dtype={'Dia': str, 'Mes':str})
df_10=pd.read_excel('RM_445_santiago.xlsx',dtype={'Dia': str, 'Mes':str})
df_11=pd.read_excel('RM_446_santiago.xlsx',dtype={'Dia': str, 'Mes':str})
df_12=pd.read_excel('RM_447_santiago.xlsx',dtype={'Dia': str, 'Mes':str})
df_13=pd.read_excel('RM_622_santiago.xlsx',dtype={'Dia': str, 'Mes':str})
df_14=pd.read_excel('RM_630_santiago.xlsx',dtype={'Dia': str, 'Mes':str})
df_15=pd.read_excel('RM_640_santiago.xlsx',dtype={'Dia': str, 'Mes':str})
df_list=[df_1,df_2,df_3,df_4,df_5,df_6,df_7,df_8,df_9,df_10,
         df_11,df_12,df_13,df_14,df_15]
df=pd.concat(df_list,sort=False)


pd.set_option('display.max_columns',100)
pd.set_option('display.max_colwidth',500)
pd.options.display.max_rows = 999
#%%
#verificar nombres de columnas para que coincidan en la concatenacion
print(df.columns)

#%%

print(df['Año'].value_counts())

#%%



#nulos y caracteres en años
df['Año'].replace("1854 (anotación: 'paginadas por fechas desde 1851 a 1858')",np.nan,inplace=True)
df['Año'].replace('Sin Fecha',np.nan,inplace=True)
df['Año'].replace('-',np.nan,inplace=True)
#cuales son nulos en año y conteo de valores nulos
print('conteo nulos:',df['Año'].isnull().values.sum())
#promedio y moda de años para determinar con qué rellenar nulos
print('promedio:',round(df['Año'].mean()))
print('moda:',df['Año'].mode()[0])
#%%
#convertir a numerico y los valores que no son convertibles los manda a nulo
pd.to_numeric(df['Año'], errors='coerce')


df['Año']=df['Año'].fillna(round(df['Año'].mean()))
df['Año']=df['Año'].astype('int64')
print(df['Año'].value_counts())
print('promedio despues de correccion:',round(df['Año'].mean()))
#%%
print(df.head(2))

#%%

#df.drop(df.index[3206],inplace=True)
df.drop(['Clasificación','Nombres A-H','Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 21'
       ,'Unnamed: 20'], axis=1,inplace = True)


#%%



df=df.fillna(0)
#print('info de marco de datos principal',df.info())
print('Dimenciones marco de datos principal', df.shape)

print(df.head(2))
print(df.dtypes)

#%%

df.rename(columns={"Ciudad origen": "Ciudad_origen"},inplace=True)
print(df.columns)
#%%
#correcciones ubicaciones
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Chamicera, Bogotá']='Bogotá, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Bogotá, Chapinero']='Bogotá, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Bogotá']='Bogotá, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'bogotá']='Bogotá, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Bogotá ']='Bogotá, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Bogotà']='Bogotá, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Bogota']='Bogotá, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Quinta de Canto']='Bogotá, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'sin ciudad']='Medellín, Antioquia, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Villavieja']='Villavieja, Huila, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Neiva']='Neiva, Huila, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Medellin']='Medellín, Antioquia, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Medellín']='Medellín, Antioquia, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Medellìn']='Medellín, Antioquia, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Medellín ']='Medellín, Antioquia, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'antioquia']='Medellín, Antioquia, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Popayàn']='Popayán, Cauca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Popayán']='Popayán, Cauca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'popayán']='Popayán, Cauca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Popayan']='Popayán, Cauca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Popayan']='Popayán, Cauca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Popayán, Sacondonoy']='Popayán, Cauca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'difícil']='Popayán, Cauca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Difícil']='Popayán, Cauca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Dificil']='Popayán, Cauca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'cauca']='Popayán, Cauca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Cauca']='Popayán, Cauca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Marinilla']='Marinilla, Antioquia, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Antioquia']='Marinilla, Antioquia, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Antioquía']='Marinilla, Antioquia, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Marinilla ']='Marinilla, Antioquia, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Riohacha']='Riohacha, La Guajira, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Santa Marta']='Santa Marta, Magdalena, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Pasto']='Pasto, Nariño, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Pastto']='Pasto, Nariño, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Pasto ']='Pasto, Nariño, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Nariño']='Pasto, Nariño, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Pipulquer (Nariño)']='Pasto, Nariño, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Cartagena']='Cartagena, Bolívar, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Cartagena ']='Cartagena, Bolívar, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Cartajena ']='Cartagena, Bolívar, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Panamá']='Panamá Viejo, Panamá'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Panamà']='Panamá Viejo, Panamá'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Pnammá ']='Panamá Viejo, Panamá'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Guaduas']='Guaduas, Cundinamarca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Honda']='Honda, Tolima, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Rionegro']='Rionegro, Antioquia, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Rionegro ']='Rionegro, Antioquia, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Barranquilla']='Barranquilla, Atlántico, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Barranquilla ']='Barranquilla, Atlántico, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Cartago']='Cartago, Valle del Cauca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Quito']='Quito, Ecuador'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Quito ']='Quito, Ecuador'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Salamina']='Salamina, Caldas, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Santarosa']='Santa Rosa de Osos, Antioquia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Santa Rosa']='Santa Rosa de Osos, Antioquia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Mompós']='Mompós, Bolívar, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Mompòs']='Mompós, Bolívar, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Cali']='Cali, Valle del Cauca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Cienaga']='Ciénaga, Magdalena, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Cienaga ']='Ciénaga, Magdalena, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Fusagasuga']='Fusagasugá, Cundinamarca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Fusagasugá']='Fusagasugá, Cundinamarca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Chagres']='Chagres, Panamá'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Union']='La Unión, Antioquia, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'La union']='La Unión, Antioquia, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Velez']='Velez, Santander, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Kingston']='Kingston, Jamaica'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Kingston Jamaica ']='Kingston, Jamaica'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Descanso']='Fusagasugá, Cundinamarca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Descanso ']='Fusagasugá, Cundinamarca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'El Descanso']='Fusagasugá, Cundinamarca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Venezuela']='Caracas, Distrito Capital, Venezuela'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Caracas']='Caracas, Distrito Capital, Venezuela'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Caracas']='Caracas, Distrito Capital, Venezuela'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Simacota']='Simacota, Santander, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Gigante']='Gigante, Huila, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Jigante ']='Gigante, Huila, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Dolores']='Dolores, Tolima, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Chochal']='Dolores, Tolima, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Socorro']='Socorro, Santander, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Tunja']='Tunja, Boyacá, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'mocoa']='Mocoa, Putumayo, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Mocoa']='Mocoa, Putumayo, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Rio de (…)']='Mocoa, Putumayo, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Putumayo']='Mocoa, Putumayo, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Purificación']='Purificación, Tolima, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Purificaciòn']='Purificación, Tolima, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Tocaima']='Tocaima, Cundinamarca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Porto cabello']='Puerto Cabello, Carabobo, Venezuela'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Yurgullo']='Puerto Cabello, Carabobo, Venezuela'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Bucaramanga']='Bucaramanga, Santander, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Guadalupe']='Guadalupe, Santander, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Ibarra']='Ibarra, Ecuador'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'La Mesa']='La Mesa, Cundinamarca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Lima']='Lima, Perú'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Laguna larga']='Pasca, Cundinamarca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Ipiales']='Ipiales, Nariño, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Ibague']='Ibagué, Tolima, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Inaguè']='Ibagué, Tolima, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'ibague']='Ibagué, Tolima, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Charalá']='Charalá, Santander, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Charala']='Charalá, Santander, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Tuquerrés']='Túquerres, Nariño, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Túquerres']='Túquerres, Nariño, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Tuquerres']='Túquerres, Nariño, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Paris']='París, Francia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Natagaima']='Natagaima, Tolima, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Sogamoso']='Sogamoso, Boyacá, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Buga']='Bugalagrande, Valle del Cauca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Londres']='Londres, Reino Unido'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Guasca']='Guasca, Cundinamarca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Natá']='Natagaima, Tolima, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Garzon']='Garzón, Huila, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Garzón']='Garzón, Huila, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Funsa']='Funza, Cundinamarca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Cucuta']='Cúcuta, Norte de Santander, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Barrancabermeja']='Barrancabermeja, Santander, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Valledupar']='Valledupar, Cesar, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Sibundoy']='Sibundoy, Putumayo, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Sibundoi']='Sibundoy, Putumayo, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Nueva York']='Nueva York, EE. UU.'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Sevilla']='Sevilla, España'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Zipaquira']='Zipaquirá, Cundinamarca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Villa de Leiva']='Villa de Leyva, Boyacá, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Villeta ']='Villeta, Cundinamarca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Villeta']='Villeta, Cundinamarca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Barbosa']='Barbosa, Antioquia, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Retiro']='Retiro, Antioquia, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Naré']='Puerto Nare, Antioquia, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Quilichao']='Santander de Quilichao, Cauca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Ocaña']='Ocaña, Norte de Santander, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Peñol']='Peñol, Antioquia, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Guayaquil']='Guayaquil, Ecuador'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Ambalema']='Ambalema, Tolima, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Curazao']='Curacao, Zulia, Venezuela'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Curasao']='Curacao, Zulia, Venezuela'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Girón']='Girón, Santander, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Sabanilla']='Puerto Colombia, Atlántico, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Sonsón']='Sonsón, Antioquia, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Sonson']='Sonsón, Antioquia, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Sonsóm']='Sonsón, Antioquia, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Remedios']='Remedios, Antioquia, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'San Jose de Ontopungos']='Panamá Viejo, Panamá'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Chepo']='Chepo, Panamá'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Bocas del Toro (Panamá)']='Bocas del Toro, Panamá'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'San Diego']='San Diego, Carabobo, Venezuela'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Mariquita']='Mariquita, Tolima, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Funza']='Funza, Cundinamarca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Chiquinquira']='Chiquinquirá, Boyacá, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Buenaventura']='Buenaventura, Valle del Cauca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Buenaventura ']='Buenaventura, Valle del Cauca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Diamante']='Colombia, Huila, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Plata']='La Plata, Huila, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Cadiz']='Cádiz, España'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Ceja']='La Ceja, Antioquia, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Puente grande']='Bogotá, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Mosqueral']='San Lorenzo, Nariño, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Subachoque']='Subachoque, Cundinamarca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Abejorral']='Abejorral, Antioquia, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Barbacoas']='Barbacoas, Nariño, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Cañasgordas']='Cañasgordas, Antioquia, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Aragoza']='Zaragoza, Antioquia, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Chia']='Chía, Cundinamarca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Barichara']='Barichara, Santander, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Chia']='Chía, Cundinamarca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Valencia']='Valencia, Carabobo, Venezuela'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Anserma']='Anserma, Caldas, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Supia']='Supía, Caldas, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Sibate']='Sibaté, Cundinamarca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Fomeque']='Fómeque, Cundinamarca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'San Sebastian (España)']='San Sebastián, España'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Maracaibo']='Maracaibo, Zulia, Venezuela'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Soledad']='Soledad, Atlántico, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Santana']='Santana, Boyacá, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Tenjo']='Tenjo, Cundinamarca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Tulua']='Tuluá, Valle del Cauca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Joya']='Tuluá, Valle del Cauca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Taboga']='Taboga, Panamá'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Aranzazu']='Aranzazu, Caldas, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Quayaquil']='Guayaquil, Ecuador'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Inglaterra']='Londres, Reino Unido'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Almaguer']='Almaguer, Cauca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'San Miguel']='San Miguel, Putumayo, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Caqueza']='Cáqueza, Cundinamarca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Cumbal']='Cumbal, Nariño, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Guadualito']='Yacopí, Cundinamarca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Guadalito']='Yacopí, Cundinamarca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Recreo']='Palmira, Valle del Cauca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Chumbanui']='Florencia, Caquetá, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Gachancipá']='Gachancipá, Cundinamarca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Gachansipa']='Gachancipá, Cundinamarca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Salina']='La Salina, Casanare, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Santa Librada']='Bojacá, Cundinamarca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Portoberrio']='Puerto Berrío, Antioquia, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'San Juan de Cuimbé']='San Juan Del Cesar, La Guajira, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Solano']='Solano, Caquetá, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Rosal']='El Rosal, Cundinamarca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Panonomé']='Penonomé, Panamá'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Zenezuela']='Cerrezuela, Panamá'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Zerrezuela']='Cerrezuela, Panamá'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Cerrezuela']='Cerrezuela, Panamá'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Neiva ']='Neiva, Huila, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Saldaña']='Saldaña, Tolima, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Guamo']='Guamo, Tolima, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Anolaima']='Anolaima, Cundinamarca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Santa Barbara']='Santa Bárbara, Antioquia, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Liverpool']='Liverpool, Reino Unido'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Agrado']='Agrado-La Plata, Agrado, Huila, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Angostura']='Angostura, Antioquia, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Belen']='Belén de Umbría, Risaralda, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Boquia']='Boquia, Salento, Quindío, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Camarones']='Camarones, Riohacha, La Guajira, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Caquetá']='San Vicente Del Caguán, Caquetá, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Caquezá']='Cáqueza, Vda. Centro, Cáqueza, Cundinamarca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Ceja del Tambo']='La Ceja, Antioquia, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Chame']='Chame, Panamá'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Cincelada']='Cincelada, Coromoro, Santander, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Coconuco']='Coconuco, Puracé, Cauca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Cocorná']='Cocorná, Antioquia, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Coro']='Coro falcon venezuela, Coro, Falcón, Venezuela'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Corozal']='Corozal, Sucre, Corozal, Sucre'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Curacao']='Curacao, Zulia, Venezuela'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Chorrera']='La Chorrera, Panamá'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'El Chocho']='El Chocho, Sucre'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'El Santuario']='El Santuario, Antioquia, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Espinal']='El Espinal, Tolima, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Guachucal']='Guachucal, Nariño, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Guarne']='Guarne, Antioquia, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Hacienda del Tambo']='El Tambo, Cauca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Hobo']='Hobo, Huila, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Ibagué']='Ibagué, Tolima, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'La Elida']='La Elida, La Belleza, Santander, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'La Union']='La Unión, Antioquia, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Limon']='Sucre, Santander, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Los Santos']='Los Santos, Santander, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Marmato']='Marmato, Caldas, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Mompox']='Mompós, Bolívar, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Naranjal']='Naranjal, Quinchía, Risaralda, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Nare']='Puerto Nare, Antioquia, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'No tiene']=0
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'no tiene']=0
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Nava']=0
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Guayota']=0
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Guaiza']=0
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Orito']='Orito, Putumayo, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Panamá ']='Panamá Viejo, Panamá'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Parita']='Parita, Panamá'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Pasca']='Pasca, Cundinamarca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Pital']='Pital, Huila, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Roldanillo']='Roldanillo, Valle del Cauca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Puentenacional']='Puente Nacional, Santander, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'San Antonio']='San Antonio, Tolima, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'San Diego Viejo']='San Diego Viejo, Puerto Caicedo, Putumayo, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'San Isidro']='San Isidro, Ecuador'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'San José']='San José de Chimbo, Ecuador'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'San Juan de Cesar']='San Juan Del Cesar, La Guajira, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'San Sebastian']='San Sebastián, España'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Santa Marta ']='Santa Marta, Magdalena, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Serrezuela']='Cerrezuela, Panamá'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Terreros']=0
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Tibacuy']='Tibacuy, Cundinamarca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Tigre']=0
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Timbío']='Timbío, Cauca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Timbio']='Timbío, Cauca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Tomarrazón']='Tomarrazón, Riohacha, La Guajira, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Ubaque']='Ubaque, Cundinamarca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Uraba']='Uraba, Yarumal, Antioquia, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Urubamba']='Urubamba, Perú'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Usme']='Bogotá, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Victoria']='La Victoria, Mosquera, Cundinamarca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Washington']='Washington, D.C., EE. UU.'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Yaguara']='Yaguara, Huila, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Yaguará']='Yaguara, Huila, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Zaragoza']='Zaragoza, Antioquia, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Zipaquirá']='Zipaquirá, Cundinamarca, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'prado']='Prado, Tolima, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Colon, Argentina']='Colon, Panamá'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'El Reposo']='El Reposo, Apartadó, Antioquia, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Mompóx']='Mompós, Bolívar, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Santa Ana']='Santa Ana, Magdalena, Colombia'
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 0]=None
df['Ciudad_origen'].loc[df['Ciudad_origen'] == 'Chapinero, Bogotá']='Bogotá, Colombia'
#rellenar con bogota los que son nulos y tienen relacion con la biblioteca
df['Ciudad_origen'].loc[(df['Ciudad_origen'].isnull()) & (df['Relacion biblioteca']==1)] = 'Bogotá, Colombia'

#%%
print(df['Ciudad_origen'].isnull().sum())
#%%
#correcciones remitentes y destinatarios

df['Remitente'].loc[df['Remitente'] == 'Emterio Gómez']='Emeterio Gómez'
df['Remitente'].loc[df['Remitente'] == 'Emeterio Gómez ']='Emeterio Gómez'
df['Destinatario'].loc[df['Destinatario'] == 'Emterio Gómez']='Emeterio Gómez'
df['Destinatario'].loc[df['Destinatario'] == 'Emeterio Gómez ']='Emeterio Gómez'
df['Remitente'].loc[df['Remitente'] == 'A.J.M.']='A.J.M'
df['Remitente'].loc[df['Remitente'] == 'A J  M']='A.J.M'
df['Remitente'].loc[df['Remitente'] == 'AG']='Ag Herrera'
df['Remitente'].loc[df['Remitente'] == 'Amador Gòmez']='Amador Gómez'
df['Remitente'].loc[df['Remitente'] == 'Ana Maria Danies']='Ana María Danies Pineda'
df['Remitente'].loc[df['Remitente'] == 'Ana María Danies']='Ana María Danies Pineda'
df['Remitente'].loc[df['Remitente'] =='Ana María Danies  Pineda']='Ana María Danies Pineda'
df['Destinatario'].loc[df['Destinatario'] == 'Ana Maria Danies']='Ana María Danies Pineda'
df['Destinatario'].loc[df['Destinatario'] == 'Ana María Danies']='Ana María Danies Pineda'
df['Destinatario'].loc[df['Destinatario'] == 'Ana María Danies ']='Ana María Danies Pineda'
df['Destinatario'].loc[df['Destinatario'] == 'Ana María Danies de Pineda']='Ana María Danies Pineda'
df['Destinatario'].loc[df['Destinatario'] == 'Ana María Danies  Pineda']='Ana María Danies Pineda'    
df['Remitente'].loc[df['Remitente'] == 'Anselmo pineda']='Anselmo Pineda'
df['Remitente'].loc[df['Remitente'] == 'Anselmo pineda y María Josefa Valencia']='Anselmo Pineda'
df['Destinatario'].loc[df['Destinatario'] == 'Anselmo pineda']='Anselmo Pineda'
df['Destinatario'].loc[df['Destinatario'] == 'Anselmo pineda ']='Anselmo Pineda'
df['Destinatario'].loc[df['Destinatario'] == 'Ancelmo Pineda']='Anselmo Pineda'
df['Destinatario'].loc[df['Destinatario'] == 'Anselmo pineda y María Josefa Valencia']='Anselmo Pineda'
df['Remitente'].loc[df['Remitente'] == 'Antonio Maria Gutierrez']='Antonio María Gutiérrez'
df['Destinatario'].loc[df['Destinatario'] == 'Antonio Maria Gutierrez']='Antonio María Gutiérrez'
df['Remitente'].loc[df['Remitente'] == 'Antonio Martínez']='Antonio R. Martínez'
df['Remitente'].loc[df['Remitente'] == 'Barbara Gómez']='Barbara Gómez de Gómez'
df['Remitente'].loc[df['Remitente'] == 'Caciano Clavo']='Caciano Calvo'
df['Remitente'].loc[df['Remitente'] == 'Carmen de Duque']='Carmen Duque de Duque'
df['Remitente'].loc[df['Remitente'] == 'Cesar Daza']='Cesar Daza Garzon'
df['Remitente'].loc[df['Remitente'] == 'Cordovez']='José María Cordovez Moure'
df['Remitente'].loc[df['Remitente'] == 'Cuervo']='Rufino José Cuervo'
df['Remitente'].loc[df['Remitente'] == 'Dfícil']='Ilegible'
df['Remitente'].loc[df['Remitente'] == 'Difícil']='Ilegible'
df['Remitente'].loc[df['Remitente'] == 'Eleutonio María Duque']='Eleuterio María Duque'
df['Remitente'].loc[df['Remitente'] == 'Eloy Concha']='Elias Concha'
df['Remitente'].loc[df['Remitente'] == 'Eugenio']='Eugenio M.'
df['Destinatario'].loc[df['Destinatario'] == 'Eugenio']='Eugenio M.'
df['Remitente'].loc[df['Remitente'] == 'Eugenio Salas']='Eujenio Salas'
df['Destinatario'].loc[df['Destinatario'] == 'Eugenio Salas']='Eujenio Salas'
df['Remitente'].loc[df['Remitente'] == 'Francisco A. Duque']='Francisco Duque'
df['Remitente'].loc[df['Remitente'] == 'Guadalupe Uribe']='Guadalupe Uribe de Riaño'
df['Remitente'].loc[df['Remitente'] == 'Ignacio (..)']='Ignacia Ballenys'
df['Destinatario'].loc[df['Destinatario'] == 'Ignacio (..)']='Ignacia Ballenys'
df['Remitente'].loc[df['Remitente'] == 'J. N. (…)']='J. N. Núñez'
df['Remitente'].loc[df['Remitente'] == 'J.J']='J.J. Flores'
df['Remitente'].loc[df['Remitente'] =='JJ']='J.J. Flores'
df['Remitente'].loc[df['Remitente'] == 'Joaquín M.']='Joaquín M. Palacio'
df['Remitente'].loc[df['Remitente'] == 'Josefa Acevedo de Gomez']='Josefa Acevedo de Gómez'
df['Destinatario'].loc[df['Destinatario'] =='Josefa Acevedo de Gomez']='Josefa Acevedo de Gómez'
df['Remitente'].loc[df['Remitente'] =='José Hilario Lopez']='José Hilario López'
df['Destinatario'].loc[df['Destinatario'] =='José Hilario Lopez']='José Hilario López'
df['Remitente'].loc[df['Remitente'] =='José Joaquin Gómez Hoyos']='José Joaquín Gómez Hoyos'
df['Remitente'].loc[df['Remitente'] =='José Joaquín Gómez']='José Joaquín Gómez Hoyos'
df['Destinatario'].loc[df['Destinatario'] =='José Joaquin Gómez Hoyos']='José Joaquín Gómez Hoyos'
df['Destinatario'].loc[df['Destinatario'] =='José Joaquín Gómez']='José Joaquín Gómez Hoyos'
df['Remitente'].loc[df['Remitente'] == 'José M. Duque Pineda']='José María Duque Pineda'
df['Destinatario'].loc[df['Destinatario'] == 'José M. Duque Pineda']='José María Duque Pineda'
df['Remitente'].loc[df['Remitente'] == 'José María Gaez']='José María González'
df['Destinatario'].loc[df['Destinatario'] == 'José María Gaez']='José María González'
df['Remitente'].loc[df['Remitente'] == 'José María Uribe']='José María Uribe Restrepo'
df['Remitente'].loc[df['Remitente'] == 'José de Obaldia']='José de Obaldía'
df['Remitente'].loc[df['Remitente'] == 'José. María Chiarri']='José María Chiarri'
df['Remitente'].loc[df['Remitente'] == 'Francisco Ortiz']='Juan Francisco Ortiz'
df['Remitente'].loc[df['Remitente'] == 'Juan de Azoàtegui']='Juan de Azoátegui'
df['Remitente'].loc[df['Remitente'] == 'Juan Z']='Juana Z'
df['Remitente'].loc[df['Remitente'] == 'Lista Porlon de Santander']='Lista Porton de Santander'
df['Remitente'].loc[df['Remitente'] == 'Manuel Garzòn']='Manuel Garzón'
df['Remitente'].loc[df['Remitente'] == 'Manuel Hurtado']='Manuel José Hurtado'
df['Remitente'].loc[df['Remitente'] =='Manuel Maria Quijano']='Manuel María Quijano'
df['Remitente'].loc[df['Remitente'] =='Manuel Marìa Quijano']='Manuel María Quijano'
df['Remitente'].loc[df['Remitente'] =='Manuel Quijano']='Manuel María Quijano'
df['Destinatario'].loc[df['Destinatario'] =='Manuel Maria Quijano']='Manuel María Quijano'
df['Destinatario'].loc[df['Destinatario'] =='Manuel Marìa Quijano']='Manuel María Quijano'
df['Destinatario'].loc[df['Destinatario'] =='Manuel Quijano']='Manuel María Quijano'
df['Remitente'].loc[df['Remitente'] == 'Manuela']='Manuela G de Zuluaga'
df['Destinatario'].loc[df['Destinatario'] == 'Maria Josefa Valencia de Pineda']='María Josefa Valencia'
df['Destinatario'].loc[df['Destinatario'] == 'María Josefa Valencia de Pineda']='María Josefa Valencia'
df['Destinatario'].loc[df['Destinatario'] == 'Maria Josefa Valencia']='María Josefa Valencia'
df['Remitente'].loc[df['Remitente'] == 'Mariana Duque']='Mariano Duque'
df['Remitente'].loc[df['Remitente'] =='Mariano Ospina y Lista Porton de Santander']='Mariano Ospina Rodríguez'
df['Destinatario'].loc[df['Destinatario'] =='Mariano Ospina y Lista Porton de Santander']='Mariano Ospina Rodríguez'
df['Remitente'].loc[df['Remitente'] =='María J']='María Josefa Valencia'
df['Destinatario'].loc[df['Destinatario'] =='María J']='María Josefa Valencia'     
df['Remitente'].loc[df['Remitente'] == 'Miguel Birbano']='Miguel Burbano'
df['Remitente'].loc[df['Remitente'] == 'Miguel Maria B']='Miguel María B'
df['Remitente'].loc[df['Remitente'] == 'Nicolasa Gonzáles de Restrepo']='Nicolasa González de Restrepo'
df['Remitente'].loc[df['Remitente'] =='Pedro Alcántara Herran']='Pedro Alcántara Herrán'
df['Remitente'].loc[df['Remitente'] =='Pedro Alcántara herran']='Pedro Alcántara Herrán'
df['Destinatario'].loc[df['Destinatario'] =='Pedro Alcántara Herran']='Pedro Alcántara Herrán'
df['Destinatario'].loc[df['Destinatario'] =='Pedro Alcántara herran']='Pedro Alcántara Herrán'
df['Remitente'].loc[df['Remitente'] =='Pedro Diaz Granados']='Pedro Díaz Granados'
df['Destinatario'].loc[df['Destinatario'] == 'Pedro Diaz Granados']='Pedro Díaz Granados'     
df['Remitente'].loc[df['Remitente'] == 'Pedro Saenz']='Pedro Sáenz'
df['Remitente'].loc[df['Remitente'] == 'Pedro Urizar']='Pedro Uribe'
df['Remitente'].loc[df['Remitente'] == 'Pedro de Marin']='Pedro de Marín'
df['Remitente'].loc[df['Remitente'] == 'Rafael Garcia']='Rafael García'
df['Remitente'].loc[df['Remitente'] == 'Sujeto J']='Sin firma'
df['Remitente'].loc[df['Remitente'] == 'Sujeto M']='Sin firma'
df['Remitente'].loc[df['Remitente'] == 'Tomàs Cipriano de Mosquera']='Tomás Cipriano de Mosquera'
df['Destinatario'].loc[df['Destinatario'] == 'Tomàs Cipriano de Mosquera']='Tomás Cipriano de Mosquera'     
df['Remitente'].loc[df['Remitente'] == 'Ulpiano Barrientos']='Ulpiana Barrientos'
df['Remitente'].loc[df['Remitente'] == 'V- Ortiz']='V. Ortiz'
df['Remitente'].loc[df['Remitente'] == 'Valentin Trujillo']='Valentín Trujillo'
df['Remitente'].loc[df['Remitente'] == 'Vicente Hoyos']='Vicente Gómez'
df['Remitente'].loc[df['Remitente'] == 'Vicente Hurtado y Carmen Hurtado']='Vicente Hurtado'
df['Remitente'].loc[df['Remitente'] == 'Victor Hoyos']='Victor Gómez'
df['Remitente'].loc[df['Remitente'] == 'josé Eusebio Caro']='José Eusebio Caro'
df['Remitente'].loc[df['Remitente'] == 'Vicente V']='Vicente Vargas'
df['Remitente'].loc[df['Remitente'] == 'difícil']='Ilegible'
df['Remitente'].loc[df['Remitente'] == 'dificil']='Ilegible'
df['Remitente'].loc[df['Remitente'] == 'ilegible']='Ilegible'
df['Remitente'].loc[df['Remitente'] == 'no esta']='Ilegible'
df['Remitente'].loc[df['Remitente'] == 'No es claro']='Ilegible'
df['Remitente'].loc[df['Remitente'] == 'por confirmar']='Ilegible'
df['Remitente'].loc[df['Remitente'] == 'sin corresponsal']='Ilegible'
df['Remitente'].loc[df['Remitente'] == 'mutilado']='Sin firma'
df['Destinatario'].loc[df['Destinatario'] == 'aSenadores y Representantes']='Senadores y Representantes'
df['Remitente'].loc[df['Remitente'] == 0]='Ilegible'
df['Destinatario'].loc[df['Destinatario'] == 0]='Ilegible'
df['Destinatario'].loc[df['Destinatario'] == 'Ballén']='Clemente Ballen'
df['Destinatario'].loc[df['Destinatario'] == 'Comadre']='Sin firma'
df['Destinatario'].loc[df['Destinatario'] == 'Hijo P.Acevedo']='Hijo Pedro Acevedo'
df['Destinatario'].loc[df['Destinatario'] == 'José María']='José María Duque Pineda'
df['Destinatario'].loc[df['Destinatario'] == 'Manuel E Acevedo']='Manuel Eusebio Acevedo'
df['Destinatario'].loc[df['Destinatario'] == 'Pedro A. Herran']='Pedro Alcántara Herrán'
df['Destinatario'].loc[df['Destinatario'] == 'Tomas Herrera']='Tomás Herrera'
df['Destinatario'].loc[df['Destinatario'] == "Comadre'"]='Rosela Duque, Ignacia Castro de Duque'
df['Destinatario'].loc[df['Destinatario'] == "Josè Marìa Posada"]='José Marìa Posada'
df['Destinatario'].loc[df['Destinatario'] == "Juan B. Soluaga"]='Juan Bautista Soluaga'
df['Destinatario'].loc[df['Destinatario'] == "Luis hoyos"]='Luis Hoyos'
df['Destinatario'].loc[df['Destinatario'] == "Mariano Ospina"]='Mariano Ospina Rodríguez'
df['Destinatario'].loc[df['Destinatario'] == "María Josefa Acevedo de Gómez"]='Josefa Acevedo de Gómez'
df['Destinatario'].loc[df['Destinatario'] == 'No es claro']='Ilegible'
df['Destinatario'].loc[df['Destinatario'] == 'dificil']='Ilegible'
df['Remitente'].loc[df['Remitente'] == 'A']='A. A.'
df['Remitente'].loc[df['Remitente'] == 'A. A']='A. A.'
df['Remitente'].loc[df['Remitente'] == 'A. A']='A. A.'
df['Remitente'].loc[df['Remitente'] == 'Ag Herrera']='A. Herrera'
df['Remitente'].loc[df['Remitente'] == 'Alfonso']='Alfonso Acevedo'
df['Remitente'].loc[df['Remitente'] == 'Alfonso A.']='Alfonso Acevedo'
df['Remitente'].loc[df['Remitente'] == 'Alfonso A.']='Alfonso Acevedo'
df['Remitente'].loc[df['Remitente'] == 'Ana Marìa Gòmez y Gonima']='Ana María Gomina de Gómez'
df['Remitente'].loc[df['Remitente'] == 'Ana Marìa Gòmina de Gòmez']='Ana María Gomina de Gómez'
df['Remitente'].loc[df['Remitente'] == 'Antonio']='Antonio B.Pineda'
df['Remitente'].loc[df['Remitente'] == 'Antonio Herran']='Antonio Herrán'
df['Remitente'].loc[df['Remitente'] == 'Carlos Gerrero']='Carlos Guerrero'
df['Remitente'].loc[df['Remitente'] == 'Carmen Duque']='Carmen Duque de Duque'
df['Remitente'].loc[df['Remitente'] == 'D. J. G']='D. J. Gómez'
df['Remitente'].loc[df['Remitente'] == 'David Cleves']='David Cleves y Alejandro Duque'
df['Remitente'].loc[df['Remitente'] == 'Dificil']='Ilegible'
df['Remitente'].loc[df['Remitente'] == 'Dificíl']='Ilegible'
df['Remitente'].loc[df['Remitente'] == 'Domingo']='Domingo Caicedo'
df['Remitente'].loc[df['Remitente'] == 'Eusebia']='Eusebia Acevedo Valencia'
df['Remitente'].loc[df['Remitente'] == 'Fernando Caicedo']='Fernando Caicedo Flórez'
df['Remitente'].loc[df['Remitente'] == 'Fernando Caicedo y Camacho']='Fernando Caicedo Flórez'
df['Remitente'].loc[df['Remitente'] == 'Firma sin nombre']='Ilegible'
df['Remitente'].loc[df['Remitente'] == 'Cuenca']='Fulgencio Cuenca'
df['Remitente'].loc[df['Remitente'] == 'Gabrriel María Gómez']='Gabriel María Gómez'
df['Remitente'].loc[df['Remitente'] == 'Guadalupe Uribe']='Guadalupe Uribe de Riaño'
df['Remitente'].loc[df['Remitente'] == 'H Botero']='H. Botero'
df['Remitente'].loc[df['Remitente'] == 'Hermenefildo Fremeda']='Hermenegildo Fremeda'
df['Remitente'].loc[df['Remitente'] == 'Hemer Gómez']='Hermes Gómez'
df['Remitente'].loc[df['Remitente'] == 'H. Cualla']='Higino Cualla'
df['Remitente'].loc[df['Remitente'] == 'Ignacio Gutierrez']='Ignacio Gutiérrez'
df['Remitente'].loc[df['Remitente'] == 'Ignacio']='Ignacio Gómez'
df['Remitente'].loc[df['Remitente'] == 'Iniciales']='Ilegible'
df['Remitente'].loc[df['Remitente'] == 'J. A.']='J. A. Duque'
df['Remitente'].loc[df['Remitente'] == 'J. Gómez']='J. Gómez de la Torre'
df['Remitente'].loc[df['Remitente'] == 'J. M. Galani']='J. M. Galán'
df['Remitente'].loc[df['Remitente'] == 'J. S']='J. Salgar'
df['Remitente'].loc[df['Remitente'] == 'J.J Flores']='J.J. Flores'
df['Remitente'].loc[df['Remitente'] == 'Jacinta']='Jacinto Corredor'
df['Remitente'].loc[df['Remitente'] == 'Joaquìn Mosquera']='Joaquín Mosquera'
df['Remitente'].loc[df['Remitente'] == 'Josefa Marìa Martìnez']='Josefa María Martínez de Aparicio'
df['Remitente'].loc[df['Remitente'] == 'Josefa de Aparicio']='Josefa María Martínez de Aparicio'
df['Remitente'].loc[df['Remitente'] == 'José B. Chiari']='José Bernardino Chiarri'
df['Remitente'].loc[df['Remitente'] == 'José J. Lopez']='José J. López'
df['Remitente'].loc[df['Remitente'] == 'José Maria Chiarri']='José María Chiarri'
df['Remitente'].loc[df['Remitente'] == 'J.M. Gómez']='José María Gómez'
df['Remitente'].loc[df['Remitente'] == 'José de Obaldía ']='José de Obaldía'
df['Remitente'].loc[df['Remitente'] == 'Juan Antonio G']='Juan Antonio Gómez'
df['Remitente'].loc[df['Remitente'] == 'Juan de Ansoategui']='Juan de Azoátegui'
df['Remitente'].loc[df['Remitente'] == 'Luis Sajan']='Luis Sufan'
df['Remitente'].loc[df['Remitente'] == 'M. L. Ramirez']='M. L. Ramírez'
df['Remitente'].loc[df['Remitente'] == 'M. Sánchez C']='M. Sanchez Caicedo'
df['Remitente'].loc[df['Remitente'] == 'Manuel E Acevedo']='Manuel Eusebio Acevedo'
df['Remitente'].loc[df['Remitente'] == 'Manuel Gómez']='Manuel Gómez de Chavez'
df['Remitente'].loc[df['Remitente'] == 'Manuel M. Diaz']='Manuel María Díaz'
df['Remitente'].loc[df['Remitente'] == 'Manuel Narvaez']='Manuel Narváez'
df['Remitente'].loc[df['Remitente'] == 'Margarita B. de Danies']='Margarita Kennedy de Danies'
df['Remitente'].loc[df['Remitente'] == 'Margarita K. de Danies']='Margarita Kennedy de Danies'
df['Remitente'].loc[df['Remitente'] == 'Maria Josefa']='Maria Josefa Valencia'
df['Remitente'].loc[df['Remitente'] == 'Maria Josefa Valencia']='María Josefa Valencia'
df['Remitente'].loc[df['Remitente'] == 'Mariano Sanchez Caicedo y Venancio Ortiz']='Mariano Sanchez Caicedo'
df['Remitente'].loc[df['Remitente'] == 'Miguel D. Grana']='Miguel D. Granados'
df['Remitente'].loc[df['Remitente'] == 'Miguel Gómez']='Miguel Gómez Restrepo'
df['Remitente'].loc[df['Remitente'] == 'Miguel Maria Giraldo']='Miguel María Giraldo'
df['Remitente'].loc[df['Remitente'] == 'N. Ortega']='Narciso Ortega'
df['Remitente'].loc[df['Remitente'] == 'N. González']='Narciso Gonzales'
df['Remitente'].loc[df['Remitente'] == 'No está']='Ilegible'
df['Remitente'].loc[df['Remitente'] == 'No tiene']='Ilegible'
df['Remitente'].loc[df['Remitente'] == 'Noticioso de Ambos Mundos, Número 465']='Ilegible'
df['Remitente'].loc[df['Remitente'] == 'Paula Valencia y Rafaela Arroyo']='Paula Valencia'
df['Remitente'].loc[df['Remitente'] == 'R. Hoyos']='Ramón Hoyos Gómez'
df['Remitente'].loc[df['Remitente'] == 'Sin firma']='Ilegible'
df['Remitente'].loc[df['Remitente'] == 'Urbano Soluaga']='Urbano Soloaga Pineda'
df['Remitente'].loc[df['Remitente'] == 'Urisami']='Urizarri'


#%%
df['Remitente'].loc[df['Remitente'] == 'Ilegible']='Anselmo Pineda'
df['Destinatario'].loc[df['Destinatario'] == 'Ilegible']='Anselmo Pineda'
#df.drop(df.loc[df['Remitente']=='Ilegible'].index, inplace=True)
#%%
#añade una columna con las frecuencisa de lugares
df['freq_ciudad_origen'] = df.groupby('Ciudad_origen')['Ciudad_origen'].transform('count')
#cuentas lugares
total_ubicaciones=df['Ciudad_origen'].nunique()
print('total de ubicaciones:',total_ubicaciones)
print(pd.Series(df['Ciudad_origen'].unique()))
print(df.head(2))

ubicaciones_counts = df.groupby(['Ciudad_origen']).size()
print(ubicaciones_counts)
print(ubicaciones_counts.sum())


#%%
#destinatarios
print(pd.Series(df['Destinatario'].unique()))
df.head(2)

destinatario_counts = df.groupby(['Destinatario']).size()
print(destinatario_counts)
#%%
#remitentes
print(pd.Series(df['Remitente'].unique()))
df.head(2)

remitente_counts = df.groupby(['Remitente']).size()
print(remitente_counts.head(100))

#%%
#subdivide el df original 
subdf=df[['Remitente','Destinatario']]

subdf.rename(columns={"Remitente": "Freq_Remitente"},inplace=True)
subdf.rename(columns={"Destinatario": "Freq_Destinatario"},inplace=True)
#variable con el conteo de valores en el df subdividido
counts_subdf=subdf.apply(pd.value_counts)
#rellenar nulos
subdf= counts_subdf.fillna(0)
#pasa el index a una columna llamada nombres
subdf['Nombres'] = subdf.index
#reset al index
subdf.reset_index(inplace = True)
#condensa el df subdividido
subdf = subdf[['Nombres','Freq_Remitente','Freq_Destinatario']]
#informacion del df subdividido
print('info de marco de datos secundario cuentas',subdf.info())
print('dimenciones marco de datos secundario cuentas',subdf.shape)
print(subdf.describe())
#ordena valores
print(subdf.sort_values(by=['Freq_Remitente','Freq_Destinatario'], ascending=False).head(100))

#añadir frecuencias como columna al df princiapl
df=pd.merge(df,subdf[['Nombres','Freq_Remitente','Freq_Destinatario']],left_on=['Remitente'], right_on = ['Nombres'], how = 'left') 

#df['frecuencia'] = pd.DataFrame(np.sort(df.values, 1)).groupby(['Remitente','Destinatario'])['Remitente'].transform('size')

print(df.head(10))
#%%
#conteo de valores de la columna seleccionada
counts1 = df['Remitente'].value_counts()
counts2= df['Destinatario'].value_counts()
#señala que estilo de grafiacas usar
plt.style.use('seaborn-dark')
#señala el tamaño de la figura base para la graficas
plt.figure(figsize=(100,50))
#señala el numero de filas, columnas y el lugar seleccionado para la primera grafica
plt.subplot(1,2,1)
#graficar el conteo de la primera columna
counts1.plot.bar(color = 'blue')
#edicion de la grafica
plt.title('Distribución variables categóricas: remitentes',fontsize=80,color='navy')
plt.xlabel('Corresponsales',fontsize=90,color='darkolivegreen')
plt.ylabel('Conteo de contactos',fontsize=90,color='darkolivegreen')
#indica en x los limites de izquierda a derecha para ser graficados
plt.xlim(0,25)
plt.xticks(rotation=70,fontsize=55)
plt.yticks(fontsize=55)
#señala el numero de filas, columnas y el lugar seleccionado para la primera grafica
plt.subplot(1,2,2)
counts2.plot.bar(color = 'red')
plt.title('Distribución variables categóricas: destinatarios',fontsize=80,color='navy')
plt.xlabel('Corresponsales',fontsize=90,color='darkolivegreen')
plt.ylabel('Conteo de contactos',fontsize=90,color='darkolivegreen')
plt.xlim(0,10)
plt.yticks(fontsize=55)
plt.xticks(rotation=65,fontsize=55)
#optimiza el espacio
plt.tight_layout()
plt.show()
#salva la imagen
plt.savefig('corresponsales_conteo.png',dpi=450)
#com´puta la media y mediana
print('Media: ', '\n',subdf.mean())
print('Mediana: ', '\n',subdf.median())
#This is within acceptable range for many purposes 
#(any analyses start to worry when skew reaches somewhere between 0.80-2.0)
print('-> Asimetría en columna Remitente: ',skew(subdf['Freq_Remitente']), ' valores máximos aceptables entre 0.80-2.0')
print('-> Asimetría en columna Destinatario: ',skew(subdf['Freq_Destinatario']), ' valores máximos aceptables entre 0.80-2.0')

#%%


num_cols_remi=['Freq_Remitente']
num_cols_desti=['Freq_Destinatario']

#identifica los valores del 3 y 1 percentil
remi_q75, remi_q25 = np.percentile(subdf[num_cols_remi], [75,25])
desti_q75, desti_q25 = np.percentile(subdf[num_cols_desti], [75,25])
print('remitente 75% ',remi_q75)
print('remitente 25% ',remi_q25)
print('destinatario 75% ',desti_q75)
print('destinatario 25% ',desti_q25)
iqr_remi = remi_q75 - remi_q25
iqr_desti = desti_q75 - desti_q25
print('iqr remitente ',iqr_remi)
print('iqr destinatario ',iqr_desti)
#computa los rangos para identificar valores atipicos
outlier_remi_below= (remi_q25-1.5*iqr_remi)
outlier_remi_above=(remi_q75+1.5*iqr_remi)
outlier_desti_below= (desti_q25-1.5*iqr_remi)
outlier_desti_above=(desti_q75+1.5*iqr_remi)
print('valores atípicos en remitente por debajo de: ',outlier_remi_below)
print('valores atípicos en remitente por encima de: ',outlier_remi_above)
print('valores atípicos en destinatario por debajo de: ',outlier_desti_below)
print('valores atípicos en destinatario por encima de: ',outlier_desti_above)


medium_remi_=subdf[np.logical_and(subdf[num_cols_remi]>outlier_remi_below,subdf[num_cols_remi]<outlier_remi_above)].drop(['Nombres','Freq_Destinatario'], axis=1)
medium_desti_=subdf[np.logical_and(subdf[num_cols_desti]>outlier_desti_below,subdf[num_cols_desti]<outlier_desti_above)].drop(['Nombres','Freq_Remitente'], axis=1)

subdf.drop(['Nombres'],axis=1)
remi_desti_count=list(subdf.count())
print('Antes de filtrado cantidad de remitentes: {}, destinatarios: {} '.format(remi_desti_count[0],remi_desti_count[1]))
medium_remi_count=list(medium_remi_.count())
print('Cantidad de remitentes después de filtrado: {}'.format(medium_remi_count[0]))
porcentaje_detotal_remi=(medium_remi_.count()*100/subdf.count()[1]).round(2)
porcentaje_detotal_remi_=list(porcentaje_detotal_remi)
reduccion_remi=list((100-porcentaje_detotal_remi).round(2))
print('En caso de remitentes después de filtrado queda el {}% disminución de {}%'.format(str(porcentaje_detotal_remi_[0]), str(reduccion_remi[0])))
medium_desti_count=list(medium_desti_.count())
print('Cantidad de destinatarios después de filtrado: {}'.format(medium_desti_count[0]))

porcentaje_detotal_desti=(medium_desti_.count()*100/subdf.count()[1]).round(2)
porcentaje_detotal_desti_=list(porcentaje_detotal_desti)
reduccion_desti=list((100-porcentaje_detotal_desti).round(2))
print('En caso de destinatarios después de filtrado queda el {}% disminución de {}%'.format(str(porcentaje_detotal_desti_[0]), str(reduccion_desti[0])))


# Display the box plots on 3 separate rows and 1 column
plt.style.use('seaborn-dark')
fig, axes = plt.subplots(nrows=2, ncols=1)
fig = plt.figure(figsize=(90,50))
medium_remi=subdf[np.logical_and(subdf[num_cols_remi]>outlier_remi_below,subdf[num_cols_remi]<outlier_remi_above)].plot(ax=axes[0],y='Remitente',kind='box')
medium_desti=subdf[np.logical_and(subdf[num_cols_desti]>outlier_desti_below,subdf[num_cols_desti]<outlier_desti_above)].plot(ax=axes[1],y='Destinatario',kind='box')
# Display the plot
axes[0].set_title('Gráficas Box-Whisker sin valores atípicos',color='navy')
axes[0].set_ylabel('Media remitente',color='darkolivegreen')
axes[1].set_ylabel('Media destinatario',color='darkolivegreen')
plt.show()
medium_remi_.dropna(inplace=True)
medium_desti_.dropna(inplace=True)
print('Asimetría: ',skew(medium_remi_['Freq_Remitente']), ' valores aceptables entre 0.80-2.0')
print('Asimetría: ',skew(medium_desti_['Freq_Destinatario']), ' valores aceptables entre 0.80-2.0')
#%%
plt.style.use('ggplot')
plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
plt.scatter(subdf['Remitente'],subdf['Destinatario'],s=subdf['Remitente'],c='blue')



# Strings
xlab = 'Contactos como remitentes'
ylab = 'Contactos como destinatarios'
title = 'Distribución contactos epistolares, magnitud: contactos en calidad de remitentes'
# etiquetas axis 

plt.xlabel(xlab,color='darkolivegreen')
plt.ylabel(ylab,color='darkolivegreen')

# titulo y posicion de textos
plt.title(title,color='navy',fontsize=12)
#plt.text(95, -110, 'Josefa Acevedo de Gómez')
#plt.text(19, 110, 'Ana María Danies Pineda')
#plt.text(110, 90, 'Juan Nepomuceno Duque')
#plt.text(100, 2500, 'Anselmo Pineda')
#plt.text(185, 5, 'Antonio María Gutiérrez')
plt.subplot(2,1,2)
plt.scatter(subdf['Remitente'],subdf['Destinatario'],s=subdf['Destinatario'],c='red')
# Strings
xlab = 'Contactos como remitentes'
ylab = 'Contactos como destinatarios'
title = 'Distribución contactos epistolares, magnitud: contactos en calidad de destinatario'
# etiquetas

plt.xlabel(xlab,color='darkolivegreen')
plt.ylabel(ylab,color='darkolivegreen')

# titulo y posicion de textos
plt.title(title,color='navy',fontsize=12)
#plt.text(95, -110, 'Josefa Acevedo de Gómez')
#plt.text(19, 110, 'Ana María Danies Pineda')
#plt.text(110, 90, 'Juan Nepomuceno Duque')
#plt.text(100, 2500, 'Anselmo Pineda')
#plt.text(180, 5, 'Antonio María Gutiérrez')
# Show plot

plt.show()
print('_________Pauta de Jacob Cohen_______')
print('| # | Correlación |  Interpretación|')
print('|---|-------------|----------------|')
print('| 1 |  0.0 - 0.1  | Insignificante |')
print('| 2 |  0.1 - 0.3  |     Pequeña    |')
print('| 3 |  0.3 - 0.5  |      Media     |')
print('| 4 |    0.5 +    |      Grande    |')
print(subdf.corr())

sns.jointplot(subdf['Remitente'],subdf['Destinatario'], data=subdf)
plt.show()

#%%
time_format = '%Y'
anos=pd.to_datetime(list(df['Año']), format=time_format)
anos_unicos=pd.to_datetime(list(df['Año']), format=time_format).year.unique()
df1=df.set_index(anos)
#print(anos_unicos)


df1=df1[['Remitente','Destinatario','Año']]
#print(df1['Remitente'].resample("Y").count())

df1['Remitente'].resample("Y").count().plot()
xlab = 'Años'
ylab = 'Número de contactos'
title = 'Número de contactos por años en el epistolario Pineda'
plt.xlabel(xlab,color='darkolivegreen')
plt.ylabel(ylab,color='darkolivegreen')
plt.title(title,color='navy')
plt.show()
plt.clf()

df1['Remitente']['1806':'1880'].resample("Y").count().plot()
xlab = 'Lapso en años'
ylab = 'Número de contactos'
title = 'Número de contactos en época más activa'
plt.xlabel(xlab,color='darkolivegreen')
plt.ylabel(ylab,color='darkolivegreen')
plt.title(title,color='navy')
plt.show()


print(len(df1.loc['1858','Remitente'])  )
print(df1.loc['1858','Remitente'])

anual_remitente=df1.loc['1829':'1866']
anual_remitente=anual_remitente['Remitente'].resample("A").count()
anual_remitente['crecimiento'] = anual_remitente.pct_change() * 100
print(' ')
print('Tasa de crecimiento anual (%)')
anual_remitente['crecimiento'] 

#%%
counts=df1['Remitente']['1806':'1880'].resample("Y").count()
counts=counts.reset_index()

p = figure(x_axis_type='datetime',x_axis_label='Años', y_axis_label='Número de contactos')
p.line(counts['index'],counts['Remitente'])
p.circle(counts['index'],counts['Remitente'],size=10,fill_color='grey', alpha=0.3, line_color=None,
         hover_fill_color='firebrick', hover_alpha=0.5,
         hover_line_color='white')
# Create a HoverTool: hover
hover = HoverTool(tooltips=None,mode='vline')

# Add the hover tool to the figure p
p.add_tools(hover)

output_file("contactos_interactivo.html")
show(p)
#%%

time_format = '%Y'
df['Año']=pd.to_datetime(df['Año'], format=time_format)

# Instantiate a new Graph: G
G = nx.Graph()

# Add nodes from each of the partitions
G.add_nodes_from(df['Remitente'], bipartite='Remitente')
G.add_nodes_from(df['Destinatario'],bipartite='Destinatario')

print('Número de nodos en la red de Anselmo Pineda: ',len(G.nodes()))

for r, d in df.iterrows():
   G.add_edge(d['Remitente'], d['Destinatario'],date=d['Año'])
   
remitente_nodes = [n for n, d in G.nodes(data=True) if d['bipartite'] == 'Remitente']

# Create the students nodes projection as a graph: G_students
G_remitentes = nx.bipartite.projected_graph(G, nodes=remitente_nodes)

dcs_remitentes = nx.degree_centrality(G_remitentes)


dcs_remitentes_importantes={key: value for key, value in dcs_remitentes.items() if key == 'Anselmo Pineda'}
print(dcs_remitentes_importantes )

plt.hist(list(dcs_remitentes.values()))
plt.yscale('log')
plt.title('Degree centrality distribution of remitentes partition')
plt.show()

#%%
#The people most popular or more liked usually are the ones who have more friends. Degree 
#centrality is a measure of the number of connections a particular node has in the network. 
#It is based on the fact that important nodes have many connections

# Get the student partition's nodes: student_nodes
destinatario_nodes = [n for n, d in G.nodes(data=True) if d['bipartite'] == 'Destinatario']

# Create the students nodes projection as a graph: G_students
G_destinatario = nx.bipartite.projected_graph(G, nodes=destinatario_nodes)
dcs_destinatario = nx.degree_centrality(G_destinatario)
#dcs_destinatarios_importantes={key: value for key, value in dcs_remitentes.items() if value > 0.94}
dcs_destinatarios_importantes=pd.DataFrame(list(dcs_destinatario.items()),columns=['Nombre','Grado'])
print(dcs_destinatarios_importantes.sort_values('Grado',ascending=False).head(30))
#%%
dcs_overal=nx.degree_centrality(G)
#una forma de obtener el maximo valor de un dicccionario
#max_key = max(dcs_overal, key=lambda k: dcs_overal[k])
#print(max_key, max(dcs_overal.values()))

#dcs_overal_sorted={k: v for k, v in sorted(dcs_overal.items(), key=lambda item: item[1])}
#print(dcs_overal_sorted)
#una forma de organizar el diccionario
#top_25=sorted(dcs_overal.items(), key=lambda x: x[1], reverse=True)[:25]
top_25_dcs=pd.DataFrame(list(dcs_overal.items()),columns=['Nombre','Grado'])
print(top_25_dcs.sort_values('Grado',ascending=False).head(30))

#%%
#It quantifies how many times a particular node comes in the shortest chosen 
#path between two other nodes. The nodes with high betweenness centrality play 
#a significant role in the communication/information flow within the network
betweenness_centrality=nx.betweenness_centrality(G,normalized=True, endpoints=True)
#max_key_betw = max(betweenness_centrality, key=lambda k: betweenness_centrality[k])
#print(max_key_betw, max(betweenness_centrality.values()))

#top_25_betw=sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:25]
top_25_betw=pd.DataFrame(list(betweenness_centrality.items()),columns=['Nombre','Centralidad'])
top_25_betw_=top_25_betw.sort_values('Centralidad',ascending=False).head(30)
print(top_25_betw_)

#plt.hist(list(betweenness_centrality.values()))
plt.bar('Nombre','Centralidad',data=top_25_betw_,color='green')
plt.xlim(0,25)
plt.yticks(fontsize=11)
plt.xticks(rotation=75,fontsize=9)
#optimiza el espacio
plt.tight_layout()
plt.show()

#know the labels of the nodes with the highest betweenness centrality using
#print(sorted(betweenness_centrality, key=betweenness_centrality.get, reverse=True)[:7])
#%%

ultimoano=datetime(1880, 1,1)
yearone=datetime(1806, 1,1)

ano_actual=yearone
td=timedelta(days=364)

n_contactos=[]


while ano_actual < ultimoano:
    if ano_actual.day == 1:
        ano_actual
    edges = [(u, v, d) for u, v, d in G.edges(data=True) if d['date'] >= ano_actual and d['date'] < ano_actual + td]
    n_contactos.append(len(edges))
    
    ano_actual += td 
   
    
plt.plot(n_contactos)  
plt.xlabel('Lapso en años',color='darkolivegreen')
plt.ylabel('Numero de indviduos contactados',color='darkolivegreen')
plt.show()

#%%
print(nx.info(G))
#Degree of a node defines the number of connections a node has
#%%


# find node with largest degree
node_and_degree = G.degree()
(largest_hub, degree) = sorted(node_and_degree, key=itemgetter(1))[-1]
# Create the CircosPlot object: c
ego_graph = nx.ego_graph(G, largest_hub,center=True)


pos = nx.spring_layout(ego_graph)

#n_color = np.asarray([degrees[n] for n in nodes])
node_color = [20000.0 * G.degree(v) for v in ego_graph]
node_size =  [v * 10000 for v in betweenness_centrality.values()]

plt.figure(figsize=(20,20))
# Draw the CircosPlot object to the screen
nx.draw(ego_graph, pos, node_size=node_size, node_color=node_color, cmap='prism',alpha=0.8)
plt.axis('off')
plt.show()
#%%

graph_sub = nx.Graph()
#graph_sub.add_nodes_from(G.nodes(data=True))
graph_sub.add_edges_from([(u, v, d) for u, v, d in G.edges(data=True) if d['date'] > datetime(1869, 1,1) and d['date'] < datetime(1879, 1,1)])
node_and_degree_sub = graph_sub.degree()
(largest_hub_sub, degree_sub) = sorted(node_and_degree_sub, key=itemgetter(1))[-1]
for n, d in graph_sub.nodes(data=True):
    graph_sub.nodes[n]['dc'] = node_and_degree_sub[n]

print(largest_hub_sub)
print(nx.info(graph_sub))
#%%
betweenness_centrality_sub = nx.betweenness_centrality(graph_sub, normalized=True, endpoints=True)
top_25_betw_sub=pd.DataFrame(list(betweenness_centrality_sub.items()),columns=['Nombre','Centralidad'])
top_25_betw_sub_=top_25_betw_sub.sort_values('Centralidad',ascending=False).head(30)
print(top_25_betw_sub_)
#%%
#esta es para graficar la red a comienzos de la vida de pineda que no es egocentrada
node_color_sub = [200000 * graph_sub.degree(v) for v in graph_sub]
node_size_sub =  [v * 10000 for v in betweenness_centrality_sub.values()]

pos = nx.spring_layout(graph_sub)
plt.figure(figsize=(20,20))
nx.draw_networkx(graph_sub,pos,node_size=node_size_sub, node_color=node_color_sub, cmap='prism')
plt.axis('off')
plt.show()

#%%
betweenness_centrality_sub = nx.betweenness_centrality(graph_sub, normalized=True, endpoints=True)
top_25_betw_sub=pd.DataFrame(list(betweenness_centrality_sub.items()),columns=['Nombre','Centralidad'])
top_25_betw_sub_=top_25_betw_sub.sort_values('Centralidad',ascending=False).head(30)
print(top_25_betw_sub_)
#%%
# find node with largest degree
node_and_degree_sub = graph_sub.degree()
(largest_hub_sub, degree_sub) = sorted(node_and_degree_sub, key=itemgetter(1))[-1]
 # Create ego graph of main hub
ego_graph_sub = nx.ego_graph(graph_sub, largest_hub_sub, center=True)
pos = nx.spring_layout(ego_graph_sub)

node_color_sub_ego = [200000 * graph_sub.degree(v) for v in ego_graph_sub]
node_size_sub_ego =  [v * 10000 for v in betweenness_centrality_sub.values()]

plt.figure(figsize=(20,20))
# Draw the CircosPlot object to the screen
nx.draw(ego_graph_sub, pos,node_size=node_size_sub_ego, node_color=node_color_sub_ego, cmap='prism',with_labels=True,font_size=9,alpha=0.9)
plt.axis('off')
plt.show()
#%%
#extraer las transcripciones
df['transcripcion_1_descr'] = df['Descripción'].str.extract(r"'([^']*)'")
df['Descripción']=df['Descripción'].str.replace(r"'([^']*)'",'')
df['transcripcion_2_desc_rela_bibl'] = df['Descripcion Relacion biblioteca'].str.extract(r"'([^']*)'")
df['Descripcion Relacion biblioteca']=df['Descripcion Relacion biblioteca'].str.replace(r"'([^']*)'",'')
print('numero de transcripciones:',df['transcripcion_1_descr'].notnull().value_counts()[1]+df['transcripcion_2_desc_rela_bibl'].notnull().value_counts()[1])
#salvar la tabla completa para reemplazar las comillas dobles por sencillas
df.head()


#%%
#proporcion de epistolas relativos a colecconismo o temas relacionados con trafico de documentos
palabras=['boletines|boletin|ordenanzas|gramática|gramatica|ciencia|geografia|historia|diccionarios|diccionario|arimética|aritmetica|Ymprenta|pliegos|cuadernito|memorias|memoria|copia|texto|ejemplares|ejemplar|cartillas|escuelas|educación|folleto|folletos|compilaciones|cartilla popular|imprenta|librito|cuaderno|manuscrito|manuscritos|gacetas|gaceta|biblioteca|colección|coleccion|publicación|impresos|impreso|tomos|tomo|artículo|articulo|papeles|papel|hojas|hoja|diario|cuaderno|borradores|catálogo|libro|libros|encuadernación|encuadernacion|archivo|volúmenes|volúmen|volumen|periódicos|periódico|periodicos|periodico|documentos|documento|legajos|legajo|obras|obra|obrita']

result = df['transcripcion_1_descr'].str.contains(pat='boletines|boletin|ordenanzas|gramática|gramatica|ciencia|geografia|historia|diccionarios|diccionario|arimética|aritmetica|Ymprenta|pliegos|cuadernito|memorias|memoria|copia|texto|ejemplares|ejemplar|cartillas|escuelas|educación|folleto|folletoscompilaciones|cartilla popular|imprenta|librito|cuaderno|manuscrito|manuscritos|cartas|gacetas|gaceta|biblioteca|colección|coleccion|publicación|impresos|impreso|tomos|tomo|artículo|articulo|papeles|papel|hojas|hoja|diario|cuaderno|borradores|catálogo|libro|libros|encuadernación|archivo|volúmenes|volúmen|volumen|periódicos|periódico|periodicos|periodico|documentos|documento|legajos|legajo|obras|obra|obrita',case=False)
result = df['transcripcion_2_desc_rela_bibl'].str.contains(pat='boletines|boletin|ordenanzas|gramática|gramatica|ciencia|geografia|historia|diccionarios|diccionario|arimética|aritmetica|Ymprenta|pliegos|cuadernito|memorias|memoria|copia|texto|ejemplares|ejemplar|cartillas|escuelas|educación|folleto|folletoscompilaciones|cartilla popular|imprenta|librito|cuaderno|manuscrito|manuscritos|cartas|gacetas|gaceta|biblioteca|colección|coleccion|publicación|impresos|impreso|tomos|tomo|artículo|articulo|papeles|papel|hojas|hoja|diario|cuaderno|borradores|catálogo|libro|libros|encuadernación|archivo|volúmenes|volúmen|volumen|periódicos|periódico|periodicos|periodico|documentos|documento|legajos|legajo|obras|obra|obrita',case=False)
result |= df['Descripcion Relacion biblioteca'].str.contains(pat='boletines|boletin|ordenanzas|gramática|gramatica|ciencia|geografia|historia|diccionarios|diccionario|arimética|aritmetica|Ymprenta|pliegos|cuadernito|memorias|memoria|copia|texto|ejemplares|ejemplar|cartillas|escuelas|educación|folleto|folletoscompilaciones|cartilla popular|imprenta|librito|cuaderno|manuscrito|manuscritos|gacetas|gaceta|biblioteca|colección|coleccion|publicación|impresos|impreso|tomos|tomo|artículo|articulo|papeles|papel|hojas|hoja|diario|cuaderno|borradores|catálogo|libro|libros|encuadernación|archivo|volúmenes|volúmen|volumen|periódicos|periódico|periodicos|periodico|documentos|documento|legajos|legajo|obras|obra|obrita',case=False)
result |= df['Descripción'].str.contains(pat='boletines|boletin|ordenanzas|gramática|gramatica|ciencia|geografia|historia|diccionarios|diccionario|arimética|aritmetica|Ymprenta|pliegos|cuadernito|memorias|memoria|copia|texto|ejemplares|ejemplar|cartillas|escuelas|educación|folleto|folletoscompilaciones|cartilla popular|imprenta|librito|cuaderno|manuscrito|manuscritos|gacetas|gaceta|biblioteca|colección|coleccion|publicación|impresos|impreso|tomos|tomo|artículo|articulo|papeles|papel|hojas|hoja|diario|cuaderno|borradores|catálogo|libro|libros|encuadernación|archivo|volúmenes|volúmen|volumen|periódicos|periódico|periodicos|periodico|documentos|documento|legajos|legajo|obras|obra|obrita',case=False)
print(result)

print("Proporción de cartas relativas al tráfico de documentos:", np.sum(result) / df.shape[0])
#print(np.sum(df['Relacion biblioteca'])/df.shape[0])

#el resultado sugiere que 793 cartas se refieren a documentos
#%%
df.to_excel('BBDD_cleaned_transc.xlsx',sheet_name='Pineda',header=True)

#%%
df=pd.read_excel('BBDD_cleaned_transc.xlsx')
print(df.columns)

#%%
#nominatim usa openstreetmap
#barra de progreso
tqdm.pandas()

geolocator = Nominatim(user_agent="UbicacionesPineda")
#para evitar el timeout error
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
#extrae valores unicos para hacer mas eficiente la busqueda
ubicaciones = df['Ciudad_origen'].unique()
#elimina la entrada None type 
index = np.argwhere(ubicaciones==None)
ubicaciones = np.delete(ubicaciones, index)
#geocodifica los valores del array de valores unicos

   #usa progress_applay para mostrar progreso
d = dict(zip(ubicaciones, pd.Series(ubicaciones).progress_apply(geocode)))


#mapea los resultados con las coincidencias con los valores unicos en el diccionario de arriba
df['location']  = df['Ciudad_origen'].map(d)

#limpia las coordenadas en sus respectivas columnas
df['coordinates']=df['location'].str[1]
df['coordinates'] = df['coordinates'].astype(str)
df['coordinates']=df['coordinates'].str.replace('(', ' ')
df['coordinates']=df['coordinates'].str.replace(')', ' ')

df['Latitud'], df['Longitud'] = df['coordinates'].str.split(',', 1).str

df['Latitud']=df['Latitud'].replace('nan',0)
df['Latitud']=df['Latitud'].replace('None',0)
df['Longitud']=df['Longitud'].replace('None',0)
df['Longitud']=df['Longitud'].fillna(0)

df['Latitud']=df['Latitud'].astype(float)
df['Longitud']=df['Longitud'].astype(float)
#si fuera un tuple pero comvertí a string no es la manera
# extract latitude to a n1ew column: lat
#df['lat'] = [loc[0] for loc in df.coordinates]

# extract longitude to a new column: lng
#df['lng'] = [loc[1] for loc in df.coordinates]


print(df.head())
#%%
df.to_excel('BBDD_cleaned_geo_transc_.xlsx',sheet_name='Pineda',header=True)
#df.to_csv('BBDD_cleaned_geo.csv',sep=',')
#%%

df=pd.read_excel('BBDD_cleaned_geo_transc_.xlsx',sheet_name='Pineda',index_col=0)
print(df.head())
#%%

#numero de cartas por lugar
ubicaciones_counts = df.groupby(['Ciudad_origen']).size()
#print(ubicaciones_counts.head(5))
counts_df = ubicaciones_counts.to_frame()
counts_df.reset_index(inplace=True)
counts_df.columns = ['Ciudad_origen', '#_epistolas']

# Merge permits_by_district and counts_df
df_and_counts = pd.merge(df,counts_df, on = 'Ciudad_origen')


print(df_and_counts.shape)
print(counts_df.shape)
print(counts_df.sort_values(by='#_epistolas',ascending=False).head(10))


#%%
#solo las coordenadas sin tipo punto para la libraria basemap para mostrar
#puntos con atributo en spyder
df_and_counts['Lat'] = df.apply(lambda x: float(x.Latitud),  axis=1)
df_and_counts['Lon'] = df.apply(lambda x: float(x.Longitud), axis=1)
#geometria de punto para usar con los mapas en jupyter con geopandas
df_and_counts['geometry'] = df.apply(lambda x: Point(float(x.Longitud), float(x.Latitud)), axis=1)
print(df_and_counts.columns)
#%%
df_and_counts_2=df_and_counts[['Ciudad_origen','#_epistolas', 'Lat', 'Lon']]
counts_df_2=df_and_counts_2.groupby(['Ciudad_origen']).agg(lambda x: stats.mode(x)[0][0]).reset_index()
print(counts_df_2.columns)
print(counts_df_2.head())
print(counts_df_2.dtypes)
print(counts_df_2.shape)
print(counts_df_2.sort_values('#_epistolas',ascending=False).head(10))


#%%
#numero de epistolas por ubicacion 
#counts_df_2=df_and_counts.groupby(['Ciudad_origen']).agg({'#_epistolas': 'max', 'Lat': 'max','Lon':'max'}).reset_index()
#counts_df_2=counts_df_2[['Ciudad_origen','#_epistolas','Lat','Lon']]
#counts_df_2['Lon']=counts_df_2['Lon'].astype('float64')
#counts_df_2['Lat']=counts_df_2['Lat'].astype('float64')
#counts_df_2['#_epistolas']=counts_df_2['#_epistolas'].astype('int')
counts_df_2['numero_cartas']=counts_df_2['#_epistolas'].astype('str')
print(counts_df_2.columns)
print(counts_df_2.head())
print(counts_df_2.dtypes)
print(counts_df_2.shape)
#%%
#esta parte solo aplica en jupyter
df_geo = gpd.GeoDataFrame(df_and_counts,crs =4326, geometry = df_and_counts.geometry)

#print(df_geo.columns)
#df_geo=df_geo.to_crs({'init': 'epsg:4326'})

print(df_geo.crs)
print(type(df_geo))

#cartografia divisiones administrativas del mundo
admin_world=gpd.read_file('admin00.shp')
admin_world=admin_world.to_crs({'init': 'epsg:4326'})
print(admin_world.crs)
#filtro a divisiones administrativas de interes
admin_world = admin_world.loc[(admin_world.CNTRY_NAME == "Colombia") | (admin_world.CNTRY_NAME == "Panama")| (admin_world.CNTRY_NAME == "Jamaica")| (admin_world.CNTRY_NAME == "Venezuela")| (admin_world.CNTRY_NAME == "United Kingdom")| (admin_world.CNTRY_NAME == "Ecuador")| (admin_world.CNTRY_NAME == "Peru")| (admin_world.CNTRY_NAME == "France")]

#admin_world.geometry=admin_world.geometry.to_crs(crs=3116)
#operacion espacial
pais_punto = gpd.sjoin(df_geo, admin_world, op = "within")
print(pais_punto.head())
print(pais_punto.crs)

admin_world_polygon=admin_world.plot(column = 'CNTRY_NAME', cmap = 'Dark2', legend=True)


pais_punto.plot(ax=admin_world_polygon, color = 'magenta',markersize=4,marker=".")
plt.xlabel('longitud')
plt.ylabel('latitud')
plt.title('Mapa correspondencia Pineda por país')
plt.show()


#%%
#group by
pais_agrupado = pais_punto[['CNTRY_NAME','#_epistolas']].groupby('CNTRY_NAME')
# Aggregate the grouped data and count within each polygon to display later
pais_agrupado_agregado=pais_agrupado.agg('count').sort_values(by = '#_epistolas', ascending = False)
pais_agrupado_agregado.reset_index(inplace=True)
print(pais_agrupado_agregado)

#%%
#filtro solo divisiones admin en colombia
admin_colombia = admin_world.loc[(admin_world.CNTRY_NAME == "Colombia")]
# join espacial departamentos con puntos de epistolas
departamento_punto = gpd.sjoin(df_geo, admin_colombia, op = "within")
admin_colombia_polygon=admin_colombia.plot(column = 'ADMIN_NAME', cmap = 'Dark2')
departamento_punto.plot(ax=admin_colombia_polygon, color = 'indigo',markersize=5,marker='o')
plt.xlabel('longitud')
plt.ylabel('latitud')
plt.title('Mapa correspondencia Pineda por departamento en Colombia')
plt.show()

#%%
#group by
departamento_agrupado = departamento_punto[['ADMIN_NAME','#_epistolas']].groupby('ADMIN_NAME')
#departamento_punto.groupby(['ADMIN_NAME']).size()
# Aggregate the grouped data and count within each polygon to display later
departamento_agrupado_agregado=departamento_agrupado.agg('count').sort_values(by = '#_epistolas', ascending = False)
departamento_agrupado_agregado.reset_index(inplace=True)

#merge df con agregado y df con division administrativa de colombia
departamento_con_agregado=pd.merge(admin_colombia,departamento_agrupado_agregado,on='ADMIN_NAME')
departamento_con_agregado.plot(column = '#_epistolas', cmap = 'BuGn',edgecolor='black',legend=True)
plt.xlabel('longitud')
plt.ylabel('latitud')
plt.title('Mapa densidad por departamento en Colombia')
plt.show()

#es necesario guardar el dataframe on merge a un geojson, este metodo es el unico
#que funcionó
with open('departamento_con_agregado.geojson', 'w') as f:
    f.write(departamento_con_agregado.to_json())
#mostrar el df con los valores agregados
departamento_agrupado_agregado
#%%
# Set up the US bounding box
us_boundingbox = [-83.7,-5.15,-63.49,13.78] 
#plt.figure(figsize=(30,30))
# Set up the Basemap object
m = Basemap(width=12000000,height=9000000,resolution='f',
            llcrnrlon = us_boundingbox[0],
            llcrnrlat = us_boundingbox[1],
            urcrnrlon = us_boundingbox[2],
            urcrnrlat = us_boundingbox[3],
            projection='merc')

# Draw continents in white,
# countries in black and coastlines and the states in gray
m.fillcontinents(color='gray', zorder= 0)
m.drawcoastlines(color='gray')
m.drawcountries(color='black')
m.drawstates(color='gray')
m.drawlsmask(land_color='0.9', ocean_color='aquamarine')


lon = [x for x in counts_df_2['Lon']]
lat = [x for x in counts_df_2['Lat']]
#epistolas= [x for x in counts_df['numero_cartas']]
size = [x * 1.5 for x in counts_df['#_epistolas']]
color= [x * 20000.0 for x in counts_df['#_epistolas']]

# Draw the points and show the plot
m.scatter(lon, lat,c=color, cmap = 'jet', s=size, latlon = True, alpha = 0.9)
plt.xlabel('longitud')
plt.ylabel('latitud')
plt.title('Mapa de epistolario Pineda agregado')

plt.savefig('Colombia_puntos_agregados.png', dpi = 300)           
plt.show()

#%%

departamento_punto['Relacion biblioteca'] = departamento_punto['Relacion biblioteca'].astype('category')

departamento_punto.plot()
plt.show()
#%%
departamento_punto_latlon=departamento_punto[['Lat', 'Lon']]
print(departamento_punto_latlon.columns)
#%%
departamento_punto['Lat_scaled']=whiten(departamento_punto['Lat'])
departamento_punto['Lon_scaled']=whiten(departamento_punto['Lon'])
#%%
print(departamento_punto.columns)
print(departamento_punto.head())
#%%
distortions = []
num_clusters = range(1, 7)

# Create a list of distortions from the kmeans function
for i in num_clusters:
    cluster_centers, distortion = kmeans(departamento_punto[['Lon','Lat']], i)
    distortions.append(distortion)

# Create a data frame with two lists - num_clusters, distortions
elbow_plot = pd.DataFrame({'num_clusters': num_clusters, 'distortions': distortions})

# Creat a line plot of num_clusters and distortions
sns.lineplot(x='num_clusters', y='distortions', data = elbow_plot)
plt.xticks(num_clusters)
plt.show()
#%%
# Compute cluster centers
#primero x luego y
cluster_centers, distortion = kmeans(departamento_punto[['Lon','Lat']], 4)

# Assign cluster labels
departamento_punto['cluster_labels'], distortion_list = vq(departamento_punto[['Lon','Lat']], cluster_centers)


# Plot the points with seaborn
sns.scatterplot(x='Lon', y='Lat', hue='cluster_labels', data=departamento_punto)
plt.show()
#%%

# Draw continents in white,
# countries in black and coastlines and the states in gray
m.fillcontinents(color='gray', zorder= 0)
m.drawcountries(color='black')
m.drawstates(color='gray')
m.drawlsmask( ocean_color='midnightblue')


lon_clusters = [x for x in departamento_punto['Lon']]
lat_clusters = [x for x in departamento_punto['Lat']]
color_clusters= [x for x in departamento_punto['cluster_labels']]

# Draw the points and show the plot
m.scatter(lon_clusters, lat_clusters,c=color_clusters, cmap = 'brg', s=10, latlon = True, alpha = 0.8)
plt.xlabel('longitud')
plt.ylabel('latitud')
plt.title('Mapa de agrupaciones en epistolario Pineda')
plt.show()

#%%
departamento_punto.to_excel('departamento_punto.xlsx',sheet_name='Pineda',header=True)

#%%

departamento_punto=pd.read_excel('departamento_punto.xlsx',sheet_name='Pineda',index_col=0)
#%%
departamento_punto=departamento_punto.drop(['Tomo', 'Folio', 'Descripción', 'Ciudad destino', 'Destinatario','Dia', 'Mes', 'Tipo relacion AMIGO',
       'Tipo relacion FAMILIA', 'Tipo relacionOtra', 'Colonizacion','Nombres','Freq_Destinatario',
       'transcripcion_1_descr', 'transcripcion_2_desc_rela_bibl', 'location',
       'coordinates', 'Latitud', 'Longitud','geometry', 'index_right', 'FIPS_ADMIN', 'GMI_ADMIN', 'ADMIN_NAME',
       'FIPS_CNTRY', 'GMI_CNTRY', 'CNTRY_NAME', 'POP_ADMIN', 'TYPE_ENG',
       'TYPE_LOC', 'SQKM', 'SQMI', 'COLOR_MAP','cluster_labels'],axis=1)
#%%
print(departamento_punto.columns)
print(departamento_punto.head())
print(departamento_punto.dtypes)
print(departamento_punto.isnull().sum())

#%%

NUMERIC_COLUMNS=['Año','#_epistolas','Lat', 'Lon','Freq_Remitente','freq_ciudad_origen']
LABELS=['Relacion biblioteca']
NUM_LAB=NUMERIC_COLUMNS+LABELS
TEXT_COLUMNS = [c for c in departamento_punto.columns if c not in NUM_LAB]
TEXT_NUM_COLUMNS=TEXT_COLUMNS+NUMERIC_COLUMNS
print(departamento_punto[TEXT_COLUMNS].head())
#%%
departamento_punto['Descripcion Relacion biblioteca']=departamento_punto['Descripcion Relacion biblioteca'].fillna('')
#borrar caracteres para limpieza de texto en la columna
#antes de parentesis, de llaves, de punto, de coma y de comillas usar \ y entre cada elemento usar | 
caracteres_reemplazar='/|"|\[...]|\[. . .]|\(|;|\)|\'|\[…]|\.|\,|\¿|…|\?'
departamento_punto['Descripcion Relacion biblioteca']=departamento_punto['Descripcion Relacion biblioteca'].str.replace(caracteres_reemplazar, '',regex=True)


# Define combine_text_columns()
def combine_text_columns(data_frame, to_drop=NUMERIC_COLUMNS + LABELS):
    """ converts all text in each row of data_frame to single vector """
    
    # Drop non-text columns that are in the df
    to_drop = set(to_drop) & set(data_frame.columns.tolist())
    text_data = data_frame.drop(to_drop,axis=1)
    
    # Replace nans with blanks
    text_data.fillna('',inplace=True)
    
    # Join all text items in a row that have a space in between
    return text_data.apply(lambda x: " ".join(x), axis=1)

stop_words_spanish=stopwords.words('spanish')
# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_BASIC = '\\S+(?=\\s+)'

# Instantiate basic CountVectorizer: vec_basic
vec_basic = CountVectorizer(token_pattern=TOKENS_BASIC,stop_words=stop_words_spanish,ngram_range=(1,2))

# Create the text vector
text_vector = combine_text_columns(departamento_punto)
# Fit to the data
vec_basic.fit(text_vector)
# Print the number of tokens and first 15 tokens
msg = "There are {} tokens in Descripcion Relacion biblioteca if we split on non-alpha numeric"
print(msg.format(len(vec_basic.get_feature_names())))
# examine the fitted vocabulary
print(vec_basic.get_feature_names()[:500])

#algunos terminos se corrigieron manualmente en el archivo excel

#%%

# Obtain the text data: get_text_data
get_text_data = FunctionTransformer(combine_text_columns, validate=False)

# Obtain the numeric data: get_numeric_data
get_numeric_data = FunctionTransformer(lambda x: departamento_punto[NUMERIC_COLUMNS], validate=False)

# Fit and transform the text data: just_text_data
just_text_data = get_text_data.fit_transform(departamento_punto)

# Fit and transform the numeric data: just_numeric_data
just_numeric_data = get_numeric_data.fit_transform(departamento_punto)

# Print head to check results
print('Text Data')
print(just_text_data.head())
print('\nNumeric Data')
print(just_numeric_data.head())
#%%

X_train, X_test, y_train, y_test = train_test_split(departamento_punto[TEXT_NUM_COLUMNS],
                                                    pd.get_dummies(departamento_punto[LABELS]),
                                                    random_state=22,test_size=0.3)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
#%%

# Create a FeatureUnion with nested pipeline: process_and_join_features
process_and_join_features = FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data)              
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer(token_pattern=TOKENS_BASIC,stop_words=stop_words_spanish,ngram_range=(1,2)))
                ]))
             ]
        )


# Instantiate nested pipeline: pl
pl = Pipeline([
        ('union', process_and_join_features),
        ('knnclf', OneVsRestClassifier(KNeighborsClassifier()))
    ])

#%%
# Fit pl to the training data
pl.fit(X_train, y_train)

# Compute and print accuracy
accuracy = pl.score(X_test, y_test)
print("\nAccuracy on sample data - all data: ", accuracy)







#%%

y = departamento_punto['Relacion biblioteca'].values
X = departamento_punto.drop(['Tomo', 'Folio','Descripción','Relacion biblioteca','Ciudad destino','Ciudad_origen', 'Remitente', 'Destinatario','Descripcion Relacion biblioteca',
       'transcripcion_1_descr','Dia', 'Mes','Tipo relacion AMIGO',
       'Tipo relacion FAMILIA', 'Tipo relacionOtra', 'Colonizacion',
       'freq_ciudad_origen', 'Nombres', 'Freq_Remitente', 'Freq_Destinatario',
       'transcripcion_1_descr', 'transcripcion_2_desc_rela_bibl', 'location',
       'coordinates', 'Latitud', 'Longitud','geometry', 'index_right', 'FIPS_ADMIN', 'GMI_ADMIN', 'ADMIN_NAME',
       'FIPS_CNTRY', 'GMI_CNTRY', 'CNTRY_NAME', 'POP_ADMIN', 'TYPE_ENG',
       'TYPE_LOC', 'SQKM', 'SQMI', 'COLOR_MAP'], axis=1).values
print(X[:5])


#%%
# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42, stratify=y)
# Create a k-NN classifier with 6 neighbors
knn = KNeighborsClassifier()

# Fit the classifier to the data
knn.fit(X_train,y_train)
print(knn.score(X_test, y_test))
#%%
# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 50)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train,y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train,y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

#%%
# Setup the pipeline
steps = [('scaler', StandardScaler()),
         ('knn', KNeighborsClassifier())]

pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'knn__n_neighbors': np.arange(1,50)}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=21)

# Instantiate the GridSearchCV object: cv
cv = GridSearchCV(pipeline,param_grid=parameters, cv=10)

# Fit to the training set
cv.fit(X_train,y_train)

# Predict the labels of the test set: y_pred
y_pred = cv.predict(X_test)

# Compute and print metrics
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))


#%%
stop_words_spanish=stopwords.words('spanish')

# Define combine_text_columns()
def combine_text_columns(data_frame, to_drop=NUMERIC_COLUMNS + LABELS):
    """ converts all text in each row of data_frame to single vector """
    
    # Drop non-text columns that are in the df
    to_drop = set(to_drop) & set(data_frame.columns.tolist())
    text_data = data_frame.drop(to_drop,axis=1)
    
    # Replace nans with blanks
    text_data.fillna('',inplace=True)
    
    # Join all text items in a row that have a space in between
    return text_data.apply(lambda x: " ".join(x), axis=1)


# Create the token pattern: TOKENS_ALPHANUMERIC
TOKENS_BASIC = '\\S+(?=\\s+)'

departamento_punto['Descripcion Relacion biblioteca'].replace(0, '', inplace=True)

# Instantiate basic CountVectorizer: vec_basic
vec_basic = CountVectorizer(token_pattern=TOKENS_BASIC,stop_words=stop_words_spanish)

# Create the text vector
text_vector = combine_text_columns(departamento_punto)
# Fit to the data
vec_basic.fit(departamento_punto['Descripcion Relacion biblioteca',])
# Print the number of tokens and first 15 tokens
msg = "There are {} tokens in Descripcion Relacion biblioteca if we split on non-alpha numeric"
print(msg.format(len(vec_basic.get_feature_names())))
# examine the fitted vocabulary
print(vec_basic.get_feature_names()[:500])




