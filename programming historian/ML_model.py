# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 11:27:26 2020

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

plt.style.use('seaborn-dark-palette')
pd.set_option('display.max_columns',100)
pd.set_option('display.max_colwidth',500)
pd.options.display.max_rows = 999
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

print(departamento_punto.describe())
print(departamento_punto.isnull().sum())

departamento_punto['Relacion biblioteca']=departamento_punto['Relacion biblioteca'].astype('category')
print(departamento_punto.dtypes)

#%%
NUMERIC_COLUMNS=['Año','#_epistolas','Lat', 'Lon','Freq_Remitente','freq_ciudad_origen','Lat_scaled','Lon_scaled']
LABELS=['Relacion biblioteca']
NUM_LAB=NUMERIC_COLUMNS+LABELS
TEXT_COLUMNS = [c for c in departamento_punto.columns if c not in NUM_LAB]
TEXT_NUM_COLUMNS=TEXT_COLUMNS+NUMERIC_COLUMNS
print(departamento_punto[TEXT_COLUMNS].head())
#%%
departamento_punto['Descripcion Relacion biblioteca']=departamento_punto['Descripcion Relacion biblioteca'].fillna('')
departamento_punto['Relacion biblioteca']=departamento_punto['Relacion biblioteca'].astype('category')

#borrar caracteres para limpieza de texto en la columna
#antes de parentesis, de llaves, de punto, de coma y de comillas usar \ y entre cada elemento usar | 
caracteres_reemplazar='/|"|\[...]|\[. . .]|\(|;|\)|\'|\[…]|\.|\,|\¿|…|\?'
departamento_punto['Descripcion Relacion biblioteca']=departamento_punto['Descripcion Relacion biblioteca'].str.replace(caracteres_reemplazar, '',regex=True)

#%%
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

#columna con texto
departamento_punto['TEXT_COLUMNS_COMB']=just_text_data
print(departamento_punto['TEXT_COLUMNS_COMB'].head())
# Print head to check results
print('Text Data')
print(just_text_data.head())
print(just_text_data.shape)
print('\nNumeric Data')
print(just_numeric_data.head())
print(just_numeric_data.shape)

#%%
#despues de eliminar las columnas hay que correr de nuevo las listas de nombres de columnas
departamento_punto=departamento_punto.drop(['Ciudad_origen','Remitente','Descripcion Relacion biblioteca'],axis=1)
print(departamento_punto.dtypes)
#%%
#algoritmo randomforest dio mejores resultados que otros
#este para la columna con todo el texto 

# Split out only the text data
X_train, X_test, y_train, y_test = train_test_split(departamento_punto[TEXT_NUM_COLUMNS],
                                                    pd.get_dummies(departamento_punto[LABELS]), 
                                                    test_size=0.2,random_state=456)

# Instantiate Pipeline object: pl
pl = Pipeline([
        ('selector', get_text_data),
        ('vec', CountVectorizer()),
        ('clf', OneVsRestClassifier(RandomForestClassifier()))
    ])

# Fit to the training data
pl.fit(X_train,y_train)
y_pred = pl.predict(X_test)
# Compute and print accuracy
accuracy = pl.score(X_test,y_test)
print(classification_report(y_test, y_pred))
print("\nAccuracy on sample data - just text data: ", accuracy)

prediccion=pl.predict_proba(departamento_punto)
print(prediccion)
# Format predictions in DataFrame: prediction_df
prediccion_df = pd.DataFrame(columns=pd.get_dummies(departamento_punto[LABELS]).columns,
                             index=departamento_punto.index,
                             data=prediccion)
print(prediccion_df)
prediccion_df.to_csv('coleccionismo.csv',index=False)
#%%
#saca una tabla de confusíón, pero no la use porque calculo las probabilidades y redondeo. En otra ocasion sirve
def confusion_matrix(default, model_prediction):
    df=pd.DataFrame(np.hstack([default, model_prediction.numpy()>0.5]),columns=['Real','Predicción'])
    confusion_matrix=pd.crosstab(df['Real'],df['Predicción'],rownames=['Real'],colnames=['Predicción'])
    heatmap(confusion_matrix,cmap='Greys',fmt='d',annot=True,cbar=False)
    plt.show()
confusion_matrix(test_targets, model_predictions)
#%%
departamento_punto['coleccionista_si']=prediccion_df['Relacion biblioteca_1']
departamento_punto['coleccionista_si'].loc[departamento_punto['coleccionista_si'] > 0]=1
departamento_punto['colab_coleccionista_def']=['No colaborador' if x == 0 else 'Colaborador' for x in departamento_punto['coleccionista_si']]
departamento_punto.to_excel('df_colaboradores_class.xlsx')
print(departamento_punto.head())
sns.scatterplot(x='Lon', y='Lat', hue='colab_coleccionista_def', data=departamento_punto)
plt.show()
#%%
print(departamento_punto['coleccionista_si'].value_counts())
print(departamento_punto['colab_coleccionista_def'].value_counts())
#%%
departamento_punto_colec_si=departamento_punto.loc[departamento_punto['coleccionista_si'] != 0.000000]
print(departamento_punto_colec_si.head())
#%%
# Set up the US bounding box
us_boundingbox = [-83.7,-5.15,-63.49,13.78] 
#plt.figure(figsize=(30,30))
# Set up the Basemap object
m = Basemap(width=12000000,height=9000000,resolution='h',
            llcrnrlon = us_boundingbox[0],
            llcrnrlat = us_boundingbox[1],
            urcrnrlon = us_boundingbox[2],
            urcrnrlat = us_boundingbox[3],
            projection='merc')

# Draw continents in white,
# countries in black and coastlines and the states in gray
m.fillcontinents(color='gray', zorder= 0)
m.drawcountries(color='black')
m.drawstates(color='gray')
m.drawlsmask( ocean_color='aquamarine')


lon = [x for x in departamento_punto['Lon']]
lat = [x for x in departamento_punto['Lat']]
#epistolas= [x for x in counts_df['numero_cartas']]
#size_colec = [x * 30 for x in departamento_punto_colec_si['coleccionista_si']]
color_colec= [x for x in departamento_punto['coleccionista_si']]

# Draw the points and show the plot
scatter_1=m.scatter(lon, lat,c=color_colec, cmap = 'brg',s=7, latlon = True, alpha = 0.8)
#plt.xlabel('longitud')
#plt.ylabel('latitud')
plt.axis('off')
plt.title('Mapa de Colaboradores (NPL)')
#para sacar una leyenda con los colores unicos de la grafica scatter
Legend=plt.legend(*scatter_1.legend_elements(),title="Clases",title_fontsize=12.5,fontsize=10.5,frameon=True,framealpha=0.7,shadow=True,loc='lower right',facecolor='beige')
#cambiar el texto de los elementos de la leyenda al texto que deseo
Legend.get_texts()[0].set_text('No colaborador')
Legend.get_texts()[1].set_text('Colaborador')

plt.savefig('Mapa_colaboradores_coleccionistas.png', dpi = 450)           
plt.show()
#%%
#svc fue el algoritmo que dio mejores resultados con datos numericos
#scalar los valores mejor

# Create training and test sets using only numeric data
X_train, X_test, y_train, y_test = train_test_split(departamento_punto[NUMERIC_COLUMNS],
                                                    pd.get_dummies(departamento_punto[LABELS]), 
                                                    test_size=0.3,random_state=456)


# Insantiate Pipeline object: pl
pl = Pipeline([
        
        ('scaler', StandardScaler()),
        ('clf', OneVsRestClassifier(SVC()))
    ])

# Fit the pipeline to the training data
pl.fit(X_train,y_train)
y_pred = pl.predict(X_test)
# Compute and print accuracy
accuracy = pl.score(X_test,y_test)
print(classification_report(y_test, y_pred))
print("\nAccuracy on sample data - all numeric, incl nans: ", accuracy)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#intentar con feature union y=dummies y stratufy

#%%
print(departamento_punto_1.columns)
print(departamento_punto['Descripcion Relacion biblioteca'].head())
#%%
#para correr esta visualizacion toca correr la lectura del df al comienzo otra vez y 
#saltar hasta aqui, y para obtener la visualizacion combinada toca correr esta primero y
#sobre esta correr la visualizacion de numero de cartas del otro codigo
#graficar el numero de menciones al coleccionismo, cuando de la coleccion
time_format = '%Y'
anos=pd.to_datetime(list(departamento_punto['Año']), format=time_format)
anos_unicos=pd.to_datetime(list(departamento_punto['Año']), format=time_format).year.unique()
departamento_punto_1=departamento_punto.set_index(anos)
print(anos_unicos)
departamento_punto_1=departamento_punto_1[['Descripcion Relacion biblioteca','Año','#_epistolas','Descripción']]
departamento_punto_1['Descripcion Relacion biblioteca']=departamento_punto_1['Descripcion Relacion biblioteca'].fillna('')

#%%
#rastreo de los terminos en la columna descripcion relacion
result = departamento_punto_1['Descripcion Relacion biblioteca'].str.contains(pat='boletines|boletin|ordenanzas|gramática|gramatica|ciencia|geografia|historia|diccionarios|diccionario|arimética|aritmetica|Ymprenta|pliegos|cuadernito|memorias|memoria|copia|texto|ejemplares|ejemplar|cartillas|escuelas|educación|folleto|folletoscompilaciones|cartilla popular|imprenta|librito|cuaderno|manuscrito|manuscritos|cartas|gacetas|gaceta|biblioteca|colección|coleccion|publicación|impresos|impreso|tomos|tomo|artículo|articulo|papeles|papel|hojas|hoja|diario|cuaderno|borradores|catálogo|libro|libros|encuadernación|archivo|volúmenes|volúmen|volumen|periódicos|periódico|periodicos|periodico|documentos|documento|legajos|legajo|obras|obra|obrita',case=False)
result |= departamento_punto_1['Descripción'].str.contains(pat='boletines|boletin|ordenanzas|gramática|gramatica|ciencia|geografia|historia|diccionarios|diccionario|arimética|aritmetica|Ymprenta|pliegos|cuadernito|memorias|memoria|copia|texto|ejemplares|ejemplar|cartillas|escuelas|educación|folleto|folletoscompilaciones|cartilla popular|imprenta|librito|cuaderno|manuscrito|manuscritos|gacetas|gaceta|biblioteca|colección|coleccion|publicación|impresos|impreso|tomos|tomo|artículo|articulo|papeles|papel|hojas|hoja|diario|cuaderno|borradores|catálogo|libro|libros|encuadernación|archivo|volúmenes|volúmen|volumen|periódicos|periódico|periodicos|periodico|documentos|documento|legajos|legajo|obras|obra|obrita',case=False)

departamento_punto_1['#_mencion_document']=result
print(departamento_punto_1.head(50))
#%%
# conteo y resample por año y suma, muestra cuantos por cada año
#print(df1['Remitente'].resample("Y").count())
suma=departamento_punto_1['#_mencion_document']['1806':'1880'].resample("Y").sum()
print(suma)
#%%
#grafica de las menciones
departamento_punto_1['#_mencion_document']['1806':'1880'].resample("Y").sum().plot(label='Menciones al coleccionismo')
xlab = 'Años'
ylab = 'Número de epístolas'
title = 'Número de epístolas y menciones al coleccionismo \n por años en el epistolario Pineda'
plt.xlabel(xlab,color='darkolivegreen',fontsize=11)
plt.ylabel(ylab,color='darkolivegreen',fontsize=11)
plt.title(title,fontsize=12,color='navy')
plt.legend(loc='best',fontsize = 'medium')
plt.yticks(fontsize=10)
plt.xticks(fontsize=10)

plt.show()
#plt.savefig('menciones_coleccionismo_y_actividad epistolar.png',dpi=300)
#%%
print(departamento_punto_1[['#_epistolas','#_mencion_document']].corr())
#%%
#singularizado por año
print(len(departamento_punto_1.loc['1849','#_mencion_document'])  )
print(departamento_punto_1.loc['1849','#_mencion_document'].sum()  )

#%%
#revision cuantos colaboradores hay
df=pd.read_excel('departamento_punto_colecc_class.xlsx')
#print(df.columns)
print(df['Remitente'].loc[df['colab_coleccionista_def']=='Colaborador'].unique())
print(df['Remitente'].loc[df['colab_coleccionista_def']=='Colaborador'].value_counts())
print(df.groupby(df['colab_coleccionista_def'])['Remitente'].nunique())


