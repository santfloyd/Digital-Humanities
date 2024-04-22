# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 16:33:26 2022

@author: Usuario
"""
#descarga de tweets (full archive con coordenadas)
#concatenar df que fueron descargados por partes  YA
#concatenar en un solo df los dataframes concatenados, mas el de vandalismo, paronacional y seguridad
#limpieza de textos (url, emojis, stop words)
#temas principales y entidades
#se entrena un modelo con la columna de autores de tweets pasada por reconocimiento de entidades primero
#data analysis mediante network analysis con NLP
#descarga de tweets con corredenadas
#analisis de sentimientos
#mapeo de sentimientos
        

#descarga por terminos
#cada df debe tener la misma cantidad de registros 200 mil registros por palabra

import snscrape.modules.twitter as sntwitter
import pandas as pd
# Setting variables to be used below
maxTweets = 100000
# Creating list to append tweet data to
tweets_list2 = []

ubicacion='-0.2298500, -78.5249500, 320km'
fecha_inicio='2022-06-13'
fecha_fin='2022-06-29'
# Using TwitterSearchScraper to scrape data and append tweets to list
#con ucibacion retorna sustancialmente menos tweets
#for i, tweet in enumerate(sntwitter.TwitterSearchScraper('CONAIE geocode:"{ubicacion}" since:"{fecha_inicio}" until:"{fecha_fin}"'.format(ubicacion=ubicacion,fecha_inicio=fecha_inicio,fecha_fin=fecha_fin)).get_items())
try:
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper('pueblo since:"{fecha_inicio}" until:"{fecha_fin}"'.format(fecha_inicio=fecha_inicio,fecha_fin=fecha_fin)).get_items()):
        if i > maxTweets:
            break
        tweets_list2.append([tweet.date, tweet.id, tweet.rawContent, tweet.user.username, tweet.likeCount, tweet.retweetCount, tweet.replyCount])
except:
    pass
# Creating a dataframe from the tweets list above
tweets_df2 = pd.DataFrame(tweets_list2, columns=['Datetime', 'Tweet Id', 'Text', 'Username', 'likes', 'retweets', 'replicas'])

#to csv
tweets_df2.to_csv(r'E:\C\Master_geomatica\proyecto_Cuenca\tweets\tweets_terminoPueblo_2.csv', encoding='utf-8')
# Display first 5 entries from dataframe
print(tweets_df2.head())
print(tweets_df2.tail())
print(tweets_df2.shape)
#%%
#concatenar dataframes cuyo nombre indica que son la continuacion del mismo termino
import os
import pandas as pd

path=r'E:\C\Master_geomatica\proyecto_Cuenca\tweets'

def concatdf_2(path):
    dfs=[]
    for root, dirs, files in os.walk(path):
        root=root #el path completo hasta el lugar de los ficheros
                    #importante que todos estén en el mismo directorio
        for i, file in enumerate(files): #utilizo enumerate para poder moverme en las posiciones de la lista de archivos usando los indices
            if i == 0 or i == (len(files)-1): #pasa si es el primer index y el último
                pass
            #comparacion de nombres de ficheros
            else:
                previous= files[i-1] #el valor previo para comparar es el valor del index menos uno
                actual= files[i] #el valor de la posicion del index acual
                #print(previous, actual)
                if os.path.splitext(previous)[0] in os.path.splitext(actual)[0]: #si el nombre sin extension del primer 
                                                                                #esta contenido en el nombre del segundo file
                    print('fichero {previous} se concatena con {actual}'.format(previous=previous, actual=actual))
                    previous_path=os.path.join(root, previous) #crea el path con los elementos que arroja os.walk
                    print(previous_path)
                    previous_df=pd.read_csv(r'{}'.format(previous_path)) #crea el df a partir del path
                                            
                    actual_path=os.path.join(root, actual)
                    print(actual_path)                        
                    actual_df=pd.read_csv(r'{}'.format(actual_path))
                    dfs=[previous_df, actual_df] #crea una lista de los dos dataframes con nombres semejantes
                    
                    df_termino=pd.concat(dfs, sort=False) #concatena
                    print('dimension del df 1: ',previous_df.shape)
                    print('dimension del df 2: ',actual_df.shape)
                    print('dimension del df concatenado: ',df_termino.shape)
                    #escribe los dfs
                    df_termino.to_csv(r'E:\C\Master_geomatica\proyecto_Cuenca\tweets\tweets_termino{file}_total.csv'.format(file=file), encoding='utf-8')
                    print('se guardo la BBDD del termino {file}'.format(file=file))
                    
print(concatdf_2(path))

#%%
#concatenar dataframes cuyo nombre indica que son la continuacion del mismo termino
import os
import pandas as pd

path=r'E:\C\Master_geomatica\proyecto_Cuenca\tweets'

def concatdf(path):
    dfs=[]
    for root, dirs, files in os.walk(path):
        root=root #el path completo hasta el lugar de los ficheros
                    #importante que todos estén en el mismo directorio
        for i, file in enumerate(files): #utilizo enumerate para poder moverme en las posiciones de la lista de archivos usando los indices
            if i == 0 or i == (len(files)-1): #pasa si es el primer index y el último
                pass
            #comparacion de nombres de ficheros
            else:
                previous= files[i-1] #el valor previo para comparar es el valor del index menos uno
                actual= files[i] #el valor de la posicion del index acual
                #print(previous, actual)
                if os.path.splitext(previous)[0] in os.path.splitext(actual)[0]: #si el nombre sin extension del primer 
                                                                                #esta contenido en el nombre del segundo file
                    print('fichero {previous} se concatena con {actual}'.format(previous=previous, actual=actual))
                    previous_path=os.path.join(root, previous) #crea el path con los elementos que arroja os.walk
                    print(previous_path)
                    previous_df=pd.read_csv(r'{}'.format(previous_path)) #crea el df a partir del path
                                            
                    actual_path=os.path.join(root, actual)
                    print(actual_path)                        
                    actual_df=pd.read_csv(r'{}'.format(actual_path))
                    dfs=[previous_df, actual_df] #crea una lista de los dos dataframes con nombres semejantes
                    
                    df_termino=pd.concat(dfs, sort=False) #concatena
                    print('dimension del df 1: ',previous_df.shape)
                    print('dimension del df 2: ',actual_df.shape)
                    print('dimension del df concatenado: ',df_termino.shape)
                    #escribe los dfs
                    df_termino.to_csv(r'E:\C\Master_geomatica\proyecto_Cuenca\tweets\tweets_termino{file}_total.csv'.format(file=file), encoding='utf-8')
                    print('se guardo la BBDD del termino {file}'.format(file=file))
                    
print(concatdf(path))
#%%

#celda 1

#merge de todos los dataframes relevantes del fichero
#como los concatenados antes y los que ya estaban completos
#identificados manualmente
import os
import pandas as pd

directory=r'E:\C\Master_geomatica\proyecto_Cuenca\tweets'

def concatdf_totales(path=directory):
    csvs=['tweets_terminoParoNacional.csv','tweets_terminoSeguridad.csv','tweets_terminoVandalismo.csv']
    dfs=[]
    for root, dirs, files in os.walk(directory):
        for i, file in enumerate(files):
            
            if os.path.splitext(file)[0].endswith('_total') or file in csvs:
                full_path=os.path.join(root, file)
                print(full_path)
                df=pd.read_csv(full_path)
                dfs.append(df)
                
    df_tweets_everything=pd.concat(dfs, sort=False)
    print(df_tweets_everything.shape)
    return df_tweets_everything
       

df_tweets=concatdf_totales(directory)
print(df_tweets.head())
#%%           
#               #%%
#             full_path=os.path.join(root, file)
            
#             #print(full_path)
#             df=pd.read_csv(full_path)
#             dfs.append(df)
    
#     df_tweets=pd.concat(dfs, sort=False)
    
    
#     print(df_tweets.head())
#     print(df_tweets.tail())
#     print(df_tweets.shape)
    
#     #print(df_tweets.head(2))
#     # df_tweets_part1=df_tweets[:500000]
#     # df_tweets_part2=df_tweets[500000:]
    
#     # df_tweets_part1.to_excel(r'E:\C\Master_geomatica\proyecto_Cuenca\tweets\df_tweets_part1.xlsx', encoding='utf-8')
#     # df_tweets_part2.to_excel(r'E:\C\Master_geomatica\proyecto_Cuenca\tweets\df_tweets_part2.xlsx', encoding='utf-8')

# csvs=['tweets_terminoVandalismo.csv','tweets_terminoParoNacional.csv','tweets_terminoSeguridad.csv'] 
#%%
#celda 2

#conteo de usuarios

import matplotlib.pyplot as plt

total_tweets=df_tweets['Text'].nunique()
total_users=df_tweets['Text'].nunique()
print('tweets unicos:',total_tweets)
print(len(df_tweets['Text'].unique()))
print(df_tweets.head(2))
print('******************###***********')
print('users unicos:',total_users)
print(len(df_tweets['Username'].unique()))

conteo_users=pd.DataFrame(df_tweets['Username'].value_counts())
conteo_users.rename(columns={'Username':'numero de tweets'},inplace=True)
conteo_users=conteo_users[:500]
print(conteo_users.head())
conteo_users.plot.bar(label='numero de tweets')
plt.title('Distribución de usuarios',fontsize=14)
plt.xlabel('Usuarios',fontsize=12)
plt.ylabel('Conteo de tweets',fontsize=12)
plt.xlim(0,10)
plt.tight_layout()
plt.show()

#%%

#celda 2A

#creacion de index por fecha para graficar el comportamiento en el dataframe total
time_format = '%Y-%m-%d %H:%M:%S+00:00'
dias=pd.to_datetime(list(df_tweets['Datetime']), format=time_format)
dias_unicos=pd.to_datetime(list(df_tweets['Datetime']), format=time_format).day.unique()
df_tweets1=df_tweets.set_index(dias)
#print(anos_unicos)

print(df_tweets1.head())
print(df_tweets1.sort_values('Datetime').head())
print(df_tweets1.columns)

#grafico

df_tweets1=df_tweets1[['Username']]
#print(df1['Remitente'].resample("Y").count())

df_tweets1['Username'].resample("D").count().plot()
xlab = 'Dias'
ylab = 'Número de tweets'
title = 'Número de tweets por dias'
plt.xlabel(xlab,color='darkolivegreen')
plt.ylabel(ylab,color='darkolivegreen')
plt.title(title,color='navy')
plt.show()
       
#%%
print(conteo_users[:101])

#%%

#celda 2B

import os
import pandas as pd
import matplotlib.pyplot as plt
#grafico de comportamiento por fecha por cada uno de los dataframes
#así rastrear los terminos que se quedaron cortos en temporalidad

directory=r'E:\C\Master_geomatica\proyecto_Cuenca\tweets'

for root, dirs, files in os.walk(directory):
    for i, file in enumerate(files):
        
            
        full_path=os.path.join(root, file)
        
        #print(full_path)
        df=pd.read_csv(full_path)
        
        time_format = '%Y-%m-%d %H:%M:%S+00:00'
        dias=pd.to_datetime(list(df['Datetime']), format=time_format)
        dias_unicos=pd.to_datetime(list(df['Datetime']), format=time_format).day.unique()
        
        df_tweets1=df.set_index(dias)
        
        
        df_tweets1=df_tweets1[['Username']]
#print(df1['Remitente'].resample("Y").count())

        df_tweets1['Username'].resample("D").count().plot()
        xlab = 'Dias'
        ylab = 'Número de tweets'
        title = 'Número de tweets por dias {}'.format(file)
        plt.xlabel(xlab,color='darkolivegreen')
        plt.ylabel(ylab,color='darkolivegreen')
        plt.title(title,color='navy')
        plt.show()
#print(df_tweets1.head())

#%%

#Celda 2C

#conteo de usuarios

import matplotlib.pyplot as plt

total_tweets=df_tweets['Text'].nunique()
total_users=df_tweets['Text'].nunique()
print('tweets unicos:',total_tweets)
print(len(df_tweets['Text'].unique()))
print(df_tweets.head(2))
print('******************###***********')
print('users unicos:',total_users)
print(len(df_tweets['Username'].unique()))

conteo_users=pd.DataFrame(df_tweets['Username'].value_counts())
conteo_users.rename(columns={'Username':'numero de tweets'},inplace=True)
conteo_users=conteo_users[:500]
print(conteo_users.head())
conteo_users.plot.bar(label='numero de tweets')
plt.title('Distribución de usuarios',fontsize=14)
plt.xlabel('Usuarios',fontsize=12)
plt.ylabel('Conteo de tweets',fontsize=12)
plt.xlim(0,25)
plt.tight_layout()
plt.show()

#%%

#celda 3

import emoji
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words_spanish=stopwords.words('spanish')

#limpieza de texto
df_tweets['Text'] = df_tweets['Text'].astype('unicode')
print(df_tweets['Text'][1490:1500])

#extraer las urls a otra columna antes de eliminarlas
df_tweets['http']=df_tweets['Text'].str.extract('(\shttps?://.*$)') 

print(df_tweets['http'][1490:1500]) 


def text_limpio(text):
    
    #remover las urls para mejor deteccion del idioma
    text = re.sub('(\shttps?://.*$)', '', text) 
    #remmueve los emojis y @
    text = emoji.replace_emoji(text, replace='')
    text = re.sub("@[A-Za-z0-9]+","", text)
    #remueve hashtags y convierte a minuscula para eliminar palabras vacias
    text = text.replace("#", "").replace("_", " ")
    text = text.lower()
    #remover las stopwords y puntuacion
    text = [word for word in text.split() 
            if word.isalpha() and word not in stop_words_spanish]     
    
    text = ' '.join(word for word in text)
    #print('TWEET: ' + str(text) + '\n')
    #devolver texto completamente limpio
    return text

#aplicar la función a cada registro del df
df_tweets['text_cleaned'] = df_tweets['Text'].map(lambda x: text_limpio(x))
print(df_tweets['text_cleaned'][1490:1500]) 

# def cleaner(tweet):
#     tweet = re.sub("@[A-Za-z0-9]+","",tweet) #Remove @ sign
#     tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet) #Remove http links
#     tweet = " ".join(tweet.split())
#     tweet = ''.join(c for c in tweet if c not in emoji.UNICODE_EMOJI) #Remove Emojis
#     tweet = tweet.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
#     tweet = " ".join(w for w in nltk.wordpunct_tokenize(tweet) \
#          if w.lower() in words or not w.isalpha())
#     return tweet
# trump_df['tweet'] = trump_df['tweet'].map(lambda x: cleaner(x))
# trump_df.to_csv('') #specify location

#%%

#celda 4

#mirar tokenizar y entrenar un modelo spacy

import spacy
import es_core_news_lg

nlp = es_core_news_lg.load()

columna_text=df_tweets['text_cleaned'][:2000]

def entidades(columna_text):
    #df a lista para iniciar el procesamiento
    lista=[t for t in columna_text.tolist()]
    
    lista_idx=df_tweets.index.tolist()
    
    #crear listas vacias para añadir los resultados por separado
    lista_palabras=[]
    lista_entidades=[]
    lista_entidades_per=[]
    lista_entidades_loc=[]
    lista_entidades_org=[]
    for text in lista:
        #print(text)
        
        tokens = nlp(''.join(text))
        ents=[]
        per=[]
        loc=[]
        org=[]
        
    
        for ent in tokens.ents:
            
            #añade todas las entidades sin distincion a una lista
            if ent.label_ in ['PER', 'LOC', 'ORG']:
            
                ents.append((ent, ent.label_))
        
            
            #print(ent.label_)
            if ent.label_ == 'PER':
                #print(ent)
                per.append(ent.text)
            if ent.label_ == 'LOC':
                #print(ent)
                loc.append(ent.text)
            if ent.label_ == 'ORG':
                #print(ent)
                org.append(ent.text)
                
        lista_entidades.append(ents)
        lista_entidades_per.append(per)
        lista_entidades_loc.append(loc)
        lista_entidades_org.append(org)
    return lista_entidades_per, lista_entidades_org, lista_entidades_loc, lista_entidades


#%%

#posibilidad a celda 4

#la única forma que fue efectiva para instalar polyglot es la siguiente
#descargar los siguientes wheels de la pagina https://www.lfd.uci.edu/~gohlke/pythonlibs/
#pycld2-0.41-cp38-cp38-win_amd64.whl / PyICU-2.9-cp38-cp38-win_amd64.whl / Morfessor-2.0.6-py3-none-any.whl
#en la ubicación donde guarde los whl
#pip install pycld2-0.41-cp38-cp38-win_amd64.whl
#pip install PyICU-2.9-cp38-cp38-win_amd64.whl
#pip install Morfessor-2.0.6-py3-none-any.whl
#luego clonar el git de polyglot
#git clone https://github.com/aboSamoor/polyglot
#cd polyglot
#python setup.py install
#descargar los modelos en español
#polyglot download LANG:es

import polyglot
from polyglot.text import Text, Word
#deteccion de idioma
from polyglot.detect import Detector
#rastreo de entidades

columna_text=df_tweets['text_cleaned'][1000:2000]
print(columna_text)
# print(LISTAINDICES[1490:1500])

def entidades(columna_text):
    #df a lista para iniciar el procesamiento
    lista=[t for t in columna_text.tolist()]
    
    lista_idx=df_tweets.index.tolist()
    
    #crear listas vacias para añadir los resultados por separado
    lista_palabras=[]
    lista_entidades=[]
    lista_entidades_per=[]
    lista_entidades_loc=[]
    lista_entidades_org=[]
    
    for text in lista:
        print(text)
        try:
            #lista para almacenar los tres idiomas posibles detectados
            lenguajes=[]
            #por cada idioma detectado añade a la lista anterios
            for language in Detector(text, quiet=True).languages:
                #print(language)
                lenguajes.append(language.name.split())
            #print(lenguajes[0])
            
            #si el primer idioma probable es español extrae las entidades
            if lenguajes[0] == ['español']:
                print('soy español')
                #extraccion
                doc=Text(text)
                #añadir todos los tokens a una lista
                lista_palabras.append(doc.words)
                
                #la entidad por cada documento
                entities = [(ent.tag, ' '.join(ent)) for ent in doc.entities]
                #print(entities)
                #añade a la lista todas las entidades encontradas
                lista_entidades.append(entities)
                #crea una variable con todas las entidades que tienen las tag de personas
                entities_per = [ent for ent in doc.entities if ent.tag=='I-PER']
                entities_org = [ent for ent in doc.entities if ent.tag=='I-LOC']
                entities_loc = [ent for ent in doc.entities if ent.tag=='I-ORG']
                #añade a la lista todas las entidades de personas
                lista_entidades_per.append(entities_per)
                lista_entidades_org.append(entities_org)
                lista_entidades_loc.append(entities_loc)
                
                
                if len(entities) > 0 and entities[0][0] == 'I-PER':
                    print(entities)
                    
                entities_loc = [ent for ent in doc.entities if ent.tag=='I-LOC']
                # # lista_entidades_loc.append(entities_loc)
                if len(entities) > 0 and entities[0][0] == 'I-LOC':
                    print(entities)
                
                entities_org = [ent for ent in doc.entities if ent.tag=='I-ORG']
                # # lista_entidades_org.append(entities_org)
                
                if len(entities) > 0 and entities[0][0] == 'I-ORG':
                    print(entities)
                                
            else:
                pass
        except:
            continue
        
    return lista_entidades_per, lista_entidades_org, lista_entidades_loc

personas, organizaciones, ubicaciones = entidades(columna_text)


#%%

#celda 4A para Spacy
personas, ubicaciones, organizaciones, entidades = entidades(columna_text)

# personas = [' '.join(item) for item in personas if len(item) > 0]
# ubicaciones = [' '.join(item) for item in ubicaciones if len(item) > 0]
# organizaciones = [' '.join(item) for item in organizaciones if len(item) > 0]


#mostrar la lista de entidades y escoger manualmente una lista de nombres
#es necesario supervisar el desempeño del algoritmo

#print(personas)
# print(ubicaciones)
# print(organizaciones)

print(entidades[:100])
#%%

#celda 4B
#celda opcional para GUARDAR el proceso de identificacion de entidades
import csv
import os


def listtoCSV(directory, file, lista):
      
    full_path=os.path.join(directory, file)
    #print(full_path)
    
    with open(full_path, 'w+', newline='', encoding="utf-8") as file:
        wr = csv.writer(file, quoting=csv.QUOTE_ALL)
        wr.writerow(lista)


        print("{} guardado con exito".format(full_path))
        
directory=r'E:\C\Master_geomatica\proyecto_Cuenca\tweets_completo\entidades'
csvs=['personas.csv', 'ubicaciones.csv', 'organizaciones.csv', 'entidades.csv']
listas=[personas, ubicaciones, organizaciones, entidades]

listas_csvs=zip(csvs, listas)

for file, lista in listas_csvs:
    listtoCSV(directory, file, lista)



#print(personas)
#print(ubicaciones)
# print(organizaciones)

#%%
#celda 4A solo para Polyglot

#extrae la entidad de la tupla devuelta por polyglot
personas = [' '.join(item) if len(item) > 1 else item[0] for sublist in personas for item in sublist ]
#print(personas)

ubicaciones = [' '.join(item) for item in ubicaciones if len(item) > 0]
#print(personas)
for ubi in ubicaciones[:20]:
    print(ubi)


#%%
#celda de debuggeo LECTURA de las listas
import csv
import os
import pandas

directory=r'E:\C\Master_geomatica\proyecto_Cuenca\tweets_completo\entidades'
csvs=['personas.csv', 'ubicaciones.csv', 'organizaciones.csv','entidades.csv']

def csvtoList(directory, csvs):
    personas_list = []
    ubicaciones_list = []
    organizaciones_list = []
    entidades_list = []
    for file in csvs:
        print(file)
        path=os.path.join(directory, file)
        with open(path, encoding='utf-8') as entidades_file:
            
            if path.endswith("personas.csv"):
                csv_reader = csv.reader(entidades_file)
                lista_per = [ent for ent_list in csv_reader for ent in ent_list]
                for per in lista_per:
                    personas_list.append(per)
                
                print(len(personas_list))
                
                   
            if path.endswith("ubicaciones.csv"):
                
                csv_reader = csv.reader(entidades_file)
                lista_org = [ent for ent_list in csv_reader for ent in ent_list]
                for org in lista_org:
                    organizaciones_list.append(org)
                
                print(len(organizaciones_list))
                
            if path.endswith("organizaciones.csv"):
                
                csv_reader = csv.reader(entidades_file)
                lista_ubi = [ent for ent_list in csv_reader for ent in ent_list]
                for ubi in lista_ubi:
                    ubicaciones_list.append(ubi)
                
                print(len(ubicaciones_list))
            
            if path.endswith("entidades.csv"):
                pass
                # columns_ents=['per', 'loc', 'org']
                # df = pd.DataFrame(path, columns=columns_ents)
                # print(df.head())
                # df["todas"] = df[df.columns].apply(lambda x: ','.join(x.dropna().astype(str)), axis=1)
                
                # print(df["todas"])
                
                return personas_list, ubicaciones_list, organizaciones_list, entidades_list
        
personas, ubicaciones, organizaciones, entidades = csvtoList(directory, csvs)

#print(len(personas))
# print(personas[:20])
# print(ubicaciones[:20])
# print(organizaciones[:20])        
#%%
print(entidades)

#%%

#celda 5

#red de usuarios


users=df_tweets['Username']

print(len(users))

#llama la funcion entidades
personas, ubicaciones, organizaciones = entidades(users)

#%%

#celda 5A

ubicaciones = [' '.join(item) if len(item) > 1 else item[0] for sublist in personas for item in sublist ]
print(ubicaciones[:20])

#%%

#celda 6 para todas las (librerias, y listas de referencia)

#UNICAMENTE
#eliminar strings iguales o muy parecidas de las listas de entidades por separado

from fuzzywuzzy import process, fuzz
import unidecode

# print("personas: ", ubicaciones)

# #unidecode para eliminar los acentos
# #choices = set(ubicaciones)
# #print("choices: ", choices)

# #entidades por cada articulo IMPORTANTE no es un set de elementos unicos
# #tiene todas las entidades para luego hacer la extraccion
# tweets_ents=[unidecode.unidecode(' '.join(item)) if len(item) > 1 else unidecode.unidecode(item[0]) for sublist in ubicaciones for item in sublist ]
# print(len(tweets_ents))

#fuzzywuzzy
#print(fuzz.token_sort_ratio("guillermo lasso", "lasso"))

from rapidfuzz.process import extract, extract_iter
from rapidfuzz.string_metric import Levenshtein, normalized_levenshtein
from rapidfuzz.fuzz import ratio

#aqui las entidades correctas
#compare_list=["guillermo","lasso","guillermito","lassito","guillermo lasso","Leonidas Iza","Iza"]
#lo hare con los lugares, son menos etiquetas y tiene que ver con la dimension espacial en los tweets

#GPE
#%%
compare_list_geo=["ecuador","loja", "cuba", "colombia", "venezuela", "azuay", "perú", "quito", "guayaquil", "cuenca", "belgica", "chile", "egipto", "mexico", "francia","cali", "latinoamerica","gualaceo","guayaquil", "pichincha","españa", "bolivia", "rusia","brasil","amazonas","ambato","africa","guatemala","miami", "bolívar", "cañar","carchi", "chimborazo","cotopaxi","el Oro", 
                  "esmeraldas", "galápagos", "guayas","imbabura","loja", "los Ríos", "manabí","morona Santiago","napo","orellana","pastaza","pichincha", "santa elena", "santo domingo de los tsáchilas", "sucumbíos", "tungurahua","zamora chinchipe"]
compare_list_per=['guillermo lasso', 'leonidas iza','chávez', 'maduro', 'gustavo petro']
compare_list_org=['conaie']

#combinar las dos listas GEO y PER para obtener cuando se vaya a mezclar ambas entidades
combined_compare_list = compare_list_geo + compare_list_per + compare_list_org
print(combined_compare_list)
#%%

#CELDA 6B para entidades de un solo tipo

#print(compare_list)
#comparar las personas y 
entidades_=[]
proba=[]

for i, ubi in enumerate(ubicaciones):
    try:
        
        #si ubi es un string extrae la entidad (evita errores nontype)
        if isinstance(ubi, str) :
            #print(i, ' ', ubi)
            
            #si la palabra tiene mas del 80 porciento de probabilidad de ser una entidad de las
            #listas con las entidades a comparar
            if process.extractOne(ubi, compare_list_geo)[1] > 90:
                entidades_.append((process.extractOne(ubi, compare_list_geo)[0]))
        
        else:
            pass
        
    except:
        continue
      
    
print(entidades_)
print(len(entidades_))

#buscar en las probabilidades mas altas
#print(proba)


#%%

#celda 6B para extraer entidades combinadas

#usar fuzzywuzzy para extraer las tuplas que coincidan a las listas de 
#palabras de referencia

#CELDA identificacion de tuplas que contengan match de más del 90


entidades_totales=[]

default_scorer = fuzz.WRatio

for i, list_ent in enumerate(entidades):
    #print(list_ent)
    
    sublist=[]
    for ent in list_ent:
        
        try:
            #si la palabra tiene mas del 80 porciento de probabilidad de ser una entidad de las
            #listas con las entidades a comparar
            #print(ent[0], 'prob: ', str(process.extractOne(str(ent[0]), compare_list_geo)[1]))
            
            #separa termino y probabilidad
            term, prob = process.extractOne(str(ent[0]), combined_compare_list)
            
            #si la probabilidad es igual a 100 porciento
            if prob == 100:
                #print(ent, 'probabilidad: ',prob)
                sublist.append(ent)
                            
        except:
            continue
        
    entidades_totales.append(sublist)

print(entidades_totales[:100])
print(len(entidades_totales))

#%%

#CELDA 6C para crear un dataframe para edges y otro para nodos con el respectivo atributo

#añadir nodos y arcos

#extraer los terminos de las entidades para formar nuevas listas 
#solo con los terminos para crear un df de edges

import pandas as pd
import numpy as np

edge_entidades_totales_list = []

def df_edges_function(lista_entidades):
    for listas in lista_entidades:
        #print(listas)
        sublista_tuple=[]
        if len(listas) > 1:
            
            for tupla in listas:
                sublista_tuple.append(str(tupla[0]))
        else:
            
            for tupla in listas:
                sublista_tuple.append(str(tupla[0]))
        
        edge_entidades_totales_list.append(sublista_tuple)
    return edge_entidades_totales_list

lista_arcos = df_edges_function(entidades_totales)    

#print(lista_arcos)

#crea un df
df_edges=pd.DataFrame(lista_arcos)

columnas = []

for i in list(df_edges.columns):
    columnas.append("nodo_{}".format(str(i)))

df_edges.columns = columnas
print(df_edges.columns)

#rellena los none con nan
df_edges = df_edges.fillna(value=np.nan)
print(df_edges.head(20))


#separa el df en dfs de dos columnas para luego concatenarlas
from more_itertools import pairwise


def column_pairs(df):
    lista_columnas = list(df.columns)
    lista_pares_columnas = list(pairwise(lista_columnas))
    return lista_pares_columnas

pares = column_pairs(df_edges)

def limpieza_edge_table(df, pares_columnas):
    lista_frames = []
    #por cada par
    for par in pares_columnas:
        #una nueva instancia del df fraccionado por cada par
        df_par = df[list(par)]
        #una lista de dataframes por cada instancia
        lista_frames.append(df_par)
        #un unico dataframe concatenado en el axis 0
        df_edges_par = pd.concat(lista_frames)
                
        #elimina los registros que tienen en todas las filas nan
        #y devuelve solo las dos primeras columnas
        df_edges_par = df_edges_par.dropna(axis=0,how='all')
        
        #FALTA conteo de vinculos exactos en otra columna
        
        return df_edges_par
        
df_edges = limpieza_edge_table(df_edges, pares)
print(len(df_edges))
print(df_edges.tail())

#crea un dataframe con los nombres de los nodos y otra columna con el 
#tipo de entidad

def df_nodes(lista_entidades):
    nodos = []
    for entity in lista_entidades: 
        
        for ent in entity:
            
            nodos.append([str(ent[0]), ent[1]])
        
    return nodos
            
lista_nodos = df_nodes(entidades_totales)

df_nodes=pd.DataFrame(lista_nodos, columns=['nodo','atributo'])       

df_nodes = df_nodes.drop_duplicates(subset=["nodo"], keep='last')

print(len(df_nodes))
print(df_nodes.tail())

#%%
import networkx as nx
import matplotlib.pyplot as plt

node_colors = {'PER': 'blue', 'LOC': 'red', 'ORG':'orange'}

df_nodes['node_color'] = df_nodes["atributo"].map(node_colors)

G = nx.from_pandas_edgelist(df_edges, source = "nodo_0", target = "nodo_1")

nodes_attr = df_nodes.set_index(["nodo"]).to_dict(orient = 'index')
nx.set_node_attributes(G, nodes_attr)
#remover las self loops
G.remove_edges_from(nx.selfloop_edges(G))

#print(G.nodes(data=True))

plt.figure(figsize=(50,50)) 
nx.draw_networkx(G, 
    pos = nx.kamada_kawai_layout(G), 
    node_size = 100, 
    node_color = nx.set_node_attributes(G, pd.Series(df_nodes.atributo, index=df_nodes.nodo).to_dict(), 'entidad'),
    #width = [G.edges[e]['years'] for e in G.edges],
    with_labels = True, font_size=36)
plt.plot()

#%%

print(pd.Series(df_nodes.atributo, index=df_nodes.nodo).to_dict())
#%%
#falta hacer que separe por color

atr=[G.nodes[n] for n in G.nodes]

for at in atr:
    print(at["atributo"])

#%%

#RECICLAGE

G = nx.Graph()

entidades_nodos=[]

#incluir nodos con el atributo que identifica cada nodo como PER, LOC, ORG

for entity in entidades_totales:
    #añade el nodo
    for ent in entity:
        entidades_nodos.append(ent[0])
        set_entidades=set(entidades_nodos)
        
        G.add_node(ent[0], tipo=ent[1])
        
        
    
    
for node in G.nodes(data=True):
    print(node)

print(set_entidades)
#%%

#celda 6B para todas las listas menos para entidades totales

#frecuencia de entidades para añadir como atributo
import pandas as pd

df = pd.DataFrame(ubicaciones, columns=['entidades'])
print(df)

#print(df.head())
df['freq_entidades'] = df.groupby('entidades')['entidades'].transform('count')
#cuentas lugares
total_ubicaciones=df['entidades'].nunique()
#print('total de ubicaciones:',total_ubicaciones)
#print(pd.Series(df['entidades'].unique()))

ubicaciones_counts = df.groupby(['entidades']).size().reset_index()
ubicaciones_counts.rename({0: "freq"},axis='columns',inplace=True)
print(ubicaciones_counts)


#%%
#print(df.head())
def frecuencia(df, columna_nueva, columna_objetivo):
    if df[columna_objetivo][1] == 'PER':
        
        #df[columna_nueva] = df.groupby(columna_objetivo)[columna_objetivo].transform('count')
        
    #cuentas lugares
    # total_ubicaciones=df[columna_objetivo].nunique()
    # #print('total de ubicaciones:',total_ubicaciones)
    # #print(pd.Series(df['entidades'].unique()))
    
    # ubicaciones_counts = df.groupby([columna_objetivo]).size().reset_index()
    # ubicaciones_counts.rename({0: "freq"},axis='columns',inplace=True)
    
    # return ubicaciones_counts
        print(df[columna_objetivo])

for ent_type in columns_ents:
    df_per=frecuencia(df, 'frecuencia_per','per')
    df_loc=frecuencia(df, 'frecuencia_per','loc')
    df_org=frecuencia(df, 'frecuencia_per','org')

print(df_per)
#%%
#celda 7

import networkx as nx
import matplotlib.pyplot as plt
#from collections import OrderedDict

G = nx.Graph()

for entity in set(entidades):
    
    
    #print(entity, ubicaciones_counts['freq'].loc[ubicaciones_counts['entidades'] == entity].to_string(index=False))
    #print(entity)
    G.add_node(entity, weight=ubicaciones_counts['freq'].loc[ubicaciones_counts['entidades'] == entity].to_string(index=False))
print(G.nodes(data=True))

#los nodos como estan conectados
#crear pares de vinculos que tengan las entidades filradas después del fuzzy
#y que conecte
#%%
#print(list(G.nodes))
# plt.figure(figsize=(10, 8))
# nx.draw(G, node_size=15)

from itertools import tee

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

for pair in pairwise(entidades):
    print(pair)
        
    G.add_edge(pair[0], pair[1])

#remover las self loops
G.remove_edges_from(nx.selfloop_edges(G))

plt.figure(figsize=(10, 8))
nx.draw_spring(G, node_size=15)

#print(G.degree())

max_1_dc = max(dict(G.degree()).items(), key = lambda x : x[1])
list_dc = sorted(dict(G.degree()).items(), key=lambda kv: (kv[1], kv[0]))
max_20_dc = list(list_dc)[len(list_dc)-20:]


print(max_1_dc)
print(list_dc)
print(max_20_dc)

#%%

#celda 7A

#degree centrality
def find_nodes_with_highest_deg_cent(G):
        
    deg_cent = nx.degree_centrality(G)
    max_1_dc = max(dict(G.degree()).items(), key = lambda x : x[1])
    max_2_dc = list(sorted(deg_cent.values()))[-2]
    max_3_dc = list(sorted(deg_cent.values()))[-3]
    
    maxnode1 = set()
    maxnode2 = set()
    maxnode3 = set()
    
      # Iterate over the degree centrality dictionary
    for k, v in deg_cent.items():
    
        # Check if the current value has the maximum degree centrality
        if v == max_1_dc:
            # Add the current node to the set of nodes
            maxnode1.add(k)
            
        if v == max_2_dc:
            # Add the current node to the set of nodes
            maxnode2.add(k)
            
        if v == max_3_dc:
            # Add the current node to the set of nodes
            maxnode3.add(k)
            
    return maxnode1,maxnode2,maxnode3



top_deg_dc,top2_deg_dc,top3_deg_dc = find_nodes_with_highest_deg_cent(G)

print(top_deg_dc,top2_deg_dc,top3_deg_dc)
#%%

#1 ATRIBUTO DE PESO DE CADA NODO ES SU FRECUENCIA celda 6A
#2 POR CADA TWEET GUARDAR EN UNA LISTA TODOS LOS TIPOS DE ENTIDADES celda 4
#3 ESTA CONECTADO A OTRO NODO SOLO SI APARECE EN EL MISMO TWEET CON OTRA ENTIDAD DE SU MISMO TIPO
#4 ESTA CONECTADO A OTRO NODO SOLO SI APARECE EN EL MISMO TWEET CON OTRA ENTIDAD DE UN TIPO DETERMINADO
#5 ESTA CONECTADO A OTRO NODO SOLO SI APARECE EN EL MISMO TWEET CON OTRA ENTIDAD DE CUALQUIER TIPO

#%%
#opcional para solo tener strings unicas eliminando las más semejantes
#con fuzzywuzzy evalua la similitud entre strings

#traducir a plano sin acentos

#eliminar strings iguales

from fuzzywuzzy import process, fuzz
import unidecode

print(fuzz.token_sort_ratio("guillermo lasso", "lasso"))

from rapidfuzz.process import extract, extract_iter
from rapidfuzz.string_metric import Levenshtein, normalized_levenshtein
from rapidfuzz.fuzz import ratio


compare_list=["guillermo","lasso","guillermito","lassito","guillermo lasso"]
print(process.extractOne("guillermo lasso", compare_list))
#%%
#descomprime las tuplas (chunks) de entidades que devuelve polyglot
#devuelve las entidades dependiendo de si son varias, las une en un solo string
#si es solo una la devuelve, lee por cada sublista en la lista de listas personas por cada item de la sublista
#finalmente genera un set de parabras unicas
#unidecode para eliminar los acentos
choices = set([unidecode.unidecode(' '.join(item)) if len(item) > 1 else unidecode.unidecode(item[0]) for sublist in personas for item in sublist ])
#entidades por cada articulo IMPORTANTE no es un set de elementos unicos
#tiene todas las entidades para luego hacer la extraccion
tweets_ents=[unidecode.unidecode(' '.join(item)) if len(item) > 1 else unidecode.unidecode(item[0]) for sublist in personas for item in sublist ]
#set
print(tweets_ents)
print(choices)
for choice in choices:
    print(choice)

cleaned_entity_list = []

for tweet_ent in tweets_ents:
    print(tweet_ent)
    
    tweet_entities = []
    for ent in set(tweet_ent):
        
        tweet_entities.append(process.extractOne(tweet_ent, choices)[0])
    
    cleaned_entity_list.append(tweet_entities)
    print(cleaned_entity_list)
    
    #%%
    
    print(fuzz.token_sort_ratio("guillermo lasso", "lasso"))
    #%%
    
print(cleaned_entity_list)
#%%

#grafico

import networkx as nx
from operator import itemgetter




        
        