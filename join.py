import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.optimize import minimize 
from datetime import timedelta
import datetime as dt
from IPython.display import Image
import requests

state_or_city = 'city'

states = [4202404, 5300108, 5002704, 5103403, 4106902, 4314902, 4205407, 1501402, 3509502, 3543402, 3548906, 3550308, 3552205, 3505708, 3506003, 3106200, 1400100, 2304400, 5208707, 3518800, 2507507, 1302603, 2408102, 1100205, 2611606, 3304557, 2927408, 3547809, 3548708, 3549904, 2111300]

df__ = pd.read_csv('results/dfs/ICU_CITIES_1302603.csv')
df__ = df__.rename(columns={'Unnamed: 0':'date'})

file_name = 'data/A175709189_28_143_208.csv'
df_covidBR = pd.read_csv(file_name)
df_covidBR = df_covidBR.rename(columns={'Quantidade_existente':'leitos'})
df_covidBR = df_covidBR.dropna(subset=['leitos'])
df_covidBR.leitos = df_covidBR.leitos.astype(int)
df_covidBR['city_IBGE_code'] = df_covidBR['Municipio'].str.split().str[0]
df_covidBR['city'] = df_covidBR['Municipio'].str.split().str[1]

df_covidBR = df_covidBR.drop(columns= ['Municipio'])


leitos_ = []
ibge_code = []
DF3 = pd.DataFrame(columns= df__.columns.values)

for name in states:

    file_ = 'results/dfs/ICU_CITIES_' + str(name) + '.csv'

    df__ = pd.read_csv(file_)
    df__ = df__.rename(columns={'Unnamed: 0':'date'})

    DF3 = DF3.append(df__.iloc[-1])

    leitos_.append(df_covidBR[df_covidBR['city_IBGE_code'] == str(name)[0:6]]['leitos'].item())
    ibge_code.append(name)

DF3['leitos'] = leitos_
DF3['city_IBGE_code'] = ibge_code

DF3 = DF3.set_index('city_IBGE_code')

df_growth = pd.read_csv('results/dfs/output.csv')

df_growth2 = df_growth[['city', 'state', 'epidemiological_week', 'city_ibge_code', 'growth_accel_rate', 'growth_rate_NORM_(%)', 'growth_accel_NORM_(%)',
                        'growth_accel_rate_NORM_(%)', 'LENTO', 'EXPONENCIAL', 'DESACELERACAO', 'ESTAVEL', 'LINEAR', 'CLASSIFICACAO' ]]


df_growth3 = df_growth2[ df_growth2['city_ibge_code'].isin(np.array(DF3.index)) ]


df_growth4 = df_growth3.set_index('city_ibge_code')

df_growth5 = df_growth4.join(DF3)

df_UTI = pd.read_csv('data/sites.csv')


df_UTI2 = df_UTI[['city_ibge_code', 'UTI_adulto_COVID19']]
df_UTI2 = df_UTI2.set_index('city_ibge_code')

df_growth6 = df_growth5.join(df_UTI2)

df_growth6.to_csv('results/dfs/results_cities_ICU_NEW_.csv')