import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from argparse import ArgumentParser
import gzip
import os

parser = ArgumentParser()

parser.add_argument('--increment', dest= 'INCREM', default= False, type= bool,
                    help= 'increment')

args = parser.parse_args()

path_data = 'data'
path_output = 'results/dfs'
path_input = 'results/dfs/csv'

if not os.path.exists(path_data):
    os.makedirs(path_data)

if not os.path.exists(path_output):
    os.makedirs(path_output)


### Function to download data
def download_df(url, filename, sep= False):
    with open(filename, 'wb') as f:
        r = requests.get(url)
        f.write(r.content)

    #if sep:
    #    out = pd.read_csv(filename, sep= ';')
    #else:
    #    out = pd.read_csv(filename)
    #return out

### ICU prob
df_age_ICU = pd.DataFrame(columns=['Age', 'ICU_prob'])
df_age_ICU['Age'] = ['0-19', '20-44', '45-54', '55-64', '65-74', '75-84', '85+']
df_age_ICU['ICU_prob'] = [0, 4.2, 10.4, 11.2, 18.8, 31, 29]

ICU_prob = [0., 0., 0., 0., 0.042, 
            0.042, 0.042, 0.042, 0.042, 0.104,
            0.104, 0.112, 0.112, 0.188, 0.188,
            0.31, 0.31, 0.29, 0.29]
### ICU period
T_ICU = 14
### p SUS
df_SUS = pd.read_csv('data/SUS_depend.csv', index_col= 'Estado')

print('Downloading update data')

url = "https://data.brasil.io/dataset/covid19/caso_full.csv.gz"
filename =  path_data + '/caso_full.csv.gz'

download_df(url, filename)
#############################################
with gzip.open(filename) as f:

    df = pd.read_csv(f)

#print(df)

##########################################


###############################################################################

#df = download_df(url, filename)

#filename = 'data/covid19-4a340474e3204299b9cec2d82f237107.csv'
#df = pd.read_csv(filename)

df = df[ df['place_type'] == 'city']

df = df.dropna(subset=['city_ibge_code'])
df.city_ibge_code = df.city_ibge_code.astype(int)

###############################################################################

#url2 = 'https://covid-insumos.saude.gov.br/paineis/insumos/lista_csv_painel.php?output=csv'
#filename2 = path_data + '/brazil_covid_insumos.csv'

#df_covidBR__ = download_df(url2, filename2, sep= True)

#df_covidBR_ = df_covidBR__[['uf', 'Leitos UTI adulto', 'UTI adulto SUS', 'Uti adulto não SUS']]

#df_covidBR = df_covidBR_.set_index('uf')

file_name = 'data/A175709189_28_143_208.csv'
df_covidBR = pd.read_csv(file_name)
df_covidBR = df_covidBR.rename(columns={'Quantidade_existente':'leitos'})
df_covidBR = df_covidBR.dropna(subset=['leitos'])
df_covidBR.leitos = df_covidBR.leitos.astype(int)

###############################################################################

def ICU_samp(df, n, n_samp_AGE_max= 100, n_samp_AGE_min= 100, n_samp_ICU_max= 100, n_samp_ICU_min= 100):

    df_samp = pd.DataFrame(columns= df['Age'])

    for j in range(n_samp_AGE_max):

    
        samp = np.random.choice(df['Age'], 
                                n, 
                                p= list(df['AGE_prob']) )

        unique, counts = np.unique(samp, return_counts= True)
    
        for l in range(len(unique)):
            df_samp.loc[j, unique[l]] = counts[l]


    df_samp = df_samp.fillna(0)

    df['n_mean'] = list(df_samp.mean(axis= 0))
    df['n_std']  = list(df_samp.std(axis =0))

    df = df.set_index('Age')


    for age in df.index:
    
        aux_ = []

        for j in range(n_samp_ICU_max):    
   
            samp = np.random.uniform(size= int(df.loc[age]['n_mean']))  
            samp_ICU = samp < df.loc[age]['ICU_prob']
            aux_.append(samp_ICU.sum())


        df.loc[age, 'n_mean_ICU']  = np.mean(aux_)
        df.loc[age, 'n_std_ICU']   = np.std(aux_)
        
    df['n_std_ICU'] = np.sqrt( df['n_std']**2 +  df['n_std_ICU']**2)
    
    return df

###############################################################################

def daily_av(df_, date, SUS= True, p_SUS= 0.6278, n_samp_max= 100, n_samp_min= 100):

    n_mean_    = []
    n_std_     = []
    n_mean_ICU_ = []
    n_std_ICU_  = []
    date_ = []

    n_mean_.append(df_['n_mean'].sum())
    n_std_.append(np.sqrt((df_['n_std']**2).sum()))
    n_mean_ICU_.append(df_['n_mean_ICU'].sum())
    n_std_ICU_.append(np.sqrt((df_['n_std_ICU']**2).sum()))
    date_.append(date)

    names = ['date', 'n_mean', 'n_std', 'n_mean_ICU', 'n_std_ICU']
    df_ICU = pd.DataFrame(columns= names)

    df_ICU['date'] = date_

    
    df_ICU['n_mean']     = n_mean_
    df_ICU['n_std']      = n_std_
    df_ICU['n_mean_ICU'] = n_mean_ICU_
    df_ICU['n_std_ICU']  = n_std_ICU_

    df_ICU = df_ICU.set_index(['date'])
    df_ICU.index = pd.to_datetime(df_ICU.index)

    if SUS:

        aux_ = []

        for j in range(n_samp_max):

            samp = np.random.uniform(size= int(df_ICU['n_mean_ICU']))  
            SUS_samp = samp <= p_SUS
            aux_.append(SUS_samp.sum())

        n_mean_ICU_SUS_ = []
        n_std_ICU_SUS_ = []

        n_mean_ICU_SUS_.append(np.mean(aux_))
        n_std_ICU_SUS_.append(np.std(aux_))

        df_ICU['n_mean_ICU_SUS'] =  n_mean_ICU_SUS_
        df_ICU['n_std_ICU_SUS']  = n_std_ICU_SUS_

        df_ICU['n_std_ICU_SUS'] = np.sqrt( df_ICU['n_std_ICU']**2 +  df_ICU['n_std_ICU_SUS']**2)

    return df_ICU

###############################################################################

def correction(x, df_, T_ICU= 14):

    df_.loc[df_.index[x] , 'n_mean_ICU_cor'] = 0.
    df_.loc[df_.index[x], 'n_std_ICU_cor'] = 0.
    
    if x <= T_ICU:
                   
        df_.loc[df_.index[x], 'n_mean_ICU_cor'] = df_.loc[df_.index[x], 'n_mean_ICU']
        df_.loc[df_.index[x], 'n_std_ICU_cor']  = df_.loc[df_.index[x], 'n_std_ICU']
             
    else:
        
        delta = df_.loc[df_.index[x], 'n_mean_ICU'] - df_.loc[df_.index[x - T_ICU], 'n_mean_ICU']
        
        df_.loc[df_.index[x], 'n_mean_ICU_cor'] = np.heaviside(delta, 0) * delta
        df_.loc[df_.index[x], 'n_std_ICU_cor']  = np.sqrt(df_.loc[df_.index[x], 'n_std_ICU']**2 + df_.loc[df_.index[x - T_ICU], 'n_std_ICU']**2)




###############################################################################

states = [4202404, 5300108, 5002704, 5103403, 4106902, 4314902, 4205407, 1501402, 3509502, 3543402, 3548906, 3550308, 3552205, 3505708, 3506003, 3106200, 1400100, 2304400, 5208707, 3518800, 2507507, 1302603, 2408102, 1100205, 2611606, 3304557, 2927408, 3547809, 3548708, 3549904, 2111300]

#states = [4202404]

column_names = ['n_mean', 'n_std', 'n_mean_ICU', 'n_std_ICU', 'uf']
DF3 = pd.DataFrame(columns= column_names)

print('Done!')

print('Running ICU estimation')

#print(df['city_ibge_code'])


for name in states:

    print('Sampling ', name)
    
    df2 = df[ df['city_ibge_code'] == name ]

    #print('df2:')

    #print(df2)

    df_I = df2.groupby('date')[['last_available_confirmed', 'last_available_deaths', 'estimated_population_2019', 'last_available_confirmed_per_100k_inhabitants', 'last_available_death_rate']].sum()
    df_I = df_I[ df_I['last_available_confirmed'] > 0]
    df_I.index = pd.to_datetime(df_I.index)

    #fit_until = df_I.index[-1].strftime('%m-%d')

    state_name =  df2['state'].iloc[0]

    file = 'data/pop_age_str_IBGE_2020_' + state_name + '.csv'
    df_age = pd.read_csv(file)
    df_age.loc[0, 'Age'] = '00-04'
    df_age.loc[1, 'Age'] = '05-09'
    df_age['AGE_prob'] = df_age['Total'] / df_age['Total'].sum().item()
    pop0 = df_age['Total'].sum().item()
    df_age['ICU_prob'] = ICU_prob   

    ########
    p_SUS = df_SUS.loc[state_name ].item()
    ########

    if not args.INCREM:

        column_names = ['n_mean', 'n_std', 'n_mean_ICU', 'n_std_ICU']
        DF2 = pd.DataFrame(columns= column_names)

        for j in range(len(df_I)):

           
            
            df1 = ICU_samp(df= df_age.reset_index(), 
                        n= int(df_I.iloc[j][0]), 
                        n_samp_AGE_max= 100, n_samp_AGE_min= 100,
                        n_samp_ICU_max= 100, n_samp_ICU_min= 100)

            DF1 = daily_av(df1, date= df_I.index[j], SUS= True, p_SUS= p_SUS, n_samp_max= 100, n_samp_min= 100)

            DF2 = DF2.append(DF1)

        for k in range(len(DF2)):
            correction(k, DF2, T_ICU= 14)

        #file1 = 'results/dfs/ICU_' + 'CITIES_' + str(name) + '_fit_until_' + fit_until + '.csv'
        file1 = 'results/dfs/ICU_' + 'CITIES_' + str(name) + '.csv'

        DF2 = DF2.join(df_I)

        DF2.to_csv(file1)

    else:

        fit_until2 = df_I.index[-2].strftime('%m-%d')

        if name == 'MG':
            fit_until2 = '06-15'

        file2 = 'results/dfs/ICU_' + 'state_' + name + '_fit_until_' + fit_until2 + '.csv'

        DF_load = pd.read_csv(file2)

        DF_NEW = ICU_samp(df= df_age.reset_index(), 
                          n= int(df_I.iloc[-1][0]), 
                          n_samp_AGE_max= 100, n_samp_AGE_min= 100,
                          n_samp_ICU_max= 100, n_samp_ICU_min= 100)

        DF_NEW1 = daily_av(DF_NEW, date= df_I.index[-1], SUS= True, p_SUS= p_SUS, n_samp_max= 100, n_samp_min= 100)

        DF2_NEW = DF_load.append(DF_NEW1)

        correction(len(DF2_NEW) - 1, DF2_NEW, T_ICU= 14)

        #file2 = 'results/dfs/df_ICU_' + 'state_' + name + '_fit_until_' + fit_until + '.csv'
        #DF2_NEW.to_pickle(file2)


        ##################################
        aux_ = []
        for l in range(len(DF2_NEW)):
            aux_.append(name)

        DF2_NEW['uf'] = aux_

        DF3 = DF3.append(DF2_NEW.iloc[-1])


if args.INCREM:

    DF4 = DF3.set_index('uf')
    DF5 = DF4.join(df_covidBR)

    DF6 = DF5.reset_index()

    file__ = 'results/dfs/df_ICU_' + 'CITIES_' + 'fit_until_' + fit_until + '.csv'
    DF6.to_csv(file__)

###########################

aux_uf_ = []
DF4 = pd.DataFrame(columns= DF2.columns.values)

for name in states:
    
    file__ = 'results/dfs/ICU_' + 'CITIES_' + str(name) + '.csv'

    df__ = pd.read_csv(file__)
    df__ = df__.rename(columns={'Unnamed: 0':'date'})

    DF4 = DF4.append(df__.iloc[-1])

    aux_uf_.append(name)

DF4['city_ibge_code'] = aux_uf_

DF4 =  DF4.set_index('city_ibge_code')


DF5 = DF4.join(df_covidBR)

file5 = 'results/dfs/df_ICU_' + 'states_ICU.csv'
DF5.to_csv(file5)
 




###########################









        





