import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from argparse import ArgumentParser
from matplotlib.dates import DateFormatter
import os
import gzip

parser = ArgumentParser()

parser.add_argument('--location', dest= 'location', default= 'Brazil', type= str,
                    help= 'Brazil, US, World')

                  
parser.add_argument('--state_or_city', dest= 'state_or_city', default= 'state', type= str,
                    help= 'Brazil: state or city')

parser.add_argument('--not_last_date', dest= 'not_last_date', default= False, type= bool,
                    help= 'Run for the last date from the data frame')

parser.add_argument('--date', dest= 'until_date', default= '2020-05-05', type= str,
                    help= 'Set --not_last_date to use it')

parser.add_argument('--show_plot', dest= 'show_plot', default= False, type= bool,
                    help= 'Show plot for each location')

parser.add_argument('--output_name', dest= 'output_name', default= 'output.csv', type= str,
                    help= 'CSV file: output.csv')

parser.add_argument('--slice', dest= 'slice', default= False, type= bool,
                    help= 'Set true and use --slice_list')


parser.add_argument('--slice_name', dest= 'slice_list', nargs='+', default=[])

parser.add_argument('--save_figdata', dest= 'save_figdata', default= False, type= bool,
                    help= 'Save figure data')


args = parser.parse_args()

last = not args.not_last_date


path_data = 'data'
path_output = 'results/dfs'

if not os.path.exists(path_data):
    os.makedirs(path_data)

if not os.path.exists(path_output):
    os.makedirs(path_output)


####### Defining useful functions

### Function to download data
def download_df(url, filename):
    with open(filename, 'wb') as f:
        r = requests.get(url)
        f.write(r.content)

    #return pd.read_csv(filename)

### Function to calculate rates
def delta(df_conf):
    list_ = []
    list_.append(0)
    for j in range(len(df_conf) - 1):
        list_.append(df_conf[j+1] - df_conf[j])
    return list_  

####### Downloading data

print('Loading data for calculating growth rate')

if args.location == 'Brazil':

    ###################################

    #url = "https://data.brasil.io/dataset/covid19/caso_full.csv.gz"
    #filename =  path_data + '/caso_full.csv.gz'
    
    #download_df(url, filename)
    #############################################
    #with gzip.open(filename) as f:
    #    df = pd.read_csv(f)
    
    ###################################
    filename = 'data/caso_full.csv.gz'

    with gzip.open(filename) as f:
        df = pd.read_csv(f)

    #df = pd.read_csv(filename)

    
    
    cases_key = 'last_available_confirmed'
    #cases_key = 'confirmed'

    #df = download_df(url, filename)

    #print('df = ', df)
    #df = pd.read_csv(filename)
    df = df[ df['place_type'] == args.state_or_city]

    if args.state_or_city == 'state':
        df['city_ibge_code'] = df['city_ibge_code'].astype(int)
        df = df.drop(columns= ['city'])
        loc_key = 'state'

    else:
        df = df.dropna()
        df['city_ibge_code'] = df['city_ibge_code'].astype(int)
        loc_key  = 'city_ibge_code'
        args.slice_list= list(map(int, args.slice_list))


elif args.location == 'World':
    url = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'
    filename = path_data + '/world_' + url.split("/")[-1]
    cases_key = 'total_cases'
    loc_key = 'location'
    df = download_df(url, filename)

else:
    url = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv"
    filename = path_data + '/US_states' + '.csv'
    cases_key = 'cases'
    loc_key = 'state'
    df = download_df(url, filename)


if args.slice:
    locs_ = np.array(args.slice_list)
else:
    locs_ = df[loc_key].unique()


print('Running')

results_ = []

### EWM
alpha = 30.
alpha2 = 7.
alpha3 = 5.

### Classification threshold
# Cases
CASES_threshold = 50
# Normalized growth rate (%)
GROWTH_RATE_threshold = 1
# Normalized growth acceleration (%)
GROWTH_ACCEL_threshold = 0.01


for locs in locs_:

    df_ = df[ df[loc_key] == locs].sort_values(by='date').set_index('date')
    df_ =  df_ [ df_[cases_key]  > 0]
    df_.index = pd.to_datetime(df_.index)


    if args.state_or_city == 'city':
        cities = df_['city'][0]
        print(cities)
    else:
        print(locs)

    # Rate
    df_['growth_rate_'] = delta(df_[cases_key])

    # Exponential weight
    df_['growth_rate'] = df_['growth_rate_'].ewm(com= alpha).mean()

    # Rate
    df_['growth_accel_'] = delta(df_['growth_rate'])

    # Exponential weight
    df_['growth_accel'] = df_['growth_accel_'].ewm(com= alpha2).mean()

    # Rate
    df_['growth_accel_rate_'] = delta(df_['growth_accel'])

    # Exponential weight
    df_['growth_accel_rate'] = df_['growth_accel_rate_'].ewm(com= alpha3).mean()

    # Dropping unsmoothed quantities
    df_ = df_.drop('growth_rate_', axis=1)
    df_ = df_.drop('growth_accel_', axis=1)
    df_ = df_.drop('growth_accel_rate_', axis=1)

    # Normalized quantities
    df_['growth_rate_NORM_(%)'] = 100*df_['growth_rate'] / df_[cases_key]
    df_['growth_accel_NORM_(%)'] = 100*df_['growth_accel'] / df_[cases_key]
    df_['growth_accel_rate_NORM_(%)'] = 100*df_['growth_accel_rate'] / df_[cases_key]

    # Classification

    df_['LENTO'] =  (df_[cases_key] < CASES_threshold)*1

    df_['EXPONENCIAL'] = ( df_['growth_accel_NORM_(%)']  >= GROWTH_ACCEL_threshold )*1

    df_['DESACELERACAO'] = ( df_['growth_accel_NORM_(%)']  <= -GROWTH_ACCEL_threshold )*1

    df_['ESTAVEL'] = ( (df_['growth_rate_NORM_(%)']  <  GROWTH_RATE_threshold) &  (df_['growth_accel_NORM_(%)']  > - GROWTH_ACCEL_threshold) & (df_['growth_accel_NORM_(%)']  < GROWTH_ACCEL_threshold) )*1

    df_['LINEAR'] = ( (df_['growth_rate_NORM_(%)']  >=  GROWTH_RATE_threshold ) &  (df_['growth_accel_NORM_(%)']  > - GROWTH_ACCEL_threshold) & (df_['growth_accel_NORM_(%)']  < GROWTH_ACCEL_threshold) )*1


    df_.loc[df_['LENTO'] == True, 'CLASSIFICACAO'] = 'LENTO'
    df_.loc[ (df_['LENTO'] != True) & (df_['DESACELERACAO'] == True), 'CLASSIFICACAO'] = 'DESACELERACAO'
    df_.loc[ (df_['LENTO'] != True) & (df_['ESTAVEL'] == True), 'CLASSIFICACAO'] = 'ESTAVEL'
    df_.loc[ (df_['LENTO'] != True) & (df_['LINEAR'] == True), 'CLASSIFICACAO' ] = 'LINEAR'
    df_.loc[ (df_['LENTO'] != True) & (df_['EXPONENCIAL'] == True), 'CLASSIFICACAO' ] = 'EXPONENCIAL'

    # Plot
    if args.show_plot:
        
        fig, axes = plt.subplots(1, 4, figsize= (24, 4))    
        axes[0].plot(df_['growth_rate'])
        if args.state_or_city == 'city':
            axes[0].set_title(cities, fontsize= 16)
        else:
            axes[0].set_title(locs, fontsize= 16)
        axes[0].set_ylabel('Daily New Cases', fontsize= 13)
        axes[0].set_xlabel('Date', fontsize= 13)
        axes[0].grid(linestyle=':')

        axes[1].plot(df_['growth_accel'], color= 'C3')
        if args.state_or_city == 'city':
            axes[1].set_title(cities, fontsize= 16)
        else:
            axes[1].set_title(locs, fontsize= 16)
        axes[1].set_ylabel('Growth acceleration', fontsize= 13)
        axes[1].set_xlabel('Date', fontsize= 13)
        axes[1].grid(linestyle=':')

        axes[2].plot(df_['growth_accel_rate'], color= 'C2')
        if args.state_or_city == 'city':
            axes[2].set_title(cities, fontsize= 16)
        else:
            axes[2].set_title(locs, fontsize= 16)
        axes[2].set_ylabel('Growth acceleration rate', fontsize= 13)
        axes[2].set_xlabel('Date', fontsize= 13)
        axes[2].grid(linestyle=':')

        axes[3].plot(df_[[cases_key]], 'o-', color= 'C1', label= 'Official data')
        axes[3].set_ylabel('Confirmed cases', fontsize= 12)
        if args.state_or_city == 'city':
            axes[3].set_title(cities, fontsize= 16)
        else:
            axes[3].set_title(locs, fontsize= 16)
        axes[3].set_xlabel('Date', fontsize= 13)
        axes[3].grid(linestyle=':')
        axes[3].legend()

        date_form = DateFormatter("%m/%d")
        axes[0].xaxis.set_major_formatter(date_form)
        axes[1].xaxis.set_major_formatter(date_form)
        axes[2].xaxis.set_major_formatter(date_form)
        axes[3].xaxis.set_major_formatter(date_form)

    if args.save_figdata:
        path_fig_data = 'results/figures'
        if not os.path.exists(path_fig_data):
            os.makedirs(path_fig_data)

        if args.state_or_city == 'city':
            df_.to_csv(path_fig_data + '/df_figures_%s.csv' % str(cities), index= True, sep= ';')
        else:
            df_.to_csv(path_fig_data + '/df_figures_%s.csv' % str(locs), index= True, sep= ';')

    df_ = df_.reset_index()

    if last:
        results_.append(df_.iloc[-1].to_dict())
    else:
        idx_ = df_.index[df_['date'] == args.until_date]
        if len(idx_) > 0:
            results_.append(df_.iloc[idx_[0]].to_dict())
        else:
            print('%s data NOT available for %s' % (args.until_date, locs))

    
    plt.show()


results = pd.DataFrame(results_)

results.to_csv(path_output + '/' + args.output_name)









