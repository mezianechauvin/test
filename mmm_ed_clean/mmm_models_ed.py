
from dotenv import load_dotenv
load_dotenv(verbose=True, override=True, interpolate=True)
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, mutual_info_score
import re
import calendar
import seaborn as sns
import matplotlib.pyplot as plt
import json
import requests
import binascii
import io
import os
import sys
import subprocess
import atexit
from urllib.request import urlopen
from bs4 import BeautifulSoup
from PIL import Image
from IPython.display import display
import gspread
import gspread_dataframe
from pprint import pprint
from time import sleep
from functools import reduce
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from permetrics.regression import RegressionMetric
import math
import pickle
from maulibs.GSheet import GSheet


print(pd.__version__)
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 100)


# gsheets must have access to googlesheet-automations@hf-mau-dev.iam.gserviceaccount.com
# channel hieararchy columns
vs_ch_cols = ['vs_campaign_type', 'vs_channel_medium', 'vs_channel_category', 'vs_channel', 'vs_channel_split', 'vs_partner']
ch_cols = [i[3:] for i in vs_ch_cols]
mel_plat_ch_cols = ['campaign_type', 'channel', 'channel_split', 'partner']

def flat_cols(self):
    """Monkey patchable function onto pandas dataframes to flatten MultiIndex column names.

    pd.DataFrame.flatten_columns = flatten_columns
    """
    df = self.copy()
    df.columns = [
        '_'.join([str(x)
                  for x in [y for y in item
                            if y]]) if not isinstance(item, str) else item
        for item in df.columns
    ]
    return df

pd.DataFrame.flat_cols = flat_cols

def read_end_to_end_data(input_gsheet_url):
    
    input_gs_client = GSheet(input_gsheet_url)
    input_title = input_gs_client.spreadsheet.title
    
    param_df = input_gs_client.read_dataframe('params', header_row_num=0)
    
    feature_col_df = input_gs_client.read_dataframe('feature_list', header_row_num=0)
    
    model_df = input_gs_client.read_dataframe('model_data', header_row_num=0)
    model_df = model_df.reset_index(drop=True)
    model_df['date'] = pd.to_datetime(model_df['date'])
    for feature in list(feature_col_df['feature'].values) + ['n_conversion']:
        model_df[feature] = model_df[feature].astype('float')
    
    spend_fee_df = input_gs_client.read_dataframe('fee_credit_data', header_row_num=0)
    spend_fee_df = spend_fee_df.reset_index(drop=True)
    for col in spend_fee_df.columns:
        if 'spend' in col:
            spend_fee_df[col] = spend_fee_df[col].astype('float')
    
    
    return input_title, param_df, feature_col_df, model_df, spend_fee_df

def adhoc_lift_test_validation():
    
    # cac_df = pd.read_csv(Path('pymc_marketing_output/weekly_data_run_2024_03_25/cac_results.csv'))
    # cac_df
    
    cac_dfs = []
    len(cac_dfs)
    for i in range(1, 5):
        cac_df = pd.read_csv(Path(f'robyn_output/weekly_data_run_2024_03_25_spend_q{i}/select_model/cac_results.csv'))
        cac_dfs.append(cac_df)
    
    cac_df = pd.concat(cac_dfs, ignore_index=True)
    cac_df
    
    cac_dfs[3]
    
    cols = ['direct_mail', 'sea_brand']
    met_cols = [(f'{col}_mel_spend', f'{col}_mel_spend_n_conversion', f'{col}_mel_spend_cac') for col in cols]
    # list of tuples to a single list of values
    met_cols = list(reduce(lambda x, y: x + y, met_cols))
    met_cols
    
    cac_df = cac_df[['out_of_sample_period', 'date', 'iso_year_week'] + met_cols]
    cac_df
    
    gsheet_url = 'https://docs.google.com/spreadsheets/d/1NH6ejWvNDDIZbOra6722ohyKQyoc_0MfO8OJb8xmbN0'
    gs_client = GSheet(gsheet_url)
    ltv_df = gs_client.read_dataframe('lift_test_validation_data', header_row_num=0).reset_index(drop=True)
    # replace dollar sign commas etc. and convert to float in the icac column
    ltv_df['spend'] = ltv_df['spend'].str.replace(r'[$,]', '', regex=True).astype(float)
    ltv_df['conversion_lift'] = ltv_df['conversion_lift'].str.replace(r'[$,]', '', regex=True).astype(float)
    ltv_df['icac'] = ltv_df['icac'].str.replace(r'[$,]', '', regex=True).astype(float)
    ltv_df = ltv_df[['group', 'start_iso_year_week', 'spend', 'conversion_lift', 'icac']]
    ltv_df
    
    dm_ltv_df = (ltv_df.loc[ltv_df['group'] == 'direct_mail', ['start_iso_year_week', 'spend', 'conversion_lift', 'icac']]
     .rename(columns={'spend': 'direct_mail_spend', 
                      'conversion_lift': 'direct_mail_conversion_lift',
                      'icac': 'direct_mail_icac'}))
    
    sea_ltv_df = (ltv_df.loc[ltv_df['group'] == 'sea_brand', ['start_iso_year_week', 'spend', 'conversion_lift', 'icac']]
     .rename(columns={'spend': 'sea_brand_spend', 
                      'conversion_lift': 'sea_brand_conversion_lift',
                      'icac': 'sea_brand_icac'}))
    
    cac_df = pd.merge(cac_df, dm_ltv_df, how='left', left_on='iso_year_week', right_on='start_iso_year_week').drop(columns='start_iso_year_week')
    cac_df = pd.merge(cac_df, sea_ltv_df, how='left', left_on='iso_year_week', right_on='start_iso_year_week').drop(columns='start_iso_year_week')
    cac_df
    
    # percent error by  mape
    cac_df['direct_mail_ape'] = ((cac_df['direct_mail_icac'] - cac_df['direct_mail_mel_spend_cac']).abs() / cac_df['direct_mail_icac'])
    cac_df['sea_brand_ape'] = ((cac_df['sea_brand_icac'] - cac_df['sea_brand_mel_spend_cac']).abs() / cac_df['sea_brand_icac'])
    
    # cac_df.to_clipboard(index=False)
    
    cac_df.groupby('out_of_sample_period')[['direct_mail_ape', 'sea_brand_ape']].mean().to_clipboard(index=True)
    
    
    
    pass

def adhoc_oos_data_range():
    
    input_gsheet_url = 'https://docs.google.com/spreadsheets/d/1NH6ejWvNDDIZbOra6722ohyKQyoc_0MfO8OJb8xmbN0'
    
    input_title, param_df, feature_col_df, model_df, spend_fee_df = read_end_to_end_data(input_gsheet_url=input_gsheet_url)
    model_df = model_df.reset_index(drop=True)
    print(input_title)
    print(model_df.head())
    
    oos_col = 'out_of_sample_q1'
    
    model_df[model_df[oos_col] == 'train'].describe()
    
    feature_cols = list(feature_col_df['feature'].values)
    
    model_df.columns.name = 'features'
    model_df
    
    oos_range_dfs = []
    for oos_col in ['out_of_sample_q1', 'out_of_sample_q2', 'out_of_sample_q3', 'out_of_sample_q4']:
        tr_min_df = model_df.loc[model_df[oos_col] == 'train', feature_cols].min()
        tr_min_df.name = 'train_feature_min'
        tr_max_df = model_df.loc[model_df[oos_col] == 'train', feature_cols].max()
        tr_max_df.name = 'train_feature_max'
        te_min_df = model_df.loc[model_df[oos_col] == 'test', feature_cols].min()
        te_min_df.name = 'test_feature_min'
        te_max_df = model_df.loc[model_df[oos_col] == 'test', feature_cols].max()
        te_max_df.name = 'test_feature_max'
        
        oos_range_df = pd.concat([tr_min_df, tr_max_df, te_min_df, te_max_df], axis=1).reset_index()
        oos_range_df['min_in_range'] = (oos_range_df['test_feature_min'] >= oos_range_df['train_feature_min']) & (oos_range_df['test_feature_min'] <= oos_range_df['train_feature_max'])
        # max_in_range
        oos_range_df['max_in_range'] = (oos_range_df['test_feature_max'] >= oos_range_df['train_feature_min']) & (oos_range_df['test_feature_max'] <= oos_range_df['train_feature_max'])
        oos_range_df['out_of_sample_period'] = oos_col
        info_cols = ['out_of_sample_period', 'features']
        oos_range_df = oos_range_df[info_cols + ['train_feature_min', 'train_feature_max', 'test_feature_min', 'test_feature_max', 'min_in_range', 'max_in_range']]
        oos_range_dfs.append(oos_range_df)
    
    
    oos_range_df = pd.concat(oos_range_dfs, axis=0, ignore_index=True)
    oos_range_df[(oos_range_df[['min_in_range', 'max_in_range']] == False).any(axis=1)].to_clipboard(index=False)
    
    pass

def example_robyn_model():
    
    from robyn_api.python_helper import *
    
    # https://github.com/facebookexperimental/Robyn/blob/main/robyn_api/robyn_python_notebook.ipynb
    
    # create folder for today's date in the robyn_output folder
    today_date = datetime.now().strftime('%y%m%d')
    today_folder = Path(f'robyn_output/run_{today_date}')
    today_folder.mkdir(parents=True, exist_ok=True)
    robyn_directory = str(Path('/Users/eddiedeane/PycharmProjects/mmm_ed/') / today_folder)
    create_files = True
    
    
    def terminate_process(proc):
        if proc.poll() is None:
            proc.kill() 

    def kill_process_and_show_message(proc):
        if proc.poll() is None:
            try:
                proc.kill()
                print(f"Process killed successfully. pid: {proc.pid}")
            except Exception as e:
                print(f"Failed to kill the process: {e}")
        else:
            print("Process is not running or already terminated.")
    
    p = subprocess.Popen(['Rscript', 'robyn_api/robynapi_call.R'])
    print('starting 30 sec sleep')
    sleep(30)
    print('done sleep')
    atexit.register(terminate_process, p)
    print('running robynapi_call.R')
    
    api_base_url = 'http://127.0.0.1:9999/{}'
    url = api_base_url[:-3] + '/openapi.json'
    html = urlopen(url).read()
    soup = BeautifulSoup(html, features='html.parser')
    apis = json.loads(soup.contents[0])
    for i in apis['paths'].keys():
        print(api_base_url.format(i[1:]))
        
    robyn_api('robyn_version')
    
    dt_simulated_weekly = pandas_builder(robyn_api('dt_simulated_weekly'))
    dt_simulated_weekly.head()
    
    dt_prophet_holidays = pandas_builder(robyn_api('dt_prophet_holidays'))
    dt_prophet_holidays.head()
    dt_prophet_holidays[dt_prophet_holidays['year'] == 2024]
    
    # Define Args for robyn_inputs()
    input_args = {
        "date_var": "DATE", # date format must be "2020-01-01"
        "dep_var": "revenue", # there should be only one dependent variable
        "dep_var_type": "revenue", # "revenue" (ROI) or "conversion" (CPA)
        "prophet_vars": ["trend", "season", "holiday"], # "trend","season", "weekday" & "holiday"
        "prophet_country": "DE", # input country code. Check: dt_prophet_holidays
        "context_vars" : ["competitor_sales_B", "events"], # e.g. competitors, discount, unemployment etc
        "paid_media_spends": ["tv_S", "ooh_S", "print_S", "facebook_S", "search_S"], # mandatory input
        "paid_media_vars": ["tv_S", "ooh_S", "print_S", "facebook_I", "search_clicks_P"], # mandatory.
        # paid_media_vars must have same order as paid_media_spends. Use media exposure metrics like
        # impressions, GRP etc. If not applicable, use spend instead.
        "organic_vars" : "newsletter", # marketing activity without media spend
        # "factor_vars" : ["events"], # force variables in context_vars or organic_vars to be categorical
        "window_start": "2016-01-01",
        "window_end": "2018-12-31",
        "adstock": "geometric" # geometric, weibull_cdf or weibull_pdf.
    }
    
    # Build the payload for the robyn_inputs()
    payload = {
        'dt_input' : asSerialisedFeather(dt_simulated_weekly), 
        'dt_holiday' : asSerialisedFeather(dt_prophet_holidays), 
        'jsonInputArgs' : json.dumps(input_args)
    }
    
    # Get response
    input_collect = robyn_api('robyn_inputs',payload=payload)
    print(input_collect.keys())
    
    # Build the payload for the hyper_names()
    payload = {
        'adstock' : input_collect['adstock'], 
        'all_media' : json.dumps(input_collect['all_media'])
    }

    # Get response
    hyper_names = robyn_api('hyper_names',payload=payload)
    print(hyper_names)
    
    # Set Args for robyn_inputs()
    input_args = {
        "hyperparameters" : {
            "facebook_S_alphas" : [0.5, 3],
            "facebook_S_gammas" : [0.3, 1],
            "facebook_S_thetas" : [0, 0.3],
            "print_S_alphas" : [0.5, 3],
            "print_S_gammas" : [0.3, 1],
            "print_S_thetas" : [0.1, 0.4],
            "tv_S_alphas" : [0.5, 3],
            "tv_S_gammas" : [0.3, 1],
            "tv_S_thetas" : [0.3, 0.8],
            "search_S_alphas" : [0.5, 3],
            "search_S_gammas" : [0.3, 1],
            "search_S_thetas" : [0, 0.3],
            "ooh_S_alphas" : [0.5, 3],
            "ooh_S_gammas" : [0.3, 1],
            "ooh_S_thetas" : [0.1, 0.4],
            "newsletter_alphas" : [0.5, 3],
            "newsletter_gammas" : [0.3, 1],
            "newsletter_thetas" : [0.1, 0.4],
            "train_size": [0.5, 0.8]
        }
    }
    
    # Build the payload for the hyper_names()
    payload = {
        'InputCollect' : json.dumps(input_collect), 
        'jsonInputArgs' : json.dumps(input_args)
    }
    
    # Get response
    input_collect = robyn_api('robyn_inputs',payload=payload)
    print(input_collect.keys())
    
    
    render_save_spendexposure(robyn_directory, InputJson=input_collect,max_size=(1000,1500))
    
    runArgs = {
        "iterations" : 2000, # NULL defaults to (max available - 1)
        "trials" : 5, # 5 recommended for the dummy dataset
        "ts_validation" : True,  # 3-way-split time series for NRMSE validation.
        "add_penalty_factor" : False, # Experimental feature. Use with caution.
    }

    # Build the payload for the robyn_run()
    payload = {
        'InputCollect' : json.dumps(input_collect), 
        'jsonRunArgs' : json.dumps(runArgs)
    }
    
    output_models = robyn_api('robyn_run',payload=payload)
    
    plot_save_outputgraphs(robyn_directory, output_models, graphtype='moo_distrb_plot', max_size=(1000, 1500), argumenttype='none')
    plot_save_outputgraphs(robyn_directory, output_models, graphtype='moo_cloud_plot', max_size=(1000, 1500), argumenttype='none')
    plot_save_outputgraphs(robyn_directory, output_models, graphtype='ts_validation_plot', max_size=(1000, 1500), argumenttype='none')
    
    
    outputs_args = {
        "pareto_fronts" : 'auto', # automatically pick how many pareto-fronts to fill min_candidates (100)
    #     "min_candidates" : 100, # top pareto models for clustering. Default to 100
    #     "calibration_constraint" : 0.1, # range [0.01, 0.1] & default at 0.1
        "csv_out" : "pareto", # "pareto", "all", or NULL (for none)
        "clusters" : True, # Set to TRUE to cluster similar models by ROAS.
        "export" : create_files, # this will create files locally
        "plot_folder" : robyn_directory, # path for plots exports and files creation
        "plot_pareto" : create_files # Set to FALSE to deactivate plotting and saving model one-pagers
    }

    # Build the payload for the robyn_outputs()
    payload = {
        'InputCollect' : json.dumps(input_collect),
        'OutputModels' : json.dumps(output_models),
        'jsonOutputsArgs' : json.dumps(outputs_args)
    }
    
    # Get response
    output_collect = robyn_api('robyn_outputs', payload=payload)
    
    
    for i in output_collect['clusters']['models']:
        print(i['solID'])
    
    output_collect['allSolutions']
    
    # load_onepager(top_pareto=True,sol='all',InputJson=input_collect,OutputJson=OutputCollect,path=robyn_directory)
    
    select_model = '5_275_3'
    
    write_robynmodel(sol=select_model, path=robyn_directory, InputJson=input_collect, OutputJson=output_collect, OutputModels=output_models)
    
    print(input_collect['paid_media_spends'])
    
    allocator_args = {
        "select_model" : select_model,
    #     "date_range" : None, # Default last month as initial period
    #     "total_budget" : None, # When NULL, default is total spend in date_range
        "channel_constr_low" : 0.7,
        "channel_constr_up" : 1.2,
        "channel_constr_multiplier" : 3,
        "scenario" : "max_response"
    }
    
    # Build the payload for the robyn_allocator()
    payload = {
        'InputCollect' : json.dumps(input_collect),
        'OutputCollect' : json.dumps(output_collect),
        "jsonAllocatorArgs": json.dumps(allocator_args),
        'dpi' : 100,
        'width' : 15,
        'height' : 15
    }
    
    # Get response
    allocator = robyn_api('robyn_allocator',payload=payload)
    
    plot_save_outputgraphs(robyn_directory, OutputJson='none', argumenttype=allocator, graphtype='allocator', max_size=(1000, 1500))
    
    
    kill_process_and_show_message(p)
    
    
    pass

def run_robyn_model():
    
    from robyn_api.python_helper import *
    
    input_gsheet_url = 'https://docs.google.com/spreadsheets/d/1NH6ejWvNDDIZbOra6722ohyKQyoc_0MfO8OJb8xmbN0'
    
    input_title, param_df, feature_col_df, model_df, spend_fee_df = read_end_to_end_data(input_gsheet_url=input_gsheet_url)
    model_df = model_df.reset_index(drop=True)
    print(input_title)
    print(model_df.head())
    
    folder_suffix = '_spend_q1'
    val_col = 'out_of_sample_q1'
    folder_name = input_title + folder_suffix
    save_path = Path('/Users/eddiedeane/PycharmProjects/mmm_ed/robyn_output') / folder_name
    save_path.mkdir(parents=True, exist_ok=True)
    
    # create folder for today's date in the robyn_output folder
    robyn_directory = str(save_path)
    print(robyn_directory)
    create_files = True
    
    
    def terminate_process(proc):
        if proc.poll() is None:
            proc.kill() 

    def kill_process_and_show_message(proc):
        if proc.poll() is None:
            try:
                proc.kill()
                print(f"Process killed successfully. pid: {proc.pid}")
            except Exception as e:
                print(f"Failed to kill the process: {e}")
        else:
            print("Process is not running or already terminated.")
    
    p = subprocess.Popen(['Rscript', 'robyn_api/robynapi_call.R'])
    atexit.register(terminate_process, p)
    print('starting 15 sec sleep')
    sleep(15)
    print('done sleep')
    print('running robynapi_call.R')
    print('open in browser http://127.0.0.1:9999/__docs__/ to see api docs')
    
    api_base_url = 'http://127.0.0.1:9999/{}'
    url = api_base_url[:-3] + '/openapi.json'
    html = urlopen(url).read()
    soup = BeautifulSoup(html, features='html.parser')
    apis = json.loads(soup.contents[0])
    for i in apis['paths'].keys():
        print(api_base_url.format(i[1:]))
        
    print('robyn_version', robyn_api('robyn_version'))
    
    # dt_simulated_weekly = pandas_builder(robyn_api('dt_simulated_weekly'))
    # print(dt_simulated_weekly.head())
    # dt_prophet_holidays = pandas_builder(robyn_api('dt_prophet_holidays'))
    # print(dt_prophet_holidays.loc[dt_prophet_holidays['country'] == 'US', 'holiday'].value_counts())
    # print(dt_prophet_holidays[dt_prophet_holidays['year'] == 2024])
    
    # features
    print(feature_col_df)
    spend_features_lst = list(feature_col_df.loc[feature_col_df['type'] == 'paid_media_spend', 'feature'].values)
    # organic_features_lst = list(feature_col_df.loc[feature_col_df['type'] == 'organic', 'feature'].values)
    # context_features_lst = list(feature_col_df.loc[feature_col_df['type'] == 'context', 'feature'].values)
    organic_features_lst = []
    context_features_lst = []
    all_features_lst = spend_features_lst + organic_features_lst + context_features_lst
    
    
    if val_col != '':
        train_test_model_df = model_df.loc[model_df[val_col].isin(('train', 'test')), ['date', 'n_conversion'] + all_features_lst].reset_index(drop=True)
        window_end = model_df.loc[model_df[val_col].isin(('train', 'test')), 'date'].max().strftime('%Y-%m-%d')
        train_test_size_df = model_df[model_df[val_col].isin(('train', 'test'))].groupby(val_col).size()
        test_size = train_test_size_df.loc['test']
        train_size = train_test_size_df.loc['train']
        per_val_test_size = 2 * math.ceil(100 * test_size / (test_size + train_size)) / 100
        per_train_size = 1 - per_val_test_size
    else:
        min_date = '2010-04-22'
        train_test_model_df = model_df.loc[model_df['date'] >= min_date, ['date', 'n_conversion'] + all_features_lst].reset_index(drop=True)
        window_end = train_test_model_df['date'].max().strftime('%Y-%m-%d')
        per_train_size = 0.6
        per_val_test_size = 1 - per_train_size
        
    window_start = train_test_model_df['date'].min().strftime('%Y-%m-%d')
    train_test_model_df['date'] = train_test_model_df['date'].astype('str')
    print(window_start, window_end, per_train_size, per_val_test_size)
    print(train_test_model_df.head())
    spend_features_lst
    organic_features_lst
    context_features_lst
    
    # Define Args for robyn_inputs()
    input_args = {
        "date_var": "date", # date format must be "2020-01-01"
        "dep_var": "n_conversion", # there should be only one dependent variable
        "dep_var_type": "conversion", # "revenue" (ROI) or "conversion" (CPA)
        "prophet_vars": ["trend", "season"], # "trend","season", "weekday" & "holiday"
        # "prophet_country": "US", # input country code. Check: dt_prophet_holidays
        "paid_media_spends": spend_features_lst, # mandatory input
        "paid_media_vars": spend_features_lst, # mandatory.
        # paid_media_vars must have same order as paid_media_spends. Use media exposure metrics like
        # impressions, GRP etc. If not applicable, use spend instead.
        # "organic_vars" : organic_features_lst, # marketing activity without media spend
        # 'organic_signs': [], # ['positive'] ['negative']
        # 'context_vars': context_features_lst,
        # "factor_vars" : [], # force variables in context_vars or organic_vars to be categorical
        "window_start": window_start,
        "window_end": window_end,
        "adstock": "geometric" # geometric, weibull_cdf or weibull_pdf.
    }
        # Build the payload for the robyn_inputs()
    payload = {
        'dt_input' : asSerialisedFeather(train_test_model_df), 
        # 'dt_holiday' : asSerialisedFeather(dt_prophet_holidays), 
        'jsonInputArgs' : json.dumps(input_args)
    }
    
    # Get response
    input_collect = robyn_api('robyn_inputs', payload=payload)
    
    # Build the payload for the hyper_names()
    payload = {
        'adstock' : input_collect['adstock'], 
        'all_media' : json.dumps(input_collect['all_media'])
    }

    # Get response
    hyper_names = robyn_api('hyper_names',payload=payload)
    print(hyper_names)
    
    hyperparameters_dict = {'train_size': per_train_size}
    
    for feat in spend_features_lst + organic_features_lst:
        hyperparameters_dict[f'{feat}_alphas'] = [0.5, 3]
        hyperparameters_dict[f'{feat}_gammas'] = [0.3, 1]
        hyperparameters_dict[f'{feat}_thetas'] = [0, 0.8]
    
    
    # Set Args for robyn_inputs()
    input_args = {"hyperparameters" : hyperparameters_dict}
    
    # Build the payload for the hyper_names()
    payload = {
        'InputCollect' : json.dumps(input_collect), 
        'jsonInputArgs' : json.dumps(input_args)
    }
    
    # Get response
    input_collect = robyn_api('robyn_inputs',payload=payload)
    print(input_collect.keys())
    
    # input_collect['exposure_vars']
    # render_save_spendexposure(robyn_directory, InputJson=input_collect, max_size=(1000,1500))
    
    # incrementality test calibration
    # calibration_input = {
    #     # channel name must in paid_media_vars
    #     "channel": ["facebook_S","tv_S","facebook_S+search_S","newsletter"],
    #     # liftStartDate must be within input data range
    #     "liftStartDate" : ["2018-05-01","2018-04-03","2018-07-01","2017-12-01"],
    #     # liftEndDate must be within input data range
    #     "liftEndDate" : ["2018-06-10","2018-06-03","2018-07-20","2017-12-31"],
    #     # Provided value must be tested on same campaign level in model and same metric as dep_var_type
    #     "liftAbs" : [400000, 300000, 700000, 200],
    #     # Spend within experiment: should match within a 10% error your spend on date range for each channel from dt_input
    #     "spend" : [421000, 7100, 350000, 0],
    #     # Confidence: if frequentist experiment, you may use 1 - pvalue
    #     "confidence" : [0.85, 0.8, 0.99, 0.95],
    #     # KPI measured: must match your dep_var
    #     "metric" : ["revenue","revenue","revenue","revenue"],
    #     # Either "immediate" or "total". For experimental inputs like Facebook Lift, "immediate" is recommended.
    #     "calibration_scope" : ["immediate","immediate","immediate","immediate"]
    # }
    # calibration_input = pd.DataFrame(calibration_input)
    # calibration_input
    # payload = {
    #     'InputCollect' : json.dumps(input_collect), 
    #     'calibration_input' : asSerialisedFeather(calibration_input), 
    # }
    # # Get response
    # input_collect = robyn_api('robyn_inputs',payload=payload)
    
    # train models
    run_args = {
        "iterations" : 4000, # NULL defaults to (max available - 1)
        "trials" : 10, # 5 recommended for the dummy dataset
        "ts_validation" : True,  # 3-way-split time series for NRMSE validation.
        "add_penalty_factor" : False, # Experimental feature. Use with caution.
    }

    # Build the payload for the robyn_run()
    payload = {
        'InputCollect' : json.dumps(input_collect), 
        'jsonRunArgs' : json.dumps(run_args)
    }
    
    output_models = robyn_api('robyn_run',payload=payload)
    
    # output training plots
    plot_save_outputgraphs(robyn_directory, output_models, graphtype='moo_distrb_plot', max_size=(1000, 1500), argumenttype='none')
    plot_save_outputgraphs(robyn_directory, output_models, graphtype='moo_cloud_plot', max_size=(1000, 1500), argumenttype='none')
    plot_save_outputgraphs(robyn_directory, output_models, graphtype='ts_validation_plot', max_size=(1000, 1500), argumenttype='none')
    
    # 
    outputs_args = {
        "pareto_fronts" : 'auto', # automatically pick how many pareto-fronts to fill min_candidates (100)
    #     "min_candidates" : 100, # top pareto models for clustering. Default to 100
    #     "calibration_constraint" : 0.1, # range [0.01, 0.1] & default at 0.1
        "csv_out" : "pareto", # "pareto", "all", or NULL (for none)
        "clusters" : True, # Set to TRUE to cluster similar models by ROAS.
        "export" : create_files, # this will create files locally
        "plot_folder" : robyn_directory, # path for plots exports and files creation
        "plot_pareto" : False # Set to FALSE to deactivate plotting and saving model one-pagers
    }

    # Build the payload for the robyn_outputs()
    payload = {
        'InputCollect' : json.dumps(input_collect),
        'OutputModels' : json.dumps(output_models),
        'jsonOutputsArgs' : json.dumps(outputs_args)
    }
    
    # Get response
    output_collect = robyn_api('robyn_outputs', payload=payload)
    
    # save input_collect, output_collect, and output_models to disk...
    with open(Path(robyn_directory) / 'input_collect.pkl', 'wb') as f:
        pickle.dump(input_collect, f)
    
    with open(Path(robyn_directory) / 'output_collect.pkl', 'wb') as f:
        pickle.dump(output_collect, f)
    
    with open(Path(robyn_directory) / 'output_models.pkl', 'wb') as f:
        pickle.dump(output_models, f)
    
    # seperate onepagers output from output_collect
    output_onepagers = False
    if output_onepagers:
        
        robyn_model_metrics_df = pd.DataFrame(output_collect['clusters']['data'])
        robyn_model_metrics_df['nrmse_sum'] = robyn_model_metrics_df['nrmse_train'] + robyn_model_metrics_df['nrmse_val'] + robyn_model_metrics_df['nrmse_test']
        robyn_model_metrics_df = robyn_model_metrics_df.sort_values('nrmse_test', ascending=True).reset_index(drop=True)
        best_test_model = robyn_model_metrics_df.loc[0, 'solID']
        best_models = [i['solID'] for i in output_collect['clusters']['models']]
        if best_test_model not in best_models:
            best_models.append(best_test_model)
        
        print('n onepagers', len(best_models))
        
        for sol_id in best_models:
            
            onepagers_args = {
                "select_model" : sol_id,
                "export" : True,
            }

            # Build the payload for the robyn_onepagers()
            payload = {
                'InputCollect' : json.dumps(input_collect),
                'OutputCollect' : json.dumps(output_collect),
                "jsonOnepagersArgs": json.dumps(onepagers_args),
                'dpi' : 100,
                'width' : 15,
                'height' : 20
            }

            # Get response
            onepager = robyn_api('robyn_onepagers',payload=payload)
    
    print('done with initial model run')

    pass

def select_robyn_mmm_model():
    
    from robyn_api.python_helper import *
    
    def terminate_process(proc):
        if proc.poll() is None:
            proc.kill() 

    def kill_process_and_show_message(proc):
        if proc.poll() is None:
            try:
                proc.kill()
                print(f"Process killed successfully. pid: {proc.pid}")
            except Exception
            as e:
                print(f"Failed to kill the process: {e}")
        else:
            print("Process is not running or already terminated.")
    
    p = subprocess.Popen(['Rscript', 'robyn_api/robynapi_call.R'])
    atexit.register(terminate_process, p)
    print('starting 30 sec sleep')
    sleep(30)
    print('done sleep')
    print('running robynapi_call.R')
    print('open in browser http://127.0.0.1:9999/__docs__/ to see api docs')
    
    api_base_url = 'http://127.0.0.1:9999/{}'
    url = api_base_url[:-3] + '/openapi.json'
    html = urlopen(url).read()
    soup = BeautifulSoup(html, features='html.parser')
    apis = json.loads(soup.contents[0])
    for i in apis['paths'].keys():
        print(api_base_url.format(i[1:]))
        
    print('robyn_version', robyn_api('robyn_version'))
    
    # folder
    read_in_data = False
    if read_in_data:
        run_folder = 'weekly_data_run_2024_03_25_spend_q3'
        robyn_directory = '/Users/eddiedeane/PycharmProjects/mmm_ed/robyn_output/' + run_folder
        print(robyn_directory)
        
        # read in input_collect, output_collect, and output_models to disk...
        with open(Path(robyn_directory) / 'input_collect.pkl', 'rb') as f:
            input_collect = pickle.load(f)
        
        with open(Path(robyn_directory) / 'output_collect.pkl', 'rb') as f:
            output_collect = pickle.load(f)
        
        with open(Path(robyn_directory) / 'output_models.pkl', 'rb') as f:
            output_models = pickle.load(f)
    
    # get the name of the model output folder
    output_save_path = Path(output_collect['plot_folder'][0])
    print(output_save_path)
    
    # which models are available to output?
    output_models_lst = [i['solID'] for i in output_collect['clusters']['models']]
    print(len(output_models_lst))
    
    # read in the pareto results csv
    # pc_df = pd.read_csv(output_save_path / 'pareto_clusters.csv')
    # print(pc_df.loc[pc_df['solID'].isin(output_models_lst)])
    robyn_model_metrics_df = pd.DataFrame(output_collect['clusters']['data'])
    robyn_model_metrics_df = robyn_model_metrics_df[['solID', 'nrmse_train', 'nrmse_val', 'nrmse_test', 'decomp.rssd']]
    robyn_model_metrics_df['nrmse_sum'] = robyn_model_metrics_df['nrmse_val'] + robyn_model_metrics_df['nrmse_test']
    robyn_model_metrics_df = robyn_model_metrics_df.sort_values('nrmse_test', ascending=True).reset_index(drop=True)
    # robyn_model_metrics_df = robyn_model_metrics_df.sort_values('nrmse_sum', ascending=True).reset_index(drop=True)
    robyn_model_metrics_df
    
    # select model with lowest RMSE
    select_model = robyn_model_metrics_df.loc[0, 'solID']
    print(select_model)
    
    # write onepager for model
    # onepagers_args = {
    #     "select_model" : select_model,
    #     "export" : True,
    # }

    # Build the payload for the robyn_onepagers()
    # payload = {
    #     'InputCollect' : json.dumps(input_collect),
    #     'OutputCollect' : json.dumps(output_collect),
    #     "jsonOnepagersArgs": json.dumps(onepagers_args),
    #     'dpi' : 100,
    #     'width' : 15,
    #     'height' : 20
    # }

    # Get response
    # robyn_api('robyn_onepagers',payload=payload)
    
    
    # write model / save to disk
    select_model_path = str(Path(robyn_directory) / 'select_model')
    select_model_path = create_robyn_directory(select_model_path)
    # write_robynmodel(sol=select_model, path=select_model_path, InputJson=input_collect, OutputJson=output_collect, OutputModels=output_models)
    
    # train validation test split
    pred_agg_df = pd.DataFrame(output_collect['xDecompAgg'])
    pred_agg_df = pred_agg_df[pred_agg_df['solID'] == select_model].reset_index(drop=True)
    train_size = np.round(pred_agg_df.loc[0, 'train_size'], 4)
    pred_agg_df
    
    # calculate regression metrics for the select_model solID by train, validation, and test
    # get the raw spend and conversion contribution to calculate CAC or CPA and regression metrics
    pred_df = pd.DataFrame(output_collect['xDecompVecCollect'])
    pred_df = pred_df[pred_df['solID'] == select_model].reset_index(drop=True)
    ndays = pred_df.shape[0]
    train_cut = int(np.round(ndays * train_size))
    val_cut = int(train_cut + np.round(ndays * (1 - train_size) / 2))
    pred_df['period'] = 'none'
    pred_df.loc[:train_cut-1, 'period'] = 'train'
    pred_df.loc[train_cut:val_cut-1, 'period'] = 'validation'
    pred_df.loc[val_cut:, 'period'] = 'test'
    pred_df
    
    out_of_sample_results = []
    for period in ['train', 'validation', 'test']:
        
        y_true = pred_df.loc[pred_df['period'] == period, 'dep_var'].values
        y_pred = pred_df.loc[pred_df['period'] == period, 'depVarHat'].values
        
        r2 = r2_score(y_true=y_true, y_pred=y_pred)
        mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
        mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
        rmse = mse ** 0.5
        mape = mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred)
        evaluator = RegressionMetric(y_true=y_true, y_pred=y_pred)
        smape = evaluator.symmetric_mean_absolute_percentage_error()
        nrmse = evaluator.normalized_root_mean_square_error()
        
        out_of_sample_results.append((period, r2, mae, mse, rmse, mape, smape, nrmse))
    
    oos_df = pd.DataFrame(out_of_sample_results, columns=['period', 'r2', 'mae', 'mse', 'rmse', 'mape', 'smape', 'nrmse'])
    oos_df['model'] = 'robyn'
    oos_df['data'] = 'weekly_data_run_2024_03_25'
    oos_df['out_of_sample_period'] = val_col
    oos_df = oos_df[['model', 'data', 'out_of_sample_period', 'period', 'r2', 'mae', 'mse', 'rmse', 'mape', 'smape', 'nrmse']]
    oos_df.to_csv(Path(select_model_path) / 'out_of_sample_results.csv', index=False)
    oos_df.to_clipboard(index=False)
    
    # join pred to spend and get CAC by channel by date
    # model_train_df = pd.DataFrame(input_collect['dt_input'])
    # model_train_df['date'] = pd.to_datetime(model_train_df['date'])
    # input_gsheet_url = 'https://docs.google.com/spreadsheets/d/1NH6ejWvNDDIZbOra6722ohyKQyoc_0MfO8OJb8xmbN0'
    # input_title, param_df, feature_col_df, model_df, spend_fee_df = read_end_to_end_data(input_gsheet_url=input_gsheet_url)
    # model_df = model_df.reset_index(drop=True)
    # print(model_df.head())
    pred_df.rename(columns={'ds': 'date'}, inplace=True)
    pred_df['date'] = pd.to_datetime(pred_df['date'])
    # change pred_df column names to _n_conversion for model features
    pred_df = pred_df.drop(columns=['dep_var', 'cluster', 'top_sol'])
    pred_df = pred_df.rename(columns={'depVarHat': 'n_conversion_pred'})
    feature_lst = [i for i in feature_col_df['feature'] if i in pred_df.columns]
    new_cols = {i: f'{i}_n_conversion' for i in pred_df.columns if i not in ('date', 'dep_var', 'depVarHat', 'solID', 'cluster', 'top_sol', 'out_of_sample_period')}
    pred_df.rename(columns=new_cols, inplace=True)
    cac_df = pd.merge(model_df, pred_df, on='date', how='outer')
    cac_df
    
    for feature in feature_lst:
        cac_df[f'{feature}_cac'] = cac_df[feature] / cac_df[f'{feature}_n_conversion']
    
    cac_df['model'] = 'robyn'
    cac_df['data'] = 'weekly_data_run_2024_03_25'
    cac_df['out_of_sample_period'] = val_col
    cac_info_cols = ['model', 'data', 'out_of_sample_period']
    cac_df = cac_df[cac_info_cols + [i for i in cac_df.columns if i not in cac_info_cols]]
    cac_df.to_csv(Path(select_model_path) / 'cac_results.csv', index=False)
    cac_df.to_clipboard(index=False)
    
    pass

def run_robyn_allocator():
    
    allocator_args = {
        "select_model" : select_model,
    #     "date_range" : None, # Default last month as initial period
    #     "total_budget" : None, # When NULL, default is total spend in date_range
        "channel_constr_low" : 0.7,
        "channel_constr_up" : 1.2,
        "channel_constr_multiplier" : 3,
        "scenario" : "max_response"
    }
    
    # Build the payload for the robyn_allocator()
    payload = {
        'InputCollect' : json.dumps(input_collect),
        'OutputCollect' : json.dumps(output_collect),
        "jsonAllocatorArgs": json.dumps(allocator_args),
        'dpi' : 100,
        'width' : 15,
        'height' : 15
    }
    
    # Get response
    allocator = robyn_api('robyn_allocator',payload=payload)
    
    plot_save_outputgraphs(robyn_directory, OutputJson='none', argumenttype=allocator, graphtype='allocator', max_size=(1000, 1500))
    
    kill_process_and_show_message(p)
    
    pass

def run_lightweight_mmm_model():
    
    # lightweight mmm
    # https://github.com/google/lightweight_mmm?tab=readme-ov-file#getting-started
    # https://github.com/google/lightweight_mmm/blob/main/examples/simple_end_to_end_demo.ipynb
    
    import jax.numpy as jnp
    import numpyro
    from lightweight_mmm import lightweight_mmm, optimize_media, plot, preprocessing, utils
    
    n_cpu = 6
    numpyro.set_host_device_count(n_cpu)
    
    seed = 42
    
    input_gsheet_url = 'https://docs.google.com/spreadsheets/d/1NH6ejWvNDDIZbOra6722ohyKQyoc_0MfO8OJb8xmbN0'
    
    today_str = date.today().strftime(r'%Y_%m_%d')
    save_path = Path('lightweight_mmm_output') / f'mmm_run_{today_str}'
    save_path.mkdir(parents=True, exist_ok=True)
    
    input_title, param_df, feature_col_df, model_df, spend_fee_df = read_end_to_end_data(input_gsheet_url=input_gsheet_url)
    
    input_title
    param_df
    feature_col_df
    model_df.info()
    spend_fee_df.info()
    model_df = model_df.sort_values('date', ascending=True).reset_index(drop=True)
    
    data_size = model_df.shape[0]
    paid_media_feauture_lst = list(feature_col_df.loc[feature_col_df['type'] == 'paid_media_spend', 'feature'].values)
    organic_context_feature_lst = list(feature_col_df.loc[feature_col_df['type'] != 'paid_media_spend', 'feature'].values)
    organic_context_feature_lst.remove('seo_discount_added')
    organic_context_feature_lst.remove('add_compliance_language')
    n_media_channels = len(paid_media_feauture_lst)
    n_extra_features = len(organic_context_feature_lst)
    split_point = data_size - 4
    split_point
    
    # media_data, extra_features, target, costs = utils.simulate_dummy_data(data_size=data_size,
    #                                                                       n_media_channels=n_media_channels,
    #                                                                       n_extra_features=n_extra_features)
    
    # costs
    costs = model_df[paid_media_feauture_lst].values.sum(axis=0)
    # Media data
    media_data = model_df[paid_media_feauture_lst].values
    media_data_train = model_df.loc[:split_point-1, paid_media_feauture_lst].values
    media_data_test = model_df.loc[split_point:, paid_media_feauture_lst].values
    media_data_train.shape[0]
    100 * media_data_test.shape[0] / media_data.shape[0]
    
    
    # Extra features
    extra_features = model_df[organic_context_feature_lst].values
    extra_features_train = model_df.loc[:split_point-1, organic_context_feature_lst].values
    extra_features_test = model_df.loc[split_point:, organic_context_feature_lst].values
    # Target
    target = model_df['n_conversion'].values
    target_train = model_df.loc[:split_point-1, 'n_conversion'].values
    target_test = model_df.loc[split_point:, 'n_conversion'].values
    
    media_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
    extra_features_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
    target_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
    cost_scaler = preprocessing.CustomScaler(divide_operation=jnp.mean)
    
    
    # scaling with means of 1
    media_data_train = media_scaler.fit_transform(media_data_train)
    extra_features_train = extra_features_scaler.fit_transform(extra_features_train)
    target_train = target_scaler.fit_transform(target_train)
    costs = cost_scaler.fit_transform(costs)
    
    
    pd.DataFrame(media_data_train).describe()
    pd.DataFrame(extra_features_train, columns=organic_context_feature_lst).describe()
    
    correlations, variances, spend_fractions, variance_inflation_factors = preprocessing.check_data_quality(
        media_data=media_scaler.transform(media_data), 
        target_data=target_scaler.transform(target), 
        cost_data=costs, 
        extra_features_data=extra_features_scaler.transform(extra_features), 
        channel_names=paid_media_feauture_lst,
        extra_features_names=organic_context_feature_lst)
    
    
    # Correlation absolute value above, say, 0.7 or so should be treated with caution. 
    # In this case you might consider dropping or merging highly correlated features.
    correlations
    
    low_variance_threshold = 0.001
    high_variance_threshold = 3.0
    variances[variances['geo_0'] <= low_variance_threshold]
    variances[variances['geo_0'] >= high_variance_threshold]
    
    low_spend_threshold = 0.01
    spend_fractions[spend_fractions['fraction of spend'] <= low_spend_threshold]
    (spend_fractions * 100).sort_values(by='fraction of spend', ascending=False)
    
    high_vif_threshold = 7.0
    variance_inflation_factors
    variance_inflation_factors[variance_inflation_factors['geo_0'] >= high_vif_threshold]
    
    # mmm = lightweight_mmm.LightweightMMM(model_name='hill_adstock')
    mmm = lightweight_mmm.LightweightMMM(model_name='adstock')
    # mmm = lightweight_mmm.LightweightMMM(model_name='carryover')
    
    # extra_features: Other variables to add to the model.
    # degrees_seasonality: Number of degrees to use for seasonality. Default is 3.
    # seasonality_frequency: Frequency of the time period used. Default is 52 as in 52 weeks per year.
    # media_names: Names of the media channels passed.
    # number_warmup: Number of warm up samples. Default is 1000.
    # number_samples: Number of samples during sampling. Default is 1000.
    # number_chains: Number of chains to sample. Default is 2.
    
    # media
    # total_costs (one value per channel)
    # target
    
    degrees_seasonality = 3
    if daily_or_weekly == 'weekly':
        seasonality_frequency = 52
    elif daily_or_weekly == 'daily':
        seasonality_frequency = 365
    number_warmup = 1000
    number_samples = 1000
    
    mmm.fit(
        media=media_data_train, 
        media_prior=costs, 
        target=target_train, 
        extra_features=extra_features_train,
        degrees_seasonality=degrees_seasonality,
        seasonality_frequency=seasonality_frequency,
        weekday_seasonality=False,
        media_names=paid_media_feauture_lst,
        number_warmup=number_warmup,
        number_samples=number_samples, 
        number_chains=4, 
        seed=seed
    )
    
    mmm.print_summary()
    media_contribution, roi_hat = mmm.get_posterior_metrics(target_scaler=target_scaler, cost_scaler=cost_scaler)
    media_contribution.shape
    media_contribution.mean(axis=0)
    roi_hat.mean(axis=0)
    
    # this doesn't seem correct...
    results_df = pd.DataFrame({'feature': paid_media_feauture_lst, 
                               'total_spend': cost_scaler.inverse_transform(costs),
                               'media_contribution': media_contribution.mean(axis=0), 
                               'roi_hat': roi_hat.mean(axis=0)})
    
    results_df['media_contribution'].sum()
    results_df['roi_hat'].sum()
    1 / results_df['roi_hat']
    
    
    media_data.sum() / target.sum()
    results_df['n_conversion'] = target.sum() * results_df['media_contribution']
    results_df['total_spend'] / results_df['n_conversion']
    
    results_df['total_spend'].sum() / results_df['n_conversion'].sum()
    
    results_df['total_spend'].sum()
    
    results_df['roi_n_conversion'] = results_df['roi_hat'] * results_df['total_spend']
    results_df['total_spend'].sum() / results_df['roi_n_conversion'].sum()
    results_df['roi_cac'] = results_df['total_spend'] / results_df['roi_n_conversion']
    results_df
    
    
    media_data.sum()
    
    new_predictions = mmm.predict(media=media_scaler.transform(media_data_test),
                                  extra_features=extra_features_scaler.transform(extra_features_test),
                                  seed=seed)
    new_predictions_point_estimate = target_scaler.inverse_transform(new_predictions.mean(axis=0))
    pd.DataFrame({'target_test': target_test, 'new_prediction': new_predictions_point_estimate})
    
    r2 = r2_score(y_true=target_test, y_pred=new_predictions_point_estimate)
    mae = mean_absolute_error(y_true=target_test, y_pred=new_predictions_point_estimate)
    mse = mean_squared_error(y_true=target_test, y_pred=new_predictions_point_estimate)
    rmse = mse ** 0.5
    mape = mean_absolute_percentage_error(y_true=target_test, y_pred=new_predictions_point_estimate)
    print(mae, mse, rmse, mape)
    
    
    plot_stuff = False
    if plot_stuff:
    
        plot.plot_media_channel_posteriors(media_mix_model=mmm)
        plt.savefig(save_path / 'media_channel_posteriors.png')
        
        plot.plot_prior_and_posterior(media_mix_model=mmm)
        plt.savefig(save_path / 'prior_and_posterior.png')
        
        plot.plot_model_fit(mmm, target_scaler=target_scaler)
        plt.savefig(save_path / 'model_fit.png')
        
        plot.plot_out_of_sample_model_fit(out_of_sample_predictions=new_predictions,
                                        out_of_sample_target=target_scaler.transform(target[split_point:]))
        plt.savefig(save_path / 'out_of_sample_model_fit.png')
        
        plot.plot_media_baseline_contribution_area_plot(media_mix_model=mmm, 
                                                        target_scaler=target_scaler, 
                                                        fig_size=(30,10))
        plt.savefig(save_path / 'media_baseline_contribution_area_plot.png')
        
        plot.plot_bars_media_metrics(metric=media_contribution, metric_name="Media Contribution Percentage")
        plt.savefig(save_path / 'media_contribution_percentage.png')
        
        plot.plot_bars_media_metrics(metric=roi_hat, metric_name="ROI hat")
        plt.savefig(save_path / 'roi_hat.png')
        
        plot.plot_response_curves(media_mix_model=mmm, target_scaler=target_scaler, seed=seed)
        plt.savefig(save_path / 'response_curves.png')
        
    
    # The optimization is meant to solve the budget allocation questions for you. 
    
    # First you need to provide for how long you want to optimize your budget (eg. 10 weeks in this case).
    n_time_periods = 4
    
    # The optimization values will be bounded by +- 20% of the max and min historic values used for training. 
    # Which means the optimization won't recommend to completely change your strategy but how to make some budget re-allocation.
    # change this with bounds_lower_pct and bounds_upper_pct
    
    # Prices are the average price you would expect for the media units of each channel. 
    # If your data is already in monetary units (eg. $) your prices should be an array of 1s.
    prices = jnp.ones(mmm.n_media_channels)
    
    budget = jnp.sum(jnp.dot(prices, media_data.mean(axis=0)))* n_time_periods
    budget
    
    # Run optimization with the parameters of choice.
    solution, kpi_without_optim, previous_media_allocation = optimize_media.find_optimal_budgets(
        n_time_periods=n_time_periods, 
        media_mix_model=mmm, 
        extra_features=extra_features_scaler.transform(extra_features_test)[:n_time_periods], 
        budget=budget, 
        prices=prices, 
        media_scaler=media_scaler, 
        target_scaler=target_scaler, 
        seed=seed
    )
    print('optimization done')
    
    # optimal weekly allocation
    optimal_buget_allocation = prices * solution.x
    optimal_buget_allocation
    
    # similar renormalization to get previous budget allocation
    previous_budget_allocation = prices * previous_media_allocation
    previous_budget_allocation
    
    pd.DataFrame({'feature': paid_media_feauture_lst, 
                  'optimal_budget_allocation': optimal_buget_allocation, 
                  'previous_budget_allocation': previous_budget_allocation})
    
    
    
    
    # Both these values should be very close in order to compare KPI
    budget, optimal_buget_allocation.sum()
    
    # Both numbers should be almost equal
    budget, jnp.sum(solution.x * prices)
    
    if plot_stuff:
        # Plot out pre post optimization budget allocation and predicted target variable comparison.
        plot.plot_pre_post_budget_allocation_comparison(media_mix_model=mmm, 
                                                        kpi_with_optim=solution['fun'], 
                                                        kpi_without_optim=kpi_without_optim,
                                                        optimal_buget_allocation=optimal_buget_allocation, 
                                                        previous_budget_allocation=previous_budget_allocation, 
                                                        figure_size=(10,10))
        plt.savefig(save_path / 'pre_post_budget_allocation_comparison.png')
    
    
    # We can use the utilities for saving models to disk.
    file_path = "media_mix_model.pkl"
    utils.save_model(media_mix_model=mmm, file_path=file_path)

    # Once saved one can load the models.
    loaded_mmm = utils.load_model(file_path=file_path)
    loaded_mmm.trace["coef_media"].shape # Example of accessing any of the model values.
    
    
    pass

def run_pymc_marketing_mmm_model(model_df):
    
    # pymc marketing pymc5
    # https://github.com/pymc-labs/pymc-marketing
    # https://github.com/pymc-labs/pymc-marketing/blob/main/docs/source/notebooks/mmm/mmm_example.ipynb
    # https://github.com/pymc-labs/pymc-marketing/blob/main/docs/source/notebooks/mmm/mmm_budget_allocation_example.ipynb
    # https://juanitorduz.github.io/mmm_roas/
    
    # https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_example.html
    # https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_budget_allocation_example.html
    
    import arviz as az
    import pymc as pm
    from pymc_marketing.mmm.transformers import geometric_adstock, logistic_saturation
    from pymc_marketing.mmm.delayed_saturated_mmm import DelayedSaturatedMMM
    from pymc_marketing.mmm.utils import apply_sklearn_transformer_across_date
    
    az.style.use("arviz-darkgrid")
    plt.rcParams["figure.figsize"] = [12, 7]
    plt.rcParams["figure.dpi"] = 100
    
    random_seed = 42
    
    input_gsheet_url = 'https://docs.google.com/spreadsheets/d/1NH6ejWvNDDIZbOra6722ohyKQyoc_0MfO8OJb8xmbN0'
    
    input_title, param_df, feature_col_df, model_df, spend_fee_df = read_end_to_end_data(input_gsheet_url=input_gsheet_url)
    print(input_title)
    
    daily_or_weekly = param_df[param_df['param'] == 'daily_or_weekly']['value'].values[0]
    today_str = date.today().strftime(r'%Y_%m_%d')
    folder_suffix = '_lv'
    folder_name = input_title + folder_suffix
    save_path = Path('/Users/eddiedeane/PycharmProjects/mmm_ed/pymc_marketing_output') / folder_name
    print(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # trend feature
    model_df['trend'] = range(model_df.shape[0])
    feature_col_df = pd.concat([feature_col_df, pd.DataFrame([('trend', 'seasonality', 0)], columns=feature_col_df.columns)], 
                               ignore_index=True)
    
    # from sklearn.model_selection import TimeSeriesSplit
    
    # spend share
    paid_media_spend_features = list(feature_col_df.loc[feature_col_df['type'] == 'paid_media_spend', 'feature'].values)
    control_features = list(feature_col_df.loc[feature_col_df['type'] != 'paid_media_spend', 'feature'].values)
    all_features = paid_media_spend_features + control_features
    
    spend_share_df = model_df[paid_media_spend_features].sum(axis=0)
    spend_share_df.index.name = 'feature'
    spend_share_df = spend_share_df.reset_index().rename(columns={0: 'spend'})
    spend_share_df['spend_share'] = spend_share_df['spend'] / spend_share_df['spend'].sum()
    spend_share_df.shape
    spend_share_df
    
    
    # How to incorporate this heuristic into the model? To begin with, it is important to note that the DelayedSaturatedMMM class scales 
    # the target and input variables through an MaxAbsScaler transformer from scikit-learn, its important to specify the priors in the 
    # scaled space (i.e. between 0 and 1). One way to do it is to use the spend share as the sigma parameter for the HalfNormal distribution. 
    # We can actually add a scaling factor to take into account the support of the distribution.
    
    halfnormal_scale = 1 / np.sqrt(1 - 2 / np.pi)
    n_channels = spend_share_df.shape[0]
    prior_sigma = halfnormal_scale * n_channels * spend_share_df['spend_share'].values
    prior_mu = np.array([1 for _ in range(n_channels)])
    prior_sigma
    prior_mu
    
    dummy_model = DelayedSaturatedMMM(date_column='', channel_columns='', adstock_max_lag=4)
    dummy_model.default_model_config
    
    custom_beta_channel_prior = {'beta_channel': {'dist': 'LogNormal',
                                              "kwargs":{# "mu": prior_mu, 
                                                        "sigma": prior_sigma},
                                              },
                            "likelihood": {
                                    "dist": "Normal",
                                    "kwargs":{
                                        "sigma": {'dist': 'HalfNormal', 'kwargs': {'sigma': 2}}
                                    # Also possible define sigma as: 
                                    # {'sigma': 5}
                                    }
                                }
                             }
    my_model_config = {**dummy_model.default_model_config, **custom_beta_channel_prior}
    print(my_model_config)
    my_sampler_config= {"progressbar": True}
    
    # python range backwards
    # import multiprocessing
    # multiprocessing.cpu_count()
    pred_metrics = []
    pred_dfs = []
    cac_dfs = []
    out_of_sample_cols = [f'out_of_sample_q{i}' for i in range(1, 5)]
    out_of_sample_col = out_of_sample_cols[0]
    out_of_sample_col
    for out_of_sample_col in out_of_sample_cols:
        
        print(out_of_sample_col)
        # x = model_df[['date'] + all_features]
        # y = model_df['n_conversion']
        x_train = model_df.loc[model_df[out_of_sample_col] == 'train', ['date'] + all_features]
        y_train = model_df.loc[model_df[out_of_sample_col] == 'train', 'n_conversion'].values
        
        x_test = model_df.loc[model_df[out_of_sample_col] == 'test', ['date'] + all_features]
        y_test = model_df.loc[model_df[out_of_sample_col] == 'test', 'n_conversion'].values
        min_test_date = x_test['date'].min()
        
        
        mmm = DelayedSaturatedMMM(
            model_config = my_model_config,
            sampler_config = my_sampler_config,
            date_column='date',
            channel_columns=paid_media_spend_features,
            control_columns=control_features,
            adstock_max_lag=8,
            yearly_seasonality=3,
        )
        
        mmm.fit(X=x_train, y=y_train, target_accept=0.95, chains=6, random_seed=random_seed)
        
        # test results
        # New dates starting from last in dataset
        # how to get posterior predictions by feature?
        # https://discourse.pymc.io/t/calculated-values-not-matching-posterior-predictive/9507
        y_test_posterior = mmm.sample_posterior_predictive(X_pred=x_test, extend_idata=False, include_last_observations=True, combined=True)
        y_test_posterior
        
        y_test_posterior = mmm.sample_posterior_predictive(X_pred=x_test, extend_idata=True, include_last_observations=True, combined=True)
        y_test_posterior
        mmm.idata
        mmm.idata.posterior
        mmm.idata.posterior_predictive
        mmm.idata.posterior_predictive['date'].shape # 25
        mmm.idata.posterior_predictive['y'].shape
        
        y_test_posterior = mmm.sample_posterior_predictive(X_pred=x_test, extend_idata=False, include_last_observations=True, combined=True, return_inferencedata=False)
        y_test_posterior['y'].mean(dim=["chain", "draw"], keep_attrs=True).data
        
        x_test.drop(columns='date').shape # 28
        y_test_posterior['y'].mean(dim=["chain", "draw"], keep_attrs=True).data
        y_test_posterior['y'].shape
        y_test_posterior['date'].shape
        
        y_test_mean_pred = mmm.predict(X_pred=x_test, extend_idata=False, include_last_observations=True)
        
        y_test_pred = y_test_posterior['y'].to_series().groupby('date').mean().values
        
        y_test_mean_pred
        y_test_pred
        
        mmm.output_var
        y_test_posterior['y']
        y_test_posterior['y'].mean(dim=["chain", "draw"], keep_attrs=True)
        y_test_posterior_means = y_test_posterior['y'].mean(dim=["chain", "draw"], keep_attrs=True)
        
        mmm.idata.posterior.intercept.shape
        mmm.idata.posterior.mean(dim=["chain", "draw"], keep_attrs=True)
        
        
        x_test = pd.concat([mmm.X.iloc[-mmm.adstock_max_lag :, :], x_test], axis=0).sort_values(by=mmm.date_column)
        mmm._data_setter(x_test)
        mmm.idata
        mmm.idata.posterior
        dir(mmm.idata)
        mmm.idata.to_dataframe()
        mmm.idata.
        
        mmm.idata
        mmm.validate_data
        mmm.data
        # post_pred = pm.sample_posterior_predictive(trace=mmm.idata, model=mmm.model)
        # list(mmm.fourier_columns)
        # var_names = mmm.channel_columns + mmm.control_columns + list(mmm.fourier_columns) + ['intercept']
        
        post_pred = pm.sample_posterior_predictive(mmm.idata, model=mmm.model, return_inferencedata=False)
        post_pred.keys()
        x_test.drop(columns='date').shape # 17 rows and 28 features
        x_test
        post_pred['y'].shape
        
        mmm.get_target_transformer().inverse_transform(post_pred['y'].mean(axis=1).mean(axis=0).reshape(-1, 1))
        
        mmm.idata.posterior
        
        
        post_pred
        post_pred.observed_data
        post_pred.constant_data
        post_pred.posterior_predictive
        
        dir(mmm)
        mmm.posterior_predictive
        dir(mmm.model)
        mmm.idata.posterior
        mmm.idata.posterior_predictive
        
        post_pred
        post_pred.observed_data
        post_pred.posterior_predictive
        post_pred.constant_data
        
        post_pred.constant_data.channel_data.data.shape
        
        
        # train results
        y_train_posterior = mmm.sample_posterior_predictive(X_pred=x_train, extend_idata=False, include_last_observations=False)
        y_train_posterior
        y_train_posterior['y'].shape
        y_train_pred = y_train_posterior['y'].to_series().groupby('date').mean().values
        
        y_train_mean_pred = mmm.predict(X_pred=x_train, extend_idata=False, include_last_observations=False)
        
        
        
        cont_df = mmm.compute_mean_contributions_over_time(original_scale=True)
        cont_df
        
        mmm.idata['posterior']
        
        
        # get predictions by spend row
        train_pred_df = pd.DataFrame({'date': x_train['date'], 'n_conversion': y_train, 'n_conversion_pred': y_train_pred})
        train_pred_df['model'] = 'pymc_marketing'
        train_pred_df['data'] = input_title
        train_pred_df['out_of_sample_period'] = out_of_sample_col
        train_pred_df['period'] = 'train'
        
        train_r2 = r2_score(y_true=y_train, y_pred=y_train_pred)
        train_mae = mean_absolute_error(y_true=y_train, y_pred=y_train_pred)
        train_mse = mean_squared_error(y_true=y_train, y_pred=y_train_pred)
        train_rmse = train_mse ** 0.5
        train_mape = mean_absolute_percentage_error(y_true=y_train, y_pred=y_train_pred)
        train_evaluator = RegressionMetric(y_true=y_train, y_pred=y_train_pred)
        train_smape = train_evaluator.symmetric_mean_absolute_percentage_error()
        train_nrmse = train_evaluator.normalized_root_mean_square_error()
        
        pred_metrics.append(('pymc_marketing', input_title, out_of_sample_col, 'train', train_r2, train_mae, train_mse, train_rmse, train_mape, train_smape, train_nrmse))
        
        
        test_pred_df = pd.DataFrame({'date': x_test['date'], 'n_conversion': y_test, 'n_conversion_pred': y_test_pred})
        test_pred_df['model'] = 'pymc_marketing'
        test_pred_df['data'] = input_title
        test_pred_df['out_of_sample_period'] = out_of_sample_col
        test_pred_df['period'] = 'test'
        
        pred_df = pd.concat([train_pred_df, test_pred_df], axis=0, ignore_index=True)
        
        test_r2 = r2_score(y_true=y_test, y_pred=y_test_pred)
        test_mae = mean_absolute_error(y_true=y_test, y_pred=y_test_pred)
        test_mse = mean_squared_error(y_true=y_test, y_pred=y_test_pred)
        test_rmse = test_mse ** 0.5
        test_mape = mean_absolute_percentage_error(y_true=y_test, y_pred=y_test_pred)
        test_evaluator = RegressionMetric(y_true=y_test, y_pred=y_test_pred)
        test_smape = test_evaluator.symmetric_mean_absolute_percentage_error()
        test_nrmse = test_evaluator.normalized_root_mean_square_error()
        
        pred_metrics.append(('pymc_marketing', input_title, out_of_sample_col, 'test', test_r2, test_mae, test_mse, test_rmse, test_mape, test_smape, test_nrmse))
        pred_dfs.append(pred_df)
        
        # validation with incrementality tests...
        get_mean_contributions_over_time_df = mmm.compute_mean_contributions_over_time(original_scale=True)
        get_mean_contributions_over_time_df.columns = [f'{i}_n_conversion' for i in get_mean_contributions_over_time_df.columns]
        cac_df = pd.merge(model_df.set_index('date'), get_mean_contributions_over_time_df, left_index=True, right_index=True).reset_index(drop=False)
        cac_df['model'] = 'pymc_marketing'
        cac_df['data'] = input_title
        cac_df['out_of_sample_period'] = out_of_sample_col
        for feature in all_features:
            cac_df[f'{feature}_cac'] = cac_df[feature] / cac_df[f'{feature}_n_conversion']
        
        cac_dfs.append(cac_df)
    
    
    # cac_df[['iso_year_week', 'date', 'direct_mail_mel_spend', 'direct_mail_mel_spend_n_conversion']].to_clipboard()
    # cac_df[['iso_year_week', 'date', 'sea_brand_mel_spend', 'sea_brand_mel_spend_n_conversion']].to_clipboard()
    print('done')
    
    oos_df = pd.DataFrame(pred_metrics, columns=['model', 'data', 'out_of_sample_period', 'period', 'r2', 'mae', 'mse', 'rmse', 'mape', 'smape', 'nrmse'])
    oos_df.to_csv(save_path / 'out_of_sample_results.csv', index=False)
    oos_df.to_clipboard(index=False)
    
    
    
    cac_df = pd.concat(cac_dfs, axis=0, ignore_index=True)
    cac_info_cols = ['model', 'data', 'out_of_sample_period']
    cac_df = cac_df[cac_info_cols + [i for i in cac_df.columns if i not in cac_info_cols]
    cac_df.to_csv(save_path / 'cac_results.csv', index=False)
    cac_df.to_clipboard(index=False)
        
    
    # Full final run
    x = model_df[['date'] + all_features]
    y = model_df['n_conversion']
    
    mmm = DelayedSaturatedMMM(
            model_config = my_model_config,
            sampler_config = my_sampler_config,
            date_column='date',
            channel_columns=paid_media_spend_features,
            control_columns=control_features,
            adstock_max_lag=8,
            yearly_seasonality=2,
        )
        
    mmm.fit(X=x, y=y, target_accept=0.95, chains=6, random_seed=random_seed)
    
    mmm.fit_result
    
    var_names = ["intercept", "likelihood_sigma", "beta_channel", "alpha", "lam", "gamma_control", "gamma_fourier", ]
    
    # summary
    mmm_summary_df = az.summary(data=mmm.fit_result, var_names=var_names)
    mmm_summary_df['mean']
    mmm_summary_df
    spend_share_df
    mmm_summary_df.loc[[i for i in mmm_summary_df.index if 'beta_channel' in i or 'gamma_control' in i], ['mean', 'hdi_3%', 'hdi_97%']]
    
    feature_col_df
    
    _ = az.plot_trace(
    data=mmm.fit_result,
    var_names=[
        "intercept",
        "likelihood_sigma",
        "beta_channel",
        "alpha",
        "lam",
        "gamma_control",
        "gamma_fourier",
    ],
    compact=True,
    backend_kwargs={"figsize": (12, 10), "layout": "constrained"},
    )
    plt.gcf().suptitle("Model Trace", fontsize=16)
    plt.show()
    
    mmm.sample_posterior_predictive(x, extend_idata=True, combined=True)
    
    mmm.plot_posterior_predictive(original_scale=True)
    plt.show()
    
    mmm.plot_components_contributions()
    plt.show()
    
    mmm.get_target_transformer()
    
    get_mean_contributions_over_time_df = mmm.compute_mean_contributions_over_time(original_scale=True)
    get_mean_contributions_over_time_df.head()
    get_mean_contributions_over_time_df.to_clipboard()
    y.sum()
    get_mean_contributions_over_time_df.columns = [f'{i}_n_conversion' for i in get_mean_contributions_over_time_df.columns]
    
    cac_df = pd.merge(model_df.drop(columns='iso_year_week').set_index('date'), 
            get_mean_contributions_over_time_df, left_index=True, right_index=True)
    
    summary_res = []
    for feature in all_features:
        cac_df[f'{feature}_cac'] = cac_df[feature] / cac_df[f'{feature}_n_conversion']
        summary_res.append((feature, cac_df[feature].sum() / cac_df[f'{feature}_n_conversion'].sum()))
    
    pd.DataFrame(summary_res, columns=['feature', 'cac']).to_clipboard(index=False)
    
    cac_df.to_clipboard()
    
    
    fig = mmm.plot_direct_contribution_curves()
    [ax.set(xlabel="x") for ax in fig.axes]
    plt.show()
    
    mmm.plot_channel_contributions_grid(start=0, stop=1.5, num=12)
    plt.show()
    
    mmm.plot_channel_contributions_grid(start=0, stop=1.5, num=12, absolute_xrange=True)
    plt.show()
    
    
    channel_contribution_original_scale = mmm.compute_channel_contribution_original_scale()
    channel_contribution_original_scale
    
    roas_samples = channel_contribution_original_scale.stack(sample=("chain", "draw")).sum("date") / model_df[paid_media_spend_features].sum().to_numpy()[..., None]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for channel in roas_samples.channel:
        sns.histplot(roas_samples.sel(channel=channel).to_numpy(), binwidth=0.05, alpha=0.3, kde=True, ax=ax)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set(title="Posterior ROAS distribution", xlabel="ROAS")
    plt.show()
    
        
        
    
    
    
    
    

    
    
    
    
    pass

def run_uber_orbit_mmm_model(model_df):
    
    # https://github.com/uber/orbit
    
    
    
    
    pass

def run_mdp_gp_mmm_model(model_df):
    
    # use conda activate mdp_mmm
    # https://github.com/hellofresh/mdp-media-mix-model-international/blob/master/notebooks/databricks/pipeline.py
    # https://github.com/hellofresh/mdp-media-mix-model-international/blob/master/notebooks/databricks/Onboarding%20Analysis.py
    # https://github.com/hellofresh/mdp-media-mix-model-international/blob/master/notebooks/jupyter/gp_mmm.ipynb
    
    from pyspark.sql import SparkSession
    from pyspark.dbutils import DBUtils    
    import os
    import mlflow
    from omegaconf import OmegaConf
    import logging
    from mdp_media_mix_model_international.configs import read_local_configs, read_deployed_configs, ConfigValidator
    from mdp_media_mix_model_international.etl.master_etl import MasterETL
    from mdp_media_mix_model_international.post_processing.post_processor import PostProcessor
    from mdp_media_mix_model_international.constants import Environments
    from mdp_media_mix_model_international.training.train import main
    from mdp_media_mix_model_international.utilities.path_maker import PathMaker
    import logging
    from copy import deepcopy
    from datetime import datetime, timedelta
    from tempfile import TemporaryDirectory

    import numpy as np
    import pandas as pd
    from mdp_media_mix_model_international.configs import extract_parameters_from_config
    from mdp_media_mix_model_international.constants import Phases
    from mdp_media_mix_model_international.metrics import ecdf_crps, mape, mae, r2
    from mdp_media_mix_model_international.model import GaussianProcessMMM, DataScaler
    from mdp_media_mix_model_international.training.trainer import BayesianTrainer
    from mdp_media_mix_model_international.training.utils import make_df, NestablePool
    from mdp_media_mix_model_international.utilities import Plotter, CutPointFinder
    from omegaconf import OmegaConf
    
    logger = logging.getLogger(__name__)
    # logger.setLevel(logging.INFO)

    
    input_gsheet_url = 'https://docs.google.com/spreadsheets/d/1NH6ejWvNDDIZbOra6722ohyKQyoc_0MfO8OJb8xmbN0'
    
    input_title, param_df, feature_col_df, model_df, spend_fee_df = read_end_to_end_data(input_gsheet_url=input_gsheet_url)
    print(input_title)
    
    config_folder = 'models/mdp_config/'
    deployment_file = os.path.join(config_folder, 'us_mdp_mmm.yaml')
    print(deployment_file)
    configs = read_local_configs(config_folder)
    configs
    
    # Produce a list of ALL configs
    all_configs = []
    for country, country_configs in configs.items():
        all_configs.extend(country_configs)
    
    len(all_configs)
    
    # Validate all configs
    validator = ConfigValidator()
    for config in all_configs:
        validator.validate_config(config)
    
    spark = SparkSession.builder.getOrCreate()
    print('done loading with spark session')
    
    # Extract data only for markets whose configs will run
    markets = [market for market, configs in configs.items() if configs]
    environment = 'experiment'
    etl = MasterETL(spark)
    runs = etl.transform(etl.extract(markets=markets), all_configs, environment)
    
    len(runs)
    run = runs[0]
    config = run["config"]
    description_text = config.get("description", {}).get("text", "")
    market = config["country"]
    run.keys()
    
    phase_dirs = {}
    for phase in [Phases.Train, Phases.TrainVal, Phases.TrainValTest]:
        phase_dirs[phase] = TemporaryDirectory()
    
    phase_dirs
        
    # with mlflow.start_run(run_name=market, description=description_text, nested=False) as mlflow_run:
    # logger.info(f"Launching run: {mlflow_run.info.run_id}")
    df = pd.read_parquet(run["mmm_features"])
    
    # replace with actual data
    # NATIVE_META_spend
    # voucher_activations_NATIVE_META
    df
    
    train, val, test = make_df(config=config, df=df)
    cpf = CutPointFinder()
    phase_configs = {}
    logger.info("Finding cutpoints")
    phase_data = zip([Phases.Train, Phases.TrainVal, Phases.TrainValTest], 
                     [train, pd.concat([train, val]), pd.concat([train, val, test])])
    
    for phase, data in phase_data:
        phase_config = deepcopy(config)
        cutpoints = cpf.find_cutpoints(data)
        phase_config["mmm_config_hierarchical"]["gp"]["baseline"]["mean"]["cutpoints"] = cutpoints
        phase_configs[phase] = phase_config
    
    phase_configs['train']
    
    log_to_mlflow = False
    
    process_args = [
        (
            phase_configs[Phases.Train],
            train,
            pd.concat([val, test]),
            Phases.Train,
            log_to_mlflow,
        ),
        (
            phase_configs[Phases.TrainVal],
            pd.concat([train, val]),
            test,
            Phases.TrainVal,
            log_to_mlflow,
        ),
        (
            phase_configs[Phases.TrainValTest],
            pd.concat([train, val, test]),
            None,
            Phases.TrainValTest,
            log_to_mlflow,
        ),
    ]
    
    config = phase_configs[Phases.Train]
    train_df = train
    test_df = pd.concat([val, test])
    phase = Phases.Train
    log_to_mlflow
            
        
    def fit_and_maybe_log(config, train_df, test_df, phase, log_to_mlflow=True):
        
        metrics = {
            "MAE": mae,
            "MAPE": mape,
            "R2": r2,
            "ECDF-CRPS": ecdf_crps,
        }
        baseline_cfg = config["mmm_config_hierarchical"]["gp"]["baseline"]
        channel_cfg = config["mmm_config_hierarchical"]["gp"]["channel"]
        baseline_scaler = DataScaler(
            time=np.asarray(pd.DatetimeIndex(train_df["week_date"])),
            lengthscale=baseline_cfg["lengthscale"],
            kernel=baseline_cfg["kernel"],
            period=baseline_cfg["period"],
        )
        channel_scaler = DataScaler(
            time=np.asarray(pd.DatetimeIndex(train_df["week_date"])),
            lengthscale=channel_cfg["lengthscale"],
            kernel=channel_cfg["kernel"],
            period=channel_cfg.get("period"),
        )
        model = GaussianProcessMMM(config["mmm_config_hierarchical"], baseline_scaler, channel_scaler)
        trainer = BayesianTrainer(model, metrics=metrics)
        # Fit params should come from config
        trace, train_metrics = trainer.fit(
            train_df,
            train_df["overall_activs_paid"].values,
            tune=2000,
            draws=1000,
            chains=4,
            random_seed=1234,
            nuts_sampler="nutpie",
        )
        
        trace
        train_metrics
        
        backtest_trace, backtest_metrics, inference, ppc_forecast = None, None, None, None
        
        if test_df is not None:
            backtest_trace, backtest_metrics = trainer.backtest(train_df, test_df, test_df["overall_activs_paid"], trace)
        
        backtest_trace
        backtest_metrics
        
        backtest_trace['posterior_predictive']['ad_channels']
        backtest_trace['posterior_predictive']['backtest_channel_contributions'].shape
        backtest_trace['posterior_predictive']['backtest_channel_contributions'].mean(axis=0).mean(axis=0).shape
        backtest_trace['posterior_predictive']['backtest_channel_contributions'].mean(axis=0).mean(axis=0)
        
        test_df
        backtest_trace['posterior_predictive']
            
        def _date_converter(d):
            d = datetime.strptime(d, "%m/%d/%y")
            return "%d-W%.2d" % (d.isocalendar()[0], d.isocalendar()[1])

        posterior_fields = [
            "spending",
            "channel_contributions",
            "spending_raw",
            "incremental_cac",
            "initial_cac_time_modulated",
            "saturation",
            "expected_delay",
            "voucher_activations",
            "intercept",
        ]
        
        inference = trainer.predict(trace, posterior_fields, config["country"])
        inference
        
        train_df[['voucher_activations_LOWER_FUNNEL', 'voucher_activations_LOWER_FUNNEL']]

        # scenario_df = pd.DataFrame(columns=train_df.columns)
        # start_date = train_df.index[-1].date() + timedelta(weeks=1)

        # end_date = pd.date_range(start=start_date, freq="Q", periods=4)[-1]
        # forecast_date_range = pd.date_range(start=start_date, end=end_date, freq="W-MON")

        # scenario_df["week_date"] = forecast_date_range
        # scenario_df["week"] = [_date_converter(d.strftime("%m/%d/%y")) for d in forecast_date_range]
        # scenario_df.set_index("week_date", drop=False, inplace=True)
        # scenario_df = scenario_df.fillna(0)
        # scenario_df

        # ppc_forecast = trainer.forecast(train_df=train_df, scenario_df=scenario_df, scenario_y=None, trace=trace)
    
    
        
    
    
    
    
    
    
    
    
    
    pass

def run_models():
    
    # data_source = 'run_end_to_end_data'
    data_source = 'gsheet'
    data_gsheet_url = 'https://docs.google.com/spreadsheets/d/1bBuFI5bSpr66pNuDAC6RzWh_bHYzZ3HGP5HuzeJFLM4'
    model_type = 'robyn_mmm'
    model_type = 'lightweight_mmm'
    model_type = 'pymc_marketing_mmm'
    model_type = 'uber_orbit_mmm'
    
    # data source either run_end_to_end_data or get data from gsheet
    if data_source == 'run_end_to_end_data':
        model_df, grp_fee_df = run_end_to_end_data()
    elif data_source == 'gsheet':
        data_gs_client = GSheet(data_gsheet_url)
        model_df = data_gs_client.read_dataframe('model_data', header_row_num=0)
        grp_fee_df = data_gs_client.read_dataframe('fee_credit_data', header_row_num=0)
        for col in model_df.columns:
            model_df[col] = pd.to_numeric(model_df[col].str.replace(',', ''), errors='ignore')
        for col in grp_fee_df.columns:
            grp_fee_df[col] = pd.to_numeric(grp_fee_df[col].str.replace(',', ''), errors='ignore')
    
    
    model_df
    grp_fee_df
    
    
    # model
    if model_type == 'robyn_mmm':
        run_robyn_model(model_df)
        # break
        # select_robyn_model()
        
    elif model_type == 'lightweight_mmm':
        run_lightweight_mmm_model(model_df)
        # break
        
    elif model_type == 'pymc_marketing_mmm':
        run_pymc_marketing_mmm_model(model_df)
        # break
    
    elif model_type == 'uber_orbit_mmm':
        run_uber_orbit_mmm_model(model_df)
        # break
    
    pass
    
    
    
    
    pass



if __name__ == '__main__':
    
    print('main')
    
    