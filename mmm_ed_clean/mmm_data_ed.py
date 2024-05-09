
from dotenv import load_dotenv
load_dotenv(verbose=True, override=True, interpolate=True)
from pathlib import Path
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import lit
import pyspark.pandas as ps
from maulibs.GSheet import GSheet
from datetime import date, timedelta
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
from datetime import datetime
from time import sleep
from maulibs.Utils import get_vault_ent_credentials
from functools import reduce
from statsmodels.stats.outliers_influence import variance_inflation_factor
from ydata_profiling import ProfileReport
import scipy.stats
import statsmodels.api as sm
import argparse


print(pd.__version__)
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 100)

spark = SparkSession.builder.getOrCreate()
print('done loading with spark session')

# which mel, static or current?
mel_source = 'static'

# gsheets must have access to googlesheet-automations@hf-mau-dev.iam.gserviceaccount.com
# channel hieararchy columns
vs_ch_cols = ['vs_campaign_type', 'vs_channel_medium', 'vs_channel_category', 'vs_channel', 'vs_channel_split', 'vs_partner']
ch_cols = [i[3:] for i in vs_ch_cols]
# mel_plat_ch_cols = ['campaign_type', 'channel', 'channel_split', 'partner']

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

if int(pd.__version__[0]) > 1:
    pd.DataFrame.iteritems = pd.DataFrame.items


def write_to_gsheet(gsheet_url, write_dfs_dict):
    
    import gspread
    import gspread_dataframe
    # needs access to service account googlesheet-automations@hf-mau-dev.iam.gserviceaccount.com
    # credentials = get_vault_credentials("gsheet", secret_path="live/data-science-spew")
    credentials = get_vault_ent_credentials(
        namespace="infrastructure/data-science-spew",
        secret_path="spew",
        secret_key="gsheet"
    )
    gc = gspread.service_account_from_dict(credentials)
    sh = gc.open_by_url(gsheet_url)

    for i_name, i_df in write_dfs_dict.items():
        ws = sh.add_worksheet(title=i_name, rows=i_df.shape[0] + 1, cols=i_df.shape[1])
        int64_cols = i_df.dtypes[i_df.dtypes == 'int64'].index
        if len(int64_cols) > 0:
            i_df[int64_cols] = i_df[int64_cols].astype('float')
        gspread_dataframe.set_with_dataframe(worksheet=ws, dataframe=i_df)

    pass

def get_last_full_week_str():
    # get last full week
    last_week_date_str = (date.today() - timedelta(days=7)).strftime('%Y-%m-%d')
    last_week_sql = f'''
    select
        date_string_backwards, 
        iso_year_week
    from dimensions.date_dimension
    where date_string_backwards = '{last_week_date_str}'
    '''
    last_week_df = spark.sql(last_week_sql).toPandas()
    last_full_week_str = last_week_df.loc[0, 'iso_year_week']
    
    return last_full_week_str

def get_us_entities():
    ent_sql = f'''
    select
        country_code, 
        country_name, 
        bob_entity_code, 
        brand_code, 
        entity_name
    from dimensions.entity_dimension ed
    where country_code = 'US'
    '''
    ent_df = spark.sql(ent_sql).toPandas()
    print(ent_df)
    return ent_df

def get_platform_tables():
    # get list of platform metric tables
    show_platform_tables_sql = "show tables from marketing_analytics_us like '*daily_view'"
    platform_tables_df = spark.sql(show_platform_tables_sql).toPandas()
    print(platform_tables_df)

    # list from github
    # bing, dv360, facebook, google, horizon, reddit, rokt, veritone
    return platform_tables_df

def get_create_static_mel_table():
    
    today_date = date.today().strftime('%Y_%m_%d')
    folder_path = 'dbfs:/mnt/data-science-mau-dev/tables/'
    # file_name = f'ed_temp_mel_static_mmm_{today_date}'
    file_name = 'ed_temp_mel_static_mmm'
    folder_file_path = folder_path + file_name
    dbutils = DBUtils(spark)
    folder_ls = dbutils.fs.ls(folder_path)
    file_names = [i.name.replace('/', '') for i in folder_ls]
    
    if file_name not in file_names:
        
        print(f'writing parquet to {folder_file_path}')
        
        table_name = 'marketing_analytics_us.marketing_expense_log'
        refresh_res = spark.catalog.refreshTable(table_name)
        
        full_mel_query = f'''
        select
            *
        from marketing_analytics_us.marketing_expense_log mel
        '''
        full_mel_sdf = spark.sql(full_mel_query)
        full_mel_sdf.write.parquet(folder_file_path, mode='overwrite')
    
    else:
        print(f'static mel ready at {folder_file_path}')

    return folder_file_path

def get_mel_channel_hierarchy(bob_entity_code, min_max_date=('2020-01-01', '2023-12-31'), min_max_data_iso_year_week=('2020-W01', '2023-W52'), 
                              ch_output_cols=['campaign_type', 'channel_medium', 'channel_category', 'channel', 'channel_split', 'partner']):
    
    # bob_entity_code = 'US'
    # min_max_date = ''
    # min_max_data_iso_year_week = ('2019-W49', '2024-W15')
    # ch_output_cols = ['campaign_type', 'channel_medium', 'channel_category', 'channel', 'channel_split', 'partner']
    
    sql_output_lst = []
    mac_sql_output_lst = []
    # convert output_ch to mel.vs_ format
    if len(ch_output_cols) > 0:
        ch_output_cols = [i.replace('vs_', '') for i in ch_output_cols]
        for col in ch_output_cols:
            if col in ch_cols:
                sql_output_lst.append(f'lower(trim(mel.vs_{col})) {col}')
                mac_sql_output_lst.append(f'lower(trim(mac.{col})) {col}')
            else:
                sql_output_lst.append(f'mel.{col}')
    else:
        sql_output_lst = [f'lower(trim(mel.vs_{col})) {col}' for col in ch_cols]
        mac_sql_output_lst = [f'lower(trim(mac.{col})) {col}' for col in ch_cols]
    
    if min_max_date != '' and min_max_data_iso_year_week == '':
        min_date, max_date = min_max_date
        where_date_stmt = f"and dd.date_string_backwards >= '{min_date}' and dd.date_string_backwards <= '{max_date}'"
    elif min_max_date == '' and min_max_data_iso_year_week != '':
        min_week, max_week = min_max_data_iso_year_week
        where_date_stmt = f"and dd.iso_year_week >= '{min_week}' and dd.iso_year_week <= '{max_week}'"
    else:
        where_date_stmt = ''
    
    output_ch_stmt = ', '.join(sql_output_lst) + ', '
    n_gb = 5
    n_gb += len(sql_output_lst)
    gb_stmt = ','.join([str(i) for i in range(1, n_gb)])
    
    if mel_source == 'static':
        mel_file_path = get_create_static_mel_table()
        mel_sdf = spark.read.parquet(mel_file_path)
        mel_sdf.createOrReplaceTempView('static_marketing_expense_log')
        mel_table_name = 'static_marketing_expense_log'
    else:
        mel_table_name = 'marketing_analytics_us.marketing_expense_log'
    
    # get mel hierarchy
    mel_channel_hierarchy_sql = f'''
    select
        mel.bob_entity_code,
        dd.year, dd.month, record_updated_by, 
        {output_ch_stmt}
        sum(mel.mktg_spend_usd) mel_spend, 
        sum(mel.agency_fees) mel_agency_fees
    from {mel_table_name} mel
    join dimensions.date_dimension dd on mel.fk_date = dd.sk_date
    where mel.bob_entity_code = '{bob_entity_code}'
        {where_date_stmt}
    group by {gb_stmt}
    '''
    table_name = 'marketing_analytics_us.marketing_expense_log'
    refresh_res = spark.catalog.refreshTable(table_name)
    print(mel_channel_hierarchy_sql)
    mel_ch_df = spark.sql(mel_channel_hierarchy_sql).toPandas()
    
    for col in [i for i in mel_ch_df.columns if i not in ['year', 'month', 'mel_spend', 'mel_agency_fees']]:
        mel_ch_df[col] = mel_ch_df[col].str.strip().str.lower()
    
    mel_ch_df['mel_spend'] = mel_ch_df['mel_spend'].astype('float')
    mel_ch_df['mel_agency_fees'] = mel_ch_df['mel_agency_fees'].astype('float')
    
    # create year_month field
    mel_ch_df['year_month'] = pd.to_datetime(mel_ch_df['year'].astype(str) + '-' + mel_ch_df['month'].astype(str).str.zfill(2))
    # min max of year month
    mel_ch_agg_df = (mel_ch_df.groupby(ch_output_cols, dropna=False, as_index=False)
                     .agg({'mel_spend': 'sum', 'mel_agency_fees': 'sum', 'year_month': ['min', 'max']}).flat_cols()
                     .rename(columns={'mel_spend_sum': 'mel_spend', 'mel_agency_fees_sum': 'mel_agency_fees'})
                     .sort_values('mel_spend', ascending=False))
    mel_ch_agg_df
    # top 3 year month spends all time
    mel_ch_top_agg_df = (mel_ch_df.groupby(ch_output_cols + ['year_month'], dropna=False, as_index=False)
     .agg({'mel_spend': 'sum'}).sort_values('mel_spend', ascending=False)
     .groupby(ch_output_cols, dropna=False, as_index=False).head(3).sort_values(ch_output_cols))
    
    mel_ch_top_agg_df['year_month'] = mel_ch_top_agg_df['year_month'].astype('str')
    mel_ch_top_agg_df = mel_ch_top_agg_df.groupby(ch_output_cols, dropna=False, as_index=False).agg({'year_month': ' | '.join}).rename(columns={'year_month': 'top_3_year_month'})
    mel_ch_top_agg_df
    
    
    # top 3 users in 23 and 24 down to partner otherwise anytime
    mel_ch_df
    mel_ch_usr_agg_df = (mel_ch_df[mel_ch_df['year'].isin((2023, 2024))].groupby(ch_output_cols + ['record_updated_by'], dropna=False, as_index=False)
     .agg({'mel_spend': 'sum'}).sort_values('mel_spend', ascending=False)
     .groupby(ch_output_cols, dropna=False, as_index=False).head(3).sort_values(ch_output_cols))
    mel_ch_usr_agg_df['mm_spend'] = (mel_ch_usr_agg_df['mel_spend'] / 1_000_000).apply(lambda x: f'${x:,.2f}MM')
    mel_ch_usr_agg_df['user_spend'] = mel_ch_usr_agg_df['record_updated_by'] + ' ' + mel_ch_usr_agg_df['mm_spend']
    mel_ch_usr_agg_df = mel_ch_usr_agg_df[~mel_ch_usr_agg_df['user_spend'].isna()]
    mel_ch_usr_agg_df = mel_ch_usr_agg_df.groupby(ch_output_cols, dropna=False, as_index=False).agg({'user_spend': ' | '.join}).rename(columns={'user_spend': 'top_3_23_24_user_spend'})
    mel_ch_usr_agg_df
    
    mel_ch_agg_df = pd.merge(mel_ch_agg_df, mel_ch_top_agg_df, how='left', on=ch_output_cols)
    mel_ch_agg_df = pd.merge(mel_ch_agg_df, mel_ch_usr_agg_df, how='left', on=ch_output_cols)
    mel_ch_agg_df = mel_ch_agg_df.sort_values('mel_spend', ascending=False).reset_index(drop=True)
    
    
    # get and outer join with mac channel hierarchy
    mel_ch_agg_df
    
    mac_output_ch_stmt = ', '.join(mac_sql_output_lst) + ', '
    n_gb = 2
    n_gb += len(mac_sql_output_lst)
    mac_gb_stmt = ','.join([str(i) for i in range(1, n_gb)])
    
    mac_channel_hierarchy_sql = f'''
    select
        {mac_output_ch_stmt}
        lower(mac.conversion_type) conversion_type, 
        count(distinct cd.customer_uuid) n_conversion
    from marketing_data_product.marketing_attribution_conversions as mac
    left join global_bi_business.date_dimension dd on mac.fk_conversion_local_date = dd.date_id
    left join global_bi_business.customer_dimension cd on mac.customer_uuid = cd.customer_uuid
    where mac.bob_entity_code = '{bob_entity_code}'
        {where_date_stmt}
    group by {mac_gb_stmt}
    '''
    print(mac_channel_hierarchy_sql)
    mac_ch_df = spark.sql(mac_channel_hierarchy_sql).toPandas()
    mac_ch_df['conversion_type'] = mac_ch_df['conversion_type'].map({'activation': 'mac_n_activation', 'reactivation': 'mac_n_reactivation'})
    mac_ch_df['n_conversion'] = mac_ch_df['n_conversion'].fillna(0)
    mac_ch_df = mac_ch_df.fillna('')
    mac_ch_piv_df = mac_ch_df.pivot_table(index=ch_output_cols, columns='conversion_type', values='n_conversion', aggfunc='sum', fill_value=0).reset_index()
    mac_ch_piv_df = mac_ch_piv_df.replace('', pd.NA)
    mel_mac_df = pd.merge(mel_ch_agg_df, mac_ch_piv_df, how='outer', on=ch_output_cols)
    mel_mac_df
    
    return mel_mac_df

def read_in_channel_hierarchy_mapping(gsheet_url, gsheet_tab_name):
    
    gs_client = GSheet(gsheet_url)
    channel_hierarchy_df = gs_client.read_dataframe(gsheet_tab_name, header_row_num=0)
    
    # remove non alphanumeric characters from group column
    channel_hierarchy_df['group'] = channel_hierarchy_df['group'].apply(lambda s: re.sub(r'\W+', '_', s))
    # channel_hierarchy_df['group'] = channel_hierarchy_df['group'] + '_mel_spend'
    
    channel_hierarchy_df.columns = [col.replace('vs_', '').lower() for col in channel_hierarchy_df.columns]
    ch_df_cols_lst = channel_hierarchy_df.columns
    df_ch_filter_lst = [i for i in ch_cols if i in ch_df_cols_lst]
    channel_hierarchy_df = channel_hierarchy_df[df_ch_filter_lst + ['group']].drop_duplicates().replace('', pd.NA).reset_index(drop=True)
    channel_hierarchy_df = channel_hierarchy_df.apply(lambda x: x.str.strip().str.lower())
    
    # create lowest level for each group mapping
    ch_lowest_level_dfs = []
    channel_hierarchy_cp_df = channel_hierarchy_df.copy(deep=True)
    for idx in range(1, len(ch_cols)+1):
        cur_ch_level = ch_cols[:idx]
        cur_ch_level
        cur_ch_level_df = channel_hierarchy_cp_df[cur_ch_level + ['group']].drop_duplicates()
        cur_ch_level_n_df = (cur_ch_level_df
                                .groupby(cur_ch_level, as_index=False, dropna=False)
                                .agg({'group': 'count'}))
        cur_ch_level_n_df = cur_ch_level_n_df.loc[cur_ch_level_n_df['group'] == 1, cur_ch_level]
        
        if idx != len(ch_cols):
            nex_ch_level = ch_cols[:idx+1]
            nex_ch_level_df = channel_hierarchy_cp_df[nex_ch_level + ['group']].drop_duplicates()
            nex_ch_level_n_df = (nex_ch_level_df
                                .groupby(nex_ch_level, as_index=False, dropna=False)
                                .agg({'group': 'count'}))
            nex_ch_level_n_df = nex_ch_level_n_df.loc[nex_ch_level_n_df['group'] == 1, nex_ch_level]
            nex_ch_level_n_df = nex_ch_level_n_df.groupby(cur_ch_level, as_index=False, dropna=False).size()
            cur_ch_level_n_df = pd.merge(cur_ch_level_n_df, nex_ch_level_n_df, how='left', on=cur_ch_level)
            cur_ch_level_n_df = cur_ch_level_n_df[cur_ch_level_n_df['size'] != 1].drop(columns='size')
        
        if cur_ch_level_n_df.shape[0] > 0:
            cur_ch_level_n_df['remove'] = 1
            channel_hierarchy_cp_df = pd.merge(channel_hierarchy_cp_df, cur_ch_level_n_df, how='left', on=cur_ch_level)
            channel_hierarchy_cp_df = channel_hierarchy_cp_df[channel_hierarchy_cp_df['remove'].isna()].drop(columns='remove')
            cur_ch_level_n_df.drop(columns='remove', inplace=True)
            cur_ch_fin_df = pd.merge(cur_ch_level_n_df, cur_ch_level_df, how='inner', on=cur_ch_level)
            cur_ch_fin_df
            ch_lowest_level_dfs.append(cur_ch_fin_df)
        
    
    return channel_hierarchy_df, ch_lowest_level_dfs

def platform_vs_mel_compare_reports():
    
    working_doc_gsheet_url = 'https://docs.google.com/spreadsheets/d/1bxiXZW4qVWbQYAi1BbqBfzRWmAMkaQ1wSUcwCxSnLd4'
    wd_gs_client = GSheet(working_doc_gsheet_url)
    
    bob_entity_code = 'US'
    
    # get last full week
    last_week_date_str = (date.today() - timedelta(days=7)).strftime('%Y-%m-%d')
    last_week_sql = f'''
    select
        date_string_backwards, 
        iso_year_week
    from dimensions.date_dimension
    where date_string_backwards = '{last_week_date_str}'
    '''
    last_week_df = spark.sql(last_week_sql).toPandas()
    last_full_week_str = last_week_df.loc[0, 'iso_year_week']
    last_week_df
    last_full_week_str
    
    # get list of platform metric tables
    show_platform_tables_sql = "show tables from marketing_analytics_us like '*daily_view'"
    platform_tables_df = spark.sql(show_platform_tables_sql).toPandas()
    platform_tables_df.columns = ['database', 'table_name', 'is_temporary']
    print(platform_tables_df)

    # list from github https://github.com/hellofresh/mau-dbt-snowflake/tree/master/dbt_project/models/media_platform
    # bing, dv360, facebook, google, horizon (digital tv ott and tv linear), reddit, rokt, veritone
    plat_ch_cols = ['campaign_type', 'channel', 'channel_split', 'partner']
    
    # for each table, create report
    iyw_tot_agg_dfs = []
    date_tot_agg_dfs = []
    iyw_ytd_stat_dfs = []
    date_ytd_stat_dfs = []
    
    plat_table_name = platform_tables_df.loc[0, 'table_name']
    plat_table_name
    
    for plat_table_name in platform_tables_df['table_name']:
        print('*'*50)
        print(plat_table_name)
        
        plat_desc_query = f'describe marketing_analytics_us.{plat_table_name}'
        plat_desc_df = spark.sql(plat_desc_query).toPandas()
        
        # [i for i in plat_desc_df['col_name'].values if i in vs_ch_cols or i in ch_cols]
        
        if 'vs_campaign_type' in plat_desc_df['col_name'].values:
            # table_cols = ['vs_campaign_type', 'vs_channel', 'vs_channel_split', 'vs_partner']
            col_sql = '''lower(trim(plat.vs_campaign_type)) campaign_type,
            lower(trim(plat.vs_channel)) channel, 
            lower(trim(plat.vs_channel_split)) channel_split,
            lower(trim(plat.vs_partner)) partner,
            '''
            
        else:
            # table_cols = ['campaign_type', 'channel', 'channel_split', 'partner']
            col_sql = '''lower(trim(plat.campaign_type)) campaign_type,
            lower(trim(plat.channel)) channel, 
            lower(trim(plat.channel_split)) channel_split,
            lower(trim(plat.partner)) partner,
            '''
        
        # platform spend
        plat_query = f'''
        select
            dd.iso_year_week, 
            dd.date_string_backwards date, 
            plat.bob_entity_code,
            {col_sql}
            sum(total_spend) plat_total_spend
        from marketing_analytics_us.{plat_table_name} plat
        join dimensions.date_dimension dd on plat.fk_date = dd.sk_date
        where bob_entity_code = '{bob_entity_code}'
            and dd.iso_year_week <= '{last_full_week_str}'
        group by 1,2,3,4,5,6,7
        '''
        plat_df = spark.sql(plat_query).toPandas()
        plat_df['date'] = pd.to_datetime(plat_df['date'])
        plat_df = plat_df.replace('', pd.NA)
        
        plat_ch_df = plat_df[plat_ch_cols].drop_duplicates().reset_index(drop=True)
        plat_ch_df = plat_ch_df[plat_ch_df.isna().all(axis=1) == False].reset_index(drop=True)
        plat_ch_df
        
        plat_table_name
        plat_df['campaign_type'].value_counts()
        
        
        plat_ch_sdf = spark.createDataFrame(plat_ch_df)
        plat_ch_sdf.printSchema()
        plat_ch_sdf.createOrReplaceTempView('plat_ch')
        
        
        if mel_source == 'static':
            mel_file_path = get_create_static_mel_table()
            mel_sdf = spark.read.parquet(mel_file_path)
            mel_sdf.createOrReplaceTempView('static_marketing_expense_log')
            mel_table_name = 'static_marketing_expense_log'
        else:
            mel_table_name = 'marketing_analytics_us.marketing_expense_log'
        
        # mel spend data
        mel_query = f'''
        select
            dd.iso_year_week, 
            dd.date_string_backwards date, 
            mel.bob_entity_code,
            lower(trim(mel.vs_campaign_type)) campaign_type, 
            lower(trim(mel.vs_channel_medium)) channel_medium, 
            lower(trim(mel.vs_channel_category)) channel_category, 
            lower(trim(mel.vs_channel)) channel, 
            lower(trim(mel.vs_channel_split)) channel_split, 
            lower(trim(mel.vs_partner)) partner, 
            sum(mel.mktg_spend_usd) mel_spend, 
            sum(mel.agency_fees) mel_agency_fees
        from {mel_table_name} mel
        join dimensions.date_dimension dd on mel.fk_date = dd.sk_date
        join plat_ch pc on lower(trim(mel.vs_campaign_type)) <=> lower(trim(pc.campaign_type))
            and lower(trim(mel.vs_channel)) <=> lower(trim(pc.channel))
            and lower(trim(mel.vs_channel_split)) <=> lower(trim(pc.channel_split))
            and lower(trim(mel.vs_partner)) <=> lower(trim(pc.partner))
        where mel.bob_entity_code = '{bob_entity_code}'
            and dd.iso_year_week <= '{last_full_week_str}'
        group by 1,2,3,4,5,6,7,8,9
        '''
        table_name = 'marketing_analytics_us.marketing_expense_log'
        refresh_res = spark.catalog.refreshTable(table_name)
        mel_df = spark.sql(mel_query).toPandas()
        mel_df['date'] = pd.to_datetime(mel_df['date'])
        
        # aggregate by time frame and merge
        # time_cols = ['date', 'iso_year_week']
        time_cols = ['iso_year_week', 'date']
        for time_col in time_cols:
        
            time_plat_ch_cols = [time_col] + plat_ch_cols
            
            plat_agg_df = (plat_df[plat_df[plat_ch_cols].notna().all(axis=1)]
                            .groupby(time_plat_ch_cols, dropna=False, as_index=False)
                            .agg({'plat_total_spend': 'sum'}))
            mel_agg_df = mel_df.groupby(time_plat_ch_cols, dropna=False, as_index=False).agg({'mel_spend': 'sum'})
            plat_mel_agg_df = pd.merge(plat_agg_df, mel_agg_df, how='outer', on=time_plat_ch_cols)
            
            # min / max available data by channel
            # plat_agg_rep_df = plat_df.groupby(plat_ch_cols, dropna=False, as_index=False).agg({'date': ['min', 'max'], 
            #                                                                  'iso_year_week': ['min', 'max'], 
            #                                                                  'plat_total_spend': 'sum'})
            # plat_agg_rep_df.columns = ['_'.join(col).rstrip('_') for col in plat_agg_rep_df.columns.values]
            
            all_tot_agg_df = (plat_mel_agg_df
            .groupby(plat_ch_cols, dropna=False, as_index=False)
            .agg({time_col: ['min', 'max'], 
                'plat_total_spend': 'sum',
                'mel_spend': 'sum'}))
            all_tot_agg_df.columns = ['_'.join(col).rstrip('_') for col in all_tot_agg_df.columns.values]
            all_tot_agg_df = all_tot_agg_df.rename(columns={f'{time_col}_min': f'{time_col}_min_all', 
                                                f'{time_col}_max': f'{time_col}_max_all', 
                                                'plat_total_spend_sum': 'plat_total_spend_sum_all', 
                                                'mel_spend_sum': 'mel_spend_sum_all'})
            
            plat_tot_agg_df = (plat_mel_agg_df[plat_mel_agg_df['plat_total_spend'].notna()]
            .groupby(plat_ch_cols, dropna=False, as_index=False)
            .agg({time_col: ['min', 'max'], 
                'plat_total_spend': 'sum',
                'mel_spend': 'sum'}))
            plat_tot_agg_df.columns = ['_'.join(col).rstrip('_') for col in plat_tot_agg_df.columns.values]
            plat_tot_agg_df = plat_tot_agg_df.rename(columns={f'{time_col}_min': f'{time_col}_min_plat', 
                                                f'{time_col}_max': f'{time_col}_max_plat', 
                                                'plat_total_spend_sum': 'plat_total_spend_sum_plat', 
                                                'mel_spend_sum': 'mel_spend_sum_plat'})
            
            mel_tot_agg_df = (plat_mel_agg_df[plat_mel_agg_df['mel_spend'].notna()]
            .groupby(plat_ch_cols, dropna=False, as_index=False)
            .agg({time_col: ['min', 'max'], 
                'plat_total_spend': 'sum',
                'mel_spend': 'sum'}))
            mel_tot_agg_df.columns = ['_'.join(col).rstrip('_') for col in mel_tot_agg_df.columns.values]
            mel_tot_agg_df = mel_tot_agg_df.rename(columns={f'{time_col}_min': f'{time_col}_min_mel', 
                                                f'{time_col}_max': f'{time_col}_max_mel', 
                                                'plat_total_spend_sum': 'plat_total_spend_sum_mel', 
                                                'mel_spend_sum': 'mel_spend_sum_mel'})
            
            
            all_tot_agg_df = pd.merge(all_tot_agg_df, plat_tot_agg_df, how='outer', on=plat_ch_cols)
            all_tot_agg_df = pd.merge(all_tot_agg_df, mel_tot_agg_df, how='outer', on=plat_ch_cols)
            all_tot_agg_df['plat_table_name'] = plat_table_name
            all_tot_agg_df = all_tot_agg_df[['plat_table_name'] + [col for col in all_tot_agg_df.columns if col != 'plat_table_name']]
            
            if time_col == 'iso_year_week':
                iyw_tot_agg_dfs.append(all_tot_agg_df)
            elif time_col == 'date':
                date_tot_agg_dfs.append(all_tot_agg_df)
            
            # aggregate accuracy by YTD from different start weeks by day and week
            plat_mel_fil_df = (plat_mel_agg_df[plat_mel_agg_df['plat_total_spend'].notna()]
                               .sort_values(time_col, ascending=True))
            plat_mel_fil_df = plat_mel_fil_df.fillna(0)
            plat_mel_fil_df
            
            uni_times = plat_mel_fil_df[time_col].unique()
            uni_times.sort()
            ytd_stat_dfs = []
            
            for min_time in uni_times:
                
                sum_df = (plat_mel_fil_df[plat_mel_fil_df[time_col] >= min_time]
                .groupby(plat_ch_cols, dropna=False, as_index=False)
                .agg({'plat_total_spend': 'sum', 'mel_spend': 'sum'}))
                
                sum_df[f'min_{time_col}'] = min_time
                
                mape_df = (plat_mel_fil_df[plat_mel_fil_df[time_col] >= min_time]
                .groupby(plat_ch_cols, dropna=False, as_index=False)
                .apply(lambda x: mean_absolute_percentage_error(x['plat_total_spend'], x['mel_spend']))
                .rename(columns={None: 'mape'}))
                
                mape_df[f'min_{time_col}'] = min_time
                
                rmse_df = (plat_mel_fil_df[plat_mel_fil_df[time_col] >= min_time]
                .groupby(plat_ch_cols, dropna=False, as_index=False)
                .apply(lambda x: mean_squared_error(x['plat_total_spend'], x['mel_spend']))
                .rename(columns={None: 'rmse'}))
                
                rmse_df['rmse'] = np.sqrt(rmse_df['rmse'])
                rmse_df[f'min_{time_col}'] = min_time
                
                mae_df = (plat_mel_fil_df[plat_mel_fil_df[time_col] >= min_time]
                .groupby(plat_ch_cols, dropna=False, as_index=False)
                .apply(lambda x: mean_absolute_error(x['plat_total_spend'], x['mel_spend']))
                .rename(columns={None: 'mae'}))
                
                mae_df[f'min_{time_col}'] = min_time
                
                ytd_stat_df = pd.merge(sum_df, mape_df, how='outer', on=[f'min_{time_col}'] + plat_ch_cols)
                ytd_stat_df = pd.merge(ytd_stat_df, rmse_df, how='outer', on=[f'min_{time_col}'] + plat_ch_cols)
                ytd_stat_df = pd.merge(ytd_stat_df, mae_df, how='outer', on=[f'min_{time_col}'] + plat_ch_cols)
                ytd_stat_df = ytd_stat_df[[f'min_{time_col}'] + plat_ch_cols + ['plat_total_spend', 'mel_spend', 'mape', 'rmse', 'mae']]
                ytd_stat_dfs.append(ytd_stat_df)

            ytd_stat_df = pd.concat(ytd_stat_dfs, axis=0)
            ytd_stat_df['plat_table_name'] = plat_table_name
            ytd_stat_df = ytd_stat_df[['plat_table_name'] + [col for col in ytd_stat_df.columns if col != 'plat_table_name']]
            
            if time_col == 'iso_year_week':
                iyw_ytd_stat_dfs.append(ytd_stat_df)
            elif time_col == 'date':
                date_ytd_stat_dfs.append(ytd_stat_df)
        
        
        print('done', plat_table_name)
        print('*'*50)
        

    iyw_tot_agg_df = pd.concat(iyw_tot_agg_dfs, axis=0)
    date_tot_agg_df = pd.concat(date_tot_agg_dfs, axis=0)
    iyw_ytd_stat_df = pd.concat(iyw_ytd_stat_dfs, axis=0)
    date_ytd_stat_df = pd.concat(date_ytd_stat_dfs, axis=0)
    
    wd_gs_client.write_dataframe(worksheet_name='plat_mel_iyw_tot_agg', dataframe=iyw_tot_agg_df)
    wd_gs_client.write_dataframe(worksheet_name='plat_mel_date_tot_agg', dataframe=date_tot_agg_df)
    wd_gs_client.write_dataframe(worksheet_name='plat_mel_iyw_ytd_stat', dataframe=iyw_ytd_stat_df)
    wd_gs_client.write_dataframe(worksheet_name='plat_mel_date_ytd_stat', dataframe=date_ytd_stat_df)
    
    pass

def get_mel_spend_data(bob_entity_code, channel_hierarchy_df=None, include_ch_group=True, ch_inner_left='inner', min_max_date='', min_max_data_iso_year_week='', time_cols=['iso_year_week', 'date'], 
                       ch_output_cols=['campaign_type', 'channel_medium', 'channel_category', 'channel', 'channel_split', 'partner']):
    
    sql_output_lst = []
    # convert output_ch to mel.vs_ format
    if len(ch_output_cols) > 0:
        ch_output_cols = [i.replace('vs_', '') for i in ch_output_cols]
        for col in ch_output_cols:
            if col in ch_cols:
                sql_output_lst.append(f'lower(trim(mel.vs_{col})) {col}')
            else:
                sql_output_lst.append(f'mel.{col}')
    
    # clean channel_hierarchy_df
    if isinstance(channel_hierarchy_df, pd.DataFrame):
        channel_hierarchy_df.columns = [col.replace('vs_', '').lower() for col in channel_hierarchy_df.columns]
        ch_df_cols_lst = channel_hierarchy_df.columns
        df_ch_filter_lst = [i for i in ch_cols if i in ch_df_cols_lst]
        if 'group' in ch_df_cols_lst and include_ch_group:
            df_grp_lst = ['group']
            sql_output_lst.append(f'ch.group group')
        else:
            df_grp_lst = []
        ch_df_cols = df_ch_filter_lst + df_grp_lst
        channel_hierarchy_df = channel_hierarchy_df[ch_df_cols].drop_duplicates().replace('', pd.NA).reset_index(drop=True)
        channel_hierarchy_df = channel_hierarchy_df.apply(lambda x: x.str.strip().str.lower())
        if ch_inner_left == 'left':
            ch_join_stmt = 'left join channel_hierarchy ch on '
        else:
            ch_join_stmt = 'join channel_hierarchy ch on '
        ch_join_stmt = ch_join_stmt + 'and '.join([f'lower(trim(mel.vs_{ch})) <=> lower(trim(ch.{ch})) ' for ch in df_ch_filter_lst])
        ch_sdf = spark.createDataFrame(channel_hierarchy_df)
        ch_sdf.createOrReplaceTempView('channel_hierarchy')
    else:
        ch_join_stmt = ''
    
    # where date statement        
    if min_max_date != '' and min_max_data_iso_year_week == '':
        min_date, max_date = min_max_date
        where_date_stmt = f"and dd.date_string_backwards >= '{min_date}' and dd.date_string_backwards <= '{max_date}'"
    elif min_max_date == '' and min_max_data_iso_year_week != '':
        min_iso_year_week, max_iso_year_week = min_max_data_iso_year_week
        where_date_stmt = f"and dd.iso_year_week >= '{min_iso_year_week}' and dd.iso_year_week <= '{max_iso_year_week}'"
    
    
    n_gb = 1
    output_ch_stmt = ', '.join(sql_output_lst) + ', '
    n_gb += len(sql_output_lst)
    
    # create time_cols_stmt
    time_sql_cols = []
    if 'date' in time_cols or 'date_string_backwards' in time_cols:
        time_sql_cols.append('dd.date_string_backwards date')
    if 'iso_year_week' in time_cols:
        time_sql_cols.append('dd.iso_year_week iso_year_week')
    if len(time_sql_cols) > 0:
        time_cols_stmt = ', '.join(time_sql_cols) + ', '
    else:
        time_cols_stmt = ''    
    
    n_gb += len(time_sql_cols)
    
    gb_stmt = ','.join([str(i) for i in range(1, n_gb)])
    
    if mel_source == 'static':
        mel_file_path = get_create_static_mel_table()
        mel_sdf = spark.read.parquet(mel_file_path)
        mel_sdf.createOrReplaceTempView('static_marketing_expense_log')
        mel_table_name = 'static_marketing_expense_log'
    else:
        mel_table_name = 'marketing_analytics_us.marketing_expense_log'
    
    # mel spend data
    mel_query = f'''
    select
        {time_cols_stmt}
        {output_ch_stmt} 
        sum(mel.mktg_spend_usd) mel_spend, 
        sum(mel.agency_fees) mel_agency_fees
    from {mel_table_name} mel
    join dimensions.date_dimension dd on mel.fk_date = dd.sk_date
    {ch_join_stmt}
    where mel.bob_entity_code = '{bob_entity_code}'
        {where_date_stmt}
    group by {gb_stmt}
    '''
    print(mel_query)
    table_name = 'marketing_analytics_us.marketing_expense_log'
    refresh_res = spark.catalog.refreshTable(table_name)
    mel_df = spark.sql(mel_query).toPandas()
    
    if 'date' in time_cols or 'date_string_backwards' in time_cols:
        mel_df['date'] = pd.to_datetime(mel_df['date'])
    
    # for string columns, remove white space (\t or \n or space) at the end of a string
    for col in ch_output_cols:
        mel_df[col] = mel_df[col].str.strip()
    
    return mel_df

def dm_month_year_regex(input_string):
    '''Extracts the month and year from the input_string using regular expressions. 
    The month is the string name and can be abbreviated. For example, january or jan, february or feb, etc.
    The year follows the month name string and can be 2 or 4 digits. For example, 2023 or 23, 2022 or 22, etc.
    There is sometimes a space between the month and year.
    Examples = ['jan 2023', 'january 2023', 'jan23', 'january23', 'jan 2022', 'january2022', 'jan 22', 'january 22',
                        'feb2023', 'february 2023', 'feb 23', 'february23', 'feb 2022', 'february 2022', 'feb22', 'february 22']
    '''
    month_map = {month.lower(): calendar.month_name[i].lower() for i, month in enumerate(calendar.month_abbr) if month}
    month_number_map = {calendar.month_name[i].lower(): i for i in range(1, 13)}

    match = re.search(r'(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|aug(ust)?|sep(tember)?|oct(ober)?|nov(ember)?|dec(ember)?)', input_string, flags=re.IGNORECASE)
    if match:
        month_match = match.group(1).lower()
        month_name = month_map.get(month_match[:3])
        month_number = month_number_map.get(month_name)
        year_match = re.search(r'\d{2,4}', input_string[match.end():])
        year = None
        if year_match:
            year = year_match.group()
            if len(year) == 2:
                year = '20' + year
        return month_match, month_name, month_number, year
    return None, None, None, None

def adjust_direct_mail_ammortized_spend(bob_entity_code, min_max_data_iso_year_week):
    
    # "direct mail" and "gift cards & flyers" are the only channels I am aware of with ammortized spend
    # direct mail un-adstock (un-ammortized, un-carryover, un-lag, un-shift) spend data
    
    working_doc_gsheet_url = 'https://docs.google.com/spreadsheets/d/1bxiXZW4qVWbQYAi1BbqBfzRWmAMkaQ1wSUcwCxSnLd4'
    wd_gs_client = GSheet(working_doc_gsheet_url)
    
    dm_ch_df = wd_gs_client.read_dataframe(worksheet_name='us_mel_direct_mail', header_row_num=0)
    dm_ch_output_cols = ['campaign_type', 'channel_medium', 'channel_category', 'channel', 'channel_split', 'partner', 'campaign_name', 'breakdown_value1', 'breakdown_value2', 'breakdown_value3', 'breakdown_value4', 'breakdown_value5', 'marketing_notes']
    dm_mel_df = get_mel_spend_data(bob_entity_code=bob_entity_code, channel_hierarchy_df=dm_ch_df, include_ch_group=False, ch_inner_left='inner', 
                                   time_cols=['iso_year_week', 'date'], min_max_data_iso_year_week=min_max_data_iso_year_week, ch_output_cols=dm_ch_output_cols)
    dm_mel_df['mel_spend'].sum()
    
    dm_mel_df['date_year'] = dm_mel_df['date'].dt.year
    dm_mel_df['date_month'] = dm_mel_df['date'].dt.month
    
    dm_mel_df['camp_cat'] = (dm_mel_df['campaign_name'].fillna('')
                             + ' ' + dm_mel_df['breakdown_value1'].fillna('') 
                             + ' ' + dm_mel_df['breakdown_value2'].fillna('')
                             + ' ' + dm_mel_df['breakdown_value3'].fillna('')
                             + ' ' + dm_mel_df['breakdown_value4'].fillna('')
                             + ' ' + dm_mel_df['breakdown_value5'].fillna('')
                             + ' ' + dm_mel_df['marketing_notes'].fillna('')).str.lower()
    
    # apply the month_year_regex function and put output to 4 different columns
    camp_regex_df = pd.DataFrame(dm_mel_df['camp_cat'].apply(dm_month_year_regex).tolist(), columns=['month_year_match', 'month_year_name', 'camp_month', 'camp_year'])
    camp_regex_df = camp_regex_df[['camp_month', 'camp_year']].astype('Int64')
    dm_mel_df = pd.concat([dm_mel_df, camp_regex_df], axis=1)
    # dm_mel_df[['camp_cat', 'iso_year_week', 'month_year']].drop_duplicates().to_clipboard(index=False)
    dm_mel_df['campaign_type'].value_counts()
    
    # create logic if there is missing a year, then use the year from the date
    # but if the month is greater than the current month, then use the previous year
    dm_mel_df.loc[(dm_mel_df['camp_year'].isna()) & (dm_mel_df['camp_month'] <= dm_mel_df['date_month']), 'camp_year'] = dm_mel_df.loc[(dm_mel_df['camp_year'].isna()) & (dm_mel_df['camp_month'] <= dm_mel_df['date_month']), 'date_year']
    dm_mel_df.loc[(dm_mel_df['camp_year'].isna()) & (dm_mel_df['camp_month'] > dm_mel_df['date_month']), 'camp_year'] = dm_mel_df.loc[(dm_mel_df['camp_year'].isna()) & (dm_mel_df['camp_month'] > dm_mel_df['date_month']), 'date_year'] - 1
    dm_mel_df.loc[(dm_mel_df['camp_month'].isna()), 'camp_month'] = dm_mel_df.loc[(dm_mel_df['camp_month'].isna()), 'date_month']
    dm_mel_df.loc[(dm_mel_df['camp_year'].isna()), 'camp_year'] = dm_mel_df.loc[(dm_mel_df['camp_year'].isna()), 'date_year']
    dm_mel_df
    
    # change all addressed to bulk solo
    dm_mel_df.loc[dm_mel_df['channel_split'] == 'addressed', 'channel_split'] = 'bulk solo'
    
    # aggregate to in-home date and create final output data
    # for channel split "bulk solo" and "addressed" mail put proportion of spend across the campaign month on in-home days
    # for all other channel splits put all spend on the mondays of the month
    in_home_date_dfs = []
    in_home_cols = ['camp_year', 'camp_month', 'campaign_type', 'channel_medium', 'channel_category', 'channel', 'channel_split']
    # (dm_mel_df.groupby(['camp_cat'] + in_home_cols, as_index=False, dropna=False)
    #  .agg({'date': ['min', 'max'], 'iso_year_week': ['min', 'max'], 'mel_spend': 'sum'})
    #  .flat_cols().to_clipboard(index=False))
    
    dm_mel_df
    
    lt_gsheet_url = 'https://docs.google.com/spreadsheets/d/152KiSkjR24up9tOJc58myHNnZ47e6Rssa6i2tAOXurM'
    lt_gs_client =GSheet(lt_gsheet_url)
    
    raw_bs_in_home_date_df = lt_gs_client.read_dataframe(worksheet_name='dm_in_home_date_raw_data', header_row_num=0)
    raw_bs_in_home_date_df['new_mover'] = raw_bs_in_home_date_df['new_mover'].astype('int')
    raw_bs_in_home_date_df = raw_bs_in_home_date_df[(raw_bs_in_home_date_df['new_mover'] == 0) & (raw_bs_in_home_date_df['bob_entity_code'] == bob_entity_code)]
    raw_bs_in_home_date_df['campaign_month'] = pd.to_datetime(raw_bs_in_home_date_df['campaign_month'])
    raw_bs_in_home_date_df['in_home_date'] = pd.to_datetime(raw_bs_in_home_date_df['in_home_date'])
    # for volume column replace commas and convert to int
    raw_bs_in_home_date_df['volume'] = raw_bs_in_home_date_df['volume'].str.replace(',', '').astype('int')
    raw_bs_in_home_date_df['camp_year'] = raw_bs_in_home_date_df['campaign_month'].dt.year
    raw_bs_in_home_date_df['camp_month'] = raw_bs_in_home_date_df['campaign_month'].dt.month
    raw_bs_in_home_date_df
    
    in_home_min_date = raw_bs_in_home_date_df['in_home_date'].min().strftime('%Y-%m-%d')
    in_home_max_date = raw_bs_in_home_date_df['in_home_date'].max().strftime('%Y-%m-%d')
    
    in_home_week_sql = f'''
    select
        dd.iso_year_week in_home_iso_year_week, 
        dd.date_string_backwards in_home_date
    from dimensions.date_dimension dd
    where dd.date_string_backwards >= '{in_home_min_date}' and dd.date_string_backwards <= '{in_home_max_date}'
    '''
    in_home_week_df = spark.sql(in_home_week_sql).toPandas()
    in_home_week_df['in_home_date'] = pd.to_datetime(in_home_week_df['in_home_date'])
    in_home_week_df
    
    raw_bs_in_home_date_df = pd.merge(raw_bs_in_home_date_df, in_home_week_df, how='left', on='in_home_date')
    
    bs_in_home_total_df = (raw_bs_in_home_date_df.groupby(in_home_cols, as_index=False, dropna=False)
     .agg({'volume': 'sum'}).rename(columns={'volume': 'total_volume'}))
    bs_in_home_total_df
    
    bs_in_home_date_df = (raw_bs_in_home_date_df.groupby(in_home_cols + ['in_home_iso_year_week', 'in_home_date'], as_index=False, dropna=False)
     .agg({'volume': 'sum'}))
     
    bs_in_home_date_df = pd.merge(bs_in_home_date_df, bs_in_home_total_df, how='left', on=in_home_cols)
    bs_in_home_date_df['volume_proportion'] = bs_in_home_date_df['volume'] / bs_in_home_date_df['total_volume']
    bs_in_home_date_df[bs_in_home_date_df['volume_proportion'] != 1]
    bs_in_home_date_df
    in_home_date_dfs.append(bs_in_home_date_df[in_home_cols + ['in_home_iso_year_week', 'in_home_date', 'volume_proportion']])
    in_home_date_dfs
    
    # for "weekly solo" "shared mailer" put on Mondays of the campaign month
    min_data_iso_year_week, max_data_iso_year_week = min_max_data_iso_year_week
    year_min_date_iso_year_week = str(int(min_data_iso_year_week[:4])-1)+min_data_iso_year_week[-4:]
    year_plus_max_date_iso_year_week = str(int(max_data_iso_year_week[:4])+1)+max_data_iso_year_week[-4:]
    monday_sql = f'''
    select
        dd.iso_year_week in_home_iso_year_week, 
        dd.date_string_backwards in_home_date
    from dimensions.date_dimension dd
    where dd.iso_year_week >= '{year_min_date_iso_year_week}' and dd.iso_year_week <= '{year_plus_max_date_iso_year_week}'
        and dd.day_of_week = 0
    '''
    monday_df = spark.sql(monday_sql).toPandas()
    monday_df['in_home_date'] = pd.to_datetime(monday_df['in_home_date'])
    monday_df['camp_year'] = monday_df['in_home_date'].dt.year
    monday_df['camp_month'] = monday_df['in_home_date'].dt.month
    monday_total_df = monday_df.groupby(['camp_year', 'camp_month'], as_index=False, dropna=False).size().rename(columns={'size': 'n_monday'})
    monday_total_df['volume_proportion'] = 1 / monday_total_df['n_monday']
    monday_df = pd.merge(monday_df, monday_total_df, how='left', on=['camp_year', 'camp_month'])
    monday_df = monday_df[(monday_df['in_home_iso_year_week'] >= min_data_iso_year_week) & 
                          (monday_df['in_home_iso_year_week'] <= max_data_iso_year_week)].reset_index(drop=True)
    monday_df
    
    dm_ch_cols = ['campaign_type', 'channel_medium', 'channel_category', 'channel', 'channel_split']
    other_dm_ch_df = (dm_mel_df.loc[dm_mel_df['channel_split'] != 'bulk solo', dm_ch_cols]
     .drop_duplicates().reset_index(drop=True))
    
    for idx, row in other_dm_ch_df.iterrows():
        cs_in_home_df = monday_df.copy(deep=True)
        for ch in dm_ch_cols:
            cs_in_home_df[ch] = row[ch]
        in_home_date_dfs.append(cs_in_home_df[in_home_cols + ['in_home_iso_year_week', 'in_home_date', 'volume_proportion']])
    
    in_home_date_df = pd.concat(in_home_date_dfs, axis=0, ignore_index=True)
    
    dm_mel_df['mel_agency_fees'] = dm_mel_df['mel_agency_fees'].astype('float')
    dm_mel_spend_agg_df = (dm_mel_df[dm_mel_df['mel_spend'] > 0].groupby(in_home_cols, as_index=False, dropna=False)
     .agg({'mel_spend': 'sum', 'mel_agency_fees': 'sum', 'iso_year_week': 'min', 'date': 'min', })
     .rename(columns={'mel_spend': 'camp_mel_spend', 'mel_agency_fees': 'camp_mel_agency_fees', 
                      'iso_year_week': 'min_iso_year_week', 'date': 'min_date'}))
    
    dm_mel_spend_agg_df = pd.merge(in_home_date_df, dm_mel_spend_agg_df, how='outer', on=in_home_cols)
    # fill in missing values in in_home_iso_year_week with min_iso_year_week
    dm_mel_spend_agg_df['in_home_iso_year_week'] = dm_mel_spend_agg_df['in_home_iso_year_week'].fillna(dm_mel_spend_agg_df['min_iso_year_week'])
    dm_mel_spend_agg_df['in_home_date'] = dm_mel_spend_agg_df['in_home_date'].fillna(dm_mel_spend_agg_df['min_date'])
    dm_mel_spend_agg_df['volume_proportion'] = dm_mel_spend_agg_df['volume_proportion'].fillna(1)
    dm_mel_spend_agg_df['mel_spend'] = dm_mel_spend_agg_df['camp_mel_spend'] * dm_mel_spend_agg_df['volume_proportion']
    dm_mel_spend_agg_df['mel_agency_fees'] = dm_mel_spend_agg_df['camp_mel_agency_fees'] * dm_mel_spend_agg_df['volume_proportion']
    dm_mel_spend_agg_df
    
    tot_dm_year_spend_df = (dm_mel_spend_agg_df.groupby(['camp_year'] + dm_ch_cols, as_index=False, dropna=False).agg({'mel_spend': 'sum'})
     .rename(columns={'mel_spend': 'year_mel_spend'}))
    tot_dm_year_spend_df
    
    dm_mel_spend_agg_df = pd.merge(dm_mel_spend_agg_df, tot_dm_year_spend_df, how='left', on=['camp_year'] + dm_ch_cols)
    dm_mel_spend_agg_df['credit_proportion'] = dm_mel_spend_agg_df['mel_spend'] / dm_mel_spend_agg_df['year_mel_spend']
    
    tot_credit_df = (dm_mel_df[dm_mel_df['mel_spend'] < 0].groupby(['camp_year'] + dm_ch_cols, as_index=False, dropna=False)
                 .agg({'mel_spend': 'sum'}).rename(columns={'mel_spend': 'tot_credit'}))
    
    dm_mel_spend_agg_df = pd.merge(dm_mel_spend_agg_df, tot_credit_df, how='left', on=['camp_year'] + dm_ch_cols)
    dm_mel_spend_agg_df['tot_credit'] = dm_mel_spend_agg_df['tot_credit'].fillna(0)
    dm_mel_spend_agg_df['new_credit'] = dm_mel_spend_agg_df['credit_proportion'] * dm_mel_spend_agg_df['tot_credit']
    # dm_mel_spend_agg_df = dm_mel_spend_agg_df[(dm_mel_spend_agg_df['mel_spend'] > 0) | (dm_mel_spend_agg_df['mel_agency_fees'] > 0)].reset_index(drop=True)
    
    dm_mel_spend_df = (dm_mel_spend_agg_df[dm_mel_spend_agg_df['mel_spend'] > 0]
                       .groupby(['in_home_iso_year_week', 'in_home_date'] + dm_ch_cols, as_index=False, dropna=False)
                       .agg({'mel_spend': 'sum', 'mel_agency_fees': 'sum'})
                       .rename(columns={'in_home_iso_year_week': 'iso_year_week', 'in_home_date': 'date'}))
    dm_mel_credit_df = (dm_mel_spend_agg_df[dm_mel_spend_agg_df['new_credit'] < 0]
                        .groupby(['in_home_iso_year_week', 'in_home_date'] + dm_ch_cols, as_index=False, dropna=False)
                        .agg({'new_credit': 'sum'})
                        .rename(columns={'in_home_iso_year_week': 'iso_year_week', 'in_home_date': 'date', 'new_credit': 'mel_spend'}))
    dm_mel_credit_df['mel_agency_fees'] = 0
    dm_mel_fin_df = pd.concat([dm_mel_spend_df, dm_mel_credit_df], axis=0, ignore_index=True)
    dm_mel_fin_df
    
    def old_in_home_logic():
        in_home_df = dm_mel_df[in_home_cols].drop_duplicates().reset_index(drop=True)
        in_home_k_dfs = []
        for dol_k in [5_000, 4_000, 3_000, 2_000, 1_000, 500, 0, -100_000_000]:
            in_home_k_dfs.append((dm_mel_df[dm_mel_df['mel_spend'] > dol_k].groupby(in_home_cols, as_index=False, dropna=False)
                                .agg({'date': 'min', 'iso_year_week': 'min'}).rename(columns={'date': f'in_home_date', 'iso_year_week': f'in_home_iso_year_week'})))
        
        in_home_k_df = pd.concat(in_home_k_dfs, axis=0, ignore_index=True).drop_duplicates(subset=in_home_cols, keep='first').reset_index(drop=True)
        in_home_df = pd.merge(in_home_df, in_home_k_df, how='left', on=['camp_year', 'camp_month', 'channel_split'])
        
        dm_mel_df = pd.merge(dm_mel_df, in_home_df, how='left', on=in_home_cols)
        dm_mel_df
        
        # ch final level
        ch_final_level = ['campaign_type', 'channel_medium', 'channel_category', 'channel', 'channel_split']
        
        # re-distribute the credit spend with the actual spend
        # this will not be correct if they are entering large spend credit at the beginning of the year
        tot_credit_df = (dm_mel_df[dm_mel_df['mel_spend'] < 0].groupby(['camp_year'] + ch_final_level, as_index=False, dropna=False)
                    .agg({'mel_spend': 'sum'}).rename(columns={'mel_spend': 'tot_credit'}))
        spend_df = (dm_mel_df[dm_mel_df['mel_spend'] > 0].groupby(['camp_year', 'in_home_iso_year_week', 'in_home_date'] + ch_final_level, as_index=False, dropna=False)
                    .agg({'mel_spend': 'sum'}))
        tot_spend_df = (spend_df.groupby(['camp_year'] + ch_final_level, as_index=False, dropna=False)
                        .agg({'mel_spend': 'sum'}).rename(columns={'mel_spend': 'tot_spend'}))
        spend_df = pd.merge(spend_df, tot_spend_df, how='left', on=['camp_year'] + ch_final_level)
        spend_df = pd.merge(spend_df, tot_credit_df, how='left', on=['camp_year'] + ch_final_level)
        spend_df['spend_percent'] = spend_df['mel_spend'] / spend_df['tot_spend']
        spend_df['new_credit'] = spend_df['spend_percent'] * spend_df['tot_credit']
        spend_df['new_credit'].sum()
        tot_credit_df['tot_credit'].sum()
        dm_cred_fin_df = spend_df.loc[spend_df['new_credit'].notna(), ['in_home_iso_year_week', 'in_home_date'] + ch_final_level + ['new_credit']]
        dm_cred_fin_df = dm_cred_fin_df.rename(columns={'new_credit': 'mel_spend'})
        dm_cred_fin_df['mel_agency_fees'] = 0
        
        dm_mel_fin_df = (dm_mel_df[dm_mel_df['mel_spend'] > 0].groupby(['in_home_iso_year_week', 'in_home_date'] + ch_final_level, as_index=False, dropna=False)
                        .agg({'mel_spend': 'sum', 'mel_agency_fees': 'sum'}))
        dm_mel_fin_df = pd.concat([dm_mel_fin_df, dm_cred_fin_df], axis=0, ignore_index=True)
        dm_mel_fin_df = dm_mel_fin_df.rename(columns={'in_home_iso_year_week': 'iso_year_week', 'in_home_date': 'date'})
        dm_mel_fin_df['partner'] = None
        dm_mel_fin_df = dm_mel_fin_df[['iso_year_week', 'date'] + ch_cols + ['mel_spend', 'mel_agency_fees']]
        dm_mel_fin_df
        
        pass
    
    # wd_gs_client.write_dataframe(worksheet_name='Adjusted DM MEL Data', dataframe=dm_mel_fin_df)
    
    return dm_mel_fin_df
    
def adjust_gc_and_fly_ammortized_spend(bob_entity_code, min_max_data_iso_year_week):
    
    working_doc_gsheet_url = 'https://docs.google.com/spreadsheets/d/1bxiXZW4qVWbQYAi1BbqBfzRWmAMkaQ1wSUcwCxSnLd4'
    wd_gs_client = GSheet(working_doc_gsheet_url)
    
    # put all GC & FLY spend from 12 weeks to 4 weeks of ammortization
    gcfly_ch_df = wd_gs_client.read_dataframe(worksheet_name='us_mel_gc_and_fly', header_row_num=0)
    gcfly_ch_output_cols = ['campaign_type', 'channel_medium', 'channel_category', 'channel', 'channel_split', 'partner', 'campaign_name', 'breakdown_value1', 'breakdown_value2', 'breakdown_value3', 'breakdown_value4', 'breakdown_value5', 'marketing_notes']
    
    gcfly_mel_df = get_mel_spend_data(bob_entity_code=bob_entity_code, channel_hierarchy_df=gcfly_ch_df, include_ch_group=False, ch_inner_left='inner',
                                      time_cols=['iso_year_week', 'date'], min_max_data_iso_year_week=min_max_data_iso_year_week, ch_output_cols=gcfly_ch_output_cols)
    date_isy_df = gcfly_mel_df[['date', 'iso_year_week']].drop_duplicates().reset_index(drop=True)
    gcfly_mel_df['date_year'] = gcfly_mel_df['date'].dt.year
    gcfly_mel_df['date_month'] = gcfly_mel_df['date'].dt.month
    
    camp_agg_df = (gcfly_mel_df.groupby(ch_cols + ['campaign_name'], as_index=False, dropna=False)
     .agg({'mel_spend': 'sum', 'mel_agency_fees': 'sum', 
          'date': ['min', 'max']}))
    
    camp_agg_df = (gcfly_mel_df[gcfly_mel_df['mel_spend'] > 0].groupby(ch_cols + ['campaign_name'], as_index=False, dropna=False)
     .agg({'mel_spend': 'sum', 'mel_agency_fees': 'sum', 'date': ['min', 'max']}).flat_cols())
    camp_agg_df

    # get number of days between min and max date
    camp_agg_df['date_diff'] = (camp_agg_df['date_max'] - camp_agg_df['date_min']).dt.days + 1
    camp_agg_df['week_diff'] = camp_agg_df['date_diff'] / 7
    # camp_agg_df.to_clipboard(index=False)
    
    # if the number of amortization days is greater than 30 change the amort to 30 (uniformally)
    # if less than 30, change to uniformly distributed
    camp_dfs = []
    for idx, row in camp_agg_df.iterrows():
        date_diff = row['date_diff']
        date_min = row['date_min']
        if date_diff <= 30:
            daily_mel_spend = row['mel_spend_sum'] / date_diff
            daily_mel_agency_fees = row['mel_agency_fees_sum'] / date_diff
            date_max = row['date_max']
        else:
            daily_mel_spend = row['mel_spend_sum'] / 30
            daily_mel_agency_fees = row['mel_agency_fees_sum'] / 30
            date_max = row['date_min'] + timedelta(days=29)
        
        # create record for each date between date_min and date_max
        camp_df = pd.DataFrame(pd.date_range(start=date_min, end=date_max, freq='D'), columns=['date'])
        camp_df = pd.merge(camp_df, date_isy_df, how='left', on='date')
        camp_df['campaign_type'] = row['campaign_type']
        camp_df['channel_medium'] = row['channel_medium']
        camp_df['channel_category'] = row['channel_category']
        camp_df['channel'] = row['channel']
        camp_df['channel_split'] = row['channel_split']
        camp_df['partner'] = row['partner']
        camp_df['campaign_name'] = row['campaign_name']
        camp_df['mel_spend'] = daily_mel_spend
        camp_df['mel_agency_fees'] = daily_mel_agency_fees
        camp_dfs.append(camp_df)
        
    
    # camp_agg_df['mel_spend_sum'].sum()
    camp_df = pd.concat(camp_dfs, axis=0, ignore_index=True)
    # camp_df['mel_spend'].sum()
    camp_df['mel_spend'] = camp_df['mel_spend'].astype('float')
    camp_df['mel_agency_fees'] = camp_df['mel_agency_fees'].astype('float')
    camp_df
    
    gcfly_fin_df = (camp_df.groupby([ 'iso_year_week', 'date', 'campaign_type', 'channel_medium', 'channel_category', 'channel'], as_index=False, dropna=False)
     .agg({'mel_spend': 'sum', 'mel_agency_fees': 'sum'}))
    
    gcfly_fin_df['channel_split'] = None
    gcfly_fin_df['partner'] = None
        
    gcfly_fin_df = gcfly_fin_df[['iso_year_week', 'date'] + ch_cols + ['mel_spend', 'mel_agency_fees']]
    gcfly_fin_df
    
    # wd_gs_client.write_dataframe(worksheet_name='Adjusted GC FLY MEL Data', dataframe=gcfly_fin_df)
    
    return gcfly_fin_df

def adjust_platform_spend():
    # get list of platform metric tables
    show_platform_tables_sql = "show tables from marketing_analytics_us like '*daily_view'"
    platform_tables_df = spark.sql(show_platform_tables_sql).toPandas()
    platform_tables_df.columns = ['database', 'table_name', 'is_temporary']
    print(platform_tables_df)

    # list from github https://github.com/hellofresh/mau-dbt-snowflake/tree/master/dbt_project/models/media_platform
    # bing, dv360, facebook, google, horizon (digital tv ott and tv linear), reddit, rokt, veritone
    plat_ch_cols = ['campaign_type', 'channel', 'channel_split', 'partner']
    
    # for each table, create report
    iyw_tot_agg_dfs = []
    date_tot_agg_dfs = []
    iyw_ytd_stat_dfs = []
    date_ytd_stat_dfs = []
    
    for plat_table_name in platform_tables_df['table_name']:
        print('*'*50)
        print(plat_table_name)
        
        plat_desc_query = f'describe marketing_analytics_us.{plat_table_name}'
        plat_desc_df = spark.sql(plat_desc_query).toPandas()
        if 'vs_campaign_type' in plat_desc_df['col_name'].values:
            # table_cols = ['vs_campaign_type', 'vs_channel', 'vs_channel_split', 'vs_partner']
            col_sql = '''lower(trim(plat.vs_campaign_type)) campaign_type,
            lower(trim(plat.vs_channel)) channel, 
            lower(trim(plat.vs_channel_split)) channel_split,
            lower(trim(plat.vs_partner)) partner,
            '''
            
        else:
            # table_cols = ['campaign_type', 'channel', 'channel_split', 'partner']
            col_sql = '''lower(trim(plat.campaign_type)) campaign_type,
            lower(trim(plat.channel)) channel, 
            lower(trim(plat.channel_split)) channel_split,
            lower(trim(plat.partner)) partner,
            '''
        
        # platform spend
        plat_query = f'''
        select
            dd.iso_year_week, 
            dd.date_string_backwards date, 
            plat.bob_entity_code,
            {col_sql}
            sum(total_spend) plat_total_spend
        from marketing_analytics_us.{plat_table_name} plat
        join dimensions.date_dimension dd on plat.fk_date = dd.sk_date
        where bob_entity_code = '{bob_entity_code}'
            and dd.iso_year_week <= '{last_full_week_str}'
        group by 1,2,3,4,5,6,7
        '''
        plat_df = spark.sql(plat_query).toPandas()
        plat_df['date'] = pd.to_datetime(plat_df['date'])
        plat_df = plat_df.replace('', pd.NA)
        
        plat_ch_df = plat_df[plat_ch_cols].drop_duplicates().reset_index(drop=True)
        plat_ch_df = plat_ch_df[plat_ch_df.isna().all(axis=1) == False].reset_index(drop=True)
    
    
    pass

def create_jc_rollup_mel_channel_grouping(bob_entity_code='US'):    

    jc_query = f'''
    SELECT
        lower(trim(vs_campaign_type)) campaign_type, lower(trim(vs_channel_medium)) channel_medium, 
        lower(trim(vs_channel_category)) channel_category, lower(trim(vs_channel)) channel, 
        lower(trim(vs_channel_split)) channel_split, lower(trim(vs_partner)) partner,
        case when vs_campaign_type = 'Acquisition' and vs_channel_category = 'Referral' then 'Referral' 
            when vs_campaign_type = 'Reactivation' then 'Reactivation'
            else 'Activation' end as jc_campaign_type,
        (case when mel.vs_channel in ('App Paid Acquisition','App CRM') then 'App Paid Acquisition + CRM'
        else mel.vs_channel end) as jc_channel,
        (case when vs_channel = 'App Paid Acquisition' and vs_partner = 'Apple' then 'Apple'
        when ((vs_channel = 'App Paid Acquisition' and vs_partner = 'Google') or vs_channel = 'App CRM') then 'Google'
        when vs_channel = 'App Paid Acquisition' and lower(vs_partner) = 'ironsource' then 'ironSource'
        when vs_channel in ('Paid Social', 'Paid Social Reactivation') and lower(vs_channel_split) = 'brkfst' and lower(vs_partner) = 'facebook' then 'Brkfst Facebook'
        when vs_channel in ('Paid Social', 'Paid Social Reactivation') and lower(vs_channel_split) = 'brkfst' and lower(vs_partner) = 'tiktok' then 'Brkfst TikTok'
        when vs_channel in ('Paid Social', 'Paid Social Reactivation') and lower(vs_channel_split) = 'msa' then 'MSA'
        when vs_channel in ('Paid Social', 'Paid Social Reactivation') and lower(vs_channel_split) = 'facebook pod' then 'Facebook Pod'
        when vs_channel in ('Paid Social','Paid Social Reactivation') and lower(vs_partner) ilike '%facebook%' then 'Facebook'
        when vs_channel in ('Paid Social','Paid Social Reactivation') and lower(vs_partner) ilike '%tiktok%' then 'TikTok'
        when vs_channel in ('Paid Social','Paid Social Reactivation') and lower(vs_partner) ilike '%reddit%' then 'Reddit'
        when vs_channel in ('Paid Social','Paid Social Reactivation') and lower(vs_partner) ilike '%pyxis%' then 'Pyxis'
        when vs_channel in ('Paid Social','Paid Social Reactivation') and lower(vs_partner) ilike '%pinterest%' then 'Pinterest'
        when vs_channel in ('Paid Social','Paid Social Reactivation') and lower(vs_partner) ilike '%snapchat%' then 'Snapchat'
        when vs_channel in ('Paid Social','Paid Social Reactivation') and lower(vs_partner) not ilike '%facebook%' and lower(vs_partner) not ilike '%tiktok%' and lower(vs_partner) not ilike '%reddit%' and lower(vs_partner) not ilike '%pyxis%' and lower(vs_partner) not ilike '%pinterest%' and lower(vs_partner) not ilike '%snapchat%' then 'Other'
        when vs_channel = 'Display' and vs_partner = 'Rokt' then 'ROKT'
        when vs_channel = 'Display' and vs_partner ilike '%LiveIntent%' then 'LiveIntent'
        when vs_channel = 'Display' and vs_partner = 'TapJoy' then 'TapJoy'
        when vs_channel = 'Display' and vs_partner = 'Outbrain' then 'Outbrain'
        when vs_channel = 'Display' and vs_partner = 'Taboola' then 'Taboola'
        when vs_channel = 'Display' and vs_channel_split = 'GSP' then 'Discovery'
        when vs_channel = 'Display' and vs_channel_split = 'GDN' then 'GDN'
        when vs_channel = 'Display' and vs_channel_split = 'DV360' then 'DV360'
        when vs_channel = 'Native' and vs_partner in ('SoYummy','First Media') then 'First Media'
        when vs_channel = 'Native' and vs_partner in ('Geist','GeistM') then 'Geist'
        when vs_channel = 'Native' and lower(vs_partner) in ('stoyo') then 'Stoyo'
        when vs_channel = 'Native' and vs_partner not in ('SoYummy','First Media','Geist','GeistM','Stoyo') then 'Other'
        when vs_channel = 'Influencer Performance' and vs_channel_split = 'Strategic Storytelling' then 'Stream Elements'
        when vs_channel = 'Influencer Performance' and vs_channel_split != 'Strategic Storytelling' then vs_channel_split
        when vs_channel = 'Partnerships' and vs_channel_split = 'B2B2C Partnerships' then 'B2B2C Partnerships'
        when vs_channel = 'Partnerships' and vs_channel_split != 'B2B2C Partnerships' then 'Other'
        when vs_channel = 'SEA Non-Brand' then vs_channel_split
        when vs_channel = 'Direct Mail Performance' and vs_channel_split not in ('Addressed', 'Bulk Solo Addressed', 'Shared Mailer (3rd party)', 'Weekly Solo Addressed') then 'Other'
        when vs_channel = 'Direct Mail CRM' and vs_channel_split not in ('Addressed', 'Bulk Solo Addressed', 'Programmatic', 'Weekly Solo Addressed') then 'Other'
        when vs_channel = 'Direct Mail Performance' and vs_channel_split in ('Addressed', 'Bulk Solo Addressed', 'Shared Mailer (3rd party)', 'Weekly Solo Addressed') then vs_channel_split
        when vs_channel = 'Direct Mail CRM' and vs_channel_split in ('Addressed', 'Bulk Solo Addressed', 'Programmatic', 'Weekly Solo Addressed') then vs_channel_split
        when vs_channel = 'Out of Home' then 'Combined'
    else 'None' end) as jc_channel_split_partner, 
        sum(mel.mktg_spend_usd) mel_spend, 
        sum(mel.agency_fees) mel_agency_fees
    FROM marketing_analytics_us.marketing_expense_log mel
    left join dimensions.date_dimension dd on mel.fk_date = dd.sk_date
    WHERE mel.bob_entity_code = '{bob_entity_code}'
    GROUP BY 1,2,3,4,5,6,7,8,9
    '''
    jc_df = spark.sql(jc_query).toPandas()
    jc_df
    
    return jc_df

def get_per_ch_to_mel_data(bob_entity_code='US', min_max_data_iso_year_week=('2021-W40', '2024-W04'), 
                                       adjust_dm=True, adjust_gc_fly=True, min_ch_spend_percent=0.03, 
                                       remove_campaign_types=['reactivation']):
    
    # add channel hierarchy level column (upper, mid upper, mid, mid lower, lower)
    # add meta facebook and instagram
    # given these channel hierarchy columns keep splitting until hit percent of spend or correlation is above 80%
    
    ch_output_cols = ['campaign_type', 'channel_medium', 'channel_category', 'channel', 'channel_split', 'partner']
    
    # if daily_or_weekly == 'daily':
    #     time_col = 'date'
    # elif daily_or_weekly == 'weekly':
    #     time_col = 'iso_year_week'
    
    # read in all mel data down to the partner level
    mel_df = get_mel_spend_data(bob_entity_code=bob_entity_code, channel_hierarchy_df=None, include_ch_group=False, 
                                min_max_data_iso_year_week=min_max_data_iso_year_week, time_cols=['iso_year_week', 'date'], 
                                ch_output_cols=ch_output_cols)
                                
    print(mel_df)
    
    # remove campaign types
    mel_df = mel_df[~mel_df['campaign_type'].isin(remove_campaign_types)].reset_index(drop=True)
    data_campaign_types = list(mel_df['campaign_type'].unique())
    # param_dict['data_campaign_types'] = data_campaign_types
    mel_df[mel_df['campaign_type'] == '']
    
    # optional
    # read in or create DM and GC & FLY adjusted spend
    # remove all DM and GC & FLY spend as it is adjusted
    if adjust_dm:
        dm_mel_df = adjust_direct_mail_ammortized_spend(bob_entity_code=bob_entity_code, min_max_data_iso_year_week=min_max_data_iso_year_week)
        dm_cols = ['campaign_type', 'channel_medium', 'channel_category', 'channel']
        dm_remove_ch_df = dm_mel_df[dm_cols].drop_duplicates().reset_index(drop=True)
        dm_remove_ch_df['dm_remove'] = True
        
        # og_dm_mel_df = mel_df[mel_df['channel'] == 'direct mail performance']
        # og_dm_mel_df = og_dm_mel_df.groupby(['channel_split', 'iso_year_week'], as_index=False, dropna=False).agg({'mel_spend': 'sum'})
        # og_dm_mel_df['year'] = og_dm_mel_df['iso_year_week'].str[:4].astype('int')
        # og_dm_mel_df['week'] = og_dm_mel_df['iso_year_week'].str[-2:].astype('int')
        
        # new_dm_agg_df = dm_mel_df.groupby(['channel_split', 'iso_year_week'], as_index=False, dropna=False).agg({'mel_spend': 'sum'}).rename(columns={'mel_spend': 'new_dm_spend'})
        # og_dm_mel_df = pd.merge(og_dm_mel_df, new_dm_agg_df, how='left', on=['channel_split', 'iso_year_week'])
        # og_dm_mel_df['new_dm_spend'] = og_dm_mel_df['new_dm_spend'].fillna(0)
        # og_dm_mel_df.to_clipboard(index=False)
        
        mel_df = pd.merge(mel_df, dm_remove_ch_df, how='left', on=dm_cols)
        mel_df = mel_df[mel_df['dm_remove'] != True]
        mel_df = mel_df.drop(columns='dm_remove')
        mel_df = pd.concat([mel_df, dm_mel_df], axis=0, ignore_index=True)
        
    if adjust_gc_fly:
        gcfly_mel_df = adjust_gc_and_fly_ammortized_spend(bob_entity_code=bob_entity_code, min_max_data_iso_year_week=min_max_data_iso_year_week)
        gcfly_cols = ['campaign_type', 'channel_medium', 'channel_category', 'channel']
        gcfly_remove_ch_df = gcfly_mel_df[gcfly_cols].drop_duplicates().reset_index(drop=True)
        gcfly_remove_ch_df['gcfly_remove'] = True
        mel_df = pd.merge(mel_df, gcfly_remove_ch_df, how='left', on=gcfly_cols)
        mel_df = mel_df[mel_df['gcfly_remove'] != True]
        mel_df = mel_df.drop(columns='gcfly_remove')
        mel_df = pd.concat([mel_df, gcfly_mel_df], axis=0, ignore_index=True)
    
    # down to the partner level, if the percent of spend is higher than the percent limit use that level
    # if not rollup 1 level and repeat until the percent limit is met
    mel_df['mel_spend'] = mel_df['mel_spend'].replace('', 0).fillna(0).astype('float')
    mel_df['mel_agency_fees'] = mel_df['mel_agency_fees'].replace('', 0).fillna(0).astype('float')
    mel_df = mel_df.fillna('')
    total_mel_spend = mel_df['mel_spend'].sum()
    sel_ch_dfs = []
    mel_df['selected_x'] = False
    mel_df['group'] = pd.NA
    
    for i in range(0, len(ch_cols)):
        
        if i != 0:
            fil_ch_cols = ch_cols[:-i]
        else:
            fil_ch_cols = ch_cols
            
        print(i, fil_ch_cols)
        ch_level = fil_ch_cols[-1]
        
        mel_agg_df = mel_df[mel_df['selected_x'] != True].groupby(fil_ch_cols, as_index=False, dropna=False).agg({'mel_spend': 'sum'})
        mel_agg_df[f'{ch_level}_percent'] = mel_agg_df['mel_spend'] / total_mel_spend
        mel_agg_df = mel_agg_df.sort_values(by=f'{ch_level}_percent', ascending=False).reset_index(drop=True)
        mel_agg_df
        sel_ch_df = mel_agg_df.loc[mel_agg_df[f'{ch_level}_percent'] >= min_ch_spend_percent, fil_ch_cols]
        
        if sel_ch_df.shape[0] != 0:
            sel_ch_df['group'] = sel_ch_df[fil_ch_cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
        else:
            sel_ch_df = mel_agg_df[fil_ch_cols]
            sel_ch_df['group'] = 'other'
        
        sel_ch_df['selected_y'] = True
        sel_ch_dfs.append(sel_ch_df)
        sel_ch_df
        
        mel_df = pd.merge(mel_df, sel_ch_df, how='left', on=fil_ch_cols)
        mel_df['selected_y'] = mel_df['selected_y'].fillna(False)
        mel_df['selected_x'].sum()
        mel_df['selected_x'] = mel_df['selected_x'] | mel_df['selected_y']
        mel_df['selected_x'].sum()
        mel_df = mel_df.drop(columns='selected_y')
        mel_df['group'] = mel_df['group_x'].fillna(mel_df['group_y'])
        mel_df = mel_df.drop(columns=['group_x', 'group_y'])
        
        if mel_df['selected_x'].sum() == mel_df.shape[0]:
            break
        
    mel_df = mel_df.drop(columns='selected_x')    
    # sel_ch_df = pd.concat(sel_ch_dfs, axis=0, ignore_index=True)
    
    return mel_df

def map_low_chs_to_ch(ch_map_low_dfs, mel_df):
    mel_df['group_x'] = pd.NA
    for ch_map_df in ch_map_low_dfs:
        col_join = [col for col in ch_map_df.columns if col != 'group']
        mel_df = pd.merge(mel_df, ch_map_df.rename(columns={'group': 'group_y'}), how='left', on=col_join)
        mel_df['group_x'] = mel_df['group_x'].fillna(mel_df['group_y'])
        mel_df = mel_df.drop(columns='group_y')
    
    mel_df = mel_df.rename(columns={'group_x': 'group'})
    mel_df['group'] = mel_df['group'].fillna('missing')
    return mel_df

def get_ch_to_mel_data(ch_map_low_dfs, bob_entity_code='US', min_max_data_iso_year_week=('2021-W40', '2024-W04'), 
                       adjust_dm=True, adjust_gc_fly=True, adjust_credits=True):
    
    ch_output_cols = ['campaign_type', 'channel_medium', 'channel_category', 'channel', 'channel_split', 'partner']
    
    # need to first read in the raw data down to partner level
    # read in all mel data down to the partner level
    mel_df = get_mel_spend_data(bob_entity_code=bob_entity_code, channel_hierarchy_df=None, include_ch_group=False, 
                                min_max_data_iso_year_week=min_max_data_iso_year_week, time_cols=['iso_year_week', 'date'], 
                                ch_output_cols=ch_output_cols)
    
    # optional
    # read in or create DM and GC & FLY adjusted spend
    # remove all DM and GC & FLY spend as it is adjusted
    # adjust DM and GC & FLY spend
    before_after_mel_spend_dfs = []
    if adjust_dm:
        dm_mel_df = adjust_direct_mail_ammortized_spend(bob_entity_code=bob_entity_code, min_max_data_iso_year_week=min_max_data_iso_year_week)
        # dm_mel_df.to_clipboard(index=False)
        dm_cols = ['campaign_type', 'channel_medium', 'channel_category', 'channel']
        dm_remove_ch_df = dm_mel_df[dm_cols].drop_duplicates().reset_index(drop=True)
        dm_remove_ch_df['dm_remove'] = True
        mel_df = pd.merge(mel_df, dm_remove_ch_df, how='left', on=dm_cols)
        mel_dm_spend = np.ceil(mel_df.loc[mel_df['dm_remove'] == True, 'mel_spend'].sum())
        adjusted_mel_dm_spend = np.ceil(dm_mel_df['mel_spend'].sum())
        dm_remove_ch_df['before_mel_spend'] = mel_dm_spend
        dm_remove_ch_df['after_mel_spend'] = adjusted_mel_dm_spend
        dm_remove_ch_df = dm_remove_ch_df.drop(columns='dm_remove')
        before_after_mel_spend_dfs.append(dm_remove_ch_df)
        
        if mel_dm_spend == adjusted_mel_dm_spend:
            print('dm spend matches', mel_dm_spend, adjusted_mel_dm_spend)
        else:
            print('dm spend does not match', mel_dm_spend, adjusted_mel_dm_spend)
        
        mel_df = mel_df[mel_df['dm_remove'].isna()]
        mel_df = pd.concat([mel_df, dm_mel_df], axis=0, ignore_index=True)
        mel_df = mel_df.drop(columns='dm_remove')
        # mel_df.loc[mel_df['channel'] == 'direct mail performance', 'mel_spend'].sum()
        
        
    if adjust_gc_fly:
        gcfly_mel_df = adjust_gc_and_fly_ammortized_spend(bob_entity_code=bob_entity_code, min_max_data_iso_year_week=min_max_data_iso_year_week)
        gcfly_cols = ['campaign_type', 'channel_medium', 'channel_category', 'channel']
        gcfly_remove_ch_df = gcfly_mel_df[gcfly_cols].drop_duplicates().reset_index(drop=True)
        gcfly_remove_ch_df['gcfly_remove'] = True
        mel_df = pd.merge(mel_df, gcfly_remove_ch_df, how='left', on=gcfly_cols)
        
        mel_gcfly_spend = np.ceil(mel_df.loc[mel_df['gcfly_remove'] == True, 'mel_spend'].sum())
        adjusted_mel_gcfly_spend = np.ceil(gcfly_mel_df['mel_spend'].sum())
        gcfly_remove_ch_df['before_mel_spend'] = mel_gcfly_spend
        gcfly_remove_ch_df['after_mel_spend'] = adjusted_mel_gcfly_spend
        gcfly_remove_ch_df = gcfly_remove_ch_df.drop(columns='gcfly_remove')
        before_after_mel_spend_dfs.append(gcfly_remove_ch_df)
        
        if mel_gcfly_spend == adjusted_mel_gcfly_spend:
            print('gcfly spend matches', mel_gcfly_spend, adjusted_mel_gcfly_spend)    
        else:
            print('gcfly spend does not match', mel_gcfly_spend, adjusted_mel_gcfly_spend)
        
        mel_df = mel_df[mel_df['gcfly_remove'].isna()]
        mel_df = mel_df.drop(columns='gcfly_remove')
        mel_df = pd.concat([mel_df, gcfly_mel_df], axis=0, ignore_index=True)
    
    
    if adjust_credits:
        # remove credits from the data
        # adjust the credits to be distributed by the year of the daily spend for the channel
        # rename credits to "mel_agency_fees"
        cred_mel_df = mel_df[mel_df['mel_spend'] < 0].reset_index(drop=True)
        cred_before_after_df = cred_mel_df.groupby(ch_cols, as_index=False, dropna=False).agg({'mel_spend': 'sum'}).rename(columns={'mel_spend': 'before_mel_spend'})
        cred_before_after_df['after_mel_spend'] = 0
        before_after_mel_spend_dfs.append(cred_before_after_df)
        mel_df = mel_df[mel_df['mel_spend'] > 0].reset_index(drop=True)
        cred_mel_df['year'] = cred_mel_df['date'].dt.year
        tot_cred_mel_df = (cred_mel_df.groupby(['year'] + ch_cols, as_index=False, dropna=False).agg({'mel_spend': 'sum'})
                           .rename(columns={'mel_spend': 'tot_credit'}))
        spend_df = (mel_df.groupby(['iso_year_week', 'date'] + ch_cols, as_index=False, dropna=False)
                    .agg({'mel_spend': 'sum'}))
        spend_df['year'] = spend_df['date'].dt.year
        tot_spend_df = (spend_df.groupby(['year'] + ch_cols, as_index=False, dropna=False)
                        .agg({'mel_spend': 'sum'}).rename(columns={'mel_spend': 'tot_spend'}))
        spend_df = pd.merge(spend_df, tot_spend_df, how='left', on=['year'] + ch_cols)
        spend_df = pd.merge(spend_df, tot_cred_mel_df, how='outer', on=['year'] + ch_cols)
        spend_df['spend_percent'] = (spend_df['mel_spend'] / spend_df['tot_spend']).fillna(1)
        spend_df['new_credit'] = (spend_df['spend_percent'] * spend_df['tot_credit']).fillna(0)
        spend_df['new_credit'].sum()
        tot_cred_mel_df['tot_credit'].sum()
        tot_cred_mel_df
        cred_fin_df = spend_df.loc[spend_df['new_credit'] < 0, ['iso_year_week', 'date'] + ch_cols + ['new_credit']].rename(columns={'new_credit': 'mel_agency_fees'})
        cred_fin_df['mel_spend'] = 0
        cred_fin_df
        mel_df = pd.concat([mel_df, cred_fin_df], axis=0, ignore_index=True)
    
    # adjust for other platforms
    
    # report on before and after adjustments
    if len(before_after_mel_spend_dfs) > 0:
        before_after_mel_spend_df = pd.concat(before_after_mel_spend_dfs, axis=0, ignore_index=True)
    else:
        before_after_mel_spend_df = pd.DataFrame()
    
    
    # then map to channel hierarchy with group using the lowest level
    # other option is to create a new channel hierarchy down to partner level from this lowest level join in sql 
    mel_df = map_low_chs_to_ch(ch_map_low_dfs=ch_map_low_dfs, mel_df=mel_df)
    
    mel_df['mel_spend'] = mel_df['mel_spend'].astype('float')
    mel_df['mel_agency_fees'] = mel_df['mel_agency_fees'].astype('float')
    
    min_data_iso_year_week, max_data_iso_year_week = min_max_data_iso_year_week
    mel_df = (mel_df[(mel_df['iso_year_week'] >= min_data_iso_year_week) & (mel_df['iso_year_week'] <= max_data_iso_year_week)]
              .sort_values('date', ascending=True).reset_index(drop=True))
    
    return before_after_mel_spend_df, mel_df

def summ_stats(df):
    dtypes_ser = df.dtypes
    dtypes_df = pd.DataFrame(dtypes_ser).T
    dtypes_df['index'] = 'dtype'
    numeric_cols = dtypes_ser[dtypes_ser.apply(lambda x: pd.api.types.is_numeric_dtype(x))].index
    tot_df = pd.DataFrame(df.loc[:, numeric_cols].sum()).T
    tot_df['index'] = 'sum'
    desc_df = df.describe().reset_index(drop=False)
    summ_df = pd.concat([dtypes_df, desc_df, tot_df], axis=0, ignore_index=True)
    summ_info_cols = ['index', 'iso_year_week', 'date']
    summ_df = summ_df[summ_info_cols + [i for i in summ_df.columns if i not in summ_info_cols]]
    summ_df = summ_df.set_index('index').transpose()
    summ_df.index.name = 'feature'
    summ_df = summ_df.reset_index(drop=False)
    return summ_df

def mel_to_spend_feature_df(mel_df, platform_cpm_df, remove_prefix='exclude', daily_or_weekly='weekly'):
    
    # show details on channel_hierarchy for all including exclude and any missing groups
    mel_df['group'] = mel_df['group'] + '_mel_spend'
    
    roll_group_df = (mel_df.groupby(ch_cols + ['group'], as_index=False, dropna=False).agg({'mel_spend': 'sum', 'mel_agency_fees': 'sum'})
     .sort_values(by='mel_spend', ascending=False)
     .reset_index(drop=True))
    
    # remove groups that start with remove_prefix
    mel_fil_df = mel_df[~mel_df['group'].str.startswith(remove_prefix)].reset_index(drop=True)
    
    # percent of spend total by group
    total_mel_spend = mel_fil_df['mel_spend'].sum()
    total_mel_fee = mel_fil_df['mel_agency_fees'].sum()
    grp_per_tot_df = mel_fil_df.groupby('group', as_index=False, dropna=False).agg({'mel_spend': 'sum', 'mel_agency_fees': 'sum'})
    grp_per_tot_df['mel_spend_percent'] = grp_per_tot_df['mel_spend'] / total_mel_spend
    grp_per_tot_df['mel_fee_percent'] = grp_per_tot_df['mel_agency_fees'] / total_mel_fee
    grp_per_tot_df = grp_per_tot_df.sort_values(by='mel_spend_percent', ascending=False)
    # add percent of total spend flag
    grp_per_tot_df['mel_spend_percent_flag'] = 0
    grp_per_tot_df.loc[grp_per_tot_df['mel_spend_percent'] < 0.01, 'mel_spend_percent_flag'] = 1
    
    # percent of spend total by group by week
    mel_fil_df
    grp_week_per_tot_df = mel_fil_df.pivot_table(index='iso_year_week', columns='group', values='mel_spend', aggfunc='sum', fill_value=0)
    grp_week_per_tot_df = grp_week_per_tot_df.div(grp_week_per_tot_df.sum(axis=1), axis=0)
    grp_week_per_tot_df = grp_week_per_tot_df.reset_index(drop=False)
    
    
    # agg
    if daily_or_weekly == 'daily':
        # aggreagte to weekly if weekly and keep daily if daily
        grp_spend_df = (mel_fil_df.pivot_table(index=['iso_year_week', 'date'], columns='group', values='mel_spend', aggfunc='sum', fill_value=0)
                        .reset_index(drop=False))
        grp_fee_df = (mel_fil_df.pivot_table(index=['iso_year_week', 'date'], columns='group', values='mel_agency_fees', aggfunc='sum', fill_value=0)
                        .reset_index(drop=False))
    elif daily_or_weekly == 'weekly':
        date_cols = ['iso_year_week', 'date']
        min_date_df = mel_fil_df.groupby('iso_year_week', as_index=False, dropna=False).agg({'date': 'min'})
        min_date_df
        grp_spend_df = (mel_fil_df.pivot_table(index=['iso_year_week'], columns='group', values='mel_spend', aggfunc='sum', fill_value=0)
                        .reset_index(drop=False))
        grp_spend_df = pd.merge(min_date_df, grp_spend_df, how='left', on='iso_year_week')
        grp_spend_df = grp_spend_df[date_cols + [col for col in grp_spend_df.columns if col not in date_cols]]
        
        grp_fee_df = (mel_fil_df.pivot_table(index=['iso_year_week'], columns='group', values='mel_agency_fees', aggfunc='sum', fill_value=0)
                     .reset_index(drop=False))
        grp_fee_df = pd.merge(min_date_df, grp_fee_df, how='left', on='iso_year_week')
        grp_fee_df = grp_fee_df[date_cols + [col for col in grp_fee_df.columns if col not in date_cols]]
    
    if platform_cpm_df is not None:
        cpm_cols = [i for i in platform_cpm_df.columns if i not in ['iso_year_week', 'date']]
        grp_spend_df = pd.merge(grp_spend_df, platform_cpm_df, how='left', on=['iso_year_week', 'date'])
        for cpm_col in cpm_cols:
            spend_col = cpm_col.replace('_cpm', '_mel_spend')
            impression_col = cpm_col.replace('_cpm', '_mel_plat_impres')
            grp_spend_df[impression_col] = grp_spend_df[spend_col] * (grp_spend_df[cpm_col] * 1000)
            grp_spend_df = grp_spend_df.drop(columns=cpm_col)
    
    # features
    spend_feature_col = [i for i in grp_spend_df.columns if i not in ['iso_year_week', 'date']]
    spend_feature_col_df = pd.DataFrame(spend_feature_col, columns=['feature'])
    spend_feature_col_df['type'] = 'paid_media_spend'
    spend_feature_col_df.loc[spend_feature_col_df['feature'].str.contains('plat_impres'), 'type'] = 'paid_media_impression'
    spend_feature_col_df['fillna'] = 0
    
    return roll_group_df, grp_per_tot_df, grp_week_per_tot_df, grp_spend_df, grp_fee_df, spend_feature_col_df

def get_platform_cpms(platform_cpm_to_impression_tables, bob_entity_code='US', min_max_data_iso_year_week=('2021-W40', '2024-W04'),
                      daily_or_weekly='weekly'):
    
    show_platform_tables_sql = "show tables from marketing_analytics_us like '*daily_view'"
    platform_tables_df = spark.sql(show_platform_tables_sql).toPandas()
    platform_tables_df.columns = ['database', 'table_name', 'is_temporary']
    print(platform_tables_df)
    potential_tables = list(platform_tables_df['table_name'].values)

    # list from github https://github.com/hellofresh/mau-dbt-snowflake/tree/master/dbt_project/models/media_platform
    # bing, dv360, facebook, google, horizon (digital tv ott and tv linear), reddit, rokt, veritone
    platform_cpm_to_impression_tables = [i for i in platform_cpm_to_impression_tables if i in potential_tables]
    
    plat_dfs = []
    for plat_table_name in platform_cpm_to_impression_tables:
    
        # plat_table_name = 'facebook_ads_platform_metrics_daily_view'
        
        plat_desc_query = f'describe marketing_analytics_us.{plat_table_name}'
        plat_desc_df = spark.sql(plat_desc_query).toPandas()
        plat_desc_df
        
        plat_ch_cols = [i for i in plat_desc_df['col_name'].values if i in vs_ch_cols + ch_cols]
        # make plat_ch_cols in the same order as vs_ch_cols
        ch_order = {value: index for index, value in enumerate(vs_ch_cols + ch_cols)}
        plat_ch_cols = sorted(plat_ch_cols, key=ch_order.get)
        plat_ch_col_sql = ', '.join([f'lower(trim(plat.{col})) {col.replace("vs_", "")}' for col in plat_ch_cols]) + ', '
        
        min_data_iso_year_week, max_data_iso_year_week = min_max_data_iso_year_week
        week_filter_stmt = f"and dd.iso_year_week >= '{min_data_iso_year_week}' and dd.iso_year_week <= '{max_data_iso_year_week}'"
        
        if daily_or_weekly == 'daily':
            date_stmt = 'dd.iso_year_week, dd.date_string_backwards date, '
            gb_stmt = ', '.join([str(i) for i in range(1, len(plat_ch_cols)+3)])
            gb_stmt
        elif daily_or_weekly == 'weekly':
            date_stmt = 'dd.iso_year_week, min(dd.date_string_backwards) date,'
            gb_stmt = ', '.join([str(i) for i in range(1, len(plat_ch_cols)+2)])
        
        # platform spend
        plat_query = f'''
        select
            {date_stmt}
            {plat_ch_col_sql}
            sum(impressions) plat_impressions,
            sum(total_spend) plat_total_spend, 
            sum(total_spend) / sum(impressions) plat_cpm
        from marketing_analytics_us.{plat_table_name} plat
        join dimensions.date_dimension dd on plat.fk_date = dd.sk_date
        where bob_entity_code = '{bob_entity_code}'
            {week_filter_stmt} 
        group by {gb_stmt}
        '''
        print(plat_table_name, '\n', plat_query)
        plat_df = spark.sql(plat_query).toPandas()
        plat_df['date'] = pd.to_datetime(plat_df['date'])
        plat_df = plat_df.replace('', pd.NA)
        plat_df['channel_medium'] = 'online'
        plat_df['channel_category'] = 'performance'
        plat_dfs.append(plat_df)
    
    plat_df = pd.concat(plat_dfs, axis=0, ignore_index=True)
    plat_df.loc[plat_df['channel'] == 'sea brand', 'channel_category'] = 'brand'
    plat_df = map_low_chs_to_ch(ch_map_low_dfs=ch_map_low_dfs, mel_df=plat_df)
    plat_df['plat_total_spend'] = plat_df['plat_total_spend'].astype('float')
    plat_df['plat_impressions'] = plat_df['plat_impressions'].astype('float')
    plat_agg_df = (plat_df[plat_df['group'] != 'missing'].groupby(['iso_year_week', 'date', 'group'], as_index=False, dropna=False)
                   .agg({'plat_total_spend': 'sum', 'plat_impressions': 'sum'}))
    plat_agg_df['plat_cpm'] = plat_agg_df['plat_total_spend'] / (plat_agg_df['plat_impressions'] / 1000)
    plat_piv_df = pd.pivot_table(plat_agg_df, index=['iso_year_week', 'date'], columns='group', values='plat_cpm')
    # plat_piv_df = plat_piv_df[['app_paid', 'display', 'native', 'paid_social_meta', 'sea_brand', 'sea_non_brand']]
    plat_piv_df.columns = [f'{i}_cpm' for i in plat_piv_df.columns]
    # sort multi index by one column, date
    plat_piv_df = plat_piv_df.sort_index(level='date', ascending=True).reset_index(drop=False)
    plat_piv_df = plat_piv_df.fillna(method='ffill')
    
    return plat_piv_df

def create_meta_mmm_agg_data():
    
    data_path = Path('raw_data/')
    # get list of .csv files in data_path
    meta_dfs = []
    for fil in data_path.iterdir():
        if fil.name[:4] == 'meta' and fil.suffix == '.csv' and fil.name != 'meta_agg.csv':
            print(f'reading in {fil.name}')
            meta_dfs.append(pd.read_csv(fil))
    
    fil.name
            
    meta_df = pd.concat(meta_dfs, axis=0, ignore_index=True)
    meta_df
    meta_df['Account name'].unique()
    meta_df['Objective'].unique()
    meta_df['Impressions'] = meta_df['Impressions'].astype('float')
    meta_df['Amount spent'] = meta_df['Amount spent'].astype('float')
    meta_agg_df = meta_df.groupby(['Account name', 'Day', 'Platform', 'Media type', 'Objective'], as_index=False, dropna=False).agg({'Impressions': 'sum', 'Amount spent': 'sum'})
    # make all columns lowercase and replace spaces with underscores
    meta_agg_df.columns = [i.lower().replace(' ', '_') for i in meta_agg_df.columns]
    meta_agg_df = meta_agg_df.rename(columns={'account_name': 'bob_entity_code', 'day': 'date'})
    meta_agg_df['bob_entity_code'] = meta_agg_df['bob_entity_code'].map({'HelloFresh US': 'US'})
    meta_agg_df['date'] = pd.to_datetime(meta_agg_df['date'])
    meta_agg_df['platform'] = meta_agg_df['platform'].str.lower()
    meta_agg_df['media_type'] = meta_agg_df['media_type'].str.lower()
    meta_agg_df['objective'] = meta_agg_df['objective'].str.lower()
    meta_agg_df.to_csv(data_path / 'meta_agg.csv', index=False)
    
    meta_agg_df = pd.read_csv(data_path / 'meta_agg.csv')
    meta_agg_df['date'] = pd.to_datetime(meta_agg_df['date'])
    meta_agg_df['platform'].value_counts()
    meta_agg_df[['platform', 'objective']].value_counts()
    meta_agg_df[['platform', 'media_type']].value_counts()
    
    meta_agg_df
    
    meta_agg_df.groupby(['platform', 'objective'], dropna=False, as_index=False).agg({'impressions': 'sum', 'amount_spent': 'sum', 
                                                                                      'date': ['min', 'max']}).to_clipboard()
    
    meta_exp_agg_df = meta_agg_df.groupby(['platform', 'media_type', 'objective'], dropna=False, as_index=False).agg({'impressions': 'sum', 'amount_spent': 'sum', 
                                                                                      'date': ['min', 'max']})
    meta_exp_agg_df['cpm'] = meta_exp_agg_df['amount_spent']['sum'] / (meta_exp_agg_df['impressions']['sum'] / 1000)
    meta_exp_agg_df.to_clipboard(index=False)
    
    meta_exp_agg_df
    
    meta_agg_df.loc[meta_agg_df['platform'] != 'instagram', 'platform'] = 'facebook'
    meta_agg_df.loc[meta_agg_df['media_type'] == 'mixed', 'media_type'] = 'video'
    date_query = f'''
    select
        dd.date_string_backwards date, 
        dd.iso_year_week
    from global_bi_business.date_dimension dd
    where dd.date_string_backwards >= '2021-01-01' and dd.date_string_backwards <= '2024-03-01'
    '''
    date_df = spark.sql(date_query).toPandas()
    date_df['date'] = pd.to_datetime(date_df['date'])
    meta_agg_df = pd.merge(meta_agg_df, date_df, how='left', on='date')
    meta_agg_df['objective'].value_counts()
    (meta_df[meta_df['Objective'].str.lower().isin(('outcome awareness', 'reach', 'video views'))]
     .groupby('Campaign name', as_index=False, dropna=False)
     .agg({'Impressions': 'sum', 'Amount spent': 'sum'}).to_clipboard(index=False))
    
    meta_agg_df.groupby('objective').agg({'amount_spent': 'sum'}).to_clipboard()
    
    fb_insta_agg_df = (meta_agg_df[meta_agg_df['objective'] != 'outcome awareness'].groupby(['iso_year_week', 'date', 'platform', 'media_type'], as_index=False, dropna=False)
                       .agg({'impressions': 'sum', 'amount_spent': 'sum'}))
    fb_insta_agg_df = fb_insta_agg_df.rename(columns={'amount_spent': 'plat_mmm_spend'})
    fb_insta_agg_df['channel'] = fb_insta_agg_df['platform'] + '_' + fb_insta_agg_df['media_type']
    meta_spend_df = (pd.pivot_table(fb_insta_agg_df, index=['iso_year_week', 'date'], columns='channel', values='plat_mmm_spend', aggfunc='sum'))
    meta_spend_df.columns = [f'{i}_plat_spend' for i in meta_spend_df.columns]
    meta_spend_df = meta_spend_df.reset_index(drop=False)
    meta_spend_df
    
    meta_impressions_df = (pd.pivot_table(fb_insta_agg_df, index=['iso_year_week', 'date'], columns='channel', values='impressions', aggfunc='sum'))
    meta_impressions_df.columns = [f'{i}_plat_impres' for i in meta_impressions_df.columns]
    meta_impressions_df = meta_impressions_df.reset_index(drop=False)
    meta_impressions_df
    
    meta_fin_df = pd.merge(meta_spend_df, meta_impressions_df, how='left', on=['iso_year_week', 'date'])
    meta_fin_df.to_csv(data_path / 'meta_agg.csv', index=False)
    
    pass

def get_meta_mmm_agg_data():
    
    data_path = Path('raw_data/')
    meta_mmm_feature_df = pd.read_csv(data_path / 'meta_agg.csv')
    meta_mmm_feature_df['date'] = pd.to_datetime(meta_mmm_feature_df['date'])
    
    feature_cols = [i for i in meta_mmm_feature_df.columns if i not in ['iso_year_week', 'date']]
    meta_mmm_feature_col_df = pd.DataFrame(feature_cols, columns=['feature'])
    meta_mmm_feature_col_df['type'] = 'paid_media_spend'
    meta_mmm_feature_col_df['fillna'] = 0
    
    return meta_mmm_feature_df, meta_mmm_feature_col_df

def create_google_mmm_agg_data():
    
    # google ads https://drive.google.com/corp/drive/folders/1cG-xrv0ZJbKPG23Y92TfhfoOdTsimp9L?resourcekey=0-f7BouR_pfFH_F5Yun3Y4-w
    # dv360 https://drive.google.com/corp/drive/folders/1c25jUS235XrId0YdErPOCED0s0XTWHGt?resourcekey=0-PgCGkLoNbtbvfqbUNH6mOw
    
    base_folder_path = Path('raw_data/')
    
    # folders that start with drive-download
    drive_folders = [i for i in base_folder_path.iterdir() if i.is_dir() and i.name[:14] == 'drive-download']
    drive_folders
    
    
    
    pass

def get_google_mmm_agg_data():
    
    
    pass

def google_search_query_volume_features():
    
    example_query = f'''
    SELECT 
        gs.reportdate
        , ddd.iso_year_week
        , gs.querylabel
        , SUM(gs.indexedqueryvolume) AS google_search_volume_index
    FROM uploads.google_search_term_traffic gs
    LEFT JOIN dimensions.date_dimension ddd ON gs.reportdate = ddd.date_string_backwards
    WHERE querylabel = 'HelloFresh'
    GROUP BY 1,2,3
    ORDER BY 1,2,3;
    '''
    
    
    
    
    pass

def get_branded_vehicle_features(bob_entity_code='US', min_max_data_iso_year_week=('2021-W40', '2024-W04'), daily_or_weekly='weekly'):
    
    min_data_iso_year_week, max_data_iso_year_week = min_max_data_iso_year_week
    week_filter_stmt = f"dd.iso_year_week >= '{min_data_iso_year_week}' and dd.iso_year_week <= '{max_data_iso_year_week}'"
    
    # keywords to filter on
    gs_url = 'https://docs.google.com/spreadsheets/d/1bxiXZW4qVWbQYAi1BbqBfzRWmAMkaQ1wSUcwCxSnLd4'
    gs_client = GSheet(gs_url)
    
    bv_df = gs_client.read_dataframe('branded_vehicles', header_row_num=0)
    bv_df = bv_df[bv_df['bob_entity_code'] == bob_entity_code]
    bv_df = bv_df.rename(columns={'brand_wrap_complete_date': 'date'})
    bv_df['date'] = pd.to_datetime(bv_df['date'])
    bv_df['n_vehicles'] = bv_df['n_vehicles'].astype('int')
    bv_agg_df = bv_df.groupby('date', as_index=False, dropna=False).agg({'n_vehicles': 'sum'})
        
    date_query = f'''
    select 
        dd.iso_year_week, dd.date_string_backwards date
    from global_bi_business.date_dimension dd
    where {week_filter_stmt}
    '''
    date_df = spark.sql(date_query).toPandas()
    date_df['date'] = pd.to_datetime(date_df['date'])
    
    mer_df = pd.merge(date_df, bv_agg_df, how='outer', on='date')
    mer_df['n_vehicles'] = mer_df['n_vehicles'].fillna(0)
    mer_df['cumulative_branded_vehicles'] = mer_df['n_vehicles'].cumsum()
    mer_df = mer_df[['iso_year_week', 'date', 'cumulative_branded_vehicles']]
    
    if daily_or_weekly == 'weekly':
        mer_df = mer_df.groupby('iso_year_week', as_index=False, dropna=False).agg({'date': 'min', 'cumulative_branded_vehicles': 'max'})
    
    vehicle_feature_col = [i for i in mer_df.columns if i not in ['iso_year_week', 'date']]
    vehicle_feature_col_df = pd.DataFrame(vehicle_feature_col, columns=['feature'])
    vehicle_feature_col_df['type'] = 'organic'
    vehicle_feature_col_df['fillna'] = 0
    
    return mer_df, vehicle_feature_col_df

def get_google_seo_features(bob_entity_code='US', min_max_data_iso_year_week=('2021-W40', '2024-W04'), daily_or_weekly='weekly', 
                            organic_features=['seo_weighted_avg_position', 'seo_discount_added', 'add_compliance_language']):
    
    
    if 'seo_weighted_avg_position' in organic_features:
    
        min_data_iso_year_week, max_data_iso_year_week = min_max_data_iso_year_week
        if min_data_iso_year_week < '2021-W40':
            min_data_iso_year_week = '2021-W40'
        
        week_filter_stmt = f"and dd.iso_year_week >= '{min_data_iso_year_week}' and dd.iso_year_week <= '{max_data_iso_year_week}'"
        
        if daily_or_weekly == 'daily':
            date_stmt = 'dd.iso_year_week, dd.date_string_backwards date, '
            gb_stmt = '1,2'
        elif daily_or_weekly == 'weekly':
            date_stmt = 'dd.iso_year_week, min(dd.date_string_backwards) date,'
            gb_stmt = '1'
        
        # keywords to filter on
        gs_url = 'https://docs.google.com/spreadsheets/d/1bxiXZW4qVWbQYAi1BbqBfzRWmAMkaQ1wSUcwCxSnLd4'
        gs_client = GSheet(gs_url)
        
        seo_ahref_keywords_df = gs_client.read_dataframe('seo_ahref_keywords', header_row_num=0)
        seo_ahref_keywords_df = seo_ahref_keywords_df[seo_ahref_keywords_df['bob_entity_code'] == bob_entity_code]
        # filter on just hellofresh.com keywords
        # seo_ahref_keywords_df = seo_ahref_keywords_df[seo_ahref_keywords_df['url'] == 'https://www.hellofresh.com/'].reset_index(drop=True)
        seo_ahref_keywords_df['keyword'] = seo_ahref_keywords_df['keyword'].str.lower()
        seo_ahref_keywords_sdf = spark.createDataFrame(seo_ahref_keywords_df)
        seo_ahref_keywords_sdf.createOrReplaceTempView('seo_ahref_keywords')
        seo_ahref_keywords_df
            
        seo_query = f'''
        select 
            {date_stmt}
            sum(gs.position * gs.impressions) / sum(gs.impressions) seo_weighted_avg_position
        from marketing_analytics_us.google_search_console_daily gs
        join global_bi_business.date_dimension dd on gs.date = dd.date_string_backwards
        join seo_ahref_keywords ahref on gs.query = ahref.keyword and gs.entity_code = ahref.bob_entity_code
        where gs.entity_code = '{bob_entity_code}'
            {week_filter_stmt}
        group by {gb_stmt}
        '''
        seo_feature_df = spark.sql(seo_query).toPandas()
        seo_feature_df['date'] = pd.to_datetime(seo_feature_df['date'])
    
    else:
        # create seo_feature_df with just date and iso_year_week
        pass
        
    
    if 'seo_discount_added' in organic_features or 'add_compliance_language' in organic_features:
        website_change_df = gs_client.read_dataframe('website_changes', header_row_num=0)
        website_change_df = website_change_df[(website_change_df['bob_entity_code'] == bob_entity_code) & 
                                              (website_change_df['event_name'].isin(organic_features))]
        website_change_df['date'] = pd.to_datetime(website_change_df['date'])
        
        for idx, row in website_change_df.iterrows():
            wc_date = row['date']
            wc_event_name = row['event_name']
            seo_feature_df[wc_event_name] = 0
            seo_feature_df.loc[seo_feature_df['date'] >= wc_date, wc_event_name] = 1
    
    
    seo_feature_col = [i for i in seo_feature_df.columns if i not in ['iso_year_week', 'date']]
    seo_feature_col_df = (seo_feature_df.loc[seo_feature_df['date'] == seo_feature_df['date'].min(), seo_feature_col]
                          .reset_index(drop=True).transpose().reset_index(drop=False).rename(columns={'index': 'feature', 0: 'fillna'}))
    seo_feature_col_df['type'] = 'organic'
    seo_feature_col_df = seo_feature_col_df[['feature', 'type', 'fillna']]
    # seo_feature_col_df['fillna'] = 0
    
    
    return seo_feature_df, seo_feature_col_df

def get_discount_features(bob_entity_code='US', min_max_data_iso_year_week=('2021-W40', '2024-W04'), daily_or_weekly='weekly', 
                          conversion_type_inclusion=['activation', 'reactivation'], channel_category_exclusion=['referral'], 
                          channel_exclusion=['b2b revenue', 'b2b conversion', 'b2b']):
    
    if len(conversion_type_inclusion) > 0:
        conversion_type_str = "', '".join([i.lower() for i in conversion_type_inclusion])
        conversion_type_stmt = f"and lower(trim(mac.conversion_type)) in ('{conversion_type_str}')"
    else:
        conversion_type_stmt = ''

    if len(channel_category_exclusion) > 0:
        channel_category_str = "', '".join([i.lower() for i in channel_category_exclusion])
        channel_category_stmt = f"and lower(trim(mac.channel_category)) not in ('{channel_category_str}')"
    else:
        channel_category_stmt = ''

    if len(channel_exclusion) > 0:
        channel_str = "', '".join([i.lower() for i in channel_exclusion])
        channel_stmt = f"and lower(trim(mac.channel)) not in ('{channel_str}')"
    else:
        channel_stmt = ''

    min_data_iso_year_week, max_data_iso_year_week = min_max_data_iso_year_week
    week_filter_stmt = f"and dd.iso_year_week >= '{min_data_iso_year_week}' and dd.iso_year_week <= '{max_data_iso_year_week}'"
    
    if daily_or_weekly == 'daily':
        date_stmt = 'dd.iso_year_week, dd.date_string_backwards date, '
        gb_stmt = '1,2'
    elif daily_or_weekly == 'weekly':
        date_stmt = 'dd.iso_year_week, min(dd.date_string_backwards) date,'
        gb_stmt = '1'
    
    discount_query = f'''
    select
        {date_stmt}
        sum(mac.total_discount_value_local_currency - mac.shipping_discount_amount_local_currency) / count(distinct cd.customer_uuid) discount_per_customer, 
        count( distinct (case when mac.shipping_discount_amount_local_currency > 0 then cd.customer_uuid end)) / count(distinct cd.customer_uuid) per_free_shipping, 
        count( distinct (case when ffl.ffl_type is not null then cd.customer_uuid end)) / count(distinct cd.customer_uuid) per_ffl
    from marketing_data_product.marketing_attribution_conversions as mac
    left join global_bi_business.date_dimension dd on mac.fk_conversion_local_date = dd.date_id
    left join global_bi_business.customer_dimension cd on mac.customer_uuid = cd.customer_uuid
    left join marketing_analytics_us.ffl_conversions ffl on mac.bob_entity_code = ffl.country and mac.customer_id = ffl.customer_id and lower(mac.conversion_type) = lower(ffl.conversion_type)
    where mac.bob_entity_code = '{bob_entity_code}'
        and mac.shipping_address_postcode is not null
        and mac.customer_uuid is not null
        and mac.flag_cancelled_24h = false
        and cd.is_test = 0
        {conversion_type_stmt} {channel_category_stmt} {channel_stmt}
        {week_filter_stmt}
    group by {gb_stmt}
    '''
    print(discount_query)
    discount_df = spark.sql(discount_query).toPandas()
    print(discount_df.head())
    
    discount_df['date'] = pd.to_datetime(discount_df['date'])
    discount_df['discount_per_customer'] = discount_df['discount_per_customer'].astype('float')
    discount_df['per_free_shipping'] = discount_df['per_free_shipping'].astype('float')
    discount_df['per_ffl'] = discount_df['per_ffl'].astype('float')
    
    discount_feature_col = [i for i in discount_df.columns if i not in ['iso_year_week', 'date']]
    discount_feature_col_df = pd.DataFrame(discount_feature_col, columns=['feature'])
    discount_feature_col_df['type'] = 'context'
    discount_feature_col_df['fillna'] = 0
    
    return discount_df, discount_feature_col_df

def get_price_features(bob_entity_code='US', min_max_data_iso_year_week=('2021-W40', '2024-W04'), daily_or_weekly='weekly'):
    
    min_data_iso_year_week, max_data_iso_year_week = min_max_data_iso_year_week
    week_filter_stmt = f"and dd.iso_year_week >= '{min_data_iso_year_week}' and dd.iso_year_week <= '{max_data_iso_year_week}'"
    
    if daily_or_weekly == 'daily':
        date_stmt = 'dd.iso_year_week, dd.date_string_backwards date, '
        gb_stmt = '1,2'
    elif daily_or_weekly == 'weekly':
        date_stmt = 'dd.iso_year_week, min(dd.date_string_backwards) date,'
        gb_stmt = '1'
    
    meal_price_query = f'''
    select
        {date_stmt}
        sum(bs.full_retail_price_usd) / sum(bs.number_of_meals) price_per_meal
    from fact_tables.boxes_shipped bs
    join global_bi_business.date_dimension dd on bs.fk_delivery_date = dd.date_id
    where bs.country = '{bob_entity_code}'
        and bs.box_shipped = 1
        and bs.is_donation = false
        and bs.is_marketplace = false
        and bs.is_wineclub = false
        {week_filter_stmt}
    group by {gb_stmt}
    '''
    mp_df = spark.sql(meal_price_query).toPandas()
    mp_df['date'] = pd.to_datetime(mp_df['date'])
    
    price_feature_col = [i for i in mp_df.columns if i not in ['iso_year_week', 'date']]
    price_feature_col_df = pd.DataFrame(price_feature_col, columns=['feature'])
    price_feature_col_df['type'] = 'context'
    price_feature_col_df['fillna'] = 0
    
    return mp_df, price_feature_col_df

def get_conversion_options(bob_entity_code='US', min_max_data_iso_year_week=('2021-W20', '2024-W04'), conversion_cols=['conversion_type', 'channel_category', 'channel']):
    
    min_data_iso_year_week, max_data_iso_year_week = min_max_data_iso_year_week
    week_filter_stmt = f"and dd.iso_year_week >= '{min_data_iso_year_week}' and dd.iso_year_week <= '{max_data_iso_year_week}'"
    conversion_cols_stmt = ', '.join([f'lower(trim(mac.{col})) {col}' for col in conversion_cols])
    gb_stmt = ', '.join([str(i) for i in range(1, len(conversion_cols)+1)])
    
    conversion_query = f'''
    select
        {conversion_cols_stmt}, 
        count(distinct cd.customer_uuid) n_conversion
    from marketing_data_product.marketing_attribution_conversions as mac
    left join global_bi_business.date_dimension dd on mac.fk_conversion_local_date = dd.date_id
    left join global_bi_business.customer_dimension cd on mac.customer_uuid = cd.customer_uuid
    where mac.bob_entity_code = '{bob_entity_code}'
        and mac.shipping_address_postcode is not null
        and mac.customer_uuid is not null
        and mac.flag_cancelled_24h = false
        and cd.is_test = 0
        {week_filter_stmt}
    group by {gb_stmt}
    '''
    print(conversion_query)
    
    conversion_df = spark.sql(conversion_query).toPandas()
    print(conversion_df.head())

    return conversion_df

def get_conversion_data(bob_entity_code='US', min_max_data_iso_year_week=('2021-W40', '2024-W04'), daily_or_weekly='weekly', 
                          conversion_type_inclusion=['activation', 'reactivation'], channel_category_exclusion=['referral'], 
                          channel_exclusion=['b2b revenue', 'b2b conversion', 'b2b']):
    
    if len(conversion_type_inclusion) > 0:
        conversion_type_str = "', '".join([i.lower() for i in conversion_type_inclusion])
        conversion_type_stmt = f"and lower(trim(mac.conversion_type)) in ('{conversion_type_str}')"
    else:
        conversion_type_stmt = ''

    if len(channel_category_exclusion) > 0:
        channel_category_str = "', '".join([i.lower() for i in channel_category_exclusion])
        channel_category_stmt = f"and lower(trim(mac.channel_category)) not in ('{channel_category_str}')"
    else:
        channel_category_stmt = ''

    if len(channel_exclusion) > 0:
        channel_str = "', '".join([i.lower() for i in channel_exclusion])
        channel_stmt = f"and lower(trim(mac.channel)) not in ('{channel_str}')"
    else:
        channel_stmt = ''

    min_data_iso_year_week, max_data_iso_year_week = min_max_data_iso_year_week
    week_filter_stmt = f"and dd.iso_year_week >= '{min_data_iso_year_week}' and dd.iso_year_week <= '{max_data_iso_year_week}'"
    
    conversion_check_query = f'''
    select
        lower(trim(mac.conversion_type)) conversion_type, 
        lower(trim(mac.channel_category)) channel_category, 
        lower(trim(mac.channel)) channel,
        count(distinct cd.customer_uuid) n_conversion
    from marketing_data_product.marketing_attribution_conversions as mac
    left join global_bi_business.date_dimension dd on mac.fk_conversion_local_date = dd.date_id
    left join global_bi_business.customer_dimension cd on mac.customer_uuid = cd.customer_uuid
    where mac.bob_entity_code = '{bob_entity_code}'
        and mac.shipping_address_postcode is not null
        and mac.customer_uuid is not null
        and mac.flag_cancelled_24h = false
        and cd.is_test = 0
        {week_filter_stmt}
    group by 1,2,3
    '''
    print(conversion_check_query)
    conversion_check_df = spark.sql(conversion_check_query).toPandas()
    print(conversion_check_df.head())
    
    conversion_check_df['n_conversion'] = conversion_check_df['n_conversion'].astype('int')
    
    if daily_or_weekly == 'daily':
        date_stmt = 'dd.iso_year_week, dd.date_string_backwards date, '
        gb_stmt = '1,2'
    elif daily_or_weekly == 'weekly':
        date_stmt = 'dd.iso_year_week, min(dd.date_string_backwards) date,'
        gb_stmt = '1'
    
    conversion_query = f'''
    select
        {date_stmt}
        count(distinct cd.customer_uuid) n_conversion
    from marketing_data_product.marketing_attribution_conversions as mac
    left join global_bi_business.date_dimension dd on mac.fk_conversion_local_date = dd.date_id
    left join global_bi_business.customer_dimension cd on mac.customer_uuid = cd.customer_uuid
    where mac.bob_entity_code = '{bob_entity_code}'
        and mac.shipping_address_postcode is not null
        and mac.customer_uuid is not null
        and mac.flag_cancelled_24h = false
        and cd.is_test = 0
        {conversion_type_stmt} {channel_category_stmt} {channel_stmt}
        {week_filter_stmt}
    group by {gb_stmt}
    '''
    print(conversion_query)
    conversion_df = spark.sql(conversion_query).toPandas()
    print(conversion_df.head())
    
    conversion_df['date'] = pd.to_datetime(conversion_df['date'])
    conversion_df['n_conversion'] = conversion_df['n_conversion'].astype('int')

    return conversion_check_df, conversion_df

def get_channel_conversion_data(bob_entity_code='US', min_max_data_iso_year_week=('2021-W40', '2024-W04'), daily_or_weekly='weekly', 
                                channel_hierarchy_df=None, conversion_type_inclusion=['activation', 'reactivation'], 
                                channel_category_exclusion=['referral'],
                                channel_exclusion=['b2b revenue', 'b2b conversion', 'b2b']):
    
    
    # channel_hierarchy_df = full_ch_df
    # channel_hierarchy_df[channel_hierarchy_df['channel'] == 'full price']
    
    if len(conversion_type_inclusion) > 0:
        conversion_type_str = "', '".join([i.lower() for i in conversion_type_inclusion])
        conversion_type_stmt = f"and lower(trim(mac.conversion_type)) in ('{conversion_type_str}')"
    else:
        conversion_type_stmt = ''

    if len(channel_category_exclusion) > 0:
        channel_category_str = "', '".join([i.lower() for i in channel_category_exclusion])
        channel_category_stmt = f"and lower(trim(mac.channel_category)) not in ('{channel_category_str}')"
    else:
        channel_category_stmt = ''

    if len(channel_exclusion) > 0:
        channel_str = "', '".join([i.lower() for i in channel_exclusion])
        channel_stmt = f"and lower(trim(mac.channel)) not in ('{channel_str}')"
    else:
        channel_stmt = ''
        
        
    # clean channel_hierarchy_df
    channel_hierarchy_df.columns = [col.replace('vs_', '').lower() for col in channel_hierarchy_df.columns]
    ch_df_cols_lst = channel_hierarchy_df.columns
    df_ch_filter_lst = [i for i in ch_cols if i in ch_df_cols_lst]
    ch_df_cols = df_ch_filter_lst + ['group']
    channel_hierarchy_df = channel_hierarchy_df[ch_df_cols].drop_duplicates().replace('', pd.NA).reset_index(drop=True)
    channel_hierarchy_df = channel_hierarchy_df.apply(lambda x: x.str.strip().str.lower())
    ch_join_stmt = 'left join channel_hierarchy ch on ' + 'and '.join([f'lower(trim(mac.{ch})) <=> lower(trim(ch.{ch})) ' for ch in df_ch_filter_lst])
    ch_sdf = spark.createDataFrame(channel_hierarchy_df)
    ch_sdf.createOrReplaceTempView('channel_hierarchy')

    min_data_iso_year_week, max_data_iso_year_week = min_max_data_iso_year_week
    week_filter_stmt = f"and dd.iso_year_week >= '{min_data_iso_year_week}' and dd.iso_year_week <= '{max_data_iso_year_week}'"
    
    if daily_or_weekly == 'daily':
        select_stmt = "dd.iso_year_week, dd.date_string_backwards date, case when ch.group is null then 'missing' else ch.group end as group, "
        gb_stmt = '1,2,3'
        date_query = f'''
        select
            dd.iso_year_week, dd.date_string_backwards date
        from global_bi_business.date_dimension dd
        where dd.iso_year_week >= '{min_data_iso_year_week}' and dd.iso_year_week <= '{max_data_iso_year_week}'
        '''
    elif daily_or_weekly == 'weekly':
        select_stmt = "dd.iso_year_week, case when ch.group is null then 'missing' else ch.group end as group, "
        gb_stmt = '1,2'
        date_query = f'''
        select
            dd.iso_year_week, min(dd.date_string_backwards) date
        from global_bi_business.date_dimension dd
        where dd.iso_year_week >= '{min_data_iso_year_week}' and dd.iso_year_week <= '{max_data_iso_year_week}'
        group by 1
        '''
    
    date_df = spark.sql(date_query).toPandas()
    date_df['date'] = pd.to_datetime(date_df['date'])
    
    conversion_query = f'''
    select
        {select_stmt}
        count(distinct cd.customer_uuid) n_conversion
    from marketing_data_product.marketing_attribution_conversions as mac
    left join global_bi_business.date_dimension dd on mac.fk_conversion_local_date = dd.date_id
    left join global_bi_business.customer_dimension cd on mac.customer_uuid = cd.customer_uuid
    {ch_join_stmt}
    where mac.bob_entity_code = '{bob_entity_code}'
        and mac.shipping_address_postcode is not null
        and mac.customer_uuid is not null
        and mac.flag_cancelled_24h = false
        and cd.is_test = 0
        {conversion_type_stmt} {channel_category_stmt} {channel_stmt}
        {week_filter_stmt}
    group by {gb_stmt}
    '''
    print(conversion_query)
    conversion_df = spark.sql(conversion_query).toPandas()
    print(conversion_df.head())
    conversion_df['n_conversion'] = conversion_df['n_conversion'].astype('int')
    
    if daily_or_weekly == 'daily':
        conversion_df['date'] = pd.to_datetime(conversion_df['date'])
        mer_df = pd.merge(date_df, conversion_df, how='outer', on=['iso_year_week', 'date']).fillna(0)
    elif daily_or_weekly == 'weekly':
        mer_df = pd.merge(date_df, conversion_df, how='outer', on=['iso_year_week']).fillna(0)


    # why is there missing conversions?
    # select_stmt
    # valid_query = f'''
    # select
    #     lower(trim(mac.conversion_type)) conversion_type, 
    #     lower(trim(mac.campaign_type)) campaign_type, 
    #     lower(trim(mac.channel_medium)) channel_medium, 
    #     lower(trim(mac.channel_category)) channel_category, 
    #     lower(trim(mac.channel)) channel, 
    #     lower(trim(mac.channel_split)) channel_split, 
    #     lower(trim(mac.partner)) partner,
    #     case when ch.group is null then 'missing' else ch.group end as group, 
    #     count(distinct cd.customer_uuid) n_conversion
    # from marketing_data_product.marketing_attribution_conversions as mac
    # left join global_bi_business.date_dimension dd on mac.fk_conversion_local_date = dd.date_id
    # left join global_bi_business.customer_dimension cd on mac.customer_uuid = cd.customer_uuid
    # {ch_join_stmt}
    # where mac.bob_entity_code = '{bob_entity_code}'
    #     and mac.shipping_address_postcode is not null
    #     and mac.customer_uuid is not null
    #     and mac.flag_cancelled_24h = false
    #     and cd.is_test = 0
    #     {conversion_type_stmt} {channel_category_stmt} {channel_stmt}
    #     {week_filter_stmt}
    # group by 1,2,3,4,5,6,7,8
    # '''
    # print(valid_query)
    # val_df = spark.sql(valid_query).toPandas()
    # print(val_df.head())
    
    # val_df[val_df['group'] == 'missing']
    # conversion_df[conversion_df['group'] == 'missing']
    # (mer_df['group'] == 'missing').sum()
    # val_df[(val_df['channel'] == 'full price') & (val_df['partner'] == 'unmapped')]


    return mer_df

def channel_conversion_transform(conversion_df, remove_prefix='exclude'):
    
    # remove groups that start with remove_prefix
    conversion_fil_df = conversion_df[~conversion_df['group'].str.startswith(remove_prefix)].reset_index(drop=True)
    conversion_fil_df['group'] = conversion_fil_df['group'] + '_mac_conversion'
    
    # percent of spend total by group
    total_n_conversion = conversion_fil_df['n_conversion'].sum()
    grp_per_tot_df = conversion_fil_df.groupby('group', as_index=False, dropna=False).agg({'n_conversion': 'sum'})
    grp_per_tot_df['n_conversion_percent'] = grp_per_tot_df['n_conversion'] / total_n_conversion
    grp_per_tot_df = grp_per_tot_df.sort_values(by='n_conversion_percent', ascending=False)
    
    # percent of spend total by group by week
    grp_week_per_tot_df = conversion_fil_df.pivot_table(index='iso_year_week', columns='group', values='n_conversion', aggfunc='sum', fill_value=0)
    grp_week_per_tot_df = grp_week_per_tot_df.div(grp_week_per_tot_df.sum(axis=1), axis=0)
    grp_week_per_tot_df = grp_week_per_tot_df.reset_index(drop=False)
    # grp_week_per_tot_df.drop(columns='iso_year_week').sum(axis=1)
    
    
    grp_conversion_df = conversion_fil_df.pivot_table(index=['iso_year_week', 'date'], columns='group', values='n_conversion', aggfunc='sum', fill_value=0).reset_index(drop=False)
    
    return grp_per_tot_df, grp_week_per_tot_df, grp_conversion_df

def create_feature_target_summary(param_dict, feature_col_df, model_df, spend_fee_df):
    
    # spend summary stats
    model_summ_df = summ_stats(model_df)
    spend_fee_summ_df = summ_stats(spend_fee_df)
    
    # add to parameters the number of rows and columns
    n_rows, n_columns = model_df[feature_col_df['feature']].shape
    param_dict['feature_shape'] = (n_rows, n_columns)
    param_dict['feature_rows_over_columns'] = n_rows / n_columns
    
    
    # add correlation by year, by quarter, and total for both target and features
    
    # correlation with target
    target_corr_dfs = []
    model_cop_df = model_df.copy(deep=True)
    model_cop_df['year'] = model_cop_df['date'].dt.year
    model_cop_df['year_quarter'] = model_cop_df['date'].dt.to_period('Q')
    
    
    years = np.sort(model_cop_df['year'].unique())
    for year in years:
        target_corr_df = (model_cop_df[model_cop_df['year'] == year].drop(columns=['iso_year_week', 'date', 'year', 'year_quarter']).corr()
                          .abs().sort_values(by='n_conversion', ascending=False)['n_conversion'].reset_index(drop=False)
        .rename(columns={'index': 'feature', 'n_conversion': 'corr_with_n_conversion'}))
        target_corr_df = target_corr_df[target_corr_df['feature'] != 'n_conversion'].reset_index(drop=True)
        target_corr_df['time_period'] = 'year'
        target_corr_df['time'] = year
        target_corr_df = target_corr_df[['time_period', 'time', 'feature', 'corr_with_n_conversion']]
        target_corr_dfs.append(target_corr_df)
    
    quarters = np.sort(model_cop_df['year_quarter'].unique())
    for quarter in quarters:
        target_corr_df = (model_cop_df[model_cop_df['year_quarter'] == quarter].drop(columns=['iso_year_week', 'date', 'year', 'year_quarter']).corr()
                          .abs().sort_values(by='n_conversion', ascending=False)['n_conversion'].reset_index(drop=False)
        .rename(columns={'index': 'feature', 'n_conversion': 'corr_with_n_conversion'}))
        target_corr_df = target_corr_df[target_corr_df['feature'] != 'n_conversion'].reset_index(drop=True)
        target_corr_df['time_period'] = 'quarter'
        target_corr_df['time'] = quarter
        target_corr_df = target_corr_df[['time_period', 'time', 'feature', 'corr_with_n_conversion']]
        target_corr_dfs.append(target_corr_df)
    
    target_corr_df = (model_cop_df.drop(columns=['iso_year_week', 'date', 'year', 'year_quarter']).corr().abs().sort_values(by='n_conversion', ascending=False)['n_conversion'].reset_index(drop=False)
    .rename(columns={'index': 'feature', 'n_conversion': 'corr_with_n_conversion'}))
    target_corr_df = target_corr_df[target_corr_df['feature'] != 'n_conversion'].reset_index(drop=True)
    target_corr_df['time_period'] = 'total'
    target_corr_df['time'] = 'total'
    target_corr_df = target_corr_df[['time_period', 'time', 'feature', 'corr_with_n_conversion']]
    target_corr_dfs.append(target_corr_df)
    
    target_corr_df = pd.concat(target_corr_dfs, axis=0, ignore_index=True)
    
    # pairwise correlation of features
    corr_pair_dfs = []
    for year in years:
        corr_df = model_cop_df[model_cop_df['year'] == year].drop(columns=['iso_year_week', 'date', 'year', 'year_quarter', 'n_conversion']).corr()
        corr_pair_df = (corr_df.abs()
                        .where(np.triu(np.ones(corr_df.shape), k=1).astype(bool))
                        .stack()
                        .sort_values(ascending=False))
        corr_pair_df.index.names = ['group_1', 'group_2']
        corr_pair_df = corr_pair_df.reset_index().rename(columns={0: 'abs_corr'})
        corr_pair_df['time_period'] = 'year'
        corr_pair_df['time'] = year
        corr_pair_df = corr_pair_df[['time_period', 'time', 'group_1', 'group_2', 'abs_corr']]
        corr_pair_dfs.append(corr_pair_df)
    
    for quarter in quarters:
        corr_df = model_cop_df[model_cop_df['year_quarter'] == quarter].drop(columns=['iso_year_week', 'date', 'year', 'year_quarter', 'n_conversion']).corr()
        corr_pair_df = (corr_df.abs()
                        .where(np.triu(np.ones(corr_df.shape), k=1).astype(bool))
                        .stack()
                        .sort_values(ascending=False))
        corr_pair_df.index.names = ['group_1', 'group_2']
        corr_pair_df = corr_pair_df.reset_index().rename(columns={0: 'abs_corr'})
        corr_pair_df['time_period'] = 'quarter'
        corr_pair_df['time'] = quarter
        corr_pair_df = corr_pair_df[['time_period', 'time', 'group_1', 'group_2', 'abs_corr']]
        corr_pair_dfs.append(corr_pair_df)
        
    corr_df = model_cop_df.drop(columns=['iso_year_week', 'date', 'year', 'year_quarter', 'n_conversion']).corr()
    corr_pair_df = (corr_df.abs()
                    .where(np.triu(np.ones(corr_df.shape), k=1).astype(bool))
                    .stack()
                    .sort_values(ascending=False))
    corr_pair_df.index.names = ['group_1', 'group_2']
    corr_pair_df = corr_pair_df.reset_index().rename(columns={0: 'abs_corr'})
    corr_pair_df['time_period'] = 'total'
    corr_pair_df['time'] = 'total'
    corr_pair_df = corr_pair_df[['time_period', 'time', 'group_1', 'group_2', 'abs_corr']]
    corr_pair_dfs.append(corr_pair_df)
    
    corr_pair_df = pd.concat(corr_pair_dfs, axis=0, ignore_index=True)
    
    
    corr_sum = corr_pair_df[corr_pair_df['time_period'] == 'total']['abs_corr'].sum()
    param_dict['abs_corr_sum'] = corr_sum
    # add correlation flag
    corr_pair_df['abs_corr_flag'] = 0
    corr_pair_df.loc[corr_pair_df['abs_corr'] > 0.80, 'abs_corr_flag'] = 1
    
    # mutual information score
    # mi_scores = []
    # for idx, row in corr_pair_df.iterrows():
    #     feat_1 = row['group_1']
    #     feat_2 = row['group_2']    
    #     c_xy = np.histogram2d(model_df[feat_1], model_df[feat_2], 10)[0]
    #     mi_scores.append((feat_1, feat_2, mutual_info_score(None, None, contingency=c_xy)))
    
    # corr_pair_df = pd.merge(corr_pair_df, pd.DataFrame(mi_scores, columns=['group_1', 'group_2', 'mi_score']), 
    #                         how='left', on=['group_1', 'group_2'])
    
    # condition number
    aaa = corr_df.to_numpy()
    www, vvv = np.linalg.eig(aaa)
    condition_number = max(www) / min(www)
    param_dict['condition_number'] = condition_number
    
    # variance inflation factor
    vif_df = pd.DataFrame([(feature, variance_inflation_factor(model_df[feature_col_df['feature'].values].values, i)) for i, feature in enumerate(feature_col_df['feature'].values)],
                          columns=['feature', 'vif'])
    # add vif flag
    vif_df['vif_flag'] = 0
    vif_df.loc[vif_df['vif'] > 7, 'vif_flag'] = 1
    
    model_summ_df = pd.merge(model_summ_df, vif_df, how='left', on='feature')
    
    # variances
    # scale with divide by mean
    scaled_feature_df = model_df[feature_col_df['feature'].values]
    scaled_feature_df = scaled_feature_df.div(model_df[feature_col_df['feature'].values].mean(), axis=1)
    variance_df = (scaled_feature_df.var(axis=0, ddof=0).reset_index(drop=False)
                   .rename(columns={'index': 'feature', 0: 'scaled_variance'}))
    low_variance_threshold = 1.0e-3
    high_variance_threshold = 3.0
    variance_df['scaled_variance_flag'] = 0
    variance_df.loc[variance_df['scaled_variance'] < low_variance_threshold, 'scaled_variance_flag'] = 1
    variance_df.loc[variance_df['scaled_variance'] > high_variance_threshold, 'scaled_variance_flag'] = 1
    model_summ_df = pd.merge(model_summ_df, variance_df, how='left', on='feature')
    
    return param_dict, model_summ_df, spend_fee_summ_df, corr_pair_df, target_corr_df

def add_validation_time(model_df):
    # train range : test range
    # q4 evaluation jan 2020 (2020-W01) to nov 2023 (2023-W48) : dec 23 (2023-W49) to jan 24 (2024-W05)
    # q3 evaluation jan 2020 (2020-W01) to aug 2023 (2023-W35) : sep 23 (2023-W36) to oct 23 (2023-W44)
    # q2 evaluation jan 2020 (2020-W01) to may 2023 (2023-W22) : jun 23 (2023-W23) to jul 23 (2023-W31)
    # q1 evaluation jan 2020 (2020-W01) to feb 2023 (2023-W09) : mar 23 (2023-W10) to apr 23 (2023-W18)
    
    # train_test_df = pd.DataFrame([['4', '2020-W01', '2023-W48', '2023-W49', '2024-W05'], 
    # ['3', '2020-W01', '2023-W35', '2023-W36', '2023-W44'], 
    # ['2', '2020-W01', '2023-W22', '2023-W23', '2023-W31'], 
    # ['1', '2020-W01', '2023-W09', '2023-W10', '2023-W18']], columns=['quarter', 'train_start', 'train_end', 'test_start', 'test_end'])
    
    gs_url = 'https://docs.google.com/spreadsheets/d/1bxiXZW4qVWbQYAi1BbqBfzRWmAMkaQ1wSUcwCxSnLd4'
    gs_client = GSheet(gs_url)
    
    train_test_df = gs_client.read_dataframe('validation_time', header_row_num=0)
    
    for idx, row in train_test_df.iterrows():
        quarter = row['quarter']
        train_start = row['train_start']
        train_end = row['train_end']
        test_start = row['test_start']
        test_end = row['test_end']
        col = f'out_of_sample_q{quarter}'
        model_df[col] = None
        model_df.loc[model_df['iso_year_week'] <= train_end, col] = 'train'
        model_df.loc[(model_df['iso_year_week'] >= test_start) & 
                    (model_df['iso_year_week'] <= test_end), col] = 'test'
        model_df[col] = model_df[col].fillna('none')
    
    
    return model_df

def check_conversions():
    
    # conversion_type_inclusion_sum = conversion_check_df.loc[(conversion_check_df['conversion_type'].isin(conversion_type_inclusion)),'n_conversion'].sum()
    # channel_category_exclusion_sum = conversion_check_df.loc[(conversion_check_df['channel_category'].isin(channel_category_exclusion)),'n_conversion'].sum()
    # channel_exclusion_sum = conversion_check_df.loc[(conversion_check_df['channel'].isin(channel_exclusion)),'n_conversion'].sum()
    # conversion_check_df[conversion_check_df['channel'].isin(channel_exclusion)]
    
    # conversion_check_df.loc[conversion_check_df['conversion_type'].isin(conversion_type_inclusion), 'n_conversion'].sum()
    # conversion_check_df.loc[~conversion_check_df['conversion_type'].isin(conversion_type_inclusion), 'n_conversion'].sum()
    
    # conversion_type_inclusion_df = conversion_check_df.loc[(conversion_check_df['conversion_type'].isin(conversion_type_inclusion)), 
    #                         'n_conversion'].sum()
    
    # conversion_type_exclusion_df = (conversion_check_df.loc[(~conversion_check_df['conversion_type'].isin(conversion_type_inclusion))]
    #  .groupby(['conversion_type'], as_index=False, dropna=False).agg({'n_conversion': 'sum'}))
    
    # channel_exclusion_df = (conversion_check_df.loc[
    #     (conversion_check_df['conversion_type'].isin(conversion_type_inclusion)) &
    #     ((conversion_check_df['channel_category'].isin(channel_category_exclusion)) | 
    #     (conversion_check_df['channel'].isin(channel_exclusion)))]
    #  .groupby(['conversion_type', 'channel_category', 'channel'], as_index=False, dropna=False).agg({'n_conversion': 'sum'}))
    
    # conversion_type_inclusion_df - channel_exclusion_df['n_conversion'].sum()
    # 9885190-2871200
    
    # conversion_check_sum
    # conversion_time_target_sum
    # conversion_time_target_sum - conversion_check_sum
    
    # print(conversion_check_sum, conversion_time_target_sum)
    # print(conversion_time_target_sum - conversion_check_sum)
    
    
    pass

def z_test_prop_two_samp(control_conv, control_impress, test_conv, test_impress, one_sided=True, ci_percent=0.90):
    
    # z_score, p_value = sm.stats.proportions_ztest([n_mail_response, n_control_response], [n_mail, n_control], alternative='larger')
        
    alpha = (1 - ci_percent) * (1 / (2 - int(one_sided)))
    
    control_p = control_conv / control_impress
    test_p = test_conv / test_impress
    
    pooled_p = (control_conv + test_conv) / (control_impress + test_impress)
    variance_diff = pooled_p * (1 - pooled_p) * (1 / control_impress + 1 / test_impress)
    std_diff = np.sqrt(variance_diff)
    
    z_score = (test_p - control_p) / std_diff
    p_value = scipy.stats.norm.sf(abs(z_score)) * (2 - int(one_sided))
    
    z_critical = scipy.stats.norm.ppf(1 - alpha)
    lift_p = test_p - control_p
    low_p = lift_p - z_critical * std_diff
    high_p = lift_p + z_critical * std_diff
    
    return variance_diff, std_diff, z_score, p_value, control_p, test_p, low_p, lift_p, high_p

def dm_validation_data(bob_entity_code='US'):
    
    # bob_entity_code = 'US'
    
    dm_gsheet_url = 'https://docs.google.com/spreadsheets/d/152KiSkjR24up9tOJc58myHNnZ47e6Rssa6i2tAOXurM'
    dm_gs_client = GSheet(dm_gsheet_url)
    
    mb_df = dm_gs_client.read_dataframe(worksheet_name='speedeon_dm_matchback_raw_data', header_row_num=0)
    in_home_date_df = dm_gs_client.read_dataframe(worksheet_name='dm_in_home_date_raw_data', header_row_num=0)
    
    mb_df = mb_df[mb_df['bob_entity_code'] == bob_entity_code].reset_index(drop=True)
    mb_df['campaign_month'] = pd.to_datetime(mb_df['campaign_month'])
    
    float_cols = ['n_mail_response', 'n_control_response', 'n_mail', 'n_control', 'spend']
    for float_col in float_cols:
        # replace $, commas, and cast to float
        mb_df[float_col] = mb_df[float_col].str.replace('$', '').str.replace(',', '')
        mb_df.loc[mb_df[float_col] == '', float_col] = 0
        mb_df[float_col] = mb_df[float_col].astype('float')
        
    mb_camp_df = (mb_df.groupby('campaign_month', as_index=False, dropna=False)
     .agg({'n_mail_response': 'sum', 'n_control_response': 'sum', 'n_mail': 'sum', 'n_control': 'sum', 'spend': 'sum'}))
    
    mb_camp_df[['rr_variance', 'rr_standard_error', 'z_score', 'p_value', 'rr_control', 'rr_mail', 'rr_lift_low', 'rr_lift', 'rr_lift_high']] = mb_camp_df.apply(lambda x: z_test_prop_two_samp(x['n_control_response'], x['n_control'], x['n_mail_response'], x['n_mail']), axis=1, result_type='expand')
    
    mb_camp_df['conversion_lift_low'] = mb_camp_df['rr_lift_low'] * mb_camp_df['n_mail']
    mb_camp_df['conversion_lift'] = mb_camp_df['rr_lift'] * mb_camp_df['n_mail']
    mb_camp_df['conversion_lift_high'] = mb_camp_df['rr_lift_high'] * mb_camp_df['n_mail']
    
    mb_camp_df['icac_low'] = mb_camp_df['spend'] / mb_camp_df['conversion_lift_high']
    mb_camp_df['icac'] = mb_camp_df['spend'] / mb_camp_df['conversion_lift']
    mb_camp_df['icac_high'] = mb_camp_df['spend'] / mb_camp_df['conversion_lift_low']
    
    in_home_date_df['campaign_month'] = pd.to_datetime(in_home_date_df['campaign_month'])
    in_home_date_df['in_home_date'] = pd.to_datetime(in_home_date_df['in_home_date'])
    in_home_date_df['volume'] = in_home_date_df['volume'].str.replace(',', '').astype('int')
    
    in_home_date_df = in_home_date_df[(in_home_date_df['bob_entity_code'] == 'US') & 
                    (in_home_date_df['campaign_type'] == 'acquisition') &
                    (in_home_date_df['channel_split'] == 'bulk solo') & 
                    (in_home_date_df['new_mover'] != '1')].reset_index(drop=True)
    
    total_volume_df = in_home_date_df.groupby(['bob_entity_code'] + ch_cols + ['campaign_month'], as_index=False, dropna=False).agg({'volume': 'sum'}).rename(columns={'volume': 'total_volume'})
    in_home_date_df = pd.merge(in_home_date_df, total_volume_df, how='left', on=['bob_entity_code'] + ch_cols + ['campaign_month'])
    in_home_date_df['volume_proportion'] = in_home_date_df['volume'] / in_home_date_df['total_volume']
    
    valid_df = pd.merge(in_home_date_df[['bob_entity_code'] + ch_cols + ['campaign_month', 'in_home_date', 'volume_proportion']], mb_camp_df, 
                        how='left', on='campaign_month')
    
    valid_df = valid_df.drop(columns='campaign_month')
    
    
    for col in ['n_mail_response', 'n_control_response', 'n_mail', 'n_control', 'spend', 'conversion_lift_low', 'conversion_lift', 'conversion_lift_high']:
        valid_df[col] = valid_df[col] * valid_df['volume_proportion']
    
    valid_df = valid_df[valid_df['n_mail'].notna()].drop(columns='volume_proportion').reset_index(drop=True)
    valid_df = valid_df.rename(columns={'in_home_date': 'start_date'})
    valid_df['end_date'] = valid_df['start_date'] + pd.DateOffset(days=45)
    
    min_date = valid_df['start_date'].min().strftime('%Y-%m-%d')
    max_date = valid_df['end_date'].max().strftime('%Y-%m-%d')
    
    date_sql = f'''
    select
        dd.date_string_backwards date,
        dd.iso_year_week
    from dimensions.date_dimension dd
    where dd.date_string_backwards >= '{min_date}' and dd.date_string_backwards <= '{max_date}'
    '''
    date_df = spark.sql(date_sql).toPandas()
    date_df['date'] = pd.to_datetime(date_df['date'])
    
    valid_df = (pd.merge(valid_df, date_df, how='left', left_on='start_date', right_on='date')
                .rename(columns={'iso_year_week': 'start_iso_year_week'}).drop(columns='date'))
    valid_df = (pd.merge(valid_df, date_df, how='left', left_on='end_date', right_on='date')
                .rename(columns={'iso_year_week': 'end_iso_year_week'}).drop(columns='date'))
    valid_df
    
    ch_gs_url = 'https://docs.google.com/spreadsheets/d/1bxiXZW4qVWbQYAi1BbqBfzRWmAMkaQ1wSUcwCxSnLd4'
    ch_gs_tab_name = 'us_ed_channel_hierarchy'
    ch_map_df, ch_map_low_dfs  = read_in_channel_hierarchy_mapping(gsheet_url=ch_gs_url, gsheet_tab_name=ch_gs_tab_name)
    valid_df = map_low_chs_to_ch(ch_map_low_dfs=ch_map_low_dfs, mel_df=valid_df)
    valid_df
    
    info_cols = ['bob_entity_code'] + ch_cols + ['group', 'start_date', 'end_date', 'start_iso_year_week', 'end_iso_year_week']
    valid_df = valid_df[info_cols + [i for i in valid_df.columns if i not in info_cols]]
    
    valid_df.to_clipboard(index=False)
    
    return valid_df

def dm_adstock_data(bob_entity_code='US'):
    
    dbutils = DBUtils(spark)
    pprint(dbutils.fs.ls("dbfs:/FileStore/eddiedeane")[:3])

    dbutils.fs.ls('s3://data-science-mau/delta/speedeon_dm_mail_lists/run_year=2023/')
    
    dbutils.fs.ls('dbfs:/data-science-mau/delta/speedeon_dm_mail_lists')
    
    dbutils.fs.ls('s3://data-science-mau/delta')
    
    dbutils.fs.ls('s3://data-science-mau/delta/speedeon_dm_mail_lists/')
    
    dbutils.fs.ls('s3://data-science-mau/delta/speedeon_dm_mail_lists/')[0].path
    
    dir(dbutils.fs.ls('s3://data-science-mau/delta/speedeon_dm_mail_lists/')[0])
    
    dbutils.fs.ls('s3://data-science-mau/delta/speedeon_dm_mail_lists/')[0].name
    
    dbutils.fs.ls('s3://data-science-mau/delta/speedeon_dm_mail_lists/CAMPAIGN_YEAR=2022')
    
    dm_path = 's3://data-science-mau/delta/speedeon_dm_mail_lists/'
    campaign_year_folders = [i.name for i in dbutils.fs.ls(dm_path) if i.name[0] == 'C']
    campaign_year_folders
    
    dm_year_month_country = []
    for year_folder in dbutils.fs.ls(dm_path):
         if year_folder.name[0] == 'C':
            for month_folder in dbutils.fs.ls(dm_path + year_folder.name):
                if month_folder.name[0] == 'C':
                    if 'COUNTRY=US/' in [c.name for c in dbutils.fs.ls(dm_path + year_folder.name + month_folder.name)]:
                        campaign_year_months.append((year_folder.name, month_folder.name, 'COUNTRY=US/'))
                
    pd.DataFrame(campaign_year_months, columns=['campaign_year', 'campaign_month', 'country'])
    
    
    dbutils.fs.ls(dm_path + ''.join(campaign_year_months[0]))

    months = ['Feb', 'Apr', 'May']

    cols_needed = ['mail_or_hold', 'fk_conversion_date', 'in_home_date']
    
    dm_agg_query = f'''
    select
        campaign_year, 
        campaign_month, 
        in_home_date, 
        fk_conversion_date - in_home_date as n_conversion_day, 
        mail_or_hold, 
        count(*) n_households
    from marketing_analytics_us.speedeon_dm_mail_lists dm
    where dm.country = 'US'
        and lower(dm.campaign_type) = 'activation'
    group by 1,2,3,4,5
    '''
    dm_agg_df = spark.sql(dm_agg_query).toPandas()
    
    # get summary stats by campaign year / month
    
    # top 10 min, top 10 max
    
    dm_agg_df['campaign_date'] = pd.to_datetime(dm_agg_df['campaign_year'] + '-' + dm_agg_df['campaign_month'] + '-01')
    dm_agg_df['in_home_date'] = pd.to_datetime(dm_agg_df['in_home_date'], format='%Y%m%d')
    dm_agg_df['n_conversion_week'] = dm_agg_df['n_conversion_day'] // 7
    dm_agg_df.loc[dm_agg_df['n_conversion_week'] == 0, 'n_conversion_week'] = 1
    dm_agg_df['n_conversion_week_bin'] = dm_agg_df['n_conversion_week']
    dm_agg_df.loc[dm_agg_df['n_conversion_week_bin'] >= 7, 'n_conversion_week_bin'] = 7
    
    # weekly distribution 1, 2, 3, 4, 5, 6, 7+
    dm_agg_df.groupby(['campaign_date', 'n_conversion_week_bin'], as_index=False, dropna=False).agg({'n_households': 'sum'}).to_clipboard()
    
    dm_agg_df['n_conversion_week'].value_counts()
    dm_agg_df[dm_agg_df['mail_or_hold'] == 'mail']
    
    dm_in_home_query = f'''
    select
        campaign_year, 
        campaign_month, 
        in_home_date, 
        mail_or_hold, 
        count(*)
    from marketing_analytics_us.speedeon_dm_mail_lists dm
    where dm.country = 'US'
    group by 1,2,3,4
    '''
    in_home_df = spark.sql(dm_in_home_query).toPandas()
    in_home_df['in_home_date'] = pd.to_datetime(in_home_df['in_home_date'], format='%Y%m%d')
    in_home_df.to_clipboard(index=False)

    mon_sdfs = []
    for month in months:
        print(month)
    parq_path = f's3://data-science-mau/delta/speedeon_dm_mail_lists/run_year=2023/run_month={month}/COUNTRY=US'
    mon_sdf = spark.read.load(parq_path, format='delta')
    mon_sdf = mon_sdf.filter("campaign_type == 'Activation'")
    mon_sdf = mon_sdf.select(cols_needed)
    mon_sdfs.append(mon_sdf)


    counts = [(mon_sdf.count(), len(mon_sdf.columns)) for mon_sdf in mon_sdfs]
    print(counts)
    sum([i[0] for i in counts])


    from functools import reduce

    def unionAll(dfs):
        return reduce(lambda df1,df2: df1.unionAll(df2.select(df1.columns)), dfs)


    sdf = unionAll(mon_sdfs)
    sdf.count()
    len(sdf.columns)
    sdf = sdf.to_pandas_on_spark()

    agg_sdf = sdf.groupby(['mail_or_hold', 'fk_conversion_date', 'in_home_date'], as_index=False, dropna=False).size()
    agg_sdf.head()
    df = agg_sdf.to_pandas().reset_index()
    df.head()
    df['fk_conversion_date'].value_counts(dropna=False)


    df['fk_conversion_date'] = pd.to_datetime(df['fk_conversion_date'].astype('Int64'), format='%Y%m%d')
    df['in_home_date'] = pd.to_datetime(df['in_home_date'].astype('Int64'), format='%Y%m%d')
    df.rename(columns={0: 'count'}, inplace=True)
    df

    ((df['fk_conversion_date'] - df['in_home_date']) / np.timedelta64(1, 'W'))
    df
    df['conv_week'] = np.floor((df['fk_conversion_date'] - df['in_home_date']) / np.timedelta64(1, 'W')).astype('Int64')
    df[df['fk_conversion_date'].isna()]
    df[df['conv_week'].isna()]

    df[df['mail_or_hold'] == 'mail'].groupby('in_home_date').agg({'count': 'sum'})

    agg_df = df.groupby(['mail_or_hold', 'in_home_date', 'conv_week'], dropna=False, as_index=False).agg({'count': 'sum'})

    agg_tot_df = agg_df.groupby(['mail_or_hold', 'in_home_date'], dropna=False, as_index=False).agg(
    {'count': 'sum'}).rename(
    columns={'count': 'total'})

    agg_df = pd.merge(agg_df, agg_tot_df, how='left', on=['mail_or_hold', 'in_home_date'])
    agg_df['cvr'] = agg_df['count'] / agg_df['total']
    agg_df

    agg_df = agg_df[agg_df['conv_week'] <= 4]
    agg_df


    inc_agg_df = pd.merge(agg_df[agg_df['mail_or_hold'] == 'mail'],
                    agg_df.loc[agg_df['mail_or_hold'] == 'hold', ['in_home_date', 'conv_week', 'cvr']],
                    how='left', on=['in_home_date', 'conv_week'], suffixes=('_mail', '_hold'))

    inc_agg_df['cvr_inc'] = inc_agg_df['cvr_mail'] - inc_agg_df['cvr_hold']
    inc_agg_df['n_inc'] = inc_agg_df['cvr_inc'] * inc_agg_df['total']
    inc_agg_df.loc[inc_agg_df['n_inc'] < 0, 'n_inc'] = 0
    inc_agg_df

    inc_agg_df = pd.merge(inc_agg_df, inc_agg_df.groupby('in_home_date', as_index=False).agg({'n_inc': 'sum'}).rename(
    columns={'n_inc': 'total_inc'}),
                    how='left', on='in_home_date')

    inc_agg_df['per_inc'] = inc_agg_df['n_inc'] / inc_agg_df['total_inc']

    inc_agg_df = pd.merge(inc_agg_df, inc_agg_df.groupby('in_home_date', as_index=False).agg({'count': 'sum'}).rename(
    columns={'count': 'total_resp'}),
                    how='left', on='in_home_date')

    inc_agg_df['per_resp'] = inc_agg_df['count'] / inc_agg_df['total_resp']

    inc_agg_df

    inc_agg_df.to_clipboard(index=False)



    agg_df = agg_df.drop(columns=['month_total_conv', 'per'])
    agg_df = agg_df[agg_df['conv_week'] <= 4]

    agg_df = pd.merge(agg_df,
                agg_df.groupby(['mail_or_hold', 'in_home_date'], as_index=False).agg({'count': 'sum'}).rename(
                    columns={'count': 'month_total_conv'}),
                how='left', on=['mail_or_hold', 'in_home_date'])

    agg_df['per'] = agg_df['count'] / agg_df['month_total_conv']
    agg_mail_df = agg_df[agg_df['mail_or_hold'] == 'mail']

    agg_mail_df.pivot(index='conv_week', columns='in_home_date', values='per').to_clipboard()
    agg_mail_df.pivot(index='conv_week', columns='in_home_date', values='count').to_clipboard()

    agg_mail_df.to_clipboard(index=False)


    
    
    
    pass

def get_static_validation_data(bob_entity_code='US'):
    
    working_doc_gsheet_url = 'https://docs.google.com/spreadsheets/d/1bxiXZW4qVWbQYAi1BbqBfzRWmAMkaQ1wSUcwCxSnLd4'
    wd_gs_client = GSheet(working_doc_gsheet_url)
    validation_df = wd_gs_client.read_dataframe(worksheet_name='lift_test_validation_data', header_row_num=0)
    validation_df = validation_df[validation_df['bob_entity_code'] == bob_entity_code].reset_index(drop=True)
    validation_df['start_date'] = pd.to_datetime(validation_df['start_date'])
    validation_df['end_date'] = pd.to_datetime(validation_df['end_date'])
    
    return validation_df

def get_static_params():
    
    static_params = []
    static_params.append(('gsheet_url', 'edit permission googlesheet-automations@hf-mau-dev.iam.gserviceaccount.com', ''))
    static_params.append(('gsheet_ch_tab_name', 'tab used to map MEL and MAC channel hierarchy to groups', 'channel_hierarchy'))
    static_params.append(('bob_entity_code', 'only US available currently', 'US'))
    static_params.append(('min_data_iso_year_week', 'yyyy-w00 format', '2021-W49'))
    static_params.append(('max_data_iso_year_week',  'yyyy-w00 format', '2024-W15'))
    static_params.append(('daily_or_weekly', 'daily or weekly', 'weekly'))
    static_params.append(('adjust_dm', 'adjust direct mail to in-home date, TRUE or FALSE', True))
    static_params.append(('adjust_gc_fly', 'adjust gift cards and flyters to 30 day flat amort, TRUE or FALSE', True))
    static_params.append(('adjust_credits', 'adjust credits to yearly based upon spend and remove from feature, TRUE or FALSE', True))
    static_params.append(('replace_with_meta_mmm', 'replace MEL data with Meta MMM platform data', False))
    static_params.append(('replace_with_google_mmm', 'replace MEL data with Google MMM platform data', False))
    static_params.append(('organic_features', 'seo_weighted_avg_position, cumulative_branded_vehicles, seo_discount_added, add_compliance_language', ''))
    static_params.append(('context_features', 'discount_per_customer, per_free_shipping, per_ffl, price_per_meal', ''))
    static_params.append(('remove_prefix', 'channel hierarchy groups prefix to remove from data', 'exclude'))
    static_params.append(('add_validation_time_cols', 'add q1, q2, q3, q4 defined validation periods', False))
    static_params.append(('conversion_type_inclusion', 'conversion types to include comma seperated, activation, reactivation', 'activation'))
    static_params.append(('channel_category_exclusion', 'channel categories to exclude comma seperated, referral', 'referral'))
    platform_cpm_tables = [
        'facebook_ads_platform_metrics_daily_view',
        'google_adwords_account_daily_view',
        'google_dv360_platform_metrics_daily_view',
        'horizon_ott_daily_view',
        'horizon_tvlinear_daily_view',
        'microsoft_bing_platform_metrics_daily_view',
        'reddit_ads_platform_metrics_daily_view',
        'rokt_platform_metrics_daily_view',
        'tiktok_ads_platform_metrics_daily_view',
        'veritone_data_daily_view'
    ]
    static_params.append(('platform_cpm_to_impression_tables', platform_cpm_tables, ''))
    param_df = pd.DataFrame(static_params, columns=['param', 'description', 'value'])
    
    return param_df

def run_channel_hierarchy(gsheet_url='', bob_entity_code='US', min_max_data_iso_year_week=('2021-W49', '2024-W15')):
    
    vs_ch_cols = ['vs_campaign_type', 'vs_channel_medium', 'vs_channel_category', 'vs_channel', 'vs_channel_split', 'vs_partner']
    ch_cols = [i[3:] for i in vs_ch_cols]
    
    min_data_iso_year_week, max_data_iso_year_week = min_max_data_iso_year_week
    
    # read in channel_hierarchy if available otherwise read in us_ed_channel_hierarchy
    gs_client = GSheet(gsheet_url)
    input_title = gs_client.spreadsheet.title
    worksheet_names_lst = [i.title for i in gs_client.spreadsheet.worksheets()]
    has_ch = 'channel_hierarchy' in worksheet_names_lst
    
    if has_ch:
        ch_map_df, ch_map_low_dfs  = read_in_channel_hierarchy_mapping(gsheet_url=gsheet_url, gsheet_tab_name='channel_hierarchy')
    else:
        working_doc_gsheet_url = 'https://docs.google.com/spreadsheets/d/1bxiXZW4qVWbQYAi1BbqBfzRWmAMkaQ1wSUcwCxSnLd4'
        ch_map_df, ch_map_low_dfs  = read_in_channel_hierarchy_mapping(gsheet_url=working_doc_gsheet_url, gsheet_tab_name='us_ed_channel_hierarchy')
    
    full_ch_df = get_mel_channel_hierarchy(bob_entity_code=bob_entity_code, min_max_data_iso_year_week=min_max_data_iso_year_week)
    full_ch_df = map_low_chs_to_ch(ch_map_low_dfs=ch_map_low_dfs, mel_df=full_ch_df)
    
    n_misisng = full_ch_df[full_ch_df['group'] == 'missing'].shape[0]
    print(f'number of channel hierarchy rows misisng: {n_misisng}')
    
    param_df = get_static_params()
    param_df.loc[param_df['param'] == 'gsheet_url', 'value'] = gsheet_url
    param_df.loc[param_df['param'] == 'min_data_iso_year_week', 'value'] = min_data_iso_year_week
    param_df.loc[param_df['param'] == 'max_data_iso_year_week', 'value'] = max_data_iso_year_week
    write_dfs_dict = {
        'input_params': param_df, 
        'new_channel_hierarchy': full_ch_df
    }
    
    print(f'writing to {gsheet_url}')
    write_to_gsheet(gsheet_url=gsheet_url, write_dfs_dict=write_dfs_dict)
    print('done writing')
    
    pass

def run_end_to_end_data(gsheet_url=''):
    
    # gsheet_url = 'https://docs.google.com/spreadsheets/d/1If45CLT1Y1j1XX3iG__TPlCXY6QA5ckXp_ZJxAOuIJ4'
    
    gs_client = GSheet(gsheet_url)
    input_param_df = gs_client.read_dataframe('input_params', header_row_num=0)
    
    def to_bool(x):
        # if TRUE, True or T or t or true or 1 return True
        # if FALSE, False or F or f or false or 0 return False
        if x.lower() in ['true', 't', '1']:
            return True
        elif x.lower() in ['false', 'f', '0', '']:
            return False
        
        return False
    
    def to_list(x):
        # using re, remove brackets and parentheses and split on space, comma, or pipe
        res = re.split(r'[\s,|]+', re.sub(r'[()\[\]]', '', x))
        if res != ['']:
            return res
        
        return []
    
    gsheet_ch_tab_name = input_param_df.loc[input_param_df['param'] == 'gsheet_ch_tab_name', 'value'].values[0]
    bob_entity_code = input_param_df.loc[input_param_df['param'] == 'bob_entity_code', 'value'].values[0]
    min_data_iso_year_week = input_param_df.loc[input_param_df['param'] == 'min_data_iso_year_week', 'value'].values[0]
    max_data_iso_year_week = input_param_df.loc[input_param_df['param'] == 'max_data_iso_year_week', 'value'].values[0]
    min_max_data_iso_year_week = (min_data_iso_year_week, max_data_iso_year_week)
    daily_or_weekly = input_param_df.loc[input_param_df['param'] == 'daily_or_weekly', 'value'].values[0]
    adjust_dm = to_bool(input_param_df.loc[input_param_df['param'] == 'adjust_dm', 'value'].values[0])
    adjust_gc_fly = to_bool(input_param_df.loc[input_param_df['param'] == 'adjust_gc_fly', 'value'].values[0])
    adjust_credits = to_bool(input_param_df.loc[input_param_df['param'] == 'adjust_credits', 'value'].values[0])
    replace_with_meta_mmm = to_bool(input_param_df.loc[input_param_df['param'] == 'replace_with_meta_mmm', 'value'].values[0])
    replace_with_google_mmm = to_bool(input_param_df.loc[input_param_df['param'] == 'replace_with_google_mmm', 'value'].values[0])
    organic_features = to_list(input_param_df.loc[input_param_df['param'] == 'organic_features', 'value'].values[0])
    context_features = to_list(input_param_df.loc[input_param_df['param'] == 'context_features', 'value'].values[0])
    remove_prefix = input_param_df.loc[input_param_df['param'] == 'remove_prefix', 'value'].values[0]
    add_validation_time_cols = to_bool(input_param_df.loc[input_param_df['param'] == 'add_validation_time_cols', 'value'].values[0])
    conversion_type_inclusion = to_list(input_param_df.loc[input_param_df['param'] == 'conversion_type_inclusion', 'value'].values[0])    
    channel_category_exclusion = to_list(input_param_df.loc[input_param_df['param'] == 'channel_category_exclusion', 'value'].values[0])
    platform_cpm_to_impression_tables = to_list(input_param_df.loc[input_param_df['param'] == 'platform_cpm_to_impression_tables', 'value'].values[0])

    print('params', gsheet_ch_tab_name, bob_entity_code, min_data_iso_year_week, max_data_iso_year_week, daily_or_weekly, adjust_dm, adjust_gc_fly, adjust_credits, replace_with_meta_mmm, replace_with_google_mmm, organic_features, context_features, remove_prefix, add_validation_time_cols, conversion_type_inclusion, channel_category_exclusion, platform_cpm_to_impression_tables)
    
    param_dict = {
        'gsheet_url': gsheet_url,
        'gsheet_ch_tab_name': gsheet_ch_tab_name,
        'bob_entity_code': bob_entity_code,
        'min_data_iso_year_week': min_data_iso_year_week,
        'max_data_iso_year_week': max_data_iso_year_week,
        'daily_or_weekly': daily_or_weekly,
        'adjust_dm': adjust_dm,
        'adjust_gc_fly': adjust_gc_fly,
        'adjust_credits': adjust_credits,
        'replace_with_meta_mmm': replace_with_meta_mmm,
        'replace_with_google_mmm': replace_with_google_mmm,
        'organic_features': organic_features,
        'context_features': context_features,
        'remove_prefix': remove_prefix,
        'add_validation_time_cols': add_validation_time_cols,
        'conversion_type_inclusion': conversion_type_inclusion,
        'channel_category_exclusion': channel_category_exclusion,
        'platform_cpm_to_impression_tables': platform_cpm_to_impression_tables
    }
    
    # channel hieararchy columns
    vs_ch_cols = ['vs_campaign_type', 'vs_channel_medium', 'vs_channel_category', 'vs_channel', 'vs_channel_split', 'vs_partner']
    ch_cols = [i[3:] for i in vs_ch_cols]
    # ch_cols = ['campaign_objective', 'channel_medium', 'channel_category', 'channel', 'channel_split', 'partner']
    # mel_plat_ch_cols = ['campaign_type', 'channel', 'channel_split', 'partner']
    
    
    # channel hierarchy
    ch_map_df, ch_map_low_dfs  = read_in_channel_hierarchy_mapping(gsheet_url=gsheet_url, gsheet_tab_name=gsheet_ch_tab_name)
    full_ch_df = get_mel_channel_hierarchy(bob_entity_code=bob_entity_code, min_max_data_iso_year_week=min_max_data_iso_year_week)
    ch_map_low_df = pd.concat(ch_map_low_dfs, axis=0, ignore_index=True)[ch_cols + ['group']]
    full_ch_df = map_low_chs_to_ch(ch_map_low_dfs=ch_map_low_dfs, mel_df=full_ch_df)
    
    
    # features and feature cols
    feature_dfs = []
    feature_col_dfs = []
    
    # replace with Meta MMM data feed
    if replace_with_meta_mmm:
        full_ch_df.loc[full_ch_df['group'] == 'paid_social_meta', 'group'] = 'exclude_paid_social_meta'
        meta_mmm_feature_df, meta_mmm_feature_col_df = get_meta_mmm_agg_data()
        feature_dfs.append(meta_mmm_feature_df)
        feature_col_dfs.append(meta_mmm_feature_col_df)
    
    # replace with Google MMM data feed
    if replace_with_google_mmm:
        print('replace with google mmm')
    
    # paid media interaction features
    # if platform data is used, change the channel hierarchy for the platform data to exclude_group so it is excluded in the spend_feature function
    # replace spend with platform integration spend / impressions
    # or just use CPM by channel hierarchy data over time
    # compare before and after spend for 'exclude_' compared to platform data
    # paramater to add just spend or spend and impressions
    # how to handle missing data???
    if len(platform_cpm_to_impression_tables) > 0:
        platform_cpm_df = get_platform_cpms(platform_cpm_to_impression_tables=platform_cpm_to_impression_tables, 
                                            bob_entity_code=bob_entity_code, min_max_data_iso_year_week=min_max_data_iso_year_week,
                                            daily_or_weekly=daily_or_weekly)
    else:
        platform_cpm_df = None
    
    # paid media spend features
    # ch_gs_client = in_gs_client
    # ch_gsheet_tab_name = ch_gsheet_tab_name
    before_after_mel_spend_df, mel_df = get_ch_to_mel_data(ch_map_low_dfs=ch_map_low_dfs,
        bob_entity_code=bob_entity_code, min_max_data_iso_year_week=min_max_data_iso_year_week, 
        adjust_dm=adjust_dm, adjust_gc_fly=adjust_gc_fly, adjust_credits=adjust_credits)
    
    n_date_missing = mel_df['date'].isna().sum()
    print(f'n date missing {n_date_missing}')
    mel_df = mel_df[~mel_df['date'].isna()].reset_index(drop=True)
    min_max_data_iso_year_week
    mel_df = mel_df.sort_values(by='date', ascending=True).reset_index(drop=True)
    mel_df
    
    roll_group_df, grp_per_tot_df, grp_week_per_tot_df, spend_feature_df, spend_fee_df, spend_feature_col_df = mel_to_spend_feature_df(
        mel_df=mel_df, platform_cpm_df=platform_cpm_df, remove_prefix=remove_prefix, daily_or_weekly=daily_or_weekly)
    feature_dfs.append(spend_feature_df)
    feature_col_dfs.append(spend_feature_col_df)
    
    mel_spend_sum = mel_df.loc[~mel_df['group'].str.contains(remove_prefix), 'mel_spend'].sum()
    mel_agency_fee_sum = mel_df.loc[~mel_df['group'].str.contains(remove_prefix), 'mel_agency_fees'].sum()
    param_dict['mel_spend_sum'] = mel_spend_sum
    param_dict['mel_agency_fee_sum'] = mel_agency_fee_sum
    
    # organic features
    # number of delivery trucks, google seo rank, cvr website changes, email / sms / push notifications, social platform views
    if 'cumulative_branded_vehicles' in organic_features:
        vehicle_feature_df, vehicle_feature_col_df = get_branded_vehicle_features(bob_entity_code=bob_entity_code, min_max_data_iso_year_week=min_max_data_iso_year_week, daily_or_weekly=daily_or_weekly)
        feature_dfs.append(vehicle_feature_df)
        feature_col_dfs.append(vehicle_feature_col_df)
    
    if 'seo_weighted_avg_position' in organic_features or 'seo_discount_added' in organic_features or 'add_compliance_language' in organic_features:
        seo_feature_df, seo_feature_col_df = get_google_seo_features(
            bob_entity_code=bob_entity_code, min_max_data_iso_year_week=min_max_data_iso_year_week, daily_or_weekly=daily_or_weekly, 
            organic_features=organic_features)
        feature_dfs.append(seo_feature_df)
        feature_col_dfs.append(seo_feature_col_df)
    
    # conversion options
    conversion_cols = ['conversion_type', 'channel_category', 'channel']
    conv_opts_df = get_conversion_options(bob_entity_code=bob_entity_code, min_max_data_iso_year_week=min_max_data_iso_year_week, 
                                          conversion_cols=conversion_cols)
    
    conv_opts_df['conversion_type'] = conv_opts_df['conversion_type'].str.lower()
    conv_opts_df['channel_category'] = conv_opts_df['channel_category'].str.lower()
    conv_opts_df['channel'] = conv_opts_df['channel'].str.lower()
    
    conv_opts_df
    for col in conversion_cols:
        print((conv_opts_df.groupby(col, as_index=False, dropna=False)
               .agg({'n_conversion': 'sum'})
               .sort_values(by='n_conversion', ascending=False)))
    
    channel_exclusion = [chan for chan in conv_opts_df['channel'].dropna().unique() if 'b2b' in chan or 'qa' in chan or 'test' in chan or 'qc' in chan]
    param_dict['channel_exclusion'] = channel_exclusion
    
    # context features
    # discount, box price, events, competitor activity, macro economic
    if 'discount_per_customer' in context_features or 'per_free_shipping' in context_features or 'per_ffl' in context_features:
        discount_feature_df, discount_feature_col_df = get_discount_features(bob_entity_code=bob_entity_code, min_max_data_iso_year_week=min_max_data_iso_year_week, 
                                            daily_or_weekly=daily_or_weekly, conversion_type_inclusion=conversion_type_inclusion,
                                            channel_category_exclusion=channel_category_exclusion, channel_exclusion=channel_exclusion)
        feature_dfs.append(discount_feature_df)
        feature_col_dfs.append(discount_feature_col_df)
    if 'price_per_meal' in context_features:
        price_feature_df, price_feature_col_df = get_price_features(bob_entity_code=bob_entity_code, min_max_data_iso_year_week=min_max_data_iso_year_week, daily_or_weekly=daily_or_weekly)
        feature_dfs.append(price_feature_df)
        feature_col_dfs.append(price_feature_col_df)
    
    # conversion target
    conversion_check_df, conversion_df = get_conversion_data(bob_entity_code=bob_entity_code, 
                                                               min_max_data_iso_year_week=min_max_data_iso_year_week, 
                                                               daily_or_weekly=daily_or_weekly, 
                                                               conversion_type_inclusion=conversion_type_inclusion, 
                                                               channel_category_exclusion=channel_category_exclusion, 
                                                               channel_exclusion=channel_exclusion)
    
    conversion_check_sum = conversion_check_df.loc[(conversion_check_df['conversion_type'].isin(conversion_type_inclusion)) &
                            (~conversion_check_df['channel_category'].isin(channel_category_exclusion)) &
                            (~conversion_check_df['channel'].isin(channel_exclusion)), 
                            'n_conversion'].sum()
    conversion_time_target_sum = conversion_df['n_conversion'].sum()
    print(conversion_check_sum, conversion_time_target_sum)
    param_dict['conversion_check_sum'] = conversion_check_sum
    param_dict['conversion_time_target_sum'] = conversion_time_target_sum
    
    # channel conversion data
    chan_conv_df = get_channel_conversion_data(bob_entity_code=bob_entity_code, min_max_data_iso_year_week=min_max_data_iso_year_week, daily_or_weekly=daily_or_weekly, 
                                               channel_hierarchy_df=full_ch_df, conversion_type_inclusion=conversion_type_inclusion, 
                                               channel_category_exclusion=channel_category_exclusion, channel_exclusion=channel_exclusion)
    chan_conv_per_tot_df, chan_conv_week_per_tot_df, chan_conv_piv_df = channel_conversion_transform(conversion_df=chan_conv_df, remove_prefix='exclude')
    
    # join all features and feature_cols
    print('number of feature dfs:', len(feature_dfs))
    model_df = reduce(lambda left, right: pd.merge(left, right, how='left', on=['iso_year_week', 'date']), feature_dfs)
    feature_col_df = pd.concat(feature_col_dfs, axis=0)
    
    # fillna with fillna value
    for idx, row in feature_col_df.iterrows():
        model_df[row['feature']] = model_df[row['feature']].fillna(row['fillna'])
    
    date_n_rows = model_df.shape[0]
    date_drop_dup_n_rows = model_df[['iso_year_week', 'date']].drop_duplicates().shape[0]
    param_dict['date_n_rows'] = date_n_rows
    param_dict['date_drop_dup_n_rows'] = date_drop_dup_n_rows
    
    # join features and target and check totals
    model_df = pd.merge(model_df, conversion_df, how='left', on=['iso_year_week', 'date'])
    model_mel_spend_sum = model_df[[i for i in model_df.columns if i[-5:] == 'spend']].sum().sum()
    model_mel_fee_sum = spend_fee_df[[i for i in spend_fee_df.columns if i[-5:] == 'spend']].sum().sum()
    model_conversion_sum = model_df['n_conversion'].sum()
    
    param_dict['model_mel_spend_sum'] = model_mel_spend_sum
    param_dict['model_mel_fee_sum'] = model_mel_fee_sum
    param_dict['model_conversion_sum'] = model_conversion_sum
    
    pprint(param_dict)
    # summary stats
    param_dict, model_summ_df, spend_fee_summ_df, corr_pair_df, target_corr_df = create_feature_target_summary(
        param_dict=param_dict, feature_col_df=feature_col_df, model_df=model_df, spend_fee_df=spend_fee_df)
    
    # add validation time
    if add_validation_time_cols:
        model_df = add_validation_time(model_df)
    
    # get lift test validation data
    validation_df = get_static_validation_data(bob_entity_code='US')
    
    # write all data tables to gsheet
    # turn the param dict dictionary to a dataframe with a column for the key and a column for the value
    param_df = pd.DataFrame(param_dict.items(), columns=['param', 'value'])
    param_df['value'] = param_df['value'].astype('str')
    param_df['step'] = 'data'
    
    model_summ_df.info()
    model_summ_df
    spend_fee_summ_df.info()
    feature_col_df.info()
    model_summ_df.info()
    model_summ_df = model_summ_df.astype('str')
    spend_fee_summ_df = spend_fee_summ_df.astype('str')
    
    write_dfs_dict = {
        'output_params': param_df, 
        'channel_hierarchy_lowest_mapping': ch_map_low_df, 
        'channel_hierarchy_full_mapping': full_ch_df, 
        'before_after_adjust_mel_spend': before_after_mel_spend_df, 
        'channel_hieararchy_group_total_spend': roll_group_df, 
        'group_percent_of_total_spend': grp_per_tot_df,
        # 'group_week_percent_of_total_spend': grp_week_per_tot_df,
        'feature_correlation': corr_pair_df,
        'target_correlation': target_corr_df,
        'model_summary_stats': model_summ_df, 
        'fee_credit_summary_stats': spend_fee_summ_df,
        'group_percent_of_total_mac_conversion': chan_conv_per_tot_df, 
        # 'group_week_percent_of_total_mac_conversion': chan_conv_week_per_tot_df, 
        'feature_list': feature_col_df,
        'model_data': model_df,
        'fee_credit_data': spend_fee_df,
        'mac_conversion_data': chan_conv_piv_df, 
        'lift_test_validation_data': validation_df
    }
    
    write_to_gsheet(gsheet_url=gsheet_url, write_dfs_dict=write_dfs_dict)
    print('done writing to gsheet')
    
    pass

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

def auto_exploratory_data_analysis(input_gsheet_url):
    
    input_title, param_df, feature_col_df, model_df, spend_fee_df = read_end_to_end_data(input_gsheet_url)
    
    profile = ProfileReport(model_df, tsmode=True, sortby='date', title=input_title, explorative=True)
    profile.config.html.minify_html = True
    profile.config.html.use_local_assets = False
    profile.config.html.inline = True
    profile.config.html.full_width = True
    profile.config.interactions.targets = ['n_conversion']
    
    report_path = Path('eda_reports')
    output_file = report_path / f'auto_eda_{input_title}.html'
    profile.to_file(output_file)
    print(f'done writing to {output_file}')
    
    pass

def run_mmm_eda_html(gsheet_url, output_folder='eda_reports'):
    
    gs_client = GSheet(gsheet_url)
    input_title = gs_client.spreadsheet.title
    
    new_file_name = f'{input_title}'
    os.system(f'cp mmm_eda_ed.ipynb {new_file_name}.ipynb')
    
    # with open('mmm_eda_ed_config.txt', 'w') as f:
    #     f.write(gsheet_url)
    os.environ['GSHEET_URL'] = gsheet_url
    
    os.system(f'jupyter nbconvert --execute --to html --no-input --no-prompt {new_file_name}.ipynb --output {new_file_name} --output-dir {output_folder}')
    
    os.system(f'rm {new_file_name}.ipynb')
    
    pass

def run_nbconvert_mmm_eda_html(gsheet_url, local_or_databricks='databricks'):
    
    # local_or_databricks = 'local'
    # gsheet_url = 'https://docs.google.com/spreadsheets/d/1FZNOFvsobaOaFBhqUUbKnqTOHK_leGeweZU64HNZB00/edit#gid=1882244808'
    
    print('nbconvert.__version__', nbconvert.__version__)

    os.environ['GSHEET_URL'] = gsheet_url
    gs_client = GSheet(gsheet_url)
    input_title = gs_client.spreadsheet.title
    new_file_name = f'{input_title}'
    print(new_file_name)
    
    if local_or_databricks == 'local':
        save_file_path = f'eda_reports/{new_file_name}.html'
        download_link = save_file_path
        template_path = 'mmm_eda_ed.ipynb'
    elif local_or_databricks == 'databricks':
        save_file_path = f'/dbfs/FileStore/eddiedeane/mmm_eda/{new_file_name}.html'
        download_link = f'https://hf-us-ma.cloud.databricks.com/files/eddiedeane/mmm_eda/{new_file_name}.html'    
        template_path = '/dbfs/FileStore/eddiedeane/mmm_eda/mmm_eda_ed.ipynb'
    
    with open(template_path) as f:
        nb = nbformat.read(f, as_version=4)

    # execute
    # execute notebook
    ep = ExecutePreprocessor(timeout=-1, kernel_name='python3')
    ep.preprocess(nb)

    # Configure the HTML exporter
    html_exporter = HTMLExporter()
    html_exporter.exclude_input = True
    html_exporter.exclude_input_prompt = True
    # html_exporter.exclude_output_prompt = True

    # Convert the notebook to HTML
    (body, resources) = html_exporter.from_notebook_node(nb)

    # Save the result to a file
    with open(save_file_path, 'w') as f:
        f.write(body)
    
    print('download eda report here:', download_link)
    
    pass

def update_channel_hierarchy():
    
    
    ch_gs_url = 'https://docs.google.com/spreadsheets/d/1bxiXZW4qVWbQYAi1BbqBfzRWmAMkaQ1wSUcwCxSnLd4'
    ch_gs_tab_name = 'original_us_mel_cj_channel_hierarchy'
    
    conversion_type_inclusion = ['activation']
    # channel_exclusion = ['b2b revenue', 'b2b conversion', 'b2b']
    channel_category_exclusion = ['referral']
    model_type = ''
    
    ch_gs_client = GSheet(ch_gs_url)
    # out_gs_client = GSheet(output_gsheet_url)
    
    old_ch_df = ch_gs_client.read_dataframe(ch_gs_tab_name, header_row_num=0)
    old_ch_df
    
    # when did the migration start? march 5th 2024 might work
    # get data before migration at mel id level
    # get latest data at mel id level
    # join and unique
    
    # compare to channel hierarchy mapping data
    # ch_map_query = f'''
    # select
    #     distinct 
    #     lower(trim(chcm.vs_campaign_type)) campaign_type, 
    #     lower(trim(chcm.vs_channel_medium)) channel_medium, 
    #     lower(trim(chcm.vs_channel_category)) channel_category, 
    #     lower(trim(chcm.vs_channel)) channel, 
    #     lower(trim(chcm.vs_channel_split)) channel_split, 
    #     lower(trim(chcm.vs_partner)) partner, 
    #     lower(trim(chcm.campaign_objective_v1)) campaign_objective_v1, 
    #     lower(trim(chcm.channel_medium_v1)) channel_medium_v1, 
    #     lower(trim(chcm.channel_category_v1)) channel_category_v1, 
    #     lower(trim(chcm.channel_v1)) channel_v1, 
    #     lower(trim(chcm.channel_split_v1)) channel_split_v1, 
    #     lower(trim(chcm.partner_v1)) partner_v1 
    # from marketing_data_product.channel_hierarchy_campaign_mapping chcm
    # '''
    # ch_map_df = spark.sql(ch_map_query).toPandas()
    # ch_map_df
    # this doesn't work
    
    # old_ch_df
    # vs_ch_cols
    # ch_cols
    # ch_map_df[ch_cols]
    # ch_map_df[ch_map_df['partner'] == 'google']
    
    
    # pd.merge(old_ch_df, ch_map_df, how='left', on=ch_cols).to_clipboard(index=False)
    
    # f'''    
    # channel_hierarchy_campaign_mapping__us_affiliates_only
    # channel_hierarchy_mapping_campaign_type
    # channel_hierarchy_mapping_channel
    # channel_hierarchy_mapping_channel_category
    # channel_hierarchy_mapping_channel_medium
    # channel_hierarchy_mapping_channel_split
    # channel_hierarchy_mapping_partner
    # '''
    
    new_ch_map_df = ch_gs_client.read_dataframe('ch_map', header_row_num=0)
    new_ch_partner_map_df = ch_gs_client.read_dataframe('ch_partner_map', header_row_num=0)
    
    for col in new_ch_map_df.columns:
        new_ch_map_df[col] = new_ch_map_df[col].str.strip().str.lower()
        
    for col in new_ch_partner_map_df.columns:
        new_ch_partner_map_df[col] = new_ch_partner_map_df[col].str.strip().str.lower()
    
    new_ch_map_df = new_ch_map_df.drop_duplicates()
    new_ch_partner_map_df = new_ch_partner_map_df.drop_duplicates()
    
    ch_mer_df = pd.merge(old_ch_df, new_ch_map_df, how='left', on=[i for i in ch_cols if i != 'partner'])
    ch_mer_df = pd.merge(ch_mer_df, new_ch_partner_map_df, how='left', on='partner')    
    
    new_ch_cols = ['campaign_type_v1', 'channel_medium_v1', 'channel_category_v1', 'channel_v1', 'channel_split_v1', 'partner_v1']
    
    old_ch_df.shape
    ch_mer_df.shape
    ch_mer_df = ch_mer_df[ch_cols + new_ch_cols + ['group']]
    ch_mer_df = ch_mer_df.reset_index(drop=True)
    ch_mer_df
    
    ch_gs_client.write_dataframe('full_ch_map', ch_mer_df)
    
    cj_ch_df = ch_mer_df[new_ch_cols + ['group']].drop_duplicates().reset_index(drop=True)
    cj_ch_df.columns = [i.replace('_v1', '') for i in cj_ch_df.columns]
    cj_ch_df
    
    ch_gs_client.write_dataframe('us_mel_cj_channel_hierarchy', cj_ch_df)
    
    # for shadow mapping
    f'''
    marketing_analytics_us.mau_channel_split_partner_logic
    '''
    
    
    pass

def ch_spend_audit():
    
    bob_entity_code = 'US'
    # fin_ch_cols = ['campaign_type', 'channel_medium', 'channel_category', 'channel', 'channel_split', 'partner']
    # fin_ch_cols = ['campaign_type', 'channel_medium', 'channel_category', 'channel', 'channel_split']
    fin_ch_cols = ['campaign_type', 'channel_medium', 'channel_category', 'channel']
    ch_lowest_level = fin_ch_cols[-1]
    
    # get the MEL and voucher conversion channel hierarchy. aggregate to campaign_type, channel_medium, channel_category, channel, channel_split
    # top spend year / months and top spend users
    ch_df = get_mel_channel_hierarchy(bob_entity_code=bob_entity_code, min_max_date='', min_max_data_iso_year_week='', 
                                      ch_output_cols=fin_ch_cols)
    ch_df
    ch_df['year_month_min'].min()
    ch_df['year_month_max'].max()
    
    min_iso_year_week = '2019-W10'
    max_iso_year_week = '2024-W14'
    min_max_data_iso_year_week = (min_iso_year_week, max_iso_year_week)
    
    dw_df = get_mel_spend_data(bob_entity_code=bob_entity_code, channel_hierarchy_df=None, include_ch_group=False, min_max_date='', 
                               min_max_data_iso_year_week=min_max_data_iso_year_week, time_cols=['iso_year_week', 'date'], 
                               ch_output_cols=fin_ch_cols)
    
    dw_df[fin_ch_cols] = dw_df[fin_ch_cols].fillna('')
    dw_df['mel_spend'] = dw_df['mel_spend'].astype('float')
    dw_df['mel_agency_fees'] = dw_df['mel_agency_fees'].astype('float')
    dw_df['mel_spend'] = dw_df['mel_spend'].fillna(0)
    dw_df['mel_agency_fees'] = dw_df['mel_agency_fees'].fillna(0)
    if len(fin_ch_cols) == 6:
        dw_df['ch_cat'] = dw_df['campaign_type'] + '|' + dw_df['channel_medium'] + '|' + dw_df['channel_category'] + '|' + dw_df['channel'] + '|' + dw_df['channel_split'] + '|' + dw_df['partner']
    elif len(fin_ch_cols) == 5:
        dw_df['ch_cat'] = dw_df['campaign_type'] + '|' + dw_df['channel_medium'] + '|' + dw_df['channel_category'] + '|' + dw_df['channel'] + '|' + dw_df['channel_split']
    elif len(fin_ch_cols) == 4:
        dw_df['ch_cat'] = dw_df['campaign_type'] + '|' + dw_df['channel_medium'] + '|' + dw_df['channel_category'] + '|' + dw_df['channel']
    dw_df['year'] = dw_df['date'].dt.year
    
    
    # total conversions
    co_df = get_conversion_options(bob_entity_code=bob_entity_code, min_max_data_iso_year_week=min_max_data_iso_year_week, conversion_cols=['conversion_type', 'channel_category', 'channel'])
    channel_exclusion = [chan for chan in co_df['channel'].dropna().unique() if 'b2b' in chan or 'qa' in chan or 'test' in chan or 'qc' in chan]
    
    cc_df, conv_df = get_conversion_data(bob_entity_code=bob_entity_code, min_max_data_iso_year_week=min_max_data_iso_year_week, daily_or_weekly='daily', 
                          conversion_type_inclusion=['activation'], channel_category_exclusion=['referral'], 
                          channel_exclusion=channel_exclusion)
    conv_df['year'] = conv_df['date'].dt.year
    
    # and voucher conversions???
    
    # mel spend and total conversions by day and week. average daily and weekly by year and total
    # fin_df = dw_df.groupby(fin_ch_cols, as_index=False, dropna=False).agg({'mel_spend': 'sum', 'mel_agency_fees': 'sum'}).rename(columns={'mel_spend': 'total_mel_spend', 'mel_agency_fees': 'total_mel_agency_fees'})
    fin_year_df = (dw_df.groupby(fin_ch_cols + ['year'], as_index=False, dropna=False).agg({'mel_spend': 'sum', 'mel_agency_fees': 'sum'})
                   .rename(columns={'mel_spend': 'total_mel_spend', 'mel_agency_fees': 'total_mel_agency_fees'})
                   )
    fin_year_df['year'] = fin_year_df['year'].astype('str')
    fin_year_df['year'] = fin_year_df['year'] + '_mel_spend'
    fin_year_df = fin_year_df.fillna('')
    fin_year_df = fin_year_df.pivot(index=fin_ch_cols, columns='year', values='total_mel_spend').reset_index(drop=False)
    fin_year_df = fin_year_df.replace('', pd.NA)
    fin_year_df
    fin_df = pd.merge(ch_df, fin_year_df, how='left', on=fin_ch_cols)
    fin_df
    
    
    # voucher conversions
    
    # correlation daily and weekly by year and total
    dw_df['year'].min()
    conv_df['year'].min()
    years = dw_df['year'].unique()
    years.sort()
    
    for year in years:
        print(year)
        date_df = pd.pivot_table(dw_df[dw_df['year'] == year], index='date', columns='ch_cat', values='mel_spend', aggfunc=np.sum).fillna(0)
        date_df = pd.merge(date_df, conv_df[['date', 'n_conversion']], how='left', on='date')
        date_corr_df = date_df.drop(columns='date').corr()['n_conversion'].reset_index(name=f'{year}_date_corr').rename(columns={'index': 'ch_cat'})
        date_ch_df = date_corr_df['ch_cat'].str.split('|', expand=True)
        date_ch_df.columns = fin_ch_cols
        date_corr_df = pd.concat([date_ch_df, date_corr_df], axis=1).drop(columns='ch_cat')
        fin_df = pd.merge(fin_df, date_corr_df, how='left', on=fin_ch_cols)
        
        week_df = pd.pivot_table(dw_df[dw_df['year'] == year], index='iso_year_week', columns='ch_cat', values='mel_spend', aggfunc=np.sum).fillna(0)
        week_df = pd.merge(week_df, conv_df.groupby('iso_year_week', as_index=False).agg({'n_conversion': 'sum'}), how='left', on='iso_year_week')
        week_corr_df = week_df.drop(columns='iso_year_week').corr()['n_conversion'].reset_index(name=f'{year}_week_corr').rename(columns={'index': 'ch_cat'})
        week_ch_df = week_corr_df['ch_cat'].str.split('|', expand=True)
        week_ch_df.columns = fin_ch_cols
        week_corr_df = pd.concat([week_ch_df, week_corr_df], axis=1).drop(columns='ch_cat')
        fin_df = pd.merge(fin_df, week_corr_df, how='left', on=fin_ch_cols)
    
    # total correlation
    date_df = pd.pivot_table(dw_df, index='date', columns='ch_cat', values='mel_spend', aggfunc=np.sum).fillna(0)
    date_df = pd.merge(date_df, conv_df[['date', 'n_conversion']], how='left', on='date')
    date_corr_df = date_df.drop(columns='date').corr()['n_conversion'].reset_index(name='total_date_corr').rename(columns={'index': 'ch_cat'})
    date_ch_df = date_corr_df['ch_cat'].str.split('|', expand=True)
    date_ch_df.columns = fin_ch_cols
    date_corr_df = pd.concat([date_ch_df, date_corr_df], axis=1).drop(columns='ch_cat')
    fin_df = pd.merge(fin_df, date_corr_df, how='left', on=fin_ch_cols)
    
    week_df = pd.pivot_table(dw_df, index='iso_year_week', columns='ch_cat', values='mel_spend', aggfunc=np.sum).fillna(0)
    week_df = pd.merge(week_df, conv_df.groupby('iso_year_week', as_index=False).agg({'n_conversion': 'sum'}), how='left', on='iso_year_week')
    week_corr_df = week_df.drop(columns='iso_year_week').corr()['n_conversion'].reset_index(name=f'total_week_corr').rename(columns={'index': 'ch_cat'})
    week_ch_df = week_corr_df['ch_cat'].str.split('|', expand=True)
    week_ch_df.columns = fin_ch_cols
    week_corr_df = pd.concat([week_ch_df, week_corr_df], axis=1).drop(columns='ch_cat')
    fin_df = pd.merge(fin_df, week_corr_df, how='left', on=fin_ch_cols)
    
    min_date_df = dw_df.groupby('iso_year_week', as_index=False, dropna=False).agg({'date': 'min'})
    week_df = pd.merge(week_df, min_date_df, how='left', on='iso_year_week')
    week_df.drop(columns='iso_year_week', inplace=True)
    week_df = week_df[['date'] + [i for i in week_df.columns if i != 'date']]
    
    
    # spend category / spend definition / CPC, CPM, CPA, other, and current group
    fin_df = fin_df.sort_values(by='mel_spend', ascending=False)
    fin_df.to_clipboard(index=False)
    
    week_df
    
    
    
    # auto eda
    input_title = f'date_{ch_lowest_level}_audit'
    type_schema = {i: 'timeseries' for i in date_df.columns if i != 'date' and date_df[i].sum() != 0}
    profile = ProfileReport(date_df, tsmode=True, type_schema=type_schema, sortby='date', title=input_title, explorative=True)
    profile.config.html.minify_html = True
    profile.config.html.use_local_assets = False
    profile.config.html.inline = False
    profile.config.html.full_width = True
    profile.config.interactions.targets = ['n_conversion']
    
    report_path = Path('eda_reports')
    output_file = report_path / f'{input_title}.html'
    profile.to_file(output_file)
    print(f'done writing to {output_file}')
    
    input_title = f'week_{ch_lowest_level}_audit'
    type_schema = {i: 'timeseries' for i in week_df.columns if i != 'date' and week_df[i].sum() != 0}
    profile = ProfileReport(week_df, tsmode=True, type_schema=type_schema, sortby='date', title=input_title, explorative=True)
    profile.config.html.minify_html = True
    profile.config.html.use_local_assets = False
    profile.config.html.inline = False
    profile.config.html.full_width = True
    profile.config.interactions.targets = ['n_conversion']
    
    report_path = Path('eda_reports')
    output_file = report_path / f'{input_title}.html'
    profile.to_file(output_file)
    print(f'done writing to {output_file}')
    
    
    
    pass




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='US MMM Data Feed', 
        epilog='''
        example of use: 
            step 1: python mmm_data_ed.py --run-ch --gsheet-url=<your_gsheet_url> --min-data-iso-year-week=2019-W49 --max-data-iso-year-week=2024-W15
            step 2: edit input_params and channel_hierarchy
            step 3: python mmm_data_ed.py --run-data --run-eda --gsheet-url=<your_gsheet_url>
        '''
    )
    
    parser.add_argument('--run-ch', action='store_true', default=False)
    parser.add_argument('--run-data', action='store_true', default=False)
    parser.add_argument('--run-eda', action='store_true', default=False)
    parser.add_argument('--gsheet-url', type=str, required=True)
    # parser.add_argument('--gsheet-ch-tab-name', type=str, default='us_ed_channel_hierarchy', required=False)
    # parser.add_argument('--bob-entity-code', type=str, default='US', required=True)
    parser.add_argument('--min-data-iso-year-week', type=str, default='2021-W49', required=False)
    parser.add_argument('--max-data-iso-year-week', type=str, default='2024-W15', required=False)
    # parser.add_argument('--daily-or-weekly', type=str, default='weekly', required=False)
    # parser.add_argument('--adjust-dm', action=argparse.BooleanOptionalAction, required=False)
    # parser.add_argument('--adjust-gc-fly', action=argparse.BooleanOptionalAction, required=False)
    # parser.add_argument('--adjust-credits', action=argparse.BooleanOptionalAction, required=False)
    # parser.add_argument('--remove-prefix', type=str, default='exclude', required=False)
    # parser.add_argument('--conversion-type-inclusion', type=str, default='activation', required=False)
    # parser.add_argument('--channel-category-exclusion', type=str, default='referral', required=False)
    parser.add_argument('--local-or-databricks', type=str, default='databricks', required=False)
    
    args = parser.parse_args()
    print(args)
    
    run_ch = args.run_ch
    run_data = args.run_data
    run_eda = args.run_eda
    gsheet_url = args.gsheet_url
    # gsheet_ch_tab_name = args.gsheet_ch_tab_name
    bob_entity_code = 'US'
    min_data_iso_year_week = args.min_data_iso_year_week
    max_data_iso_year_week = args.max_data_iso_year_week
    # daily_or_weekly = args.daily_or_weekly
    # adjust_dm = args.adjust_dm
    # adjust_gc_fly = args.adjust_gc_fly
    # adjust_credits = args.adjust_credits
    # remove_prefix = args.remove_prefix
    # conversion_type_inclusion = re.split(r'[ ,|]+', args.conversion_type_inclusion)
    # channel_category_exclusion = re.split(r'[ ,|]+', args.channel_category_exclusion)
    local_or_databricks = args.local_or_databricks
    
    if local_or_databricks == 'databricks':
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'nbconvert', '--upgrade'])
        import nbformat
        import nbconvert
        from nbconvert.preprocessors import ExecutePreprocessor
        from nbconvert import HTMLExporter

    
    os.environ['GSHEET_URL'] = gsheet_url
    
    if run_ch:
        run_channel_hierarchy(gsheet_url=gsheet_url, bob_entity_code=bob_entity_code, min_max_data_iso_year_week=(min_data_iso_year_week, max_data_iso_year_week))
    
    if run_data:
        run_end_to_end_data(gsheet_url=gsheet_url)
    
    if run_eda:
        run_nbconvert_mmm_eda_html(gsheet_url=gsheet_url, local_or_databricks=local_or_databricks)
        