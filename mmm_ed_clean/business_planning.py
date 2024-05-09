
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
from maulibs.GSheet import GSheet
from datetime import date, timedelta
from mmm_data_ed import get_mel_channel_hierarchy, read_in_channel_hierarchy_mapping, map_low_chs_to_ch, get_mel_spend_data, get_conversion_options, get_conversion_data, get_channel_conversion_data, write_to_gsheet
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt
import pymc as pm
import pymc.sampling_jax
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.preprocessing import MaxAbsScaler
import xarray as xr
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
    

# plt.style.use("bmh")
plt.rcParams["figure.figsize"] = [10, 6]
plt.rcParams["figure.dpi"] = 100


spark = SparkSession.builder.getOrCreate()
print('done loading with spark session')

# ccv and discount logic
    # discount = take average of last n weeks
    # ccv = simple timeseries model SARIMA
    # with forecasted ccv and discount, get max cac by week for roi = 1

# mel logic
    # other_ and referral_ = still acquisition channel spend that we need to forecast but not as a part of paid impression channel spend
        # 1. average last x weeks
        # 2. last year spend same week
        # 3. known actuals as some are in contracts
        # 4. ARIMA or other model
    # exclude = do nothing with this
    # paid impression channel = remove "other_" "referral_" "exclude_" 
        # from this total spend and forecast with total MMM model. 
            # challenges = spend logged incorrectly (dm / gc&fly), low funnel spend (result of other spend), no changing mix understanding (single spend), no time-varying coefficient prediction (same CAC overtime)
            # option 1 use the pymc marketing mmm all built in
            # option 2 use more complicated asdr model and create simple frequentist formula with last or averge of last time-varying coefficients
            # option 3 use saturation model due to complicated adstock, create frequentist, and create closed form solution
                # set trend level
                # run with ccv and discount forecast and ROIS
                # now try with adjusted direct mail and gcfly spend data
                # add back adstock
                # add time varying intercept. for freq forecast, take last value
                # add time vary spend coefficient. for freq forecast, take last value
                
        # by channel, use historical % of total spend by month
        # other ideas https://github.com/google/amss
    # total spend = other_ + referral_ + paid impression channel. does not include exclude_
        # Think about how to suggest spend level from model. ROI 1, 1.2, 0.8
        # zero out trend...
        # one step at a time as the past spend impacts the future conversions and spend

# mac logic
    # total all conversion_type = activations excluding channel_category = referral. including other 24h cancel flag
    # trend + seasonality + spend effect conversion forecast
    # v1 v2 v3 total spend levels with max cac seasonality 
    # slit by channel, use historical % of total activations
    # referral use historical % of total activations
    
# scenarios
# 1. future spend as a percent of last year (60 65 70)
# 2. prescribe spend with ROI (0.7 0.8 0.9)

# reporting
# 1. change to a dataframe for every model
# 2. create a model column so can filter on which model
# 3. also create create the final pivot mel mac outputs at channel level for each model



def get_na_entities():
    ent_sql = f'''
    select 
        entity_code, 
        entity_name, 
        country_code, 
        bob_entity_code
    from dimensions.entity_dimension ed
    where cluster = 'North America'
        and is_entity_active = 1
        and (brand_name ilike 'Hello%' or brand_name ilike 'Every%' or brand_name ilike 'Green%' or brand_name ilike 'Chefs%' or brand_name ilike 'Factor%')
    '''
    print(ent_sql)
    ent_df = spark.sql(ent_sql).toPandas()
    return ent_df

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

def get_dates(min_max_iso_year_week):
    
    min_iso_year_week, max_iso_year_week = min_max_iso_year_week
    
    # weeks and quarters
    date_sql = f'''
    select
        dd.iso_year_week, 
        min(date_string_backwards) as min_date, 
        max(date_string_backwards) as max_date, 
        min(quarter) as min_quarter, 
        max(quarter) as max_quarter, 
        min(year) as min_year,
        max(year) as max_year
    from dimensions.date_dimension dd
    where dd.iso_year_week >= '{min_iso_year_week}' and dd.iso_year_week <= '{max_iso_year_week}'
    group by 1
    '''
    dates_df = spark.sql(date_sql).toPandas()
    print(dates_df)
    dates_df['min_date'] = pd.to_datetime(dates_df['min_date'])
    dates_df['max_date'] = pd.to_datetime(dates_df['max_date'])
    dates_df['min_year'] = dates_df['min_year'].astype(int)
    dates_df['max_year'] = dates_df['max_year'].astype(int)
    dates_df['date'] = dates_df['min_date']
    dates_df['year'] = dates_df['iso_year_week'].str[:4].astype(int)
    dates_df['n_week'] = dates_df['iso_year_week'].str[-2:].astype(int)
    dates_df = dates_df.sort_values('date', ascending=True).reset_index(drop=True)
    dates_df['index'] = dates_df.index
    dates_df
    
    return dates_df

def get_ccv_discount(bob_entity_code='US', min_max_iso_year_week=('2020-W45', '2024-W52')):
        
        min_iso_year_week, max_iso_year_week = min_max_iso_year_week
        # ccv
        # discount
        ccv_query = f'''
        select
            mp.country, 
            dd.iso_year_week,
            count(distinct mp.customer_id) n_customer, 
            sum(mp.discount_redeemed_prediction_52w) discount_redeemed_prediction_52w,  
            sum(mp.ccv_mpc2_prediction_52w) ccv_mpc2_prediction_52w
        from morpheus.predictions mp
        left join dimensions.date_dimension dd on mp.conversion_date = dd.sk_date
        left join marketing_data_product.marketing_attribution_conversions as mac on mp.customer_id = mac.customer_id 
            and mp.country = mac.bob_entity_code
            and lower(mp.conversion_type) = lower(mac.conversion_type)
        left join global_bi_business.customer_dimension cd on mac.customer_uuid = cd.customer_uuid
        where mp.country = '{bob_entity_code}'
            and mp.maturity = mp.prediction_after_x_weeks
            and dd.iso_year_week >= '{min_iso_year_week}' and dd.iso_year_week <= '{max_iso_year_week}'
            and lower(mp.conversion_type) = 'activation'
            and lower(mp.channel_category) != 'referral'
            and mac.shipping_address_postcode is not null
            and mac.customer_uuid is not null
            and mac.flag_cancelled_24h = false
            and cd.is_test = 0
        group by 1, 2
        '''
        print(ccv_query)
        table_name = 'morpheus.predictions'
        refresh_res = spark.catalog.refreshTable(table_name)
        ccv_df = spark.sql(ccv_query).toPandas()
        
        ccv_df['ccv'] = ccv_df['ccv_mpc2_prediction_52w'] / ccv_df['n_customer']
        ccv_df['discount'] = ccv_df['discount_redeemed_prediction_52w'] / ccv_df['n_customer']
        
        return ccv_df

def create_channel_hierarchy(bob_entity_code='US', min_max_data_iso_year_week=('2020-W45', '2024-W52'), write_to_gs_bool=True, gsheet_url=''):
    
    # create channel hierarchy
    if write_to_gs_bool:
        ch_df = get_mel_channel_hierarchy(bob_entity_code=bob_entity_code, min_max_date='', min_max_data_iso_year_week=min_max_data_iso_year_week)
        write_to_gsheet(gsheet_url=gsheet_url, write_dfs_dict={f'{bob_entity_code.lower()}_ch': ch_df})

    else:
        # channel hierarchy
        gsheet_ch_tab_name = f'{bob_entity_code.lower()}_ch'
        ch_map_df, ch_map_low_dfs  = read_in_channel_hierarchy_mapping(gsheet_url=gsheet_url, gsheet_tab_name=gsheet_ch_tab_name)
        full_ch_df = get_mel_channel_hierarchy(bob_entity_code=bob_entity_code, min_max_data_iso_year_week=min_max_data_iso_year_week)
        full_ch_df = map_low_chs_to_ch(ch_map_low_dfs=ch_map_low_dfs, mel_df=full_ch_df)
        # full_ch_df.drop(columns='group', inplace=True)
        n_missing = (full_ch_df['group'] == 'missing').sum()
        print('n_missing', n_missing)
        if n_missing > 0:
            print(full_ch_df[full_ch_df['group'] == 'missing'])
            write_to_gsheet(gsheet_url=gsheet_url, write_dfs_dict={f'new_{bob_entity_code.lower()}_ch': full_ch_df})
            
        return full_ch_df

def create_data(bob_entity_code='US', min_actual_iso_year_week='2020-W45', max_pred_iso_year_week='2024-W52', write_to_gs_bool=False, gsheet_url='', 
                full_ch_df=''):
    
    # dates
    max_actual_iso_year_week = get_last_full_week_str()
    # max_actual_iso_year_week = max_actual_iso_year_week[:-1] + str(int(max_actual_iso_year_week[-1]) - 1)
    min_max_actual_iso_year_week = (min_actual_iso_year_week, max_actual_iso_year_week)
    max_actual_iso_year_week
    
    dates_df = get_dates(min_max_iso_year_week=(min_actual_iso_year_week, max_pred_iso_year_week))
    max_actual_index = dates_df[dates_df['iso_year_week'] == max_actual_iso_year_week].index[0]
    min_pred_iso_year_week = dates_df.loc[max_actual_index+1, 'iso_year_week']
    min_max_pred_iso_year_week = (min_pred_iso_year_week, max_pred_iso_year_week)
    max_pred_iso_year_week
    
    dates_df['time_period'] = 'actual'
    dates_df.loc[dates_df['iso_year_week'] >= min_pred_iso_year_week, 'time_period'] = 'pred'
    dates_df
    
    pred_weeks_df = dates_df[(dates_df['iso_year_week'] >= min_pred_iso_year_week) & (dates_df['iso_year_week'] <= max_pred_iso_year_week)].copy(deep=True)
    
    if write_to_gs_bool:
        write_to_gsheet(gsheet_url=gsheet_url, write_dfs_dict={'dates': dates_df})
    

    # full_ch_df[full_ch_df['partner'].str.contains('parse', regex=False).fillna(False)]
    # full_ch_df.loc[full_ch_df['partner'].str.contains('parse', regex=False).fillna(False), 'mel_spend'].sum()
    # full_ch_df.loc[full_ch_df['partner'] == 'parse.ly', 'partner'] = 'parse.ly'
    
    # ccv and discount
    ccv_df = get_ccv_discount(bob_entity_code=bob_entity_code, min_max_iso_year_week=min_max_actual_iso_year_week)
    if write_to_gs_bool:
        write_to_gsheet(gsheet_url=gsheet_url, write_dfs_dict={f'{bob_entity_code.lower()}_ccv_discount': ccv_df})


    # spend by week
    mel_df = get_mel_spend_data(bob_entity_code, channel_hierarchy_df=full_ch_df, include_ch_group=True, ch_inner_left='left', 
                                min_max_data_iso_year_week=min_max_actual_iso_year_week, time_cols=['iso_year_week'], ch_output_cols=[])
    mel_df['mel_spend'] = mel_df['mel_spend'].fillna(0)
    mel_df['mel_agency_fees'] = mel_df['mel_agency_fees'].fillna(0)

    mel_spend_na = mel_df.loc[mel_df['group'].isna(), 'mel_spend'].sum()
    print('-'*50 + '\n' + '-'*50 + '\n' + '-'*50 + '\n' + '-'*50)
    print('mel_spend_na:', mel_spend_na)
    print('-'*50 + '\n' + '-'*50 + '\n' + '-'*50 + '\n' + '-'*50)
    
    
    mel_df['group'] = mel_df['group'].fillna('exclude')
    if write_to_gs_bool:
        write_to_gsheet(gsheet_url=gsheet_url, write_dfs_dict={f'{bob_entity_code.lower()}_ch_mel': mel_df})

    # mel_df = pd.merge(mel_df, dates_df, on='iso_year_week', how='left')
    # gs_client.write_dataframe('us_mel_spend', mel_df)

    # conversions
    # co_df = get_conversion_options(bob_entity_code=bob_entity_code, min_max_data_iso_year_week=min_max_actual_iso_year_week, conversion_cols=['conversion_type', 'channel_category', 'channel'])
    # channel_exclusion = [i for i in co_df['channel'].dropna().unique() if 'b2b' in i or 'qc' in i or 'qa' in i]
    channel_exclusion = []

    # ch_df, c_df = get_conversion_data(bob_entity_code=bob_entity_code, min_max_data_iso_year_week=min_max_actual_iso_year_week, daily_or_weekly='weekly', 
    #                           conversion_type_inclusion=['activation'], channel_category_exclusion=['referral'], 
    #                           channel_exclusion=channel_exclusion)

    # mac_df = get_channel_conversion_data(bob_entity_code=bob_entity_code, min_max_data_iso_year_week=min_max_actual_iso_year_week, daily_or_weekly='weekly', 
    #                                 channel_hierarchy_df=full_ch_df, conversion_type_inclusion=['activation'], 
    #                                 channel_category_exclusion=['referral'],
    #                                 channel_exclusion=channel_exclusion)
    # mac_df.drop(columns='date', inplace=True)
    
    mac_df = get_channel_conversion_data(bob_entity_code=bob_entity_code, min_max_data_iso_year_week=min_max_actual_iso_year_week, daily_or_weekly='weekly', 
                                    channel_hierarchy_df=full_ch_df, conversion_type_inclusion=['activation'], 
                                    channel_category_exclusion=[],
                                    channel_exclusion=channel_exclusion)
    mac_df.drop(columns='date', inplace=True)
    mac_df.loc[mac_df['group'] == 0, 'group'] = 'exclude'
    mac_df['group'].fillna('exclude', inplace=True)
    
    if write_to_gs_bool:
        write_to_gsheet(gsheet_url=gsheet_url, write_dfs_dict={f'{bob_entity_code.lower()}_ch_mac': mac_df})
    
    mel_df
    mac_df
    
    # aggreagate channel, other, and referral spend
    # us_agg_mel_mac
    # "other_" "referral_" "exclude_" 
    
    channel_mel_agg_df = (mel_df[(mel_df['group'].str[:5] != 'other') & (mel_df['group'].str[:8] != 'referral') & (mel_df['group'].str[:7] != 'exclude')]
     .groupby('iso_year_week', as_index=False, dropna=False)
     .agg({'mel_spend': 'sum', 'mel_agency_fees': 'sum'})
     .rename(columns={'mel_spend': 'channel_mel_spend', 'mel_agency_fees': 'channel_mel_agency_fees'})) 
    channel_mel_agg_df['channel_total_spend'] = channel_mel_agg_df['channel_mel_spend'] + channel_mel_agg_df['channel_mel_agency_fees']
    channel_mel_agg_df.drop(columns=['channel_mel_spend', 'channel_mel_agency_fees'], inplace=True)
    
    other_mel_agg_df = (mel_df[(mel_df['group'].str[:5] == 'other')]
     .groupby('iso_year_week', as_index=False, dropna=False)
     .agg({'mel_spend': 'sum', 'mel_agency_fees': 'sum'})
     .rename(columns={'mel_spend': 'other_mel_spend', 'mel_agency_fees': 'other_mel_agency_fees'})) 
    other_mel_agg_df['other_total_spend'] = other_mel_agg_df['other_mel_spend'] + other_mel_agg_df['other_mel_agency_fees']
    other_mel_agg_df.drop(columns=['other_mel_spend', 'other_mel_agency_fees'], inplace=True)
    
    referral_mel_agg_df = (mel_df[(mel_df['group'].str[:8] == 'referral')]
     .groupby('iso_year_week', as_index=False, dropna=False)
     .agg({'mel_spend': 'sum', 'mel_agency_fees': 'sum'})
     .rename(columns={'mel_spend': 'referral_mel_spend', 'mel_agency_fees': 'referral_mel_agency_fees'})) 
    referral_mel_agg_df['referral_total_spend'] = referral_mel_agg_df['referral_mel_spend'] + referral_mel_agg_df['referral_mel_agency_fees']
    referral_mel_agg_df.drop(columns=['referral_mel_spend', 'referral_mel_agency_fees'], inplace=True)
    
    
    mel_agg_df = pd.merge(channel_mel_agg_df, other_mel_agg_df, how='outer', on='iso_year_week').fillna(0)
    mel_agg_df = pd.merge(mel_agg_df, referral_mel_agg_df, how='outer', on='iso_year_week').fillna(0)
    mac_agg_df = (mac_df[~mac_df['group'].str.contains('referral')].groupby('iso_year_week', as_index=False, dropna=False)
                  .agg({'n_conversion': 'sum'}))
    ref_agg_df = (mac_df[mac_df['group'].str.contains('referral')].groupby('iso_year_week', as_index=False, dropna=False)
                  .agg({'n_conversion': 'sum'}).rename(columns={'n_conversion': 'n_referral'}))
    
    mel_mac_agg_df = pd.merge(mel_agg_df, mac_agg_df, how='outer', on='iso_year_week').fillna(0)
    mel_mac_agg_df = pd.merge(mel_mac_agg_df, ref_agg_df, how='outer', on='iso_year_week').fillna(0)
    mel_mac_agg_df['total_spend'] = mel_mac_agg_df['channel_total_spend'] + mel_mac_agg_df['other_total_spend'] + mel_mac_agg_df['referral_total_spend']
    mel_mac_agg_df['total_cac'] = mel_mac_agg_df['total_spend'] / mel_mac_agg_df['n_conversion']
    mel_mac_agg_df['referral_percent'] = mel_mac_agg_df['n_referral'] / mel_mac_agg_df['n_conversion']
    
    mel_mac_agg_df = pd.merge(mel_mac_agg_df, dates_df, on='iso_year_week', how='left')
    # mel_mac_agg_df[['date', 'n_referral', 'n_conversion', 'referral_percent']].to_clipboard(index=False)

    mel_mac_agg_df = pd.merge(mel_mac_agg_df, ccv_df[['iso_year_week', 'ccv', 'discount']], on='iso_year_week', how='left')
    # mel_mac_agg_df['channel_roi'] = mel_mac_agg_df['ccv'] / (mel_mac_agg_df['channel_cac'] + mel_mac_agg_df['discount'])
    mel_mac_agg_df['total_roi'] = mel_mac_agg_df['ccv'] / (mel_mac_agg_df['total_cac'] + mel_mac_agg_df['discount'])
    mel_mac_agg_df
    if write_to_gs_bool:
        write_to_gsheet(gsheet_url=gsheet_url, write_dfs_dict={f'{bob_entity_code.lower()}_agg_mel_mac': mel_mac_agg_df})
    
    print('done creating data')
    
    res = (dates_df, pred_weeks_df, ccv_df, mel_df, mac_df, mel_mac_agg_df)
    
    return res

def forecast_ccv_discount(bob_entity_code='US', ccv_df='', pred_weeks_df='', hyperparam_search=False, flatten_forecast=True, write_to_gs_bool=False, gsheet_url=''):
    
    # fix ccv prediction to be same ccv at beginning of next year
    discount_n_weeks = 10
    # discount forecast
    mel_mac_agg_df
    avg_discount = mel_mac_agg_df.iloc[-discount_n_weeks:]['discount'].mean()
    avg_discount
    
    ccv_d_pred_df = pred_weeks_df.copy(deep=True)
    ccv_d_pred_df['discount'] = avg_discount
    ccv_d_pred_df

    if hyperparam_search:
        train = mel_mac_agg_df.iloc[:int(mel_mac_agg_df.shape[0] * 0.7)]['ccv'].values
        test = mel_mac_agg_df.iloc[int(mel_mac_agg_df.shape[0] * 0.7):]['ccv'].values
        
        orders = [[1,1,1], [4,1,1], [4, 2, 1], [2,2,2], [4, 4, 4], [6, 3, 1], [6, 6, 6]]
        seasonal_orders = [[0,0,0,52], [1,1,1,52], [2,2,2,52]]
        
        results = []
        for order in orders:
            for seasonal_order in seasonal_orders:
                model = SARIMAX(train, order=order, seasonal_order=seasonal_order, trend='ct')
                model_fit = model.fit(disp=False)
                forecast = model_fit.predict(len(train), len(train) + len(test) - 1)
                rmse = np.sqrt(mean_squared_error(test, forecast))
                print(f'order={order}, seasonal_order={seasonal_order}, rmse={rmse}')
                results.append((order, seasonal_order, rmse))
        
        results_df = pd.DataFrame(results, columns=['order', 'seasonal_order', 'rmse']).sort_values('rmse').reset_index(drop=True)
        print(results_df.head(20))
    else:
        order = (4,2,1)
        seasonal_order = (1,1,1,52)
    
    # Define the model
    # order = results_df.loc[0, 'order']
    # seasonal_order = results_df.loc[0, 'seasonal_order']
    order = (4,2,1)
    seasonal_order = (1,1,1,52)
    ccv_val = mel_mac_agg_df['ccv'].values
    model = SARIMAX(ccv_val, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(maxiter=500, disp=True)
    pred_weeks_df
    ccv_forecast = model_fit.predict(len(ccv_val), len(ccv_val) + pred_weeks_df.shape[0] - 1)
    ccv_d_pred_df['ccv'] = ccv_forecast
    ccv_d_pred_df
    
    
    ccv_d_act_df = mel_mac_agg_df.copy(deep=True)
    ccv_d_fin_df = pd.concat([ccv_d_act_df, ccv_d_pred_df], axis=0)
    ccv_d_fin_df
    
    if flatten_forecast:
        act_first_ccv = ccv_d_fin_df.loc[ccv_d_fin_df['iso_year_week'] == '2024-W01', 'ccv'].values[0]
        pred_first_ccv = ccv_d_fin_df.loc[ccv_d_fin_df['iso_year_week'] == '2025-W01', 'ccv'].values[0]
        act_first_ccv
        pred_first_ccv
        
        # create even spaced numpy array from 1 to 0.91
        ccv_d_fin_df
        n_ones = ccv_d_fin_df[ccv_d_fin_df['iso_year_week'] <= '2024-W01'].shape[0]
        year_adjust = np.linspace(1, (act_first_ccv / pred_first_ccv), 52)
        n_more = ccv_d_fin_df.shape[0] - n_ones - year_adjust.shape[0]
        x_step = year_adjust[-2] - year_adjust[-1]
        x_start = year_adjust[-1] - x_step
        end_adjust = x_start + np.arange(0, n_more) * x_step
        ccv_d_fin_df['ccv_pred_adj'] = np.concatenate((np.ones(n_ones), year_adjust, end_adjust))
        
        # make n more predictions at same spacing
        ccv_d_fin_df.loc[ccv_d_fin_df['time_period'] == 'actual', 'ccv_pred_adj'] = 1
        ccv_d_fin_df['ccv'] * ccv_d_fin_df['ccv_pred_adj']
        ccv_d_fin_df['ccv'] = ccv_d_fin_df['ccv'] * ccv_d_fin_df['ccv_pred_adj']
        ccv_d_fin_df
    
    act_first_ccv = ccv_d_fin_df.loc[ccv_d_fin_df['iso_year_week'] == '2024-W01', 'ccv'].values[0]
    new_pred_first_ccv = ccv_d_fin_df.loc[ccv_d_fin_df['iso_year_week'] == '2025-W01', 'ccv'].values[0]
    print(f'2024-W01 forecast {act_first_ccv} 2025-W01 forecast {new_pred_first_ccv}')
    
    # ccv_d_fin_df['max_cac'] = ccv_d_fin_df['ccv'] - ccv_d_fin_df['discount']
    # ccv_d_fin_df.to_clipboard(index=False)
    
    ccv_d_fin_df
    pred_weeks_df
    # pred_weeks_df.drop(columns=['ccv', 'discount', 'max_cac'], inplace=True, errors='ignore')
    # pred_weeks_df['ccv'] = 315
    # pred_weeks_df['discount'] = 90
    
    pred_weeks_df = pd.merge(pred_weeks_df, ccv_d_fin_df[['iso_year_week', 'ccv', 'discount']], how='left', on='iso_year_week')
    pred_weeks_df
    
    if write_to_gs_bool:
        write_to_gsheet(gsheet_url=gsheet_url, write_dfs_dict={f'{bob_entity_code.lower()}_ccv_discount': ccv_d_fin_df})
        
    print('done ccv and discount')
    
    return pred_weeks_df

def other_and_referral_mel_model(bob_entity_code='US', mel_df='', pred_weeks_df='', write_to_gs_bool=False, gsheet_url=''):
    
    # average last n weeks and flat by channel
    model_name = 'flat_average_last_n_weeks'
    min_pred_week = '2024-W01'
    
    # filter on other and referral group
    or_mel_df = mel_df[(mel_df['group'].str[:5] == 'other') | (mel_df['group'].str[:8] == 'referral')].copy(deep=True)
    or_mel_df['total_spend'] = or_mel_df['mel_spend'] + or_mel_df['mel_agency_fees']
    
    or_agg_df = (or_mel_df[or_mel_df['iso_year_week'] >= min_pred_week].groupby('group', as_index=False, dropna=False)
                    .agg({'total_spend': 'mean'}))
    
    or_mel_pred_df = pd.merge(pred_weeks_df, or_agg_df, how='cross')
    
    
    other_mel_pred_df = (or_mel_pred_df[or_mel_pred_df['group'].str.contains('other')].groupby('iso_year_week', as_index=False, dropna=False)
     .agg({'total_spend': 'sum'}).rename(columns={'total_spend': 'other_total_spend'}))
    
    ref_mel_pred_df = (or_mel_pred_df[or_mel_pred_df['group'].str.contains('referral')].groupby('iso_year_week', as_index=False, dropna=False)
     .agg({'total_spend': 'sum'}).rename(columns={'total_spend': 'referral_total_spend'}))
    
    # pred_weeks_df.drop(columns=['other_total_spend', 'referral_total_spend'], inplace=True, errors='ignore')
    pred_weeks_df = pd.merge(pred_weeks_df, other_mel_pred_df, how='left', on='iso_year_week')
    pred_weeks_df = pd.merge(pred_weeks_df, ref_mel_pred_df, how='left', on='iso_year_week')
    
    if write_to_gs_bool:
        ws_name = f'{bob_entity_code.lower()}_other_mel_spend_pred'
        write_to_gsheet(gsheet_url=gsheet_url, write_dfs_dict={ws_name: or_agg_df})
    
    print('done mel other and referral')
    
    return pred_weeks_df

def yoy_mel_summary(mel_mac_agg_df):
    
    this_year = mel_mac_agg_df.loc[mel_mac_agg_df['time_period'] == 'actual', 'year'].max()
    latest_df = (mel_mac_agg_df.loc[mel_mac_agg_df['year'] == this_year, ['iso_year_week', 'time_period', 'year', 'n_week', 'channel_total_spend']]
             .rename(columns={'year': 'latest_year', 'channel_total_spend': 'latest_channel_total_spend'}).copy(deep=True))
    latest_df = pd.concat([latest_df, pred_weeks_df[['iso_year_week', 'time_period', 'year', 'n_week']].rename(columns={'year': 'latest_year'})], axis=0)
    pred_weeks_df[['iso_year_week', 'time_period', 'year', 'n_week']].tail(20)
    
    past_df = (mel_mac_agg_df[['year', 'n_week', 'channel_total_spend']]
             .rename(columns={'year': 'past_year', 'channel_total_spend': 'past_channel_total_spend'}).copy(deep=True))
    past_df['latest_year'] = past_df['past_year'] + 1
    
    yoy_df = pd.merge(latest_df, past_df, how='left', on=['latest_year', 'n_week'])
    yoy_df['yoy_per'] = yoy_df['latest_channel_total_spend'] / yoy_df['past_channel_total_spend']
    yoy_df['yoy_per_cumsum'] = yoy_df['latest_channel_total_spend'].cumsum() / yoy_df['past_channel_total_spend'].cumsum()
    
    return yoy_df

def geometric_adstock(x, alpha: float = 0.0, l_max: int = 12):
        """Geometric adstock transformation."""
        cycles = [
            pt.concatenate(
                [pt.zeros(i), x[: x.shape[0] - i]]
            )
            for i in range(l_max)
        ]
        x_cycle = pt.stack(cycles)
        w = pt.as_tensor_variable([pt.power(alpha, i) for i in range(l_max)])
        return pt.dot(w, x_cycle)

def logistic_saturation(x, lam: float = 0.5):
    """Logistic saturation transformation."""
    return (1 - pt.exp(-lam * x)) / (1 + pt.exp(-lam * x))

def inverse_logistic_saturation(y, lam: float = 0.5):
    """Inverse logistic saturation transformation."""
    return -pt.log((1 - y) / (1 + y)) / lam

def np_logistic_saturation(x, lam: float = 0.5):
    """Logistic saturation transformation."""
    return (1 - np.exp(-lam * x)) / (1 + np.exp(-lam * x))

def np_inverse_logistic_saturation(y, lam: float = 0.5):
    """Inverse logistic saturation transformation."""
    return -np.log((1 - y) / (1 + y)) / lam

def geometric_adstock_vectorized(x, alpha, l_max: int = 12):
    """Vectorized geometric adstock transformation."""
    cycles = [
        pt.concatenate(tensor_list=[pt.zeros(shape=x.shape)[:i], x[: x.shape[0] - i]])
        for i in range(l_max)
    ]
    x_cycle = pt.stack(cycles)
    x_cycle = pt.transpose(x=x_cycle, axes=[1, 2, 0])
    w = pt.as_tensor_variable([pt.power(alpha, i) for i in range(l_max)])
    w = pt.transpose(w)[None, ...]
    return pt.sum(pt.mul(x_cycle, w), axis=2)

def freq_saturation_model_spend_req(saturation_model_trace, t_pred, fourier_features_pred, channel_scaler, endog_scaler, chan_spend, fixed_spend, chan_conv):
    
    freq_a = float(saturation_model_trace.posterior['a'].mean())
    freq_b_trend = float(saturation_model_trace.posterior['b_trend'].mean())
    freq_lambda = float(saturation_model_trace.posterior['lam'].mean())
    freq_b_z = float(saturation_model_trace.posterior['b_z'].mean())
    freq_b_fourier = saturation_model_trace.posterior['b_fourier'].mean(dim=('chain', 'draw')).values
    
    freq_trend = freq_a + freq_b_trend * t_pred
    freq_seasonality = pm.math.dot(fourier_features_pred, freq_b_fourier).eval()
    freq_intercept = freq_trend + freq_seasonality
    
    if isinstance(chan_spend, list):
        chan_spend = np.array(chan_spend)
    elif isinstance(chan_spend, pd.Series):
        chan_spend = chan_spend.values
    
    if isinstance(fixed_spend, list):
        fixed_spend = np.array(fixed_spend)
    elif isinstance(fixed_spend, pd.Series):
        fixed_spend = fixed_spend.values
    
    if isinstance(chan_conv, list):
        chan_conv = np.array(chan_conv)
    elif isinstance(chan_conv, pd.Series):
        chan_conv = chan_conv.values
    
    fixed_spend_scaled = channel_scaler.transform(fixed_spend.reshape(-1,1)).flatten()
    
    chan_spend_scaled = channel_scaler.transform(chan_spend.reshape(-1,1))
    # chan_spend_scaled_saturated = logistic_saturation(chan_spend_scaled, lam=freq_lambda).eval()
    chan_conv_scaled = endog_scaler.transform(chan_conv.reshape(-1, 1))
    cac_scaled = (chan_spend_scaled / chan_conv_scaled).flatten()
    cac_scaled
    
    freq_spend_scaled = (fixed_spend_scaled - cac_scaled * freq_intercept) / (cac_scaled * freq_b_z - 1)
    # freq_spend_scaled = inverse_logistic_saturation(y=freq_spend_scaled_saturated, lam=freq_lambda).eval()
    freq_spend_scaled
    
    freq_spend = channel_scaler.inverse_transform(freq_spend_scaled.reshape(-1, 1)).flatten()
    
    return freq_spend

def freq_saturation_model_conv_pred(saturation_model_trace, t_pred, fourier_features_pred, channel_scaler, endog_scaler, chan_spend):
    
    freq_a = float(saturation_model_trace.posterior['a'].mean())
    freq_b_trend = float(saturation_model_trace.posterior['b_trend'].mean())
    freq_lambda = float(saturation_model_trace.posterior['lam'].mean())
    freq_b_z = float(saturation_model_trace.posterior['b_z'].mean())
    freq_b_fourier = saturation_model_trace.posterior['b_fourier'].mean(dim=('chain', 'draw')).values
    
    freq_trend = freq_a + freq_b_trend * t_pred
    freq_seasonality = pm.math.dot(fourier_features_pred, freq_b_fourier).eval()
    freq_intercept = freq_trend + freq_seasonality
    
    if isinstance(chan_spend, list):
        chan_spend = np.array(chan_spend)
    elif isinstance(chan_spend, pd.Series):
        chan_spend = chan_spend.values
    
    new_z_scaled = channel_scaler.transform(chan_spend.reshape(-1, 1)).flatten()
    freq_z_saturated = logistic_saturation(x=new_z_scaled, lam=freq_lambda).eval()
    freq_z_effect = freq_b_z * freq_z_saturated
    
    freq_mu = freq_intercept + freq_z_effect
    
    freq_trend_pred = endog_scaler.inverse_transform(X=freq_trend.reshape(-1, 1)).flatten()
    freq_seasonality_pred = endog_scaler.inverse_transform(X=freq_seasonality.reshape(-1, 1)).flatten()
    freq_z_effect_pred = endog_scaler.inverse_transform(X=freq_z_effect.reshape(-1, 1)).flatten()
    freq_total_pred = endog_scaler.inverse_transform(X=freq_mu.reshape(-1, 1)).flatten()
    # set any prediction less than 0 to 1
    freq_total_pred[freq_total_pred < 0] = 1
    
    return freq_trend_pred, freq_seasonality_pred, freq_z_effect_pred, freq_total_pred

def freq_saturation_model_heur_spend_req(saturation_model_trace, t_pred, fourier_features_pred, channel_scaler, endog_scaler, fixed_spend_pred, max_cac):
    
    if isinstance(fixed_spend_pred, list):
        fixed_spend_pred = np.array(fixed_spend_pred)
    elif isinstance(fixed_spend_pred, pd.Series):
        fixed_spend_pred = fixed_spend_pred.values
    
    chan_spend = np.array([0 for i in range(len(fixed_spend_pred))])
    
    n_iter = int(20_000_000 / 50_000)
    print(f'n_iter max: {n_iter}')
    
    freq_a = float(saturation_model_trace.posterior['a'].mean())
    freq_b_trend = float(saturation_model_trace.posterior['b_trend'].mean())
    freq_lambda = float(saturation_model_trace.posterior['lam'].mean())
    freq_b_z = float(saturation_model_trace.posterior['b_z'].mean())
    freq_b_fourier = saturation_model_trace.posterior['b_fourier'].mean(dim=('chain', 'draw')).values
    
    freq_trend = freq_a + freq_b_trend * t_pred
    freq_seasonality = pm.math.dot(fourier_features_pred, freq_b_fourier).eval()
    freq_intercept = freq_trend + freq_seasonality
    
    
    for i in range(n_iter):
        if i % 20 == 0:
            print(f'iteration {i}')
        
        new_z_scaled = channel_scaler.transform(chan_spend.reshape(-1, 1)).flatten()
        freq_z_saturated = logistic_saturation(x=new_z_scaled, lam=freq_lambda).eval()
        freq_z_effect = freq_b_z * freq_z_saturated
        freq_mu = freq_intercept + freq_z_effect
        freq_total_pred = endog_scaler.inverse_transform(X=freq_mu.reshape(-1, 1)).flatten()
        # set any prediction less than 0 to 1
        freq_total_pred[freq_total_pred < 0] = 1
        
        cacs = (chan_spend + fixed_spend_pred) / freq_total_pred
        off_target = max_cac - cacs
        n_off_target = (off_target > 0).sum()
        if i % 20 == 0:
            print(f'n_off_target: {n_off_target}')
        if n_off_target > 0:
            # where off target is greater than 0, increase chan_spend by 50_000
            chan_spend[off_target > 0] += 50_000
        else:
            print('done solving')
            break
        
    return chan_spend
    
def plot_saturation(lam_mean, lam_std):
    lams = np.array([lam_mean - lam_std, lam_mean, lam_mean + lam_std])
    x = np.linspace(0, 5, 100)
    ax = plt.subplot(111)
    for l in lams:
        y = np_logistic_saturation(x, lam=l)
        plt.plot(x, y, label=f'lam = {l}')
    plt.xlabel('spend', fontsize=12)
    plt.ylabel('f(spend)', fontsize=12)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    pass

def pymc_paid_total_saturation_model_fit(mel_mac_agg_df, min_model_week='2021-W25'):
    
    model_df = mel_mac_agg_df[['iso_year_week', 'date', 'n_week', 'channel_total_spend', 'n_conversion']].copy(deep=True)
    model_df = model_df.sort_values('date', ascending=True).reset_index(drop=True)
    model_df = model_df[model_df['iso_year_week'] >= min_model_week].reset_index(drop=True)
    model_df
    
    # t = ((model_df.index - model_df.index.min()) / (model_df.index.max() - model_df.index.min())).values
    t = np.arange(model_df.shape[0])
    
    model_df['t'] = t
    
    n_order = 7
    periods = model_df['n_week'] / 52
    
    fourier_features = pd.DataFrame(
        {
            f"{func}_order_{order}": getattr(np, func)(2 * np.pi * periods * order)
            for order in range(1, n_order + 1)
            for func in ("sin", "cos")
        }
    )
    
    fourier_features

    model_df.shape
    t.shape
    fourier_features.shape
    
    date = model_df['date'].to_numpy()
    date_index = model_df.index
    y = model_df['n_conversion'].to_numpy()
    z = model_df['channel_total_spend'].to_numpy()
    n_obs = y.size

    endog_scaler = MaxAbsScaler()
    endog_scaler.fit(y.reshape(-1, 1))
    y_scaled = endog_scaler.transform(y.reshape(-1, 1)).flatten()

    channel_scaler = MaxAbsScaler()
    channel_scaler.fit(z.reshape(-1, 1))
    z_scaled = channel_scaler.transform(z.reshape(-1, 1)).flatten()
    
    z_scaled / y_scaled

    coords = {"date": date, "fourier_mode": np.arange(2 * n_order)}
    
    palette = "viridis_r"
    cmap = plt.get_cmap(palette)
    percs = np.linspace(51, 99, 100)
    colors = (percs - np.min(percs)) / (np.max(percs) - np.min(percs))
    
    
    with pm.Model() as saturation_model:
        # --- coords ---
        saturation_model.add_coord(name="date", values=date, mutable=True)
        saturation_model.add_coord(name="fourier_mode", values=np.arange(2 * n_order), mutable=False)
        
        # --- data containers ---
        z_scaled_ = pm.MutableData(name="z_scaled", value=z_scaled, dims="date")
        y_scaled_ = pm.MutableData(name="y_scaled", value=y_scaled, dims="date")
        t_ = pm.Data(name="t", value=t, dims="date")
        fourier_features_ = pm.MutableData(name="fourier_features", value=fourier_features, dims=("date", "fourier_mode"))
        
        # --- priors ---
        ## intercept
        a = pm.Normal(name="a", mu=0, sigma=4)
        ## trend
        b_trend = pm.Normal(name="b_trend", mu=0, sigma=2)
        ## seasonality
        b_fourier = pm.Laplace(name="b_fourier", mu=0, b=2, dims="fourier_mode")
        ## saturation effect
        lam = pm.Gamma(name="lam", alpha=3, beta=1)
        ## regressor
        b_z = pm.HalfNormal(name="b_z", sigma=2)
        ## standard deviation of the normal likelihood
        sigma = pm.HalfNormal(name="sigma", sigma=0.5)
        # degrees of freedom of the t distribution
        nu = pm.Gamma(name="nu", alpha=25, beta=2)

        # --- model parametrization ---
        trend = pm.Deterministic("trend", a + b_trend * t, dims="date")
        seasonality = pm.Deterministic(name="seasonality", var=pm.math.dot(fourier_features_, b_fourier), dims="date")
        
        z_saturated = pm.Deterministic(
            name="z_saturated",
            var=logistic_saturation(x=z_scaled_, lam=lam),
            dims="date",
        )
        z_effect = pm.Deterministic(name="z_effect", var=b_z * z_saturated, dims="date")
        
        mu = pm.Deterministic(name="mu", var=trend + seasonality + z_effect, dims="date")

        # --- likelihood ---
        pm.StudentT(name="likelihood", nu=nu, mu=mu, sigma=sigma, observed=y_scaled, dims="date")

        # --- prior samples
        saturation_model_prior_predictive = pm.sample_prior_predictive()
    
    
    # pm.model_to_graphviz(model=adstock_saturation_model)

    # fit
    with saturation_model:
        saturation_model_trace = pm.sample(
            nuts_sampler="numpyro",
            draws=8_000,
            chains=6,
            idata_kwargs={"log_likelihood": True},
        )
        saturation_model_posterior_predictive = pm.sample_posterior_predictive(trace=saturation_model_trace)

    # lam_mean = saturation_model_trace.posterior['lam'].mean()
    # lam_std = saturation_model_trace.posterior['lam'].std()
    # plot_saturation(lam_mean, lam_std)

    
    posterior_predictive_likelihood = az.extract(
        data=saturation_model_posterior_predictive,
        group="posterior_predictive",
        var_names="likelihood",
    )

    posterior_predictive_likelihood_inv = endog_scaler.inverse_transform(X=posterior_predictive_likelihood)
    posterior_predictive_likelihood_inv.shape
    posterior_predictive_likelihood_inv.mean(axis=1)
    
    # compute HDI for all the model parameters
    # model_hdi = az.hdi(ary=saturation_model_trace)
    
    z_effect_posterior_samples = xr.apply_ufunc(
        lambda x: endog_scaler.inverse_transform(X=x.reshape(1, -1)),
        saturation_model_trace.posterior["z_effect"],
        input_core_dims=[["date"]],
        output_core_dims=[["date"]],
        vectorize=True,
    )
    trend_posterior_samples = xr.apply_ufunc(
        lambda x: endog_scaler.inverse_transform(X=x.reshape(1, -1)),
        saturation_model_trace.posterior["trend"],
        input_core_dims=[["date"]],
        output_core_dims=[["date"]],
        vectorize=True,
    )
    seasonality_posterior_samples = xr.apply_ufunc(
        lambda x: endog_scaler.inverse_transform(X=x.reshape(1, -1)),
        saturation_model_trace.posterior["seasonality"],
        input_core_dims=[["date"]],
        output_core_dims=[["date"]],
        vectorize=True,
    )
    likelihood_posterior_samples = endog_scaler.inverse_transform(
        X=az.extract(
            data=saturation_model_posterior_predictive,
            group="posterior_predictive",
            var_names=["likelihood"],
        )
    )
    
    as_z_effect_mean = z_effect_posterior_samples.mean(dim=("chain", "draw"))
    # z_effect_hdi = az.hdi(ary=z_effect_posterior_samples)["z_effect"]
    as_trend_mean = trend_posterior_samples.mean(dim=("chain", "draw"))
    # trend_hdi = az.hdi(ary=trend_posterior_samples)["trend"]
    as_seasonality_mean = seasonality_posterior_samples.mean(dim=("chain", "draw"))
    # seasonality_hdi = az.hdi(ary=seasonality_posterior_samples)["seasonality"]
    as_likelihood_mean = likelihood_posterior_samples.mean(axis=1)
    
    # create model_df as copy of mel_mac_agg_df with the 
    # add contribution to model_df
    mel_mac_agg_df['trend'] = np.concatenate((np.zeros(mel_mac_agg_df.shape[0] - model_df.shape[0]), t))
    mel_mac_agg_df[f'mmm_conv_channel_total_spend'] = np.concatenate((np.zeros(mel_mac_agg_df.shape[0] - model_df.shape[0]), as_z_effect_mean))
    mel_mac_agg_df[f'mmm_conv_trend'] = np.concatenate((np.zeros(mel_mac_agg_df.shape[0] - model_df.shape[0]), as_trend_mean))
    mel_mac_agg_df[f'mmm_conv_seasonality'] = np.concatenate((np.zeros(mel_mac_agg_df.shape[0] - model_df.shape[0]), as_seasonality_mean))
    mel_mac_agg_df[f'mmm_conv_total'] = np.concatenate((np.zeros(mel_mac_agg_df.shape[0] - model_df.shape[0]), as_z_effect_mean + as_trend_mean + as_seasonality_mean))
    # mel_mac_agg_df.to_clipboard(index=False)
    
    # with create pred_weeks_df in same format
    # add forecast and contribution
    # create future t, fourier_features, 
    # endog_scaler.inverse_transform(X=freq_trend.reshape(-1, 1)).flatten()
    # endog_scaler.inverse_transform(X=freq_seasonality.reshape(-1, 1)).flatten().min()
    
    return mel_mac_agg_df, saturation_model_trace, channel_scaler, endog_scaler

def pymc_paid_total_saturation_model_roi(saturation_model_trace, channel_scaler, endog_scaler, mel_mac_agg_df, pred_weeks_df, trend_flat_continue='flat', trend_flat_week='2024-W15', 
                                         rois=[0.7, 0.8, 0.9], referral_percent=0.3):
    
    
    model_name= 'roi'
    
    n_order = 7
    
    if trend_flat_continue == 'flat':
        t_pick = mel_mac_agg_df.loc[mel_mac_agg_df['iso_year_week'] == trend_flat_week, 'trend'].values[0]
        t_pred = np.array([t_pick for i in range(pred_weeks_df.shape[0])])
    else:
        trend_continue_last_t = mel_mac_agg_df['trend'].max()
        t_pred = np.arange(trend_continue_last_t+1, trend_continue_last_t+1+pred_weeks_df.shape[0])
    
    periods_pred = pred_weeks_df['n_week'] / 52
    fourier_features_pred = pd.DataFrame(
        {
            f"{func}_order_{order}": getattr(np, func)(2 * np.pi * periods_pred * order)
            for order in range(1, n_order + 1)
            for func in ("sin", "cos")
        }
    )
    
    act_pred_dfs = []
    for roi in rois:
        
        mod_pred_df = pred_weeks_df.copy(deep=True)
        
        model_name_cat = f'{roi*100:.0f}_{model_name}'
        mod_pred_df['model'] = model_name_cat
        
        max_cac = (mod_pred_df['ccv'] / roi) - mod_pred_df['discount']
        mod_pred_df['max_cac'] = max_cac
    
        fixed_spend_pred = mod_pred_df['other_total_spend'] + mod_pred_df['referral_total_spend']
        
        mod_pred_df['trend'] = t_pred
        
        paid_spend_req = freq_saturation_model_heur_spend_req(
            saturation_model_trace=saturation_model_trace, t_pred=t_pred, fourier_features_pred=fourier_features_pred, 
            channel_scaler=channel_scaler, endog_scaler=endog_scaler, fixed_spend_pred=fixed_spend_pred, 
            max_cac=max_cac)
        paid_spend_req[paid_spend_req < 0] = 0
        
        # subtract otehr and referral total spend to get actual spend
        # change paid_spend_req values from negative to 0
        mod_pred_df['channel_total_spend'] = paid_spend_req
        mod_pred_df['total_spend'] = mod_pred_df['channel_total_spend'] + mod_pred_df['other_total_spend'] + mod_pred_df['referral_total_spend']
        
        freq_trend_pred, freq_seasonality_pred, freq_z_effect_pred, freq_total_pred = freq_saturation_model_conv_pred(
            saturation_model_trace=saturation_model_trace, t_pred=t_pred, fourier_features_pred=fourier_features_pred, 
            channel_scaler=channel_scaler, endog_scaler=endog_scaler, chan_spend=paid_spend_req)
        
        mod_pred_df['mmm_conv_channel_total_spend'] = freq_z_effect_pred
        mod_pred_df['mmm_conv_trend'] = freq_trend_pred
        mod_pred_df['mmm_conv_seasonality'] = freq_seasonality_pred
        mod_pred_df['mmm_conv_total'] = freq_total_pred
        mod_pred_df['n_referral'] = mod_pred_df['mmm_conv_total'] * referral_percent
        mod_pred_df['total_cac'] = mod_pred_df['total_spend'] / mod_pred_df['mmm_conv_total']
        mod_pred_df['total_roi'] = mod_pred_df['ccv'] / (mod_pred_df['total_cac'] + mod_pred_df['discount'])
        mod_pred_df.head(20)
        
        act_df = mel_mac_agg_df.copy(deep=True)
        act_df['model'] = model_name_cat
        
        act_pred_df = pd.concat([act_df, mod_pred_df], axis=0, ignore_index=True)
        act_pred_dfs.append(act_pred_df)
            
    
    act_pred_df = pd.concat(act_pred_dfs, axis=0, ignore_index=True)
    
    return act_pred_df

def paid_total_yoy_percent_mel_mac(saturation_model_trace, channel_scaler, endog_scaler, mel_mac_agg_df, pred_weeks_df, trend_flat_continue='flat', trend_flat_week='2024-W15', 
                                   yoy_pers=[0.7, 0.8, 0.9], referral_percent=0.3):
    
    model_name = 'yoy_per'
    
    n_order = 7
    
    if trend_flat_continue == 'flat':
        t_pick = mel_mac_agg_df.loc[mel_mac_agg_df['iso_year_week'] == trend_flat_week, 'trend'].values[0]
        t_pred = np.array([t_pick for i in range(pred_weeks_df.shape[0])])
    else:
        trend_continue_last_t = mel_mac_agg_df['trend'].max()
        t_pred = np.arange(trend_continue_last_t+1, trend_continue_last_t+1+pred_weeks_df.shape[0])
    
    periods_pred = pred_weeks_df['n_week'] / 52
    fourier_features_pred = pd.DataFrame(
        {
            f"{func}_order_{order}": getattr(np, func)(2 * np.pi * periods_pred * order)
            for order in range(1, n_order + 1)
            for func in ("sin", "cos")
        }
    )
    
    # what is the current yoy percent
    # use that as the new level
    this_year = mel_mac_agg_df.loc[mel_mac_agg_df['time_period'] == 'actual', 'year'].max()
    latest_df = (mel_mac_agg_df.loc[mel_mac_agg_df['year'] == this_year, ['iso_year_week', 'time_period', 'year', 'n_week', 'channel_total_spend']]
             .rename(columns={'year': 'latest_year', 'channel_total_spend': 'latest_channel_total_spend'}).copy(deep=True))
    latest_df = pd.concat([latest_df, pred_weeks_df[['iso_year_week', 'time_period', 'year', 'n_week']].rename(columns={'year': 'latest_year'})], axis=0)
    pred_weeks_df[['iso_year_week', 'time_period', 'year', 'n_week']].tail(20)
    
    past_df = (mel_mac_agg_df[['year', 'n_week', 'channel_total_spend']]
             .rename(columns={'year': 'past_year', 'channel_total_spend': 'past_channel_total_spend'}).copy(deep=True))
    past_df['latest_year'] = past_df['past_year'] + 1
    
    yoy_df = pd.merge(latest_df, past_df, how='left', on=['latest_year', 'n_week'])
    yoy_df['yoy_per'] = yoy_df['latest_channel_total_spend'] / yoy_df['past_channel_total_spend']
    yoy_df['yoy_per_cumsum'] = yoy_df['latest_channel_total_spend'].cumsum() / yoy_df['past_channel_total_spend'].cumsum()
    
    act_pred_dfs = []
    for yoy_per in yoy_pers:
        
        mod_pred_df = pred_weeks_df.copy(deep=True)
        
        model_name_cat = f'{yoy_per*100:.0f}_{model_name}'
        mod_pred_df['model'] = model_name_cat
    
        paid_spend_req = (yoy_df.loc[yoy_df['time_period'] == 'pred', 'past_channel_total_spend'] * yoy_per).values
        paid_spend_req[paid_spend_req < 0] = 0
        
        mod_pred_df['trend'] = t_pred
        mod_pred_df['channel_total_spend'] = paid_spend_req
        mod_pred_df['total_spend'] = mod_pred_df['channel_total_spend'] + mod_pred_df['other_total_spend'] + mod_pred_df['referral_total_spend']
        
        freq_trend_pred, freq_seasonality_pred, freq_z_effect_pred, freq_total_pred = freq_saturation_model_conv_pred(
            saturation_model_trace=saturation_model_trace, t_pred=t_pred, fourier_features_pred=fourier_features_pred, 
            channel_scaler=channel_scaler, endog_scaler=endog_scaler, chan_spend=paid_spend_req)
        
        mod_pred_df['mmm_conv_channel_total_spend'] = freq_z_effect_pred
        mod_pred_df['mmm_conv_trend'] = freq_trend_pred
        mod_pred_df['mmm_conv_seasonality'] = freq_seasonality_pred
        mod_pred_df['mmm_conv_total'] = freq_total_pred
        mod_pred_df['n_referral'] = mod_pred_df['mmm_conv_total'] * referral_percent
        mod_pred_df['total_cac'] = mod_pred_df['total_spend'] / mod_pred_df['mmm_conv_total']
        mod_pred_df['total_roi'] = mod_pred_df['ccv'] / (mod_pred_df['total_cac'] + mod_pred_df['discount'])
        
        act_df = mel_mac_agg_df.copy(deep=True)
        act_df['model'] = model_name_cat

        act_pred_df = pd.concat([act_df, mod_pred_df], axis=0)
        act_pred_dfs.append(act_pred_df)
    
    act_pred_df = pd.concat(act_pred_dfs, axis=0, ignore_index=True)
    
    return act_pred_df, yoy_df

def paid_total_sarima_mel_mac(saturation_model_trace, channel_scaler, endog_scaler, mel_mac_agg_df, pred_weeks_df, trend_flat_continue='flat', trend_flat_week='2024-W15', 
                              hyperparam_search=False, referral_percent=0.3):
    
    model_name = 'time_series'
    
    n_order = 7
    
    if trend_flat_continue == 'flat':
        t_pick = mel_mac_agg_df.loc[mel_mac_agg_df['iso_year_week'] == trend_flat_week, 'trend'].values[0]
        t_pred = np.array([t_pick for i in range(pred_weeks_df.shape[0])])
    else:
        trend_continue_last_t = mel_mac_agg_df['trend'].max()
        t_pred = np.arange(trend_continue_last_t+1, trend_continue_last_t+1+pred_weeks_df.shape[0])
    
    periods_pred = pred_weeks_df['n_week'] / 52
    fourier_features_pred = pd.DataFrame(
        {
            f"{func}_order_{order}": getattr(np, func)(2 * np.pi * periods_pred * order)
            for order in range(1, n_order + 1)
            for func in ("sin", "cos")
        }
    )
    
    # alternatively can do prophet like model in pymc
    # https://www.pymc.io/projects/examples/en/latest/time_series/Air_passengers-Prophet_with_Bayesian_workflow.html
    
    
    if hyperparam_search:
        
        train = mel_mac_agg_df.iloc[:int(mel_mac_agg_df.shape[0] * 0.7)]['channel_total_spend'].values
        test = mel_mac_agg_df.iloc[int(mel_mac_agg_df.shape[0] * 0.7):]['channel_total_spend'].values
        
        orders = [[1,1,1], [4,1,1], [4, 2, 1], [2,2,2], [4, 4, 4], [6, 3, 1], [6, 6, 6]]
        seasonal_orders = [[0,0,0,52], [1,1,1,52], [2,2,2,52]]
        
        results = []
        for order in orders:
            for seasonal_order in seasonal_orders:
                model = SARIMAX(train, order=order, seasonal_order=seasonal_order, trend='ct')
                model_fit = model.fit(disp=False)
                forecast = model_fit.predict(len(train), len(train) + len(test) - 1)
                rmse = np.sqrt(mean_squared_error(test, forecast))
                print(f'order={order}, seasonal_order={seasonal_order}, rmse={rmse}')
                results.append((order, seasonal_order, rmse))
        
        results_df = pd.DataFrame(results, columns=['order', 'seasonal_order', 'rmse']).sort_values('rmse').reset_index(drop=True)
        print(results_df.head(20))
        order = results_df.loc[0, 'order']
        seasonal_order = results_df.loc[0, 'seasonal_order']
    
    else:
        order = (4,2,1)
        seasonal_order = (1,1,1,52)
        
    
    # Define the model
    # order = results_df.loc[0, 'order']
    # seasonal_order = results_df.loc[0, 'seasonal_order']
    
    past_spend = mel_mac_agg_df['channel_total_spend'].values
    model = SARIMAX(past_spend, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(maxiter=500, disp=True)
    paid_spend_req = model_fit.predict(len(past_spend), len(past_spend) + pred_weeks_df.shape[0] - 1)
    paid_spend_req[paid_spend_req < 0] = 0
        
    mod_pred_df = pred_weeks_df.copy(deep=True)
    
    mod_pred_df['model'] = model_name

    mod_pred_df['trend'] = t_pred
    mod_pred_df['channel_total_spend'] = paid_spend_req
    mod_pred_df['total_spend'] = mod_pred_df['channel_total_spend'] + mod_pred_df['other_total_spend'] + mod_pred_df['referral_total_spend']
    
    freq_trend_pred, freq_seasonality_pred, freq_z_effect_pred, freq_total_pred = freq_saturation_model_conv_pred(
        saturation_model_trace=saturation_model_trace, t_pred=t_pred, fourier_features_pred=fourier_features_pred, 
        channel_scaler=channel_scaler, endog_scaler=endog_scaler, chan_spend=paid_spend_req)
    
    mod_pred_df['mmm_conv_channel_total_spend'] = freq_z_effect_pred
    mod_pred_df['mmm_conv_trend'] = freq_trend_pred
    mod_pred_df['mmm_conv_seasonality'] = freq_seasonality_pred
    mod_pred_df['mmm_conv_total'] = freq_total_pred
    mod_pred_df['n_referral'] = mod_pred_df['mmm_conv_total'] * referral_percent
    mod_pred_df['total_cac'] = mod_pred_df['total_spend'] / mod_pred_df['mmm_conv_total']
    mod_pred_df['total_roi'] = mod_pred_df['ccv'] / (mod_pred_df['total_cac'] + mod_pred_df['discount'])
    
    act_df = mel_mac_agg_df.copy(deep=True)
    act_df['model'] = model_name

    act_pred_df = pd.concat([act_df, mod_pred_df], axis=0)
    
    return act_pred_df

def paid_channel_mel_mac_logic(mel_df, mac_df, pred_weeks_df, min_mix_week='2024-W04', max_mix_week='2024-W17', model_name='sat_mod'):
    
    
    # channel spend mix
    chan_mel_mix_agg_df = (mel_df[(mel_df['iso_year_week'] >= min_mix_week) & (mel_df['iso_year_week'] <= max_mix_week) & 
                             (~mel_df['group'].str.contains('referral')) & (~mel_df['group'].str.contains('other')) & 
                             (mel_df['group'] != 'exclude')]
                      .groupby('group', as_index=False).agg({'mel_spend': 'sum', 'mel_agency_fees': 'sum'}))
    chan_mel_mix_agg_df['total_mel_spend'] = chan_mel_mix_agg_df['mel_spend'] + chan_mel_mix_agg_df['mel_agency_fees']
    chan_mel_mix_agg_df['mix_percent'] = chan_mel_mix_agg_df['total_mel_spend'] / chan_mel_mix_agg_df['total_mel_spend'].sum()
    chan_mel_mix_agg_df.sort_values('mix_percent', ascending=False, inplace=True)
    chan_mel_mix_agg_df = chan_mel_mix_agg_df[['group', 'mix_percent']]
    chan_mel_mix_df = pd.merge(chan_mel_mix_agg_df[['group', 'mix_percent']], pred_weeks_df[['iso_year_week', 'channel_total_spend']], how='cross')
    chan_mel_mix_df['channel_spend'] = chan_mel_mix_df['mix_percent'] * chan_mel_mix_df['channel_total_spend']
    chan_mel_mix_df = chan_mel_mix_df[['iso_year_week', 'group', 'channel_spend']]
    chan_mel_mix_df
    
    # other spend mix
    other_mel_mix_agg_df = (mel_df[(mel_df['iso_year_week'] >= min_mix_week) & (mel_df['iso_year_week'] <= max_mix_week) & 
                             (mel_df['group'].str.contains('other'))]
                      .groupby('group', as_index=False).agg({'mel_spend': 'sum', 'mel_agency_fees': 'sum'}))
    other_mel_mix_agg_df['total_mel_spend'] = other_mel_mix_agg_df['mel_spend'] + other_mel_mix_agg_df['mel_agency_fees']
    other_mel_mix_agg_df['mix_percent'] = other_mel_mix_agg_df['total_mel_spend'] / other_mel_mix_agg_df['total_mel_spend'].sum()
    other_mel_mix_agg_df.sort_values('mix_percent', ascending=False, inplace=True)
    other_mel_mix_agg_df = other_mel_mix_agg_df[['group', 'mix_percent']]
    other_mel_mix_df = pd.merge(other_mel_mix_agg_df[['group', 'mix_percent']], pred_weeks_df[['iso_year_week', 'other_total_spend']], how='cross')
    other_mel_mix_df['channel_spend'] = other_mel_mix_df['mix_percent'] * other_mel_mix_df['other_total_spend']
    other_mel_mix_df = other_mel_mix_df[['iso_year_week', 'group', 'channel_spend']]
    other_mel_mix_df
    
    # referral spend mix
    referral_mel_mix_df = pred_weeks_df[['iso_year_week', 'referral_total_spend']].copy(deep=True)
    referral_mel_mix_df['group'] = 'referral'
    referral_mel_mix_df.rename(columns={'referral_total_spend': 'channel_spend'}, inplace=True)
    referral_mel_mix_df = referral_mel_mix_df[['iso_year_week', 'group', 'channel_spend']]
    referral_mel_mix_df
    
    # mel combine
    mel_chan_mix_df = pd.concat([chan_mel_mix_df, other_mel_mix_df, referral_mel_mix_df], axis=0, ignore_index=True)
    mel_chan_mix_df = mel_chan_mix_df[['iso_year_week', 'group', 'channel_spend']]
    mel_chan_mix_df
    
    
    # channel conv mix
    chan_mac_mix_agg_df = (mac_df[(mac_df['iso_year_week'] >= min_mix_week) & (mac_df['iso_year_week'] <= max_mix_week) & 
                             (~mac_df['group'].str.contains('referral'))]
                      .groupby('group', as_index=False).agg({'n_conversion': 'sum'}))
    chan_mac_mix_agg_df['group'] = chan_mac_mix_agg_df['group'].str.replace('other_', '')
    chan_mac_mix_agg_df['mix_percent'] = chan_mac_mix_agg_df['n_conversion'] / chan_mac_mix_agg_df['n_conversion'].sum()
    chan_mac_mix_agg_df.sort_values('mix_percent', ascending=False, inplace=True)
    chan_mac_mix_agg_df = chan_mac_mix_agg_df[['group', 'mix_percent']]
    chan_mac_mix_agg_df
    
    chan_mac_mix_df = pd.merge(chan_mac_mix_agg_df[['group', 'mix_percent']], pred_weeks_df[['iso_year_week', 'mmm_conv_total']], how='cross')
    chan_mac_mix_df['channel_conversion'] = chan_mac_mix_df['mix_percent'] * chan_mac_mix_df['mmm_conv_total']
    chan_mac_mix_df = chan_mac_mix_df[['iso_year_week', 'group', 'channel_conversion']]
    chan_mac_mix_df
    
    # referral conv mix
    referral_mac_mix_df = pred_weeks_df[['iso_year_week', 'n_referral']].copy(deep=True)
    referral_mac_mix_df['group'] = 'referral'
    referral_mac_mix_df.rename(columns={'n_referral': 'channel_conversion'}, inplace=True)
    referral_mac_mix_df = referral_mac_mix_df[['iso_year_week', 'group', 'channel_conversion']]
    
    # mac combine
    mac_chan_mix_df = pd.concat([chan_mac_mix_df, referral_mac_mix_df], axis=0, ignore_index=True)
    mac_chan_mix_df = mac_chan_mix_df[['iso_year_week', 'group', 'channel_conversion']]
    mac_chan_mix_df
    
    
    cac_chan_mix_df = pd.merge(mel_chan_mix_df, mac_chan_mix_df, on=['iso_year_week', 'group'], how='outer')
    cac_chan_mix_df['voucher_cac'] = cac_chan_mix_df['channel_spend'] / cac_chan_mix_df['channel_conversion']
    cac_chan_mix_df['model'] = model_name
    
    
    # pivots
    run_pivot = False
    if run_pivot:
        mel_chan_mix_piv_df = mel_chan_mix_df.pivot(index='group', columns='iso_year_week', values='channel_spend').reset_index(drop=False)
        mel_chan_mix_piv_df['model'] = model_name
        mel_chan_mix_piv_df = mel_chan_mix_piv_df[['model', 'group'] + [i for i in mel_chan_mix_piv_df.columns if i not in ('model', 'group')]]
        mel_chan_mix_piv_df
        
        mac_chan_mix_piv_df = mac_chan_mix_df.pivot(index='group', columns='iso_year_week', values='channel_conversion').reset_index(drop=False)
        mac_chan_mix_piv_df['model'] = model_name
        mac_chan_mix_piv_df = mac_chan_mix_piv_df[['model', 'group'] + [i for i in mac_chan_mix_piv_df.columns if i not in ('model', 'group')]]
        mac_chan_mix_piv_df
        
        cac_chan_mix_piv_df = cac_chan_mix_df.pivot(index='group', columns='iso_year_week', values='voucher_cac').reset_index(drop=False)
        cac_chan_mix_piv_df['model'] = model_name
        cac_chan_mix_piv_df = cac_chan_mix_piv_df[['model', 'group'] + [i for i in cac_chan_mix_piv_df.columns if i not in ('model', 'group')]]
        cac_chan_mix_piv_df.head(20)
            
        return cac_chan_mix_df, mel_chan_mix_piv_df, mac_chan_mix_piv_df, cac_chan_mix_piv_df

    
    return cac_chan_mix_df

def actual_channel_cacs(mel_df, mac_df):
        mel_cop_df = mel_df.copy(deep=True)
        mac_cop_df = mac_df.copy(deep=True)
        mel_cop_df.loc[mel_cop_df['group'].str.contains('referral'), 'group'] = 'referral'
        mel_cop_df['total_spend'] = mel_cop_df['mel_spend'] + mel_cop_df['mel_agency_fees']
        mac_cop_df.loc[mac_cop_df['group'].str.contains('referral'), 'group'] = 'referral'
        cac_df = pd.merge(mel_cop_df[['iso_year_week', 'group', 'total_spend']], mac_cop_df, on=['iso_year_week', 'group'], how='outer')
        cac_df.rename(columns={'total_spend': 'channel_spend', 'n_conversion': 'channel_conversion'}, inplace=True)
        cac_df['voucher_cac'] = cac_df['channel_spend'] / cac_df['channel_conversion']
        return cac_df

def summary_report(total_model_detail_df, act_pred_cac_df):
    
    # total past / future spend also by group and channel
    # total past / future conversions also by group and channel
    # total past / future cac also by group and channel
    # total past / future roi also by group and channel
    quarter_agg = (total_model_detail_df.groupby(['model', 'time_period', 'min_year', 'min_quarter'], as_index=False)
                   .agg({'total_spend': 'sum', 'n_conversion': 'sum', 'mmm_conv_total': 'sum'}))
    quarter_agg['actual_voucher_cac'] = quarter_agg['total_spend'] / quarter_agg['n_conversion']
    quarter_agg['pred_voucher_cac'] = quarter_agg['total_spend'] / quarter_agg['mmm_conv_total']
    quarter_agg
    
    group_agg = (act_pred_cac_df.groupby(['model', 'time_period', 'min_year', 'min_quarter', 'group'], as_index=False)
                 .agg({'channel_spend': 'sum', 'channel_conversion': 'sum'}))
    group_agg['voucher_cac'] = group_agg['channel_spend'] / group_agg['channel_conversion']
    
    return quarter_agg, group_agg


def main():
    
    ent_df = get_na_entities()
    ent_df.shape
    print(ent_df)
    
    gsheet_urls = {
        'US': 'https://docs.google.com/spreadsheets/d/1fHNozGMHKKeWSzLwvejxX8-m5wXwUDYZKqo3UvaH6kg', 
        'FJ': 'https://docs.google.com/spreadsheets/d/1CmKSMJDDH1L7asOTKvRrInOZCKOzksdKh9SiIdXuQOE', 
        'ER': 'https://docs.google.com/spreadsheets/d/1zPZv5k4ApMi3ih-qQ7w9h5mbCuha-RYmXa_Zq_x9JpQ', 
        'CG': 'https://docs.google.com/spreadsheets/d/1FhNtGGfsaEP_tqUwcszz9_C_SR5rbE6gb_nl8srvzAA', 
        'CA': 'https://docs.google.com/spreadsheets/d/1hoU69ScImlGk8UKnAtuFPhxot8fNQXpznyQ746TLBmo', 
        'CF': 'https://docs.google.com/spreadsheets/d/1YJgx-wXySAUfIM8rhyZahc8drp3rfHr_plIPXjjp_dU', 
        'CK': 'https://docs.google.com/spreadsheets/d/1LP2ATG9JqnwvZtVp7jrnf5TBWNZK9dyKzsrmF841kCc', 
        
    }
    min_actual_iso_year_weeks = {
        'US': '2020-W45', 
        'FJ': '2021-W30', 
        'ER': '2021-W30', 
        'CG': '2021-W45',
        'CA': '2021-W30',
        'CF': '2022-W45',
        'CK': '2021-W30'
    }
    min_model_weeks = {
        'US': '2022-W30', # '2021-W30'
        'FJ': '2021-W30', 
        'ER': '2021-W30', 
        'CG': '2021-W45',
        'CA': '2021-W30',
        'CF': '2022-W45',
        'CK': '2021-W30'
        
    }
    trend_flat_continues = {
        'US': 'flat', 
        'FJ': 'continues', 
        'ER': 'continues', 
        'CG': 'continues',
        'CA': 'continues',
        'CF': 'continues',
        'CK': 'continues'
    }
    trend_flat_weeks = {
        'US': '2024-W01', 
        'CA': '2024-W01'
    }
    rois_dict = {
        'US': [0.9], 
        'FJ': [1],
        'ER': [1],
        'CG': [1],
        'CA': [],
        'CF': [],
        'CK': []
    }
    
    
    bob_entity_code = 'US'
    min_actual_iso_year_week = min_actual_iso_year_weeks[bob_entity_code]
    max_pred_iso_year_week = '2025-W17'
    write_to_gs_bool = False
    plots_action = 'na' # show, save, or not_run
    gsheet_url = gsheet_urls[bob_entity_code]
    min_model_week = min_model_weeks[bob_entity_code]
    trend_flat_continue = trend_flat_continues[bob_entity_code]
    trend_flat_week = trend_flat_weeks.get(bob_entity_code, '2024-W01')
    rois = rois_dict.get(bob_entity_code, [])
    
    # write channel hierarchy if new
    write_ch = False
    if write_ch:
        nothing_ = create_channel_hierarchy(
            bob_entity_code=bob_entity_code, min_max_data_iso_year_week=(min_actual_iso_year_week, max_pred_iso_year_week), 
            write_to_gs_bool=True, gsheet_url=gsheet_url)
    
    # read in channel hierarchy mapping
    full_ch_df = create_channel_hierarchy(
        bob_entity_code=bob_entity_code, min_max_data_iso_year_week=(min_actual_iso_year_week, max_pred_iso_year_week), 
        write_to_gs_bool=False, gsheet_url=gsheet_url)
        
    # read in data
    dates_df, pred_weeks_df, ccv_df, mel_df, mac_df, mel_mac_agg_df = create_data(
        bob_entity_code=bob_entity_code, min_actual_iso_year_week=min_actual_iso_year_week, max_pred_iso_year_week=max_pred_iso_year_week,
        full_ch_df=full_ch_df, write_to_gs_bool=False, gsheet_url=gsheet_url)
    
    # yoy summary
    yoy_mel_sum_df = yoy_mel_summary(mel_mac_agg_df=mel_mac_agg_df)
    print(yoy_mel_sum_df.head(20))
    yoy_pers_dict = {
        'US': [0.5, 0.55, 0.60, 0.65, 0.70], 
        'FJ': [1, 1.5, 2, 2.5], 
        'ER': [0.8, 0.9, 1, 1.1, 1.2],
        'CG': [0.8, 0.9, 1, 1.1, 1.2],
        'CA': [0.8, 0.9, 1, 1.1, 1.2],
        'CF': [],
        'CK': [0.8, 0.9, 1, 1.1, 1.2],
    }
    yoy_pers = yoy_pers_dict[bob_entity_code]
    
    if plots_action == 'show':
        # view ccv discount
        ax = ccv_df['ccv'].plot()
        ccv_df['discount'].plot(ax=ax)
        ax.set_ylim(bottom=0)
        plt.show()
        
        ccv_df.loc[ccv_df['ccv'] == ccv_df['ccv'].max(), 'ccv'] = 177
        ccv_df.loc[ccv_df['discount'] == ccv_df['discount'].max(), 'discount'] = 79
        
        # view referral percent
        ac = mel_mac_agg_df['referral_percent'].plot()
        ax.set_ylim(bottom=0)
        plt.show()
    
    referral_percents = {
        'US': 0.30, 
        'FJ': 0.30, 
        'ER': 0.25,
        'CG': 0.15,
        'CA': 0.30,
        'CF': 0.30,
        'CK': 0.22    
    }
    referral_percent = referral_percents[bob_entity_code]
    
    
    # forecast ccv (sarima) and discount (flat)
    pred_weeks_df = forecast_ccv_discount(
        bob_entity_code=bob_entity_code, ccv_df=ccv_df, pred_weeks_df=pred_weeks_df, hyperparam_search=False, flatten_forecast=True, 
        write_to_gs_bool=write_to_gs_bool, gsheet_url=gsheet_url)
    pred_weeks_df
    
    if plots_action == 'show':
        ax = pred_weeks_df['ccv'].plot()
        pred_weeks_df['discount'].plot(ax=ax)
        ax.set_ylim(bottom=0)
        plt.show()
    
    # forecast mel referral and other (flat)
    pred_weeks_df = other_and_referral_mel_model(
        bob_entity_code=bob_entity_code, mel_df=mel_df, pred_weeks_df=pred_weeks_df, 
        write_to_gs_bool=write_to_gs_bool, gsheet_url=gsheet_url)
    pred_weeks_df
    
    # fit mmm model (saturation, seasonality, trend, and intercept)
    mel_mac_agg_df, saturation_model_trace, channel_scaler, endog_scaler = pymc_paid_total_saturation_model_fit(
        mel_mac_agg_df=mel_mac_agg_df, min_model_week=min_model_week)
    mel_mac_agg_df
    
    if plots_action == 'show':
        ax = mel_mac_agg_df['mmm_conv_trend'].plot()
        mel_mac_agg_df['mmm_conv_seasonality'].plot(ax=ax)
        mel_mac_agg_df['mmm_conv_total'].plot(ax=ax)
        mel_mac_agg_df['n_conversion'].plot(ax=ax)
        ax.legend()
        plt.show()
        
        mel_mac_agg_df['n_conversion'].plot()
        plt.legend()
        plt.show()
    
    
    write_dfs = dict()
    
    total_model_detail_dfs = []
    
    # prescribe spend based upon roi and mmm forecast
    if len(rois) > 0:
        roi_act_pred_df = pymc_paid_total_saturation_model_roi(
            saturation_model_trace=saturation_model_trace, channel_scaler=channel_scaler, endog_scaler=endog_scaler, 
            mel_mac_agg_df=mel_mac_agg_df, pred_weeks_df=pred_weeks_df, 
            trend_flat_continue=trend_flat_continue, trend_flat_week=trend_flat_week, rois=rois, 
            referral_percent=referral_percent)
        total_model_detail_dfs.append(roi_act_pred_df)
        
    
    # forecast spend with yoy percent
    if len(yoy_pers) > 0:
        yoy_act_pred_df, yoy_df = paid_total_yoy_percent_mel_mac(
            saturation_model_trace=saturation_model_trace, channel_scaler=channel_scaler, endog_scaler=endog_scaler, 
            mel_mac_agg_df=mel_mac_agg_df, pred_weeks_df=pred_weeks_df, 
            trend_flat_continue=trend_flat_continue, trend_flat_week=trend_flat_week, yoy_pers=yoy_pers, 
            referral_percent=referral_percent)
        total_model_detail_dfs.append(yoy_act_pred_df)
    
    # forecast spend with time series model
    ts_act_pred_df = paid_total_sarima_mel_mac(
        saturation_model_trace=saturation_model_trace, channel_scaler=channel_scaler, endog_scaler=endog_scaler, 
        mel_mac_agg_df=mel_mac_agg_df, pred_weeks_df=pred_weeks_df, 
        trend_flat_continue=trend_flat_continue, trend_flat_week=trend_flat_week, hyperparam_search=False, 
        referral_percent=referral_percent)
    total_model_detail_dfs.append(ts_act_pred_df)
    
    total_model_detail_df = pd.concat(total_model_detail_dfs, axis=0, ignore_index=True)
    total_model_detail_df = total_model_detail_df[['model'] + [i for i in total_model_detail_df.columns if i not in ('model')]]
    write_dfs['total_model_detail'] = total_model_detail_df
    
    # mel channel mix
    cac_chan_dfs = []
    
    # mac channel mix
    yoy_models = yoy_act_pred_df['model'].unique()
    for yoy_model in yoy_models:
        cac_chan_df = paid_channel_mel_mac_logic(
            mac_df=mac_df, mel_df=mel_df, pred_weeks_df=yoy_act_pred_df[(yoy_act_pred_df['model'] == yoy_model) & (yoy_act_pred_df['time_period'] == 'pred')].copy(deep=True), 
            min_mix_week='2024-W04', max_mix_week='2024-W17', model_name=yoy_model
        )
        cac_chan_dfs.append(cac_chan_df)
    
    ts_models = ts_act_pred_df['model'].unique()
    for ts_model in ts_models:
        cac_chan_df = paid_channel_mel_mac_logic(
            mac_df=mac_df, mel_df=mel_df, pred_weeks_df=ts_act_pred_df[(ts_act_pred_df['model'] == ts_model) & (ts_act_pred_df['time_period'] == 'pred')].copy(deep=True), 
            min_mix_week='2024-W04', max_mix_week='2024-W17', model_name=ts_model
        )
        cac_chan_dfs.append(cac_chan_df)
    
    
    actual_cac_df = actual_channel_cacs(mel_df=mel_df, mac_df=mac_df)
    cac_chan_df = pd.concat(cac_chan_dfs, axis=0, ignore_index=True)
    act_pred_cac_df = pd.concat([actual_cac_df, cac_chan_df], axis=0, ignore_index=True)
    act_pred_cac_df = pd.merge(act_pred_cac_df, dates_df, how='left', on='iso_year_week')
    act_pred_cac_df['model'].fillna('actual', inplace=True)
    act_pred_cac_df = act_pred_cac_df[['model'] + [i for i in act_pred_cac_df.columns if i not in ('model')]]
    write_dfs['cac_channels'] = act_pred_cac_df
    
    quarter_sum_df, group_sum_df = summary_report(total_model_detail_df=total_model_detail_df, act_pred_cac_df=act_pred_cac_df)
    quarter_sum_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    group_sum_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    quarter_sum_df = quarter_sum_df[['model'] + [i for i in quarter_sum_df.columns if i not in ('model')]]
    group_sum_df = group_sum_df[['model'] + [i for i in group_sum_df.columns if i not in ('model')]]
    write_dfs['quarter_summary'] = quarter_sum_df
    write_dfs['group_summary'] = group_sum_df
    
    write_to_gsheet(gsheet_url=gsheet_url, write_dfs_dict=write_dfs)
    
    pass



