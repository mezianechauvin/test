

from dotenv import load_dotenv
load_dotenv(verbose=True, override=True, interpolate=True)
from pathlib import Path
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import lit
import pyspark.pandas as ps
from maulibs.GSheet import GSheet
from datetime import date, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import re
import calendar
import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.dbutils import DBUtils
import pyspark.pandas as ps

print(pd.__version__)
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 100)

spark = SparkSession.builder.getOrCreate()
print('done loading with spark session')


# start with hellofresh us
bob_entity_code = 'US'
working_doc_gsheet_url = 'https://docs.google.com/spreadsheets/d/1bxiXZW4qVWbQYAi1BbqBfzRWmAMkaQ1wSUcwCxSnLd4'
wd_gs_client = GSheet(working_doc_gsheet_url)

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


from feature_exploration import get_mel_spend_data


def facebook_investigation():

    # -----------------------
    # start with facebook
    # -----------------------

    # read in facebook channel hieararchy
    fb_ch_df = wd_gs_client.read_dataframe(worksheet_name='us_mel_facebook_ch', header_row_num=0)
    fb_ch_df = fb_ch_df[vs_ch_cols]
    fb_ch_df.replace('', pd.NA, inplace=True)
    fb_ch_df.head(10)

    fb_ch_sdf = spark.createDataFrame(fb_ch_df)
    fb_ch_sdf.printSchema()
    fb_ch_sdf.createOrReplaceTempView('fb_ch')


    # facebook mel spend data
    fb_mel_query = f'''
    select
        dd.iso_year_week, 
        dd.year, 
        dd.month, 
        dd.day_of_month, 
        dd.date_string_backwards date, 
        mel.bob_entity_code,
        trim(mel.vs_campaign_type) campaign_type, 
        trim(mel.vs_channel_medium) channel_medium, 
        trim(mel.vs_channel_category) channel_category, 
        trim(mel.vs_channel) channel, 
        trim(mel.vs_channel_split) channel_split, 
        trim(mel.vs_partner) partner, 
        sum(mel.mktg_spend_usd) mktg_spend, 
        sum(mel.agency_fees) agency_fees
    from marketing_analytics_us.marketing_expense_log mel
    join dimensions.date_dimension dd on mel.fk_date = dd.sk_date
    join fb_ch on mel.vs_campaign_type <=> fb_ch.vs_campaign_type 
        and mel.vs_channel_medium <=> fb_ch.vs_channel_medium 
        and mel.vs_channel_category <=> fb_ch.vs_channel_category 
        and mel.vs_channel <=> fb_ch.vs_channel 
        and mel.vs_channel_split <=> fb_ch.vs_channel_split 
        and mel.vs_partner <=> fb_ch.vs_partner
    where mel.bob_entity_code = 'US'
    group by 1,2,3,4,5,6,7,8,9,10,11,12
    '''
    fb_mel_df = spark.sql(fb_mel_query).toPandas()
    print(fb_mel_df)
    fb_mel_df['date'] = pd.to_datetime(fb_mel_df['date'])
    fb_mel_df

    fb_mel_df['channel_split'].value_counts(dropna=False)
    fb_mel_df['partner'].value_counts(dropna=False)

    daily_mel_fb_spend_df = (fb_mel_df.groupby(mel_plat_ch_cols + ['iso_year_week', 'date'], as_index=False, dropna=False)
                            .agg({'mktg_spend': 'sum', 'agency_fees': 'sum'})
                            .sort_values(['date'], ascending=[True])
                            .reset_index(drop=True))
    daily_mel_fb_spend_df


    # facebook platform spend data
    fb_plat_query = f'''
    select
        dd.iso_year_week, 
        dd.year, 
        dd.month, 
        dd.day_of_month, 
        dd.date_string_backwards date, 
        fb.bob_entity_code,
        trim(fb.vs_campaign_type) campaign_type,
        trim(fb.vs_channel) channel, 
        trim(fb.vs_channel_split) channel_split,
        trim(fb.vs_partner) partner,
        sum(total_spend) fb_total_spend,
        sum(impressions) fb_impressions,
        sum(clicks) fb_clicks,
        sum(platform_conversions) fb_platform_conversions
    from marketing_analytics_us.facebook_ads_platform_metrics_daily_view fb
    join dimensions.date_dimension dd on fb.fk_date = dd.sk_date
    where bob_entity_code = '{bob_entity_code}'
    group by 1,2,3,4,5,6,7,8,9,10
    '''
    fb_plat_df = spark.sql(fb_plat_query).toPandas()
    fb_plat_df['date'] = pd.to_datetime(fb_plat_df['date'])
    fb_plat_df


    daily_plat_fb_spend_df = (fb_plat_df.groupby(mel_plat_ch_cols + ['iso_year_week', 'date'], as_index=False, dropna=False)
                            .agg({'fb_total_spend': 'sum', 'fb_impressions': 'sum'}))

    # compare hierarchies
    mel_uni_chans_df = fb_mel_df[mel_plat_ch_cols].drop_duplicates()
    fb_plat_uni_chans_df = fb_plat_df[mel_plat_ch_cols].drop_duplicates()
    pd.merge(mel_uni_chans_df, fb_plat_uni_chans_df, how='outer', on=mel_plat_ch_cols, indicator=True).to_clipboard(index=False)


    # merge and compare daily spend data
    daily_mel_fb_spend_df
    daily_plat_fb_spend_df
    mer_fb_spend_df = (pd.merge(daily_mel_fb_spend_df, daily_plat_fb_spend_df, how='outer', on=mel_plat_ch_cols + ['iso_year_week', 'date']))
    mer_fb_spend_df

    mer_fb_spend_fil_df = mer_fb_spend_df[mer_fb_spend_df['fb_total_spend'].notna()]
    mer_fb_spend_fil_df
    mer_week_agg_df = (mer_fb_spend_fil_df.groupby(mel_plat_ch_cols + ['iso_year_week'], as_index=False, dropna=False)
                    .agg({'mktg_spend': 'sum', 'agency_fees': 'sum', 'fb_total_spend': 'sum', 'fb_impressions': 'sum'}))

    spend_ranking_df = (mer_week_agg_df.groupby(mel_plat_ch_cols, as_index=False, dropna=False)
                        .agg({'fb_total_spend': 'sum'})
                        .sort_values('fb_total_spend', ascending=False)
                        .reset_index(drop=True))

    spend_ranking_df['rank'] = spend_ranking_df.index + 1
    spend_ranking_df
    mer_week_agg_df = pd.merge(mer_week_agg_df, spend_ranking_df, how='left', on=mel_plat_ch_cols)
    mer_week_agg_df.sort_values(['rank', 'iso_year_week'], ascending=[True, True], inplace=True)
    mer_week_agg_df.to_clipboard(index=False)


    # -----------------------
    # just start with in-house facebook spend by day, campaign_name, breakdown_fields etc.
    # -----------------------

    # read in facebook channel hieararchy
    fb_ch_df = wd_gs_client.read_dataframe(worksheet_name='us_mel_in_house_facebook_ch', header_row_num=0)
    fb_ch_df = fb_ch_df[vs_ch_cols]
    fb_ch_df.replace('', pd.NA, inplace=True)
    fb_ch_df.head(10)

    fb_ch_sdf = spark.createDataFrame(fb_ch_df)
    fb_ch_sdf.printSchema()
    fb_ch_sdf.createOrReplaceTempView('fb_ch')


    # facebook mel spend data
    fb_mel_query = f'''
    select
        dd.iso_year_week, 
        dd.date_string_backwards date, 
        mel.bob_entity_code,
        trim(mel.vs_campaign_type) campaign_type, 
        trim(mel.vs_channel_medium) channel_medium, 
        trim(mel.vs_channel_category) channel_category, 
        trim(mel.vs_channel) channel, 
        trim(mel.vs_channel_split) channel_split, 
        trim(mel.vs_partner) partner, 
        trim(mel.campaign_name) campaign_name,
        breakdown_value1, 
        breakdown_value2,
        breakdown_value3, 
        breakdown_value4,
        breakdown_value5,
        cost_distribution,
        cac_spend, 
        accrued, 
        written_off, 
        finance_vendor, 
        sum(mel.mktg_spend_usd) mktg_spend, 
        sum(mel.agency_fees) agency_fees
    from marketing_analytics_us.marketing_expense_log mel
    join dimensions.date_dimension dd on mel.fk_date = dd.sk_date
    join fb_ch on mel.vs_campaign_type <=> fb_ch.vs_campaign_type 
        and mel.vs_channel_medium <=> fb_ch.vs_channel_medium 
        and mel.vs_channel_category <=> fb_ch.vs_channel_category 
        and mel.vs_channel <=> fb_ch.vs_channel 
        and mel.vs_channel_split <=> fb_ch.vs_channel_split 
        and mel.vs_partner <=> fb_ch.vs_partner
    where mel.bob_entity_code = 'US'
    group by 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
    '''
    fb_mel_df = spark.sql(fb_mel_query).toPandas()
    fb_mel_df['date'] = pd.to_datetime(fb_mel_df['date'])
    fb_mel_df

    # facebook platform spend data
    fb_plat_query = f'''
    select
        dd.iso_year_week, 
        dd.year, 
        dd.month, 
        dd.day_of_month, 
        dd.date_string_backwards date, 
        fb.bob_entity_code,
        trim(fb.vs_campaign_type) campaign_type,
        trim(fb.vs_channel) channel, 
        trim(fb.vs_channel_split) channel_split,
        trim(fb.vs_partner) partner,
        raw_platform_campaign_name, 
        sum(total_spend) fb_total_spend,
        sum(impressions) fb_impressions,
        sum(clicks) fb_clicks,
        sum(platform_conversions) fb_platform_conversions
    from marketing_analytics_us.facebook_ads_platform_metrics_daily_view fb
    join dimensions.date_dimension dd on fb.fk_date = dd.sk_date
    where bob_entity_code = '{bob_entity_code}'
    group by 1,2,3,4,5,6,7,8,9,10,11
    '''
    fb_plat_df = spark.sql(fb_plat_query).toPandas()
    fb_plat_df['date'] = pd.to_datetime(fb_plat_df['date'])
    fb_plat_df[(fb_plat_df['month'] == 9) & (fb_plat_df['year'] == 2023)].to_clipboard(index=False)



    # aggregate by internal only by week, then figure out which weeks to investigate (more recent the better)
    fb_mel_df

    fb_mel_df.groupby(['finance_vendor', 'partner'], as_index=False, dropna=False).agg({'mktg_spend': 'sum', 'agency_fees': 'sum'}).sort_values('mktg_spend', ascending=False).head(20)

    mel_plat_ch_cols

    weekly_mel_fb_spend_df = (fb_mel_df.groupby(mel_plat_ch_cols + ['iso_year_week'], as_index=False, dropna=False)
                            .agg({'mktg_spend': 'sum', 'agency_fees': 'sum'})
                            .sort_values(['iso_year_week'], ascending=[True])
                            .reset_index(drop=True))
    weekly_mel_fb_spend_piv_df = weekly_mel_fb_spend_df.pivot(index='iso_year_week', columns=mel_plat_ch_cols, values='mktg_spend').fillna(0).reset_index()
    weekly_mel_fb_spend_piv_df

    new_mel_cols = []
    for col in weekly_mel_fb_spend_piv_df.columns:
        if col[0] != 'iso_year_week':
            new_mel_cols.append(' '.join([i for i in col if isinstance(i, str)]).strip() + ' MEL')
        else:
            new_mel_cols.append(col[0])
    new_mel_cols
    weekly_mel_fb_spend_piv_df.columns = new_mel_cols
    weekly_mel_fb_spend_piv_df


    weekly_plat_fb_spend_df = (fb_plat_df.groupby(mel_plat_ch_cols + ['iso_year_week'], as_index=False, dropna=False)
                            .agg({'fb_total_spend': 'sum'})
                            .sort_values(['iso_year_week'], ascending=[True])
                            .reset_index(drop=True))
    weekly_plat_fb_spend_piv_df = weekly_plat_fb_spend_df.pivot(index='iso_year_week', columns=mel_plat_ch_cols, values='fb_total_spend').fillna(0).reset_index()
    weekly_plat_fb_spend_piv_df

    new_plat_cols = []
    for col in weekly_plat_fb_spend_piv_df.columns:
        if col[0] != 'iso_year_week':
            new_plat_cols.append(' '.join([i for i in col if isinstance(i, str)]).strip() + ' PLAT')
        else:
            new_plat_cols.append(col[0])
    weekly_plat_fb_spend_piv_df.columns = new_plat_cols
    weekly_plat_fb_spend_piv_df


    weekly_mel_plat_df = pd.merge(weekly_mel_fb_spend_piv_df, weekly_plat_fb_spend_piv_df, how='outer', on='iso_year_week')
    weekly_mel_plat_df.to_clipboard(index=False)

    (fb_plat_df[fb_plat_df['iso_year_week'] == '2023-W37'].groupby(mel_plat_ch_cols + ['raw_platform_campaign_name'], as_index=False, dropna=False)
    .agg({'fb_total_spend': 'sum'})
    .sort_values('fb_total_spend', ascending=False).to_clipboard(index=False))

    (fb_mel_df[fb_mel_df['iso_year_week'] == '2023-W37'].groupby(mel_plat_ch_cols + ['cac_spend', 'written_off'], as_index=False, dropna=False)
    .agg({'mktg_spend': 'sum', 'agency_fees': 'sum'})
    .sort_values('mktg_spend', ascending=False).to_clipboard(index=False))


    fb_mel_df[(fb_mel_df['iso_year_week'] == '2023-W37') & (fb_mel_df['mktg_spend'] < 0)]
    fb_mel_df[(fb_mel_df['iso_year_week'] == '2023-W37')]

    fb_mel_df[(fb_mel_df['iso_year_week'].isin(('2023-W35', '2023-W36', '2023-W37', '2023-W38', '2023-W39'))) & (fb_mel_df['mktg_spend'] < 0)]
    (fb_mel_df[(fb_mel_df['mktg_spend'] < 0)].groupby(['iso_year_week', 'partner'], as_index=False)
    .agg({'mktg_spend': 'sum'}).to_clipboard(index=False))
    fb_mel_df



    # just 2023-W37 investigation
    fb_mel_df
    mel_dets_cols = ['campaign_name', 'breakdown_value1', 'breakdown_value2', 'breakdown_value3', 'breakdown_value4', 'breakdown_value5', 'cost_distribution', 'cac_spend', 'accrued', 'written_off', 'finance_vendor']
    (fb_mel_df[(fb_mel_df['iso_year_week'] == '2023-W37')].groupby(['iso_year_week'] + mel_plat_ch_cols + mel_dets_cols, dropna=False, as_index=False)
    .agg({'mktg_spend': 'sum', 'agency_fees': 'sum'})).to_clipboard(index=False)

    plat_dets_cols = ['raw_platform_campaign_name']
    (fb_plat_df[(fb_plat_df['iso_year_week'] == '2023-W37')].groupby(['iso_year_week'] + mel_plat_ch_cols + plat_dets_cols, dropna=False, as_index=False)
    .agg({'fb_total_spend': 'sum'})).to_clipboard(index=False)


    # facebook mel details

    # facebook mel spend data
    new_fb_mel_query = f'''
    select
        dd.iso_year_week, 
        dd.date_string_backwards date, 
        mel.expense_log_id, 
        mel.bob_entity_code,
        trim(mel.vs_campaign_type) campaign_type, 
        trim(mel.vs_channel_medium) channel_medium, 
        trim(mel.vs_channel_category) channel_category, 
        trim(mel.vs_channel) channel, 
        trim(mel.vs_channel_split) channel_split, 
        trim(mel.vs_partner) partner, 
        trim(mel.campaign_name) campaign_name,
        breakdown_value1, 
        breakdown_value2,
        breakdown_value3, 
        breakdown_value4,
        breakdown_value5,
        cost_distribution,
        cac_spend, 
        accrued, 
        written_off, 
        finance_vendor, 
        sum(mel.mktg_spend_usd) mktg_spend, 
        sum(mel.agency_fees) agency_fees
    from marketing_analytics_us.marketing_expense_log mel
    join dimensions.date_dimension dd on mel.fk_date = dd.sk_date
    where mel.bob_entity_code = 'US' and dd.iso_year_week = '2023-W37'
        and (lower(mel.vs_channel) ilike '%meta%' or lower(mel.vs_channel) ilike '%facebook%'
            or lower(mel.vs_channel_split) ilike '%meta%' or lower(mel.vs_channel_split) ilike '%facebook%'
            or lower(mel.vs_partner) ilike '%meta%' or lower(mel.vs_partner) ilike '%facebook%'
            or lower(mel.finance_vendor) ilike '%meta%' or lower(mel.finance_vendor) ilike '%facebook%')
    group by 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21
    '''
    new_mel_df = spark.sql(new_fb_mel_query).toPandas()
    new_mel_df['date'] = pd.to_datetime(new_mel_df['date'])
    new_mel_df

    new_mel_df.to_clipboard(index=False)

    fb_mel_df[(fb_mel_df['iso_year_week'] == '2023-W37')].to_clipboard(index=False)



    # facebook mel spend data
    sept_fb_mel_query = f'''
    select
        dd.iso_year_week, 
        dd.date_string_backwards date, 
        mel.expense_log_id, 
        mel.bob_entity_code,
        trim(mel.vs_campaign_type) campaign_type, 
        trim(mel.vs_channel_medium) channel_medium, 
        trim(mel.vs_channel_category) channel_category, 
        trim(mel.vs_channel) channel, 
        trim(mel.vs_channel_split) channel_split, 
        trim(mel.vs_partner) partner, 
        trim(mel.campaign_name) campaign_name,
        breakdown_value1, 
        breakdown_value2,
        breakdown_value3, 
        breakdown_value4,
        breakdown_value5,
        cost_distribution,
        cac_spend, 
        accrued, 
        written_off, 
        finance_vendor, 
        sum(mel.mktg_spend_usd) mktg_spend, 
        sum(mel.agency_fees) agency_fees
    from marketing_analytics_us.marketing_expense_log mel
    join dimensions.date_dimension dd on mel.fk_date = dd.sk_date
    where dd.iso_year_week >= '2023-W34' and dd.iso_year_week <= '2023-W40'
        and (lower(mel.vs_channel) ilike '%meta%' or lower(mel.vs_channel) ilike '%facebook%'
            or lower(mel.vs_channel_split) ilike '%meta%' or lower(mel.vs_channel_split) ilike '%facebook%'
            or lower(mel.vs_partner) ilike '%meta%' or lower(mel.vs_partner) ilike '%facebook%'
            or lower(mel.finance_vendor) ilike '%meta%' or lower(mel.finance_vendor) ilike '%facebook%')
    group by 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21
    '''
    sept_fb_mel_df = spark.sql(sept_fb_mel_query).toPandas()
    sept_fb_mel_df['date'] = pd.to_datetime(sept_fb_mel_df['date'])
    sept_fb_mel_df

    sept_fb_mel_df['year'] = sept_fb_mel_df['date'].dt.year
    sept_fb_mel_df['month'] = sept_fb_mel_df['date'].dt.month
    sept_fb_mel_df['day_of_month'] = sept_fb_mel_df['date'].dt.day
    sept_fb_mel_df.to_clipboard(index=False)








    def fb_one_query_rushi():
        # one query

        platform_spend_query = f'''
        SELECT
            ed.bob_entity_code AS country
            , dd.iso_year_week
            , fb.vs_channel
            , fb.vs_channel_split
            , fb.vs_partner
            , SUM(fb.total_spend) AS platform_spend
        FROM
            marketing_analytics_us.facebook_ads_platform_metrics_daily_view AS fb
            INNER JOIN dimensions.date_dimension AS dd
            ON fb.fk_date = dd.sk_date
            LEFT JOIN dimensions.entity_dimension AS ed
            ON fb.fk_entity = ed.sk_entity
        GROUP BY 1, 2, 3, 4, 5
        '''
        platform_spend_sdf = spark.sql(platform_spend_query)
        platform_spend_sdf.createOrReplaceTempView('platform_spend')

        mel_spend_query = '''
        SELECT
            mel.bob_entity_code AS country
            , dd.iso_year_week
            , mel.vs_channel
            , mel.vs_channel_split
            , mel.vs_partner
            , SUM(mel.mktg_spend_local_cur) AS mel_spend
        FROM
            marketing_analytics_us.marketing_expense_log AS mel
        INNER JOIN dimensions.date_dimension AS dd
            ON mel.fk_date = dd.sk_date
        WHERE
            -- (mel.vs_channel ILIKE '%facebook%'
            -- OR mel.vs_channel_split ILIKE '%facebook%'
            -- OR mel.vs_partner ILIKE '%facebook%')
            mel.vs_channel LIKE '%Paid%Social%'
            AND mel.vs_partner ILIKE 'Facebook'
            AND mel.bob_entity_code = 'US'
            AND dd.iso_year_week BETWEEN '2023-W01' AND '2023-W52'
            -- AND mel.vs_channel_split NOT LIKE '%Non-Facebook%'
        GROUP BY 1, 2, 3, 4, 5
        '''
        mel_spend_sdf = spark.sql(mel_spend_query)
        mel_spend_sdf.createOrReplaceTempView('mel_spend')


        one_fb_query = f'''
        SELECT
        platform_spend.country
        , platform_spend.iso_year_week
        , platform_spend.vs_channel
        , platform_spend.vs_channel_split
        , platform_spend.vs_partner
        , platform_spend.platform_spend
        , mel_spend.mel_spend
        , mel_spend.mel_spend - platform_spend.platform_spend AS mel_minus_platform
        , platform_spend.platform_spend / mel_spend.mel_spend AS platform_over_mel
        FROM platform_spend
        INNER JOIN mel_spend
        ON platform_spend.country = mel_spend.country
        AND platform_spend.iso_year_week = mel_spend.iso_year_week
        AND platform_spend.vs_channel = mel_spend.vs_channel
        AND platform_spend.vs_channel_split = mel_spend.vs_channel_split
        AND platform_spend.vs_partner = mel_spend.vs_partner
        AND platform_spend.country = 'US'
        ORDER BY 1, 2, 3, 4;
        '''
        one_df = spark.sql(one_fb_query).toPandas()
        one_df.to_clipboard(index=False)
        
        pass

def seo():
    
    # seo
    
    seo_ch_df = wd_gs_client.read_dataframe(worksheet_name='us_mel_seo', header_row_num=0)
    seo_ch_df
    df = get_mel_spend_data(bob_entity_code=bob_entity_code, channel_hierarchy_df=seo_ch_df, min_max_iso_year_week=('2019-W01', '2024-W04'))
    df
    
    df['year'] = df['date'].dt.year
    df_agg = df.groupby(['year'] + ch_cols, as_index=False, dropna=False).agg({'mel_spend': 'sum', 'mel_agency_fees': 'sum'})
    df_agg.to_clipboard(index=False)
    
    
    
    
    
    
    
    
    
    pass

def current_features():
    
    from pyspark.dbutils import DBUtils
    
    s3_bucket = 'data-science-mau'
    bob_entity_code = 'US'
    
    dbutils = DBUtils(spark)
    dbutils.fs.ls('/mnt')
    dbutils.fs.ls('/mnt/data-science-mau-dev/')
    dbutils.fs.ls('/mnt/data-science-mau-dev/analyses/mmm_assets/mmm_main_model/')
    
    
    marketing_forecasting_features = spark.read.csv(f"/mnt/{s3_bucket}/analyses/mmm_assets/processed/country={bob_entity_code}", header=True).toPandas()
    marketing_forecasting_features
    
    marketing_forecasting_features['week_date'] = pd.to_datetime(marketing_forecasting_features['week_date'])
    marketing_forecasting_features['week_date'].max()
    
    wd_gs_client.write_dataframe(worksheet_name='Current Features', dataframe=marketing_forecasting_features)
    
    bucket_name = '/mnt/data-science-mau/analyses/mmm_assets/processed/country='
    brand_name = 'US'
    
    df_raw = spark.read.csv(bucket_name + brand_name + '/marketing_forecasting_features_dataset.csv', header='true', inferSchema="true").toPandas()
    df_covid = spark.read.csv(bucket_name + brand_name + '/covid_national_table.csv', header='true', inferSchema="true").toPandas()
    df_gt = spark.read.csv(bucket_name + brand_name + '/Google Trends.csv', header='true', inferSchema="true").toPandas()
    df_raw = df_raw.loc[df_raw['is_past'] == True]
    
    df_raw
    
    df_raw['week_date'] = pd.to_datetime(df_raw['week_date'])
    df_raw['week_date'].min()
    df_raw['week_date'].max()
    df_covid
    df_gt
    df_raw
    
    
    HF_channel_list = ['SEM_PERFORMANCE',	'NATIVE',	'DISPLAY',	'OFFLINE_OTHER_LOWER',	'PAID_SOCIAL_FB',	'PAID_SOCIAL_NFB',	'TV_LINEAR',	'TV_DIGITAL',	'RADIO',	'DIRECT_MAIL',	'INFLUENCER',	'OFFLINE_OTHER_UPPER',	'VIDEO',	'PODCAST',	'LOWER_FUNNEL_RS']
    from pprint import pprint
    
    pprint(HF_channel_list)

    
    
                                                    
    
    
    
    pass

def meta_mmm_data_to_s3():
    
    download_path = Path('/Users/eddiedeane/Downloads')
    
    year_lst = [2021, 2022, 2023, 2024]
    
    for year in year_lst:
        df = pd.read_csv(download_path / 'HF-US-MMM-Data-ED-LZ-_2021.csv')
        df
        # write to s3
        
        dbutils = DBUtils(spark)
        dbutils.fs.mkdirs('/mnt/data-science-mau/analyses/mmm_assets/geoMMM/meta/')
        dbutils.fs.ls('s3://data-science-mau/analyses/mmm_assets/geoMMM/meta')
        dbutils.fs.cp('file:/Users/eddiedeane/Downloads/HF-US-MMM-Data-ED-LZ-_2021.csv', 's3://data-science-mau/analyses/mmm_assets/geoMMM/meta/HF-US-MMM-Data-ED-LZ-_2021.csv')
        
        sdf = spark.createDataFrame(df)
        sdf.write.csv('s3://data-science-mau/analyses/mmm_assets/geoMMM/meta/HF-US-MMM-Data-ED-LZ-_2021.csv')
        
        
        
        's3://data-science-mau/analyses/mmm_assets/geoMMM/meta/'
    
    
    
    pass

def fix_channel_hierarchies():
    
    us_mel_channel_hierarchy_df = get_channel_hierarchy(bob_entity_code='US', min_max_date='', min_max_iso_year_week='')
    us_mel_channel_hierarchy_df
    wd_gs_client.write_dataframe(worksheet_name='us_mel_channel_hierarchy', dataframe=us_mel_channel_hierarchy_df)
    
    
    us_mel_jc_channel_hierarchy_df = create_jc_rollup_mel_channel_grouping(bob_entity_code='US')
    us_mel_jc_channel_hierarchy_df['group'] = us_mel_jc_channel_hierarchy_df['jc_campaign_type'] + ' - ' + us_mel_jc_channel_hierarchy_df['jc_channel'] + ' - ' + us_mel_jc_channel_hierarchy_df['jc_channel_split_partner']
    us_mel_jc_channel_hierarchy_df = pd.merge(us_mel_jc_channel_hierarchy_df, us_mel_channel_hierarchy_df.drop(columns=['mel_spend', 'mel_agency_fees']), how='left', on=ch_cols)
    wd_gs_client.write_dataframe(worksheet_name='us_mel_jc_channel_hierarchy', dataframe=us_mel_jc_channel_hierarchy_df)
    
    us_mel_cj_channel_hierarchy_df = wd_gs_client.read_dataframe('us_mel_cj_channel_hierarchy', header_row_num=0)
    us_mel_cj_channel_hierarchy_df.drop(columns=['bob_entity_code', 'mktg_spend', 'agency_fees', 'total_spend', 'percent_of_total'], inplace=True)
    us_mel_cj_channel_hierarchy_df
    
    def clean_channel_hierarchy(df, add_cols=['spend_category', 'group']):
        pot_ch_cols = ['campaign_type', 'channel_medium', 'channel_category', 'channel', 'channel_split', 'partner']
        df.columns = [col.replace('vs_', '').lower() for col in df.columns]
        ch_df_cols_lst = df.columns
        df_ch_filter_lst = [i for i in pot_ch_cols if i in ch_df_cols_lst]
        df = df[df_ch_filter_lst + add_cols].drop_duplicates().replace('', pd.NA).reset_index(drop=True)
        df = df.apply(lambda x: x.str.strip().str.lower())
        
        return df
    
    us_mel_cj_channel_hierarchy_df = clean_channel_hierarchy(us_mel_cj_channel_hierarchy_df, add_cols=['spend_category', 'group'])
    us_mel_cj_channel_hierarchy_df
    us_mel_cj_channel_hierarchy_cp_df = pd.merge(us_mel_channel_hierarchy_df, us_mel_cj_channel_hierarchy_df, how='left', on=ch_cols)
    us_mel_cj_channel_hierarchy_cp_df[us_mel_cj_channel_hierarchy_cp_df['group'].isna()]
    wd_gs_client.write_dataframe(worksheet_name='us_mel_cj_channel_hierarchy', dataframe=us_mel_cj_channel_hierarchy_cp_df)
    
    
    pass

def meta_mmm_investigation():
    
    data_path = Path('raw_data/')
    meta_df = pd.read_csv(data_path / 'HF-US-MMM-Data-ED-LZ-_2024.csv')
    meta_df
    meta_df.groupby(['Platform', 'Placement', 'Media type'], dropna=False, as_index=False).agg({'Impressions': 'sum', 
                                                                  'Amount spent': 'sum'}).to_clipboard()
    
    meta_df.groupby(['Platform', 'Media type'], dropna=False, as_index=False).agg({'Impressions': 'sum', 
                                                                  'Amount spent': 'sum'}).to_clipboard()
    
    
    pass

def seo_mmm_investigation():
    
    search_summary_query = f'''
    select 
        dd.iso_year_week, 
        case when ahref.url = 'https://www.hellofresh.com/' then 'hellofresh' else 'hellofresh_fil' end entity_code,
        count(distinct gs.query) n_queries, 
        avg(gs.position) avg_position, 
        sum(gs.impressions) tot_impr,
        sum(gs.position * gs.impressions) / sum(gs.impressions) weighted_avg_position
    from marketing_analytics_us.google_search_console_daily gs
    join global_bi_business.date_dimension dd on gs.date = dd.date_string_backwards
    join seo_ahref_keywords ahref on gs.query = ahref.keyword and gs.entity_code = ahref.bob_entity_code
    where gs.entity_code = 'US'
    group by 1, 2
    order by date asc
    '''
    search_summary_fil_df = spark.sql(search_summary_query).toPandas()
    search_summary_fil_df['date'] = pd.to_datetime(search_summary_fil_df['date'])
    search_summary_fil_df.rename(columns={'n_queries': 'n_queries_hf', 'weighted_avg_position': 'weighted_avg_position_hf', 'avg_position': 'avg_position_hf'}, inplace=True)
    
    pass

def plat_tables_info():
    
    plat_dfs = []
    for plat_table_name in platform_tables_df['table_name']:
        plat_desc_query = f'describe marketing_analytics_us.{plat_table_name}'
        plat_desc_df = spark.sql(plat_desc_query).toPandas()
        plat_ch_cols = [i for i in plat_desc_df['col_name'].values if i in vs_ch_cols + ch_cols]
        plat_ch_cols_sql = ', '.join(plat_ch_cols)
        plat_query = f'''
        select
            distinct {plat_ch_cols_sql}
        from marketing_analytics_us.{plat_table_name} plat
        where bob_entity_code = '{bob_entity_code}'
        '''
        print(plat_query)
        plat_df = spark.sql(plat_query).toPandas()
        plat_df['table_name'] = plat_table_name
        plat_dfs.append(plat_df)
        
    pd.concat(plat_dfs, axis=0, ignore_index=True).to_clipboard(index=False)
    
    
    
    pass












