#!/usr/bin/env python
# app_update.py


#############################
#!/usr/bin/env python
# db_conn.py

import psycopg2
import pymysql
import sys
from functools import reduce
import pandas as pd
from datetime import datetime, date, timedelta
from dateutil.relativedelta import *
from calendar import monthrange
import pickle
import os.path


class PostGreSQLConnection:
    def __init__(self, dbname, host, port, user, password):
        self.connection = None
        self.dbname = dbname
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        
    def __enter__(self):
        self.connection = psycopg2.connect(
                dbname = self.dbname,
                host = self.host,
                port = self.port,
                user = self.user,
                password = self.password
        )
        return self.connection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.commit()
        self.connection.close()
        
        
class MySQLConnection:
    def __init__(self, db, host, port, user, password):
        self.connection = None
        self.db = db
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        
    def __enter__(self):
        self.connection = pymysql.connect(
                db = self.db,
                host = self.host,
                port = self.port,
                user = self.user,
                password = self.password
        )
        return self.connection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.connection.commit()
        self.connection.close()
        
####################################################################################################################

#!/usr/bin/env python
# database.py

import getpass
import pandas as pd
from datetime import datetime
dbname = 'bids'
host1 = 'ignite-bid-cluster.ce0ui3hfe3ue.us-west-2.redshift.amazonaws.com'
port1 = '5439'
user = 'jasonhsiao'
password1 = 'r8MiC08x07@Z'
db = 'fuelAsset'
host2 = 'fuelasset.db.fuel451.com'
port2 = 3306
user = 'jasonhsiao'
password2 = 'y1U$hcSPVsQW'

def get_multiple_tbs(param_hr):
    """
    rtype: pandas dataframe
    """

    with PostGreSQLConnection(dbname, host1, port1, user, password1) as conn1:
        with MySQLConnection(db, host2, port2, user, password2) as conn2:
            print('connections successful')
            sql_query = """
                    SELECT /* update on 6/24, add two cols, adtype & ctype */
                        tbl1.bid,
                        tbl1.cid,
                        tbl1.date,
                        tbl1.adtype,
                        tbl1.ctype,
                        tbl1.imp,
                        tbl1.spd,
                        tbl1.clk,
                        tbl1.wclk,
                      --  tbl1.ctr,
                        tbl1.vtc,
                        tbl1.ctc,
                        tbl1.tc,
                        tbl1.tc_ov,
                        tbl1.cost,
                        tbl2.vtc_windows,
                        tbl2.ctc_windows,
                        tbl2.campaign_type,
                        tbl2.is_lower_funnel,
                        tbl4.dailylimit
                    FROM (
                        SELECT /* get adx performance data */
                            date,
                            cast(bid as int) as bid,
                            cast(cid as int) as cid,
                            cast(adtype as int) as adtype,
                            cast(ctype as int) as ctype,
                            sum(imp) as imp,
                            sum(spd) as spd,
                            sum(clk) as clk,
                            sum(wclk) as wclk,
                           -- sum(clk)/sum(imp) as ctr,
                            sum(vtc) as vtc,
                            sum(ctc) as ctc,
                            sum(tc) as tc,
                            sum(tc_ov) as tc_ov,
                            sum(cost) as cost
                        FROM adxgenericreportd
                        GROUP BY 1, 2, 3, 4, 5
                        ORDER BY 1 DESC
                         ) as tbl1
                         
                    INNER JOIN (
                        SELECT /* get funnel data on 5/2/18 */
                            tb1.date,
                            tb1.bid,
                            tb1.cid,
                            tb1.vtc_windows,
                            tb1.ctc_windows,
                            tb3.campaign_type,
                            tb3.is_lower_funnel
                        FROM (
                            SELECT *
                            FROM daily_campaign_status
                            WHERE 1 = 1
                            AND status = 1
                             ) as tb1
                        LEFT JOIN campaign_conversion_attribution as tb2
                        ON 1 = 1
                        AND tb1.bid = tb2.bid
                        
                        LEFT JOIN (
                            select
                                bid,
                                id as "cid",
                                case when type = 1 then 1
                                      else 0 end as is_lower_funnel,
                                case when type = 1 then 'lower_funnel'
                                     when type = 2 then 'top_funnel'
                                     else NULL end as campaign_type,
                                to_char(insertion_timestamp, 'YYYY-mm-dd') as date
                            FROM campaignlog
                                  ) as tb3
                        ON (tb1.bid = tb3.bid and tb1.cid = tb3.cid and tb1.date = tb3.date)
                              ) as tbl2
                    ON 1 = 1
                    AND tbl1.bid = tbl2.bid
                    AND tbl1.cid = tbl2.cid
                    AND tbl1.date = tbl2.date
                    
                    INNER JOIN (
                        SELECT /*get dailylimit info on 5/2/18*/
                            c.cid,
                            c.bid,
                            c.dailylimit,
                            date(c.insertion_timestamp)
                        FROM (
                                SELECT
                                    datesk,
                                    id as cid,
                                    bid,
                                    dailylimit,
                                    insertion_timestamp
                                FROM campaignlog
                                where 1 = 1
                                AND CAST(datepart(hour,insertion_timestamp) AS CHAR(2)) = {}
                                AND status = 1
                             ) as c
                        INNER JOIN (
                                SELECT
                                    id,
                                    MAX(insertion_timestamp) AS insertion_timestamp
                                FROM campaignlog
                                GROUP BY
                                    id,
                                    datesk
                                   ) as t
                        ON 1 = 1
                        AND c.cid=t.id
                        AND c.insertion_timestamp=t.insertion_timestamp
                              ) as tbl4
                    ON 1 = 1
                    AND tbl1.bid = tbl4.bid
                    AND tbl1.cid = tbl4.cid
                    AND tbl1.date = tbl4.date
                    WHERE tbl1.date >= '2017-04-01'
                """.format(param_hr)
            
            df1 = pd.read_sql(sql_query, conn1)
            df1.to_csv('/home/jhsiao_fuelx_com/Projects/p-p-balancer-project/data/adx_funnel_combo_v2_{}.csv'.format(datetime.now().date()),index=False)
            #df1.to_csv('adx_funnel_combo_{}.csv'.format(datetime.now().date()), index=False)
            #df1.to_csv('data/adx_funnel_combo_{}.csv'.format(datetime.now().date()), index=False)
            print('Done querying and save to the local')
            
            sql_query = """
                    SELECT /* get objective info on 5/3 */
                        bid,
                        max(objective) as objective
                    FROM campaign_conversion_attribution
                    GROUP BY bid
                """
            df2 = pd.read_sql(sql_query, conn1)
	    #df2.to_csv('data/objective_{}.csv'.format(datetime.now().date()),index=False)
            #df2.to_csv('objective_{}.csv'.format(datetime.now().date()), index=False)
            df2.to_csv('/home/jhsiao_fuelx_com/Projects/p-p-balancer-project/data/objective_v2_{}.csv'.format(datetime.now().date()), index=False)
            
            sql_query = """
                    SELECT /* get roi_goal info on 5/2/18 */
                        distinct a.bid,
                        a.roi_goal
                    FROM campaign_conversion_attribution_log as a
                    LEFT JOIN campaign_conversion_attribution_log as b
                    ON 1 = 1
                    AND a.bid = b.bid
                    AND a.effective_date < b.effective_date
                    WHERE b.effective_date is NULL
                """
            df3 = pd.read_sql(sql_query, conn2)
	    #df3.to_csv('data/roi_goal_{}.csv'.format(datetime.now().date()),index=False)
            #df3.to_csv('roi_goal_{}.csv'.format(datetime.now().date()), index=False)
            df3.to_csv('/home/jhsiao_fuelx_com/Projects/p-p-balancer-project/data/roi_goal_v2_{}.csv'.format(datetime.now().date()),index=False)
            
    return df1, df2, df3

####################################################################################################################

#!/usr/bin/env python
# data_transformation.py

import pandas as pd
from datetime import datetime, date
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os.path


__author__ = 'jhsiao'


def get_cid_bidCPM_dic():
    """
    """
    db3 = 'fuelData'
    host3 = '162.222.180.40'
    port3 = 3306
    user = 'jasonhsiao'
    password3 = 'a0X@vHLsMzVaz'
    with MySQLConnection(db3, host3, port3, user, password3) as conn3:
        print('connections successful')

        sql_query = """
                SELECT /* get bid_CPM data from bidding tb in fuelData */
                       date_format(created, '%Y-%m-%d') as date,
                       bid,
                       cid,
                       bidMaxCPM
                FROM bidding
                -- WHERE date_format(created, '%Y-%m-%d') >= '2017-04-01' # update on 10/12
                GROUP BY 1, 2, 3
                ORDER BY 1 DESC
            """
        df_bidCPM = pd.read_sql(sql_query, conn3)
#         df_bidCPM.to_csv('bidCPM_{}.csv'.format(datetime.now().date()), index=False)
        df_bidCPM.to_csv('/home/jhsiao_fuelx_com/Projects/p-p-balancer-project'
                      '/data/bidCPM_{}.csv'.format(datetime.now().date()),
                      index=False)
    #Production fuelAsset DB
    db1 = 'fuelAsset'
    host1 = 'fuelasset.db.fuel451.com'
    port1 = 3306
    user = 'jasonhsiao'
    password1 = 'y1U$hcSPVsQW'
    with MySQLConnection(db1, host1, port1, user, password1) as conn1:
        print('connections successful')
        sql_query1 = """
            SELECT c.id as cid,
                   c.bid,
                   cbl.value,
                   c.created
                   -- c.date_format(created, '%Y-%m-%d') as date
            FROM campaign c
            LEFT JOIN
             (SELECT c2.cid,
                     c2.value,
                     c2.created
              FROM campaign_billing_log c2
              INNER JOIN (SELECT cid,
                                 MAX(created) AS created
                          FROM campaign_billing_log
                          GROUP BY cid) t USING(cid, created)) cbl
            ON c.id = cbl.cid
            GROUP BY c.id
            -- ORDER BY c.date_format(created, '%Y-%m-%d') as date DESC;
            ORDER BY c.created DESC;
        """
        df_fac = pd.read_sql(sql_query1, conn1)
#         df_fac.to_csv('dailylimit_factor_{}.csv'.format(datetime.now().date()), index=False)
        df_fac.to_csv('/home/jhsiao_fuelx_com/Projects/p-p-balancer-project'
                      '/data/dailylimit_factor_{}.csv'.format(datetime.now().date()),
                      index=False)
        cid_fac_dic = {cid: fac for cid, fac in zip(df_fac.cid.values, df_fac.value.values)}
        df_bidCPM['fac'] = df_bidCPM['cid'].map(cid_fac_dic)
        df_bidCPM['bid_CPM'] = df_bidCPM['bidMaxCPM']/(1 - df_bidCPM['fac'])
        df_bidCPM.bid_CPM.replace(np.nan, 0, inplace=True)

        cid_bidCPM_dic = {cid: bidCPM for cid, bidCPM in 
                          zip(df_bidCPM.cid.values, df_bidCPM.bid_CPM.values)}
    return cid_bidCPM_dic

def clean_raw_data():
    ### update on 7/24 to include bid_cpm, weekly seasonality, ctype & adtype
    
    path1 = '/home/jhsiao_fuelx_com/Projects/p-p-balancer-project/data/adx_funnel_combo_v2_{}.csv'.format(datetime.now().date())
    path2 = '/home/jhsiao_fuelx_com/Projects/p-p-balancer-project/data/objective_v2_{}.csv'.format(datetime.now().date())
    path3 = '/home/jhsiao_fuelx_com/Projects/p-p-balancer-project/data/roi_goal_v2_{}.csv'.format(datetime.now().date())
#     path1 = '/home/jhsiao_fuelx_com/Projects/p-p-balancer-project/data/adx_funnel_combo_{}.csv'.format(datetime.now().date())
#     path2 = '/home/jhsiao_fuelx_com/Projects/p-p-balancer-project/data/objective_{}.csv'.format(datetime.now().date())
#     path3 = '/home/jhsiao_fuelx_com/Projects/p-p-balancer-project/data/roi_goal_{}.csv'.format(datetime.now().date())
    if os.path.isfile(path1) and os.path.isfile(path2) and os.path.isfile(path3):
        df1, df2, df3 = pd.read_csv(path1).drop_duplicates(), pd.read_csv(path2), pd.read_csv(path3)
    else:
        for param in reversed(range(24)):
            df1, df2, df3 = get_multiple_tbs(param)
            if df1.date.max() == datetime.now().date() - timedelta(1):
                print(param)
                break
                
        df1 = df1.drop_duplicates()
            
    df4 = pd.merge(df2, df3, how='inner', on='bid')
    df_new = pd.merge(df1, df4, how='inner', on='bid')
    d = {1: 'CPC', 2: 'ROI', 3: 'CPM', 4: 'PCTR', 6: 'CPA'}
    df_new['price_model'] = df_new['objective'].map(d)
    
    ############# update on 7/24, add adtype, and ctype cols ##########################################
    df_new['ctype'] = [3 if c==8 else c for c in df_new.ctype.values]
    # use 3 instead of 8 to represent FULL-funnel if we treat ctype col as a numerical variable
    # normally, it's treated as a categorical col that has to be one-hot-encoded, but for sake of 
    # simplicity, change from 8 to 3 to reduce the impact.
    #######################################################################
    
    ############## update on 7/24 add bid_CPM col #########################################
    cid_bidCPM_dic = get_cid_bidCPM_dic()
    df_new['bidCPM'] = df_new['cid'].map(cid_bidCPM_dic)
    df_new.bidCPM.replace(np.nan, 0, inplace=True)
    #######################################################################
    
    ############### update on 7/24 add weekly seasonality col ##############################
    if type(df_new.date[0]) == str:
        df_new['dow'] = df_new['date'].apply(lambda x: datetime.strptime(x,'%Y-%m-%d').date().weekday())
    else:
        df_new['dow'] = df_new['date'].apply(lambda x: x.weekday())
    #df_new['dow'] = df_new['date'].apply(lambda x: x.weekday())
    ########################################################################################

    final_idx1, pace_clf_lst = pace_clf(df_new)
    df_new['pace_clf'] = pace_clf_lst
    final_idx2, perf_clf_lst, delta_lst = perf_clf(df_new)
    df_new2 = df_new.iloc[final_idx2,:]
    df_new2['perf_clf'], df_new2['perf_delta'] = perf_clf_lst, delta_lst
    df_new2.dailylimit.replace(np.nan, 0, inplace=True)
    df_new3 = df_new2[
          (df_new2.dailylimit >= 0) &
          (df_new2.cost >= 0) &
          (df_new2.tc_ov >= 0)
   ]
    df_new3['perf_delta'] = df_new3['perf_delta'].apply(lambda x: abs(x))
    df_new3['perf_delta'] = [9999 if row >= 9999 else row for row in df_new3.perf_delta.values] # overwrite perf_delta w/ hard-coded value, 9999 for infinity, which is 100 times worse perf
    df_new3.to_csv('/home/jhsiao_fuelx_com/Projects/p-p-balancer-project'
                   '/data/all_mdf_v2_{}.csv'.format(datetime.now().date()), index=False)
    return df_new3

def pace_clf(df):
    pace_clf_lst = []
    final_idx = []
    spds, dailylimits = df.spd.values, df.dailylimit.values
    for idx, (spd, dailylimit) in enumerate(zip(spds, dailylimits)):

        if abs(spd-dailylimit) <= 0.10*dailylimit and spd != 0 and dailylimit != 0:
            pace_clf_lst.append(1)
            final_idx.append(idx)
        elif spd == 0 and dailylimit == 0:
            pace_clf_lst.append(0)
            final_idx.append(idx)
        else:
            pace_clf_lst.append(0)
            final_idx.append(idx)
    return final_idx, pace_clf_lst

def perf_clf(df): ### update on 10/01/18
    perf_clf_lst = []
    final_idx = []
    delta_lst = []
    costs, tcs, clks, imps, tc_ovs, goals, models = \
    df.cost.values, df.tc.values, df.clk.values, df.imp.values, \
    df.tc_ov.values, df.roi_goal.values, df.price_model.values
    for idx ,(cost, tc, clk, imp, tc_ov, goal, model) in enumerate(
      zip(costs, tcs, clks, imps, tc_ovs, goals, models)
    ):
        if model == 'ROI':
            if cost == 0: # denominator == 0 --> infinity
                perf_clf_lst.append(0) # change to 0 on 5/21/18
                final_idx.append(idx)
#                 delta_lst.append(0)
#                 delta_lst.append(float("inf"))
                delta_lst.append(sys.maxsize)
#                 delta_lst.append(1000)
            elif tc_ov == 0 and cost == 0: # both numerator and denominator == 0 --> whole term --> infinity
                perf_clf_lst.append(0)
                final_idx.append(idx)
                delta_lst.append(sys.maxsize)
#                 delta_lst.append(float("inf"))
#                 delta_lst.append(1000)
            else:
                roi_actual = tc_ov / cost
                delta = roi_actual - goal
                if delta >= 0:
                    perf_clf_lst.append(1)
                    final_idx.append(idx)
                    delta_lst.append(delta)
                else:
                    perf_clf_lst.append(0)
                    final_idx.append(idx)
                    delta_lst.append(delta)
        if model == 'CPA':
            if tc == 0: # denominator == 0 --> infinity
                perf_clf_lst.append(0)
                final_idx.append(idx)
                delta_lst.append(sys.maxsize)
#                 delta_lst.append(float("inf"))
#                 delta_lst.append(1000)
            elif cost == 0 and tc == 0: # added this edge case on 5/21/18 # both numerator and denominator == 0 --> whole term --> infinity
                perf_clf_lst.append(0)
                final_idx.append(idx)
                delta_lst.append(sys.maxsize)
#                 delta_lst.append("inf")
#                 delta_lst.append(1000)
            else:
                cpa_actual = cost / tc
                delta = cpa_actual - goal
                if delta <= 0:
                    perf_clf_lst.append(1)
                    final_idx.append(idx)
                    delta_lst.append(delta)
                else:
                    perf_clf_lst.append(0)
                    final_idx.append(idx)
                    delta_lst.append(delta)
        if model == 'CPC':
            if clk == 0: # denominator == 0 --> infinity
                perf_clf_lst.append(0)
                final_idx.append(idx)
                delta_lst.append(sys.maxsize)
#                 delta_lst.append(float("inf"))
#                 delta_lst.append(1000)
            elif cost == 0 and clk == 0: # added this edge case on 5/21/18
                perf_clf_lst.append(0)
                final_idx.append(idx)
                delta_lst.append(sys.maxsize)
#                 delta_lst.append(float("inf"))
#                 delta_lst.append(1000)
            else:
                cpc_actual = cost / clk
                delta = cpc_actual - goal
                if delta <= 0:
                    perf_clf_lst.append(1)
                    final_idx.append(idx)
                    delta_lst.append(delta)
                else:
                    perf_clf_lst.append(0)
                    final_idx.append(idx)
                    delta_lst.append(delta)
        if model == 'CPM':
            if imp == 0: # denominator == 0 --> infinity
                perf_clf_lst.append(0)
                final_idx.append(idx)
                delta_lst.append(sys.maxsize)
#                 delta_lst.append(float("inf"))
#                 delta_lst.append(1000)
            elif cost == 0 and imp == 0: # added this edge case on 5/21/18
                perf_clf_lst.append(0)
                final_idx.append(idx)
                delta_lst.append(sys.maxsize)
#                 delta_lst.append(float("inf"))
#                 delta_lst.append(1000)
            else:
                cpm_actual = cost / (imp*1000)
                delta = cpm_actual - goal
                if delta <= 0:
                    perf_clf_lst.append(1)
                    final_idx.append(idx)
                    delta_lst.append(delta)
                else:
                    perf_clf_lst.append(0)
                    final_idx.append(idx)
                    delta_lst.append(delta)
    return final_idx, perf_clf_lst, delta_lst

def get_data():
#     path = 'BudgetCalculator%2Fadx_funnel_combo2018-05-03_v3.csv'
#     path = 'all_mdf3_2018-05-03.csv'
    #path = 'all_mdf3_2018-05-14.csv'
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    return df

def get_cid_data(data, CID):
    """
    :type df: pandas.dataframe
    :rtype: pandas.dataframe
    get data for a specific cid w/ data sorted in asecnding order
    """
    df = data[data.cid==CID].sort_values(by='date', ascending=True)
    global last_day_in_df
    last_day_in_df = df.date.max()
    print('last date in df: ', last_day_in_df)
    columns_list = [col for col in df.columns
                    if col not in [
                    'bid','cid','date','campaign_type',
                    'is_lower_funnel', 'spd', 'objective',
                    'roi_goal','price_model'
                    ]]
    y = 'spd'
    df = df[[y] + columns_list]
    # make sure spd'd be the first col in df
    return df, last_day_in_df

def cids_list(BID):
    """
    rtype: CIDs_lst: cid list for TRAINING given a BID
    """
#     df = pd.read_csv('all_mdf_2018-07-09.csv')
    df = pd.read_csv(
        '/home/jhsiao_fuelx_com/Projects/p-p-balancer-project/data/all_mdf_v2_{}.csv'.format(datetime.now().date())
    )
    final_date = df.date.max()
    df2 = df[df.bid == BID]
    CIDs_lst = []
    CIDs_lst_wo = []
    for CID in list(set(df2.cid.values)):
#     for CID in list(dedupe(df2.cid.values)):
        df3 = df2[df2.cid == CID]
        df4 = df3.sort_values(by='date', ascending=True)
        if df4.date.max() == final_date:
            CIDs_lst_wo.append(CID)
        if df4.date.max() == final_date and \
            df4.shape[0] >= 30:
            CIDs_lst.append(CID)
    CIDs_lst = sorted(CIDs_lst)
    CIDs_lst_wo = sorted(CIDs_lst_wo)
    return CIDs_lst, CIDs_lst_wo, len(CIDs_lst)==len(CIDs_lst_wo)

def cids_list_all(df):
    """
    rtype: CIDs_lst: a full cid list for TRAINING
    """
    final_date = df.date.max()
    CIDs_lst = []
    for CID in df.cid.unique().tolist():
        df2 = df[df.cid==CID]
        df3 = df2.sort_values(by='date', ascending=True)
        if df3.date.max() == final_date and df3.shape[0] >= 30:
            CIDs_lst.append(CID)
    CIDs_lst = sorted(CIDs_lst)
    return CIDs_lst

def cids_list_all_wo_fil(df):
    """
    rtype: CIDs_lst: a full cid list for TRAINING
    rtype: CIDs_lst: a full cid list currently running up to date
    """
    final_date = df.date.max()
    CIDs_lst = []
    for CID in df.cid.unique().tolist():
        df2 = df[df.cid==CID]
        df3 = df2.sort_values(by='date', ascending=True)
        if df3.date.max() == final_date:
            CIDs_lst.append(CID)
    CIDs_lst = sorted(CIDs_lst)
    return CIDs_lst

def get_fully_trained_bids():
    
    data = pd.read_csv(
        '/home/jhsiao_fuelx_com/Projects/p-p-balancer-project/data/all_mdf_v2_{}.csv'.format(datetime.now().date())
    )
    #data = pd.read_csv('all_mdf_2018-07-09.csv')

    from collections import defaultdict
    bid_cid_dic = defaultdict(list)
    for k, v in zip(data.bid.values, data.cid.values):
        bid_cid_dic[k].append(v)
    bid_cid_dic2 = {
        k: list(set(v)) for k, v in bid_cid_dic.items()
    }
    cids_lst_all = cids_list_all(data)
    print('cids running/lasting for 30 days: ', len(cids_lst_all))
#     cids_lst_all = cids_list_all_wo_fil(data)
#     print('cids running on last day in df: ', len(cids_lst_all))
    bids_lst_update = []
    for k, v in bid_cid_dic2.items():
        for ele in v:
            if ele in cids_lst_all:
                bids_lst_update.append(k)
    new_bids_lst = list(set(bids_lst_update))

    bids_ls = new_bids_lst # 92 bids currently running up to date

    #bids_boo = [get_params(bid)[0] for bid in bids_ls]
    bids_boo = [cids_list(bid)[-1] for bid in bids_ls]
    print(bids_boo, len(bids_boo))

    bids_bool_tup = zip(new_bids_lst, bids_boo)
    bids_lst_boo_true = [bid for bid, boo in bids_bool_tup
                         if boo==True]
    bids_lst_boo_false = [bid for bid in new_bids_lst
                          if bid not in bids_lst_boo_true]
    
    cids_lst_for_bids_lst_boo_true = [cids for bid, cids in bid_cid_dic2.items() 
                                      if bid in bids_lst_boo_true]
    
    cids_lst_all_lst_boo_true = [cid for sublist in cids_lst_for_bids_lst_boo_true 
                                 for cid in sublist]
    print(len(cids_lst_all_lst_boo_true))
    
    cids_up_running_all_in_bids = [cid for cid in cids_lst_all_lst_boo_true 
                                   if cid in cids_lst_all]
    print(len(cids_up_running_all_in_bids))
    
    assert len(new_bids_lst) == len(bids_lst_boo_true) + \
                                len(bids_lst_boo_false)

    assert all(cid in cids_lst_all for cid in cids_up_running_all_in_bids)
    
    return bids_lst_boo_true, bids_lst_boo_false, cids_up_running_all_in_bids

def dedupe(items):
    seen = set()
    for item in items:
        if item not in seen:
            return item
            seen.add(item)
    return seen


def synthesize_feature_data(df, m, last_day_in_df):
    """
    :type df: pandas.dataframe
    :type m: int
    :type last_day_in_df: str
    update on 5/29
    """
#     df_tail = df.tail(m)
    from datetime import datetime, date, timedelta
#     if any(ele for ele in df.perf_clf.values) == 1 and any(ele for ele in df.pace_clf.values) == 1:
    if any(ele == 1 for ele in df.perf_clf.values) and any(ele == 1 for ele in df.pace_clf.values):
        df_tail = df[(df.perf_clf==1)&(df.pace_clf==1)].tail(3) ### update on 5/25
#     elif any(ele for ele in df.pace_clf.values) == 1 and all(ele for ele in df.perf_clf.values) == 0:
    elif any(ele == 1 for ele in df.pace_clf.values) and all(ele == 0 for ele in df.perf_clf.values):
        df_tail = df[df.pace_clf==1].tail(3)
#     elif any(ele for ele in df.perf_clf.values) == 1 and all(ele for ele in df.pace_clf.values) == 0:
    elif any(ele == 1 for ele in df.perf_clf.values) and all(ele == 0 for ele in df.pace_clf.values):
        df_tail = df[df.perf_clf==1].tail(3)
    else:
        df_tail = df.tail(3)

    np_row1 = df_tail.mean().values.reshape(1,-1)
#     np_row1[:,-2], np_row1[:,-1] = round(np_row1[:,-2]), round(np_row1[:,-1])
#    np_row1[:,-2], np_row1[:,-1] = 1, 1
    np_row1[:,-3], np_row1[:,-2] = 1, 1 # force pace_clf & perf_clf to be 1
    dow = datetime.strptime(last_day_in_df, '%Y-%m-%d').date().weekday()
    np_row1[:,-4] = dow
    cnt_inc = (lambda x: x+1 if x != 6 else 0)

    ########## update on 10/3/2018 ##########################
    np_row1[:,-6] = df.dailylimit.values[-1] # -6: dailylimit col, grab the most recent dailylimit from the date b4
    np_row1[:,-9] = df.cost.values[-1] # -9: cost col
    ########################################################

    np_mrows = np.array([np_row1 for i in range(m)]).reshape(m, len(df.columns))
    np_mrows[:,-4] = [cnt_inc((i+dow)%7) for i in range(m)]
    df_mrows = pd.DataFrame(np_mrows, columns=df.columns)
    df_total = pd.concat([df, df_mrows], axis=0)
    df = df_total
    return df

def transformer_data(df):
    train_test_split_rat = 0.88
#     train_test_split_rat = 0.70
    ### update on 6/3/18
    assert isinstance(df, pd.DataFrame)
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    df = df[indices_to_keep].astype(np.float64)
    ###
    global train_idx
    train_idx = int(train_test_split_rat*df.shape[0])
    training_set = df.iloc[:train_idx,:].values
    # feature sacling
    sc = MinMaxScaler(feature_range=(0,1))
    training_set_scaled = sc.fit_transform(training_set)
    assert training_set_scaled.shape == training_set.shape
    # creating a data structure w/ k previous timesteps
    global D
    global k
    D = len(df.columns)
#     k = 15
#     k = 7
    k = 1
    X_train, y_train = [], []
    for i in range(k, len(training_set)):
        X_train.append(training_set_scaled[i-k:i, :])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # reshaping
    X_train = np.reshape(
        X_train,
        (X_train.shape[0], X_train.shape[1], D)
    )
    # getting the real spd
    dataset_test = df.iloc[train_idx:, :].values
    real_spd = dataset_test[:, 0:1]
    # getting the predicted spd inputs
    dataset_total = df.iloc[:,:].values
    inputs = dataset_total[len(dataset_total)-len(dataset_test)-k:, :]
    inputs = inputs.reshape(-1, D)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(k, inputs.shape[0]):
        X_test.append(inputs[i-k:i,:])
    X_test = np.array(X_test)
    X_test = np.reshape(
        X_test,
        (X_test.shape[0], X_test.shape[1], D)
    )
    
    return X_train, y_train, X_test, real_spd, sc, D, train_idx

#################################################################################################################

#!/usr/bin/env python
# model.py
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import math
from sklearn.metrics import mean_squared_error
__author__ = 'jhsiao'
def model_train(X_train, y_train):
    # Initializing the RNN
    regressor = Sequential()
    # Adding the first LSTM layer and some dropout reg
    # w/ dropout rate: p
    regressor.add(
        LSTM(
            units=50,
            return_sequences=True,
            input_shape=(X_train.shape[1], X_train.shape[2])
        )
    )
    p = 0.2
    regressor.add(Dropout(p))
    # Adding a 2nd LSTM layer and some dropout reg
    regressor.add(
        LSTM(
            units=50,
            return_sequences=True,
        )
    )
    regressor.add(Dropout(p))
    # Adding a 3rd LSTM layer and some dropout reg
    regressor.add(
        LSTM(
            units=50,
            return_sequences=True,
        )
    )
    regressor.add(Dropout(p))
    # Adding a 4th LSTM layer and some dropout reg
    regressor.add(
        LSTM(
            units=50,
            return_sequences=False,
        )
    )
    regressor.add(Dropout(p))
    # Adding the output layer
    regressor.add(Dense(units=1))
    # Compiling the RNN
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    # Fitting the RNN to the training set
    regressor.fit(X_train, y_train, epochs=100, batch_size=8)
    return regressor


def model_test(X_test, regressor):
    predicted_spd = regressor.predict(X_test)
    return predicted_spd

#################################################################################################################

#!/usr/bin/env python
# plot_metrics.py
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import random
from datetime import datetime, timedelta
__author__ = 'jhsiao'

def cmp_table(predicted_spd, real_spd, sc, X_test, df, n, D, train_idx, m, last_day_in_df):
    """
    update on 5/29/18
    """
    dummy = np.random.randn(len(X_test), D-1)
    print(dummy.shape)
    predicted_spd = np.concatenate((predicted_spd, dummy), axis=1)
    print(predicted_spd.shape)
    predicted_spd = sc.inverse_transform(predicted_spd)
    print(predicted_spd.shape)
    predicted_spd_list = [ele for ele in predicted_spd[:,0]]
    real_spd_list = [ele for ele in real_spd.reshape(-1)]
    df_comp = pd.DataFrame(data = df.iloc[train_idx:, :1].values, columns=['real_spd'])
    df_comp['prediction spd'] = predicted_spd_list
    df_comp = df_comp.iloc[-m-n:,:]
    if type(last_day_in_df) == str:
        start = (datetime.strptime(last_day_in_df, '%Y-%m-%d') - timedelta(n-1)).strftime('%Y-%m-%d')
        end = (datetime.strptime(last_day_in_df, '%Y-%m-%d') + timedelta(m)).strftime('%Y-%m-%d')
    else:
        start = (last_day_in_df - timedelta(n-1)).strftime('%Y-%m-%d')
        end = (last_day_in_df + timedelta(m)).strftime('%Y-%m-%d')
    df_comp['date'] = pd.date_range(start, end)
    df_comp.iloc[-m:,:]['real_spd'] = np.nan
    
    return df_comp, predicted_spd

def model_metrics(real_spd, predicted_spd):
    rmse = math.sqrt(mean_squared_error(real_spd, predicted_spd[:,0:1]))
    print("MSE:", mean_squared_error(real_spd, predicted_spd[:,0:1]))
    print("RMSE:", rmse)
    
#def vis_plot(real_spd, predicted_spd):
#    plt.plot(real_spd, color = 'red', label = 'Real Spd')
#    plt.plot(predicted_spd[:,0:1], color = 'blue', label = 'Predicted Spd')
#    plt.title('Spd Prediction')
#    plt.xlabel('Time')
#    plt.ylabel('Spd')
#    plt.legend()
#    plt.show()

def dailylimit_sum(df_lst):
    df_total = pd.concat([df for df in df_lst], axis=0)
    df_gp = df_total[['date', 'real_spd', 'prediction spd']].groupby(by='date', as_index=False).sum()
    return df_gp

################################################################################################################

def main():
    
    global CID
    # df = get_data()
    df = clean_raw_data()
    print('Done loading the whole data')
    # CIDs = [2496,2671,2812,3546,4021]
    df, last_day_in_df = get_cid_data(df, CID)
    print('Successful for a given cid')
    df = synthesize_feature_data(df, m, last_day_in_df)
    print('Successfully added feature sets')
    X_train, y_train, X_test, real_spd, sc, D, train_idx =         transformer_data(df)
    print('Done transferring data')
    regressor = model_train(X_train, y_train)
    print('Done training')
    predicted_spd = model_test(X_test, regressor)
    print('Done predictions')
    # df_comp, predicted_spd = plot_metrics.cmp_table(predicted_spd, real_spd, sc, X_test, df)
    # print('Done comparasion table')
    # print(df_comp)
    # update on 5/29/18
    df_comp, predicted_spd = cmp_table(
        predicted_spd, real_spd, sc,
        X_test, df, n, D, train_idx, m, last_day_in_df
    )
    print('Done comparasion table')
    print(df_comp)
    model_metrics(real_spd, predicted_spd)
    print('Done Metrics')
    # plot_metrics.vis_plot(real_spd, predicted_spd)
    # print('Done plots')
    
    return df_comp


def get_iotarget_dailylimit_frm_db():
    """
    :rtype: cid_total_spd dict: {cid: total_spd in present day}
    """
    #Production fuelAsset DB
    db1 = 'fuelAsset'
    host1 = 'fuelasset.db.fuel451.com'
    port1 = 3306
    user = 'jasonhsiao'
    password1 = 'y1U$hcSPVsQW'
    # read iotarget(current month) as well as dailylimit from campaign for a PRESENT day
    with MySQLConnection(db1, host1, port1, user, password1) as conn1:
        print('connections successful')
        sql_query1 = """
            SELECT
                   id as cid,
                   ioTarget,
                   daily_limit as adjusted_spd
            FROM campaign
            WHERE 1 = 1
            ORDER BY created DESC
        """
        df1 = pd.read_sql(sql_query1, conn1)
        df1.to_csv('iotarget_dailylimit_v2_{}.csv'.format(datetime.now().date()),
                   index=False)
        comp_dic = {cid:(iotarget,adjusted_spd)
                    for  cid, iotarget, adjusted_spd in
                    zip(df1.cid.values, df1.ioTarget.values, df1.adjusted_spd.values)}
#         df1.to_csv('/home/jhsiao_fuelx_com/Projects/p-p-balancer-project/data/dailylimit_factor_{}.csv'.format(datetime.now().date()),
#                    index=False)
    print(df1.head(2))
    print(df1.shape)
    return df1, comp_dic

def get_daily_spd(): # update on 9/4/18
    """
    rtype: daily_spd dict: {cid: (raw_spd_tmr, raw_spd_pt, ie. current dailylimit)}
    """
    df = pd.read_csv('dailyio_adjusted_{}.csv'.format(datetime.now().date()-timedelta(1)))
#     df_tmr = df[df.pred_date==datetime.now().date().strftime('%Y-%m-%d')][['cid','adjusted_spd']]
    df_tmr = df[df.pred_date==df.pred_date.max()][['cid','adjusted_spd']]
    print(df_tmr.shape)

#     #metho1 directly read from model outputs
#     df_pt = df[df.pred_date==df.pred_date.min()][['cid','adjusted_spd']]
#     print(df_pt.shape)

    #method2 directly read from campaign table in MYSQL to get present day's dailylimit
    df_pt, _ = get_iotarget_dailylimit_frm_db()
    df_pt = df_pt[['cid','adjusted_spd']]
    print(df_pt.shape)


    df = df_tmr.merge(df_pt, on='cid', how='left')
    print(df.shape)
    assert df.shape[0] == df_tmr.shape[0]

    daily_spd = {cid: (raw_spd_tmr, raw_spd_pt)
                 for cid, raw_spd_tmr, raw_spd_pt in zip(df.cid.values,
                                                         df.adjusted_spd_x.values,
                                                         df.adjusted_spd_y.values)}
    return daily_spd


def get_daily_limit_dic(filename, CIDs):
    """
    :type filename: directory, mainly in csv
    :type BID: int
    :type CIDs: List[int]
    :rtype: Dict{cid[int]:daily_limit[float]}
    """
    df = pd.read_csv(filename)
    clt_df = df.iloc[[idx for idx, cid in enumerate(df.cid.values) if cid in CIDs], :] \
               .sort_values(by=['bid','cid','date'], ascending=[1,1,0])

    clt_df_one = clt_df[clt_df.date==clt_df.date.max()]
    print(clt_df.date.max())

    daily_limit_dic = {k:v for k, v in zip(clt_df_one.cid.values, clt_df_one.dailylimit.values)}
    return daily_limit_dic, clt_df


def get_daily_pred_spd_dic(filename, CIDs):
    """
    :type filename: dir, mainly in pkl
    :type CIDs: List[int]
    :rtype: Dict{cid[int]:daily_pred_spd[float]}
    """
    with open(filename, 'rb') as fh:
        d = pickle.load(fh)

    data_np_lst = []
    df_all = []
    for k, df in d.items():
        df['cid'] = [k for i in range(df.shape[0])]
        df['date'] = df['date'].apply(lambda x: x.strftime('%Y%m%d'))
        df['given_date_forcast'] = [datetime.now().date().strftime('%Y%m%d')
                                    for i in range(df.shape[0])]
        df['real_spd'] = df['real_spd'].apply(lambda x: round(x, 2))
        df.fillna('null', inplace=True)
        df['prediction spd'] = df['prediction spd'].apply(lambda x: round(x, 2))
        df = df[['cid','given_date_forcast','date','real_spd','prediction spd']]
        df_all.append(df)
        data_np = df.values
        data_np_lst.append(data_np)

    data = reduce(lambda x, y: np.concatenate((x,y), axis=0), data_np_lst).tolist()

    df_new = pd.DataFrame(data=data, columns=['cid','given_date_forcast','date','real_spd','prediction spd'])
    df_new.head(2)

    df_new.drop(columns='given_date_forcast')
    df_new2 = df_new[df_new.date==df_new.date.max()]
    df_new2 = df_new2.iloc[[idx for idx, cid in enumerate(df_new2.cid.values) if cid in CIDs], :]

    print(df_new.date.max())
    daily_pred_spd_dic = {k:v for k, v in zip(df_new2.cid.values, df_new2['prediction spd'].values)}

    return daily_pred_spd_dic


def consume_dailyIO3(daily_pred_spd_dic, daily_limit_dic):
    """
    :type daily_pred_spd_dic: Dict{cid[int]:raw_spd[float]}
    :type daily_limit_dic: Dict{cid[int]: daily_limit[float]}
    :rtype actual_spd_dic: Dict{cid[int]: actual_spd[float]}
    """
#     boo, _, _, daily_limit_dic, _, daily_pred_spd_dic = get_params(BID) # bypass boolean check

    daily_pred_spd_dic_tup_lst = sorted([(k,v) for k, v in daily_pred_spd_dic.items()],
                                        key=lambda x: x[1],
                                        reverse=True)
    boo = True # force to be true for manually adjust
    print('daily_limit_dic: ', daily_limit_dic)
    print('daily_limit_sum: ', sum([v for v in daily_limit_dic.values()]))
    print('daily_pred_spd_dic: ', daily_pred_spd_dic)
    print('daily_pred_spd_sum: ', sum([v for v in daily_pred_spd_dic.values()]))
    cids = [cid for cid,_ in daily_pred_spd_dic_tup_lst]

    if boo:
        dailyIO_lst = [v for v in daily_limit_dic.values()]
        dailyIO_sum = sum(dailyIO_lst)
        #############################
        dailyIO = dailyIO_sum
        #############################
        pred_dailyIO_lst = [v for _, v in daily_pred_spd_dic_tup_lst]
        pred_dailyIO_sum = sum(pred_dailyIO_lst)

        delta = dailyIO_sum - pred_dailyIO_sum
        cnt = 0
        actual_spd = []

        if delta >= 0: # underspend case: basically ml-solution + evenly increment to be assigned to each campaign
            print('more to spend')
            pred_dailyIO_lst = [ele+(delta/len(pred_dailyIO_lst)) for ele in pred_dailyIO_lst]
            actual_spd = pred_dailyIO_lst
            dailyIO_sum = 0
        else: # delta<0: overspend case:
            for spd in pred_dailyIO_lst:
                if dailyIO_sum >= 0 and dailyIO_sum >= spd:
                    dailyIO_sum -= spd
                    cnt += 1
                    actual_spd.append(spd)
                    print('actual_spd: ',actual_spd)
                elif dailyIO_sum >= 0 and dailyIO_sum < spd:
                    spd = dailyIO_sum
                    dailyIO_sum -= spd
                    cnt += 1
                    actual_spd.append(spd)
                    print('actual_spd: ', actual_spd)
                else:
                    break
            print('cnt: ', cnt)
            if any([x==0 for x in actual_spd]):
                length_zeros = len([x for x in actual_spd if x == 0])

                pct = 1-(length_zeros*min([val for val in daily_pred_spd_dic.values()])/dailyIO-0.01)
                print('calculated pct: ', pct)
                if pct < 0.7:
                    pct = 0.7
                    print('modified pct: ', pct)
                if pct >= 0.98:
                    pct = 0.98
                    print('modified pct: ', pct)

                print('# of zeros: ', length_zeros)
                ################
                actual_spd = [pct*ele for ele in actual_spd[:-length_zeros]]+length_zeros*[0]

                print('sum of actual_spd: ', sum(actual_spd))

                print('last elements spd: ', (1-pct)*dailyIO/length_zeros)
                print('last_elements_lst: ', length_zeros*[(1-pct)*dailyIO/length_zeros])

                actual_spd[-length_zeros:] = [(1-pct)*dailyIO/length_zeros
                                              for _ in range(length_zeros)][::-1]
                print('ACTUAL SPENDING: ', actual_spd)
#################################################################################
        print('ACTUAL SPENDING: ', actual_spd)
        if actual_spd[-1] < 5.0:
            old = actual_spd.pop()
            print('old:', old)
            diff = 5.0 - old
            print('diff:', diff)
            actual_spd.append(5.0)
#             actual_spd[-1] == 4
            print('last element in actual_spd:', actual_spd[-1])
            print('size of actual_spd:', len(actual_spd))
            sub = diff / (len(actual_spd)-1)
            print('sub:', sub)
            ####### update on 10/3/18 customized layer #########
            if actual_spd[0] >= 0.4 * sum(actual_spd):
                actual_spd[0] -= diff
            else:
                actual_spd[:-1] = [ele - sub for ele in actual_spd[:-1]]

            actual_spd = sorted(actual_spd, reverse=True)
#             actual_spd[:-1] = [ele-sub for ele in actual_spd[:-1]]
            #################################################
#            actual_spd[:-1] = [ele-sub for ele in actual_spd[:-1]]

            print(actual_spd)
#                 assert sum(actual_spd) == dailyIO
        actual_spd_dic = {k:v for k,v in zip(cids, actual_spd)}
        print('dailyIO_sum after: ',dailyIO_sum)
        print('actual_spd_sum: ', sum([v for v in actual_spd_dic.values()]))
        return actual_spd_dic, round(dailyIO_sum, 2)==0
##################################################################################

#         actual_spd_dic = {k:v for k,v in zip(cids, actual_spd)}
#         print('dailyIO_sum after consumption: ',dailyIO_sum)
#         print('actual_spd_sum: ', sum([v for v in actual_spd_dic.values()]))
#         print('actual_spd_dic: ', actual_spd_dic)

#         return actual_spd_dic, round(dailyIO_sum, 2)==0
    else:
        print('not a fully trained client')
        print('please reinsert a bid:')



def get_daily_spd_two_days(d_tmr, d_pt):
    """
    :type: d_tmr: Dict{cid[int]: raw_spd_tmr[float]}
    :type: d_pt: Dict{cid:[int]: raw_spd_pt[float] }
    :rtype: daily_spd: Dict{cid[int]: (raw_spd_tmr, raw_spd_pt)}
    """
    data = pd.DataFrame(data=[(k,v) for k, v in d_tmr.items()], columns=['cid', 'adjuster'])
    data['adjuster_old'] = data['cid'].map(d_pt)
    print(data.adjuster.sum(), data.adjuster_old.sum())
    daily_spd = {cid: (raw_spd_tmr, raw_spd_pt)
                 for cid, raw_spd_tmr, raw_spd_pt in zip(data.cid.values,
                                                         data.adjuster.values,
                                                         data.adjuster_old.values)}

    return daily_spd


def get_dailylimit_factor_frm_db():
    """
    """
    #Production fuelAsset DB
    db1 = 'fuelAsset'
    host1 = 'fuelasset.db.fuel451.com'
    port1 = 3306
    user = 'jasonhsiao'
    password1 = 'y1U$hcSPVsQW'
    with MySQLConnection(db1, host1, port1, user, password1) as conn1:
        print('connections successful')
        sql_query1 = """
            SELECT c.id as cid,
                   c.bid,
                   cbl.value
            FROM campaign c
            LEFT JOIN
             (SELECT c2.cid,
                     c2.value,
                     c2.created
              FROM campaign_billing_log c2
              INNER JOIN (SELECT cid,
                                 MAX(created) AS created
                          FROM campaign_billing_log
                          GROUP BY cid) t USING(cid, created)) cbl
            ON c.id = cbl.cid
            GROUP BY c.id
            ORDER BY c.created DESC;
        """
        df1 = pd.read_sql(sql_query1, conn1)
        print(df1.head(2))
        df1.to_csv('dailylimit_factor_v2_{}.csv'.format(datetime.now().date()),
                   index=False)
#         df1.to_csv('/home/jhsiao_fuelx_com/Projects/p-p-balancer-project/data/dailylimit_factor_{}.csv'.format(datetime.now().date()),
#                    index=False)
        cid_fac = {cid: value for cid, value in zip(df1.cid.values, df1.value.values)}

    return cid_fac


def uniq(array):
    output = []
    idx_lst = []
    for idx, x in enumerate(array):
        if x not in output:
            output.append(x)
            idx_lst.append(idx)
    return idx_lst, output


def get_total_spd_frm_db():
    """
    """
    #Production fuelAsset DB
    db1 = 'fuelAsset'
    host1 = 'fuelasset.db.fuel451.com'
    port1 = 3306
    user = 'jasonhsiao'
    password1 = 'y1U$hcSPVsQW'
    with MySQLConnection(db1, host1, port1, user, password1) as conn1:
        print('connections successful')
        sql_query1 = """
            SELECT bid,
                   cid,
                   total_spend
            FROM campaign_budget
            WHERE 1 = 1
            -- AND bid in ({})
            -- AND MONTH(updated) = MONTH(current_date())
            -- AND YEAR(updated) = YEAR(current_date())
            AND is_ongoing = 0
            ORDER BY updated DESC;
        """
        df1 = pd.read_sql(sql_query1, conn1)
        df1.to_csv('total_spd_v2_{}.csv'.format(datetime.now().date()),
                   index=False)
#         df1.to_csv('/home/jhsiao_fuelx_com/Projects/p-p-balancer-project/data/dailylimit_factor_{}.csv'.format(datetime.now().date()),
#                    index=False)
    print(df1.head(2))
    print(df1.shape)

    idx_lst, _ = uniq(df1.cid.values)
    df1 = df1.iloc[[idx for idx in idx_lst], :]
    print(df1.shape)

    cid_total_spd = {cid: total_spd for cid, total_spd in zip(df1.cid.values, df1.total_spend.values)}
    return cid_total_spd


def data_grab(daily_spd, cid_fac, cid_total_spd): # IMPORTANT fun to get the up to date dailylimit
    """
    :type daily_spd: DICT {cid: (adjuster_spd_tmr, adjuster_spd_pt)}, load from model output
    :type cid_fac: DICT {cid: fac}, a larger dict for look-up
    :type cid_total_spd: DICT {cid: total_spd}, a larger dict for look-up
    :rtype: data: List[List]]
    for example: row [DATE, CID, RAW_PRED, TOTAL_SPD_CUR, TOTAL_SPD_FUT]
    """

    records = []
    cid_tmr_pt = []

    for cid, (pred_tmr, pred_pt) in daily_spd.items():
        DATE = datetime.now().date()

        CUR_YEAR, CUR_MON, CUR_DAY, NEXT_MON = int(DATE.strftime('%Y-%m-%d').split('-')[0]), \
                                               int(DATE.strftime('%Y-%m-%d').split('-')[1]), \
                                               int(DATE.strftime('%Y-%m-%d').split('-')[2]), \
                                               int((DATE + relativedelta(months=+1)).strftime('%Y-%m-%d').split('-')[1])

        DATE = datetime.now().date().strftime('%Y%m%d') # update DATE to a specific format

        DAYS_in_CURMON = monthrange(CUR_YEAR, CUR_MON)[1]
        DAYS_in_NEXTMON = monthrange(CUR_YEAR, NEXT_MON)[1]
        DAYS_left_CURMON = DAYS_in_CURMON - CUR_DAY + 1
        DAYS_passed_CURMON = CUR_DAY - 1

#         DAYS_left_CURMON = DAYS_in_CURMON - CUR_DAY
#         DAYS_passed_CURMON = CUR_DAY
        assert DAYS_in_CURMON == (DAYS_left_CURMON+DAYS_passed_CURMON)

#         spd_mdf = adjuster_spd / (1 - cid_fac[cid])

        spd_mdf_tmr = pred_tmr / (1 - cid_fac[cid])
        spd_mdf_pt = pred_pt / (1 - cid_fac[cid])
        print(cid,'-',spd_mdf_tmr,'-',spd_mdf_pt)
        cid_tmr_pt.append([cid, spd_mdf_tmr, spd_mdf_pt])

        # method1
#         TOTAL1 = spd_mdf*DAYS_left_CURMON + DAYS_passed_CURMON*cid_total_spd[cid] / DAYS_in_CURMON

        # method2
#         TOTAL1 = cid_total_spd[cid] - (DAYS_left_CURMON * spd_mdf_pt) + DAYS_left_CURMON * spd_mdf_tmr
        if CUR_DAY == 1:
            TOTAL1 = spd_mdf_tmr * DAYS_left_CURMON
        else:
            TOTAL1 = cid_total_spd[cid] + DAYS_left_CURMON * (spd_mdf_tmr - spd_mdf_pt)
#        TOTAL1 = cid_total_spd[cid] + DAYS_left_CURMON * (spd_mdf_tmr - spd_mdf_pt)

        TOTAL2 = spd_mdf_tmr * DAYS_in_NEXTMON
#         TOTAL2 = spd_mdf*DAYS_in_NEXTMON

        records.append([DATE, cid, pred_tmr, TOTAL1, TOTAL2])

    return records, cid_tmr_pt


####################### update on 9/24 #################################################################
def calActualGoalPct(df_gp):
    """
    @df_gp: pandas.DataFrame: aggregated df at clt level
    @return: actual: List[float], actual performance
    @return: pcts: List[float], percentage in terms of goal
    """
    actual = []
    pcts = []
    for idx, (price_model, imp, clk, tc, tc_ov, cost, goal) in enumerate(zip(df_gp.price_model.values,
                                                                             df_gp.imp.values,
                                                                             df_gp.clk.values,
                                                                             df_gp.tc.values,
                                                                             df_gp.tc_ov.values,
                                                                             df_gp.cost.values,
                                                                             df_gp.roi_goal.values
                                                                            )):
        if price_model == 'ROI':
            val = tc_ov / cost
            pct = ((val - goal) / goal) * 100
            actual.append(val)
            pcts.append(round(pct, 2))
        elif price_model == 'CPA':
            val = cost / tc
            pct = ((goal - val) / goal) * 100
            actual.append(val)
            pcts.append(round(pct, 2))
        elif price_model == 'CPC':
            val = cost / clk
            pct = ((goal - val) / goal) * 100
            actual.append(val)
            pcts.append(round(pct, 2))
        else:
            val = cost / (imp * 1000)
            pct = ((goal - val) / goal) * 100
            actual.append(val)
            pcts.append(round(pct, 2))
            
    return actual, pcts


# output the bid that currently running (up to yesterday)

def get_active_clt_for_imp(df):
    """
    @df: pandas.DataFrame
    @return: BIDs_active: List[int], clients currently running
    @return: BIDs_imp: List[int], clients DO NOT meet the criteria, performance above 5% for 3 days in the row, ready for values push
    """
    print(df.shape)
    BIDs_active = []
    max_date = df.date.max()
    for BID in df.bid.unique().tolist():
        if any([val == max_date for val in df[df.bid==BID].date.values]):
            BIDs_active.append(BID)
            
    df = df.iloc[[idx for idx, val in enumerate(df.bid.values) if val in BIDs_active],:]
    print(df.shape)
    
    df_new = df[['bid','date','imp','clk','tc','tc_ov','cost','price_model', 'roi_goal']]
    df_gp = df_new.groupby(by=['bid','date','price_model','roi_goal'], as_index=False).sum()
    
    df_gp['actual'] = calActualGoalPct(df_gp)[0]
    df_gp['pcts'] = calActualGoalPct(df_gp)[1]
    print(df_gp.shape)
    
    df_gp = df_gp[['bid','date','imp','clk','tc','tc_ov','cost','price_model','roi_goal','actual','pcts']]
    ################################################################
    import copy
#    df_save = copy.deepcopy(df_gp)
#    df_save = df_save[df_save.date>='2018-01-01']
#    df_save = df_save.rename(index=str, columns={"roi_goal": "goal"})
#    df_save.to_csv('perf_5pct_check_{}.csv'.format(datetime.now().date()), index=False)
    #######################################################################
    BIDs_drop = []
    for BID in df_gp.bid.unique().tolist():
        if all([val >= 5 for val in 
                df_gp[(df_gp.bid==BID)].sort_values(by='date',ascending=False).head(3).pcts.values]): # 5% & 3 days in a row
            BIDs_drop.append(BID)
    assert len(BIDs_active) > len(BIDs_drop)
    BIDs_imp = list(set(BIDs_active).difference(set(BIDs_drop)))
    
    return BIDs_active, BIDs_imp


# output the cids that currently running (up to yesterday)
# 3 cases: 1) running 30 days and above; 
#          2) running greater than 8 days and less than 30 days; 
#          3) running less than 8 days ---> not ready for training

# 1st phase output 3 distinct groups of cids

def get_active_cids_for_imp(df):
    """
    @df: pandas.DataFrame
    @return: CIDs_gt30: List[int], strategies currently running over 30 days up to date
    @return: CIDs_inBtw: List[int], strategies currently running btw 8~30 days up to date
    @return: CIDs_lt08: List[int], strategies currently running less than 8 days up to update, (Important! actual min. = 8 days)
    @return: df: pandas.DataFrame, outputs dataframe whose CIDS are currently running
    """
    print(df.shape)
    BIDs_active = []
    max_date = df.date.max()
    for BID in df.bid.unique().tolist():
        if any([val == max_date for val in df[df.bid==BID].date.values]):
            BIDs_active.append(BID)
            
    df = df.iloc[[idx for idx, val in enumerate(df.bid.values) if val in BIDs_active],:] #overwrite df to BIDs currently running
    print(df.shape)
    
    CIDs_active = []
    final_date = df.date.max()
    for CID in list(set(df.cid.values)): #
        if any([val == final_date for val in df[df.cid==CID].date.values]):
            CIDs_active.append(CID)
    df = df.iloc[[idx for idx, val in enumerate(df.cid.values) if val in CIDs_active], :] # overwrite df to CIDs currently running

    CIDs_gt30, CIDs_inBtw, CIDs_lt08 = set(), set(), set()
    for CID in CIDs_active: # iterate through cids that are active up to date
        df2 = df[df.cid == CID]
        df3 = df2.sort_values(by='date',ascending=False)
        
        if df3.shape[0] >= 30:
            CIDs_gt30.add(CID)
        elif df3.shape[0] >= 8 and df3.shape[0] < 30:
            CIDs_inBtw.add(CID)
        else:
            CIDs_lt08.add(CID)

    print(len(set(df.cid.values)))
    assert len(CIDs_gt30) + len(CIDs_inBtw) + len(CIDs_lt08) == len(set(df.cid.values))
    
    return CIDs_gt30, CIDs_inBtw, CIDs_lt08, df


def get_bid_cid_cnt_dic(df):
    """
    @df: pandas.DataFrame, df has CIDs currently running
    @return: bid_cid_cnt_dic: Dict{bid:(cid, cnt)}
    """
  
    bid_cids_lst = [(BID, df[df.bid==BID].cid.unique().tolist()) for BID in df.bid.unique().tolist()] # one-line list comprehension
    cid_cnt_lst = [(CID, df[df.cid==CID].shape[0])for CID in df.cid.unique().tolist()] # one-line list comprehension
    cid_cnt_dic ={cid: cnt for (cid, cnt) in cid_cnt_lst} # convert cid_cnt list to dict

    from collections import defaultdict 
    bid_cid_cnt_dic = defaultdict(list)
    for (bid, cids) in bid_cids_lst:
        for cid, cnt in cid_cnt_dic.items():
            if cid in cids:
                bid_cid_cnt_dic[bid].append((cid,cnt))
                
    return bid_cid_cnt_dic


def get_cids_all(bid_cid_cnt_dic, bids_imp):
    """
    @bid_cid_cnt_dic: Dict{bid[int]: (cid[int], cnt[int])}
    @bids_imp: List[int]
    @return: cids_all, List[List[int]]
    """
    tmp_dic = {bid: cid_cnt for bid, cid_cnt in bid_cid_cnt_dic.items() if bid in bids_imp}
    tmp_cids = [cid for bid in tmp_dic.keys() for cid, cnt in tmp_dic[bid] if cnt >= 8]
    from collections import defaultdict
    tmp_dic2 = defaultdict(list)
    for bid, cid_cnt in bid_cid_cnt_dic.items():
        for cid, cnt in cid_cnt:
            if cid in tmp_cids:
                tmp_dic2[bid].append(cid)

    cids_all = [cids_lst for cids_lst in tmp_dic2.values()]
    
    return cids_all


########
####### dev DBs updates
########

def db_update_dev(data):

    # dev DB in MYSQL
    connection = pymysql.connect(
        db='fuelAsset',
        host='35.226.210.107',
        port=3306,
        user='jasonhsiao',
        password='123'
    )
    print('connection successful')


    cursor = connection.cursor()
    # query1 = """
    # ........
    # """
    # cursor.execute(query1)
    # connection.commit()

    # first updates on campaign_budget table in MYSQL
    cnt = 0
    for row in data:
    #     if row[2] in ['20180606', '20180607', '20180609', '20180612']:
        # if row[2] == '20180608':
        # if row[2] == datetime.now().date().strftime('%Y-%m-%d'):
        update_totalSpd_query = \
        """
            UPDATE campaign_budget
            SET total_spend = {},
                updated = now()
            WHERE 1 = 1
            AND is_ongoing = 0
            AND cid = {}
            AND created = (
                SELECT cb.created
                FROM (
                    SELECT MAX(created) as created
                    FROM campaign_budget
                    WHERE cid = {} and is_ongoing = 0
                    GROUP BY cid
                ) cb
            )
        """.format(row[3], row[1], row[1])
        cursor.execute(update_totalSpd_query)
        connection.commit()
        cnt += 1
    print('updated', cnt, 'row(s) and rendered it in UI')
    print(cnt)

    cnt = 0
    for row in data:
    #     if row[2] in ['20180606', '20180607', '20180609', '20180612']:
        # if row[2] == '20180608':
        # if row[2] == datetime.now().date().strftime('%Y-%m-%d'):
        update_totalSpd_query2 = \
        """
            UPDATE campaign_budget
            SET total_spend = {},
                updated = now()
            WHERE 1 = 1
            AND is_ongoing = 1
            AND cid = {}
            AND created = (
                SELECT cb.created
                FROM (
                    SELECT MAX(created) as created
                    FROM campaign_budget
                    WHERE cid = {} and is_ongoing = 1
                    GROUP BY cid
                ) cb
            )
        """.format(row[4], row[1], row[1])
        cursor.execute(update_totalSpd_query2)
        connection.commit()
        cnt += 1
    print('updated', cnt, 'row(s) and rendered it in UI')
    print(cnt)

    # second update on spending table in MYSQL
    cnt = 0
    for row in data:
        # if row[2] == '20180608':
        # if row[2] == datetime.now().date().strftime('%Y-%m-%d'):
        update_rawDailyLimit_query = \
        """
            UPDATE fuelBidding.spending /*update OriginalDailyLimit */
            SET originalDailyLimit = {}, /* this value / (1-fac) equal to dailylimit rendered in ui */
                  updated = now()
            WHERE 1 = 1
            AND override = 1
            AND cid = {}
            AND updated = (
                SELECT cb.updated
                FROM (
                         SELECT MAX(updated) as updated
                         FROM fuelBidding.spending
                         WHERE cid = {} and override = 1
                         GROUP BY cid
                         ORDER BY updated DESC
                         LIMIT 1
                ) cb
            );
        """.format(row[2], row[1], row[1])
        cursor.execute(update_rawDailyLimit_query)
        connection.commit()
        cnt += 1
    print('updated', cnt, 'row(s) and rendered it in UI')
    print(cnt)

    # third updates on campaign table in MYSQL
#     connection = pymysql.connect(
#         db='fuelAsset',
#         host='fuelasset.db.fuel451.com',
#         port=3306,
#         user='jasonhsiao',
#         password='y1U$hcSPVsQW'
#     )
    print('connection successful')

    cursor = connection.cursor()

    cnt = 0
    for row in data:

    #     print(row[2],row[1])
        update_dailylimit_monthio_query = \
        """
            UPDATE campaign
            SET daily_limit = {},
                iotarget = {},
                created = now()
            WHERE 1 = 1
            AND id = {}
            AND created = (
                SELECT cb.created
                FROM (
                    SELECT MAX(created) as created
                    FROM campaign
                    WHERE id = {}
                    GROUP BY id
                ) cb
            )
        """.format(row[2], row[3], row[1], row[1])

        cursor.execute(update_dailylimit_monthio_query)
        connection.commit()
    #     connection.close()
        cnt += 1
    print('updated', cnt, 'row(s) and rendered it in UI')
    print(cnt)


######################
##### production DBs updates
#######################

def db_update_prod(data):

    # production DB in MYSQL
    connection = pymysql.connect(
        db='fuelAsset',
        host='fuelasset.db.fuel451.com',
        port=3306,
        user='jasonhsiao',
        password='y1U$hcSPVsQW'
    )

    print('connection successful')

    cursor = connection.cursor()
    # query1 = """
    # ........
    # """
    # cursor.execute(query1)
    # connection.commit()

    # first updates on campaign_budget table in MYSQL
    cnt = 0
    for row in data:
    #     if row[2] in ['20180606', '20180607', '20180609', '20180612']:
        # if row[2] == '20180608':
        # if row[2] == datetime.now().date().strftime('%Y-%m-%d'):
        update_totalSpd_query = \
        """
            UPDATE campaign_budget
            SET total_spend = {},
                updated = now()
            WHERE 1 = 1
            AND is_ongoing = 0
            AND cid = {}
            AND created = (
                SELECT cb.created
                FROM (
                    SELECT MAX(created) as created
                    FROM campaign_budget
                    WHERE cid = {} and is_ongoing = 0
                    GROUP BY cid
                ) cb
            )
        """.format(row[3], row[1], row[1])
        cursor.execute(update_totalSpd_query)
        connection.commit()
        cnt += 1
    print('updated', cnt, 'row(s) and rendered it in UI')
    print(cnt)

    cnt = 0
    for row in data:
    #     if row[2] in ['20180606', '20180607', '20180609', '20180612']:
        # if row[2] == '20180608':
        # if row[2] == datetime.now().date().strftime('%Y-%m-%d'):
        update_totalSpd_query2 = \
        """
            UPDATE campaign_budget
            SET total_spend = {},
                updated = now()
            WHERE 1 = 1
            AND is_ongoing = 1
            AND cid = {}
            AND created = (
                SELECT cb.created
                FROM (
                    SELECT MAX(created) as created
                    FROM campaign_budget
                    WHERE cid = {} and is_ongoing = 1
                    GROUP BY cid
                ) cb
            )
        """.format(row[4], row[1], row[1])
        cursor.execute(update_totalSpd_query2)
        connection.commit()
        cnt += 1
    print('updated', cnt, 'row(s) and rendered it in UI')
    print(cnt)

    # second update on spending table in MYSQL

    # produection db, fuelData, table, spending to update original daily limit records
    connection = pymysql.connect(
        db='fuelData',
        host='162.222.180.40',
        port=3306,
        user='jasonhsiao',
        password='a0X@vHLsMzVaz'
    )
    print('connection successful')

    cursor = connection.cursor()

    cnt = 0
    for row in data:
        # if row[2] == '20180608':
        # if row[2] == datetime.now().date().strftime('%Y-%m-%d'):
        update_rawDailyLimit_query = \
        """
            UPDATE spending /*update col, OriginalDailyLimit */
            -- UPDATE fuelBidding.spending /*update OriginalDailyLimit */
            SET originalDailyLimit = {}, /* this value / (1-fac) equal to dailylimit rendered in ui */
                  updated = now()
            WHERE 1 = 1
            AND override = 1
            AND cid = {}
            AND updated = (
                SELECT cb.updated
                FROM (
                         SELECT MAX(updated) as updated
                         FROM spending
                         WHERE cid = {} and override = 1
                         GROUP BY cid
                         ORDER BY updated DESC
                         LIMIT 1
                ) cb
            );
        """.format(row[2], row[1], row[1])
        cursor.execute(update_rawDailyLimit_query)
        connection.commit()
        cnt += 1
    print('updated', cnt, 'row(s) and rendered it in UI')
    print(cnt)

    # third updates on campaign table in MYSQL
    connection = pymysql.connect(
        db='fuelAsset',
        host='fuelasset.db.fuel451.com',
        port=3306,
        user='jasonhsiao',
        password='y1U$hcSPVsQW'
    )

    print('connection successful')

    cursor = connection.cursor()

    cnt = 0
    for row in data:

    #     print(row[2],row[1])
        update_dailylimit_monthio_query = \
        """
            UPDATE campaign
            SET daily_limit = {},
                iotarget = {},
                created = now()
            WHERE 1 = 1
            AND id = {}
            AND created = (
                SELECT cb.created
                FROM (
                    SELECT MAX(created) as created
                    FROM campaign
                    WHERE id = {}
                    GROUP BY id
                ) cb
            )
        """.format(row[2], row[3], row[1], row[1])

        cursor.execute(update_dailylimit_monthio_query)
        connection.commit()
    #     connection.close()
        cnt += 1
    print('updated', cnt, 'row(s) and rendered it in UI')
    print(cnt)












if __name__ == '__main__':

    t0 = datetime.now()
    df = clean_raw_data()
    #df = pd.read_csv('/home/jhsiao_fuelx_com/Projects/p-p-balancer-project/data/all_mdf_v2_2018-09-24.csv')
    BIDs_active, BIDs_imp = get_active_clt_for_imp(df)
    CIDs_gt30, CIDs_inBtw, CIDs_lt08, df_active = get_active_cids_for_imp(df)
    bid_cid_cnt_dic = get_bid_cid_cnt_dic(df_active)

    #BIDs_input = [2168,2129,2023,1985,1940]
    #BIDs_input = [2168,2023] # overwrite the BIDs_imp above to manually control the input
    #BIDs_input = [1913,2088,2043,1941,1781,1674,1729,2124,2111,1897,2046,1978,2170,2040,1793,2065] # ph1
    #BIDs_input = [1782,1841,1701,1926,1580,1641,2131,2139,2009,1867,2030,2084,1798,1912,1840,1757,1975,2032,2024] # ph2
    #BIDs_input = [2134,1882,2168,2132,1921,2023,2052,967,2099,2149,1985,1899,1630,2116,1906,2191,1940,1874,2180,2129] # ph3
    #BIDs_input = [2192,1947,1989,1960,2068,1919,1752,1845,1932,1939,1802,1821,2039,2119,2160] # ph4
#    BIDs_input = [2088,2043,1941,1781,1674,1729,2124,2111,1897,2046,2040,1793,2065,
#                  1782,1841,1701,1926,1580,1641,2131,2139,1867,2084,1798,1912,1840,1757,1975,2032,2024,
#                  2134,1882,2168,2132,1921,2023,2052,967,2099,2149,1985,1899,1630,2116,1906,2191,1940,1874,2180,2129,
#                  2192,1947,1989,1960,2068,1919,1752,1845,1932,1939,1802,1821,2039,2119,2160] #ph1&2&3&4
##############################################################################################
#    BIDs_input = [2088,2043,1941,1781,1674,1729,2124,2111,1897,2046,2040,1793,2065,
#                  1841,1701,1926,1580,1641,2131,2139,1867,2084,1798,1912,1840,1757,1975,2032,2024,
#                  2134,1882,2168,2132,1921,2023,2052,967,2099,2149,1985,1899,1630,2116,1906,2191,1940,1874,2180,2129,
#                  2192,1947,1989,1960,2068,1919,1752,1845,1932,1939,1802,1821,2039,2119,2160] #ph1&2&3&4
#    BIDs_all = [bid for bid in BIDs_input if bid in BIDs_imp]
##############################################################################################
    BIDs_rmd = [2085,1810,1751,2170,1913,1782,1978,2009,2030,
                1538,1827,2092,1885,1898,
                1935,2180,2149,2150,2123] # being removed from training list
    BIDs_all = list(set(BIDs_imp).difference(set(BIDs_rmd)))
#############################################################################################
    BIDs_all = BIDs_all[25:]


    #import copy
    #BIDs_all = copy.deepcopy(BIDs_imp) # for 100% client list implementation
    print(BIDs_all)
    print(len(BIDs_all))
    CIDs_all = get_cids_all(bid_cid_cnt_dic, BIDs_all) # get CIDs_all function
#    CIDs_all = [[3815,3817,3819,3821]]
    print(CIDs_all)
    print(len([cid for sublist in CIDs_all for cid in sublist]))

    records = []
    cid_tmr_pt_all = []
    #df_lst_all = []
    d_all = dict()
    cnt_clt = 0
    cnt2 = 0
    for CIDs in CIDs_all:
        cnt_clt += 1
        df_lst = []
        cnt = 0
        for CID in CIDs[:]:
    #       cnt = cnt + 1
            print('CID: ', CID)
    #         m, n = 32, 1
            m, n = 1, 1
            try:
                df_comp = main()
                df_lst.append(df_comp)
            except Exception:
                print('CID to drop: ', CID)
                CIDs.remove(CID)
                print('shape mismatch')
            cnt += 1

            if cnt == 1:
                print('{}st train set: '.format(cnt))
            elif cnt == 2:
                print('{}nd train set: '.format(cnt))
            elif cnt == 3:
                print('{}rd train set: '.format(cnt))
            else:
                print('{}th train set: '.format(cnt))
        cnt2 += cnt
        print('CURRENTLY TRAINING ON: ', cnt2, 'th strategy overall')
        #print('\n')
        #print('TOTAL TRAINING SETS: ', cnt2)
        #print('\n')

        print(len(df_lst))
        assert len(df_lst) == len(CIDs)
        #df_lst_all.append(df_lst)
        dic = dict(zip(CIDs, df_lst))
        d_all.update(dic)
        #import copy
        #dic_all = copy.deepcopy(dic)
        #df_lst_all.append(dic)
        with open('/home/jhsiao_fuelx_com/Projects/p-p-balancer-project/data/spd_pred_2ndPool_v3_{}.pkl'\
                  .format(datetime.now().date()-timedelta(1)),'wb') as fh:
            pickle.dump(dic, fh)
        with open('/home/jhsiao_fuelx_com/Projects/p-p-balancer-project/data/spd_pred_all_v3_{}.pkl'\
                  .format(datetime.now().date()),'wb') as fh:
            pickle.dump(d_all, fh)
        #with open('spd_pred_v2_{}.pkl'.format(datetime.now().date()),'wb') as fh:
        #with open('data/spd_pred_v2_{}.pkl'.format(datetime.now().date()), 'wb') as fh:
    #        pickle.dump(dic, fh)
    #    print('write predicitons to local')
    #    print('running time: ', datetime.now() - t0)

        _, comp_dic = get_iotarget_dailylimit_frm_db()
        cid_dailylimit_dic = {cid: dailylimit for cid, (_, dailylimit) in comp_dic.items()}
        daily_limit_dic = {cid: dailylimit for cid, dailylimit in cid_dailylimit_dic.items()
                           if cid in CIDs}

        filename = '/home/jhsiao_fuelx_com/Projects/p-p-balancer-project/data/spd_pred_2ndPool_v3_{}.pkl'.format(datetime.now().date()-timedelta(1))
        daily_pred_spd_dic = get_daily_pred_spd_dic(filename, CIDs)

        d_tmr, _ = consume_dailyIO3(daily_pred_spd_dic, daily_limit_dic)

        import copy
        d_pt = copy.deepcopy(daily_limit_dic)

        daily_spd = get_daily_spd_two_days(d_tmr, d_pt)
        cid_fac = get_dailylimit_factor_frm_db()
        cid_total_spd = get_total_spd_frm_db()

        data, cid_tmr_pt = data_grab(daily_spd, cid_fac, cid_total_spd)
        print(len(data))
        data = sorted(data, key=lambda x: x[1])
        print(data)
        records.append(data)
        cid_tmr_pt_all.append(cid_tmr_pt)
        db_update_dev(data)
#        db_update_prod(data)

#    with open('/home/jhsiao_fuelx_com/Projects/p-p-balancer-project/data/records_{}.pkl'.format(datetime.now().date()-timedelta(1)),'wb') as fh:

    with open('/home/jhsiao_fuelx_com/Projects/p-p-balancer-project/data/records_v3_{}_allph.pkl'.format(datetime.now().date()),'wb') as fh:
        pickle.dump(records, fh)

    with open('/home/jhsiao_fuelx_com/Projects/p-p-balancer-project/data/cid_tmr_pt_vals_v3_{}_allph.pkl'.format(datetime.now().date()), 'wb') as fh:
        pickle.dump(cid_tmr_pt_all, fh)

    print('NUMBER of CLIENTS for TRAINING: ', cnt_clt)
    print('\n')
    assert cnt2 == len([cid for sublist in CIDs_all for cid in sublist])
    print('TOTAL TRAINING SETS at CID level: ', cnt2)
    print('\n')
    print(cid_tmr_pt_all)
    print('\n')
    print(records)
    print('\n')
    print('RUNNING TIME: ', datetime.now() - t0)
    print('\n')
