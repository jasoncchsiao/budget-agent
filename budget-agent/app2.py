#!/usr/bin/env python
# app2.py


#############################
#!/usr/bin/env python
# db_conn.py

import psycopg2
import pymysql

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
dbname = 'xxxxxx'
host1 = 'xxxxxxxxxxxxxxxxxxxxxxxxx'
port1 = 'xxxxxxxxxxxxxxxxxxxxxxxxx'
user = 'xxxxxxxxxxxxxxxxxxxxxxxxx'
password1 = 'xxxxxxxxxxxxxxxxxxxxxxxxx'
db = 'xxxxxxxxxxxxxxxxxxxxxxxxx'
host2 = 'xxxxxxxxxxxxxxxxxxxxxxxxx'
port2 = xxxxxxxxxxxxxxxxxxxxxxxxx
user = 'xxxxxxxxxxxxxxxxxxxxxxxxx'
password2 = 'xxxxxxxxxxxxxxxxxxxxxxxxx'

def get_multiple_tbs():
    """
    rtype: pandas dataframe
    """
    with PostGreSQLConnection(dbname, host1, port1, user, password1) as conn1:
        with MySQLConnection(db, host2, port2, user, password2) as conn2:
            print('connections successful')
            sql_query = """
                    SELECT
                        tbl1.bid,
                        tbl1.cid,
                        tbl1.date,
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
                        GROUP BY 1, 2, 3
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
                                AND CAST(datepart(hour,insertion_timestamp) AS CHAR(2)) = 23
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
                """
            df1 = pd.read_sql(sql_query, conn1)
            df1.to_csv('/home/jhsiao_fuelx_com/Projects/p-p-balancer-project/data/adx_funnel_combo_{}.csv'.format(datetime.now().date()),index=False)
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
            df2.to_csv('/home/jhsiao_fuelx_com/Projects/p-p-balancer-project/data/objective_{}.csv'.format(datetime.now().date()), index=False)
            
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
            df3.to_csv('/home/jhsiao_fuelx_com/Projects/p-p-balancer-project/data/roi_goal_{}.csv'.format(datetime.now().date()),index=False)
            
    return df1, df2, df3

####################################################################################################################

#!/usr/bin/env python
# data_transformation.py

import pandas as pd
from datetime import datetime
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os.path


__author__ = 'jhsiao'

def clean_raw_data():

    path1 = '/home/jhsiao_fuelx_com/Projects/p-p-balancer-project/data/adx_funnel_combo_{}.csv'.format(datetime.now().date())
    path2 = '/home/jhsiao_fuelx_com/Projects/p-p-balancer-project/data/objective_{}.csv'.format(datetime.now().date())
    path3 = '/home/jhsiao_fuelx_com/Projects/p-p-balancer-project/data/roi_goal_{}.csv'.format(datetime.now().date())
    if os.path.isfile(path1) and os.path.isfile(path2) and os.path.isfile(path3):
        df1, df2, df3 = pd.read_csv(path1).drop_duplicates(), pd.read_csv(path2), pd.read_csv(path3)
    else:
        df1, df2, df3 = get_multiple_tbs()
        df1 = df1.drop_duplicates()
    df4 = pd.merge(df2, df3, how='inner', on='bid')
    df_new = pd.merge(df1, df4, how='inner', on='bid')
    d = {1: 'CPC', 2: 'ROI', 3: 'CPM', 4: 'PCTR', 6: 'CPA'}
    df_new['price_model'] = df_new['objective'].map(d)
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

def perf_clf(df): ### update on 5/21/18
    perf_clf_lst = []
    final_idx = []
    delta_lst = []
    costs, tcs, clks, imps, tc_ovs, goals, models = df.cost.values, df.tc.values, df.clk.values, df.imp.values, df.tc_ov.values, df.roi_goal.values, df.price_model.values
    for idx ,(cost, tc, clk, imp, tc_ov, goal, model) in enumerate(
      zip(costs, tcs, clks, imps, tc_ovs, goals, models)
    ):
        if model == 'ROI':
            if cost == 0:
                perf_clf_lst.append(0) # change to 0 on 5/21/18
                final_idx.append(idx)
                delta_lst.append(0)
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
            if tc == 0:
                perf_clf_lst.append(0)
                final_idx.append(idx)
                delta_lst.append(0)
            elif cost == 0: # added this edge case on 5/21/18
                perf_clf_lst.append(0)
                final_idx.append(idx)
                delta_lst.append(0)
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
            if clk == 0:
                perf_clf_lst.append(0)
                final_idx.append(idx)
                delta_lst.append(0)
            elif cost == 0: # added this edge case on 5/21/18
                perf_clf_lst.append(0)
                final_idx.append(idx)
                delta_lst.append(0)
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
            if imp == 0:
                perf_clf_lst.append(0)
                final_idx.append(idx)
                delta_lst.append(0)
            elif cost == 0: # added this edge case on 5/21/18
                perf_clf_lst.append(0)
                final_idx.append(idx)
                delta_lst.append(0)
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
        '/home/jhsiao_fuelx_com/Projects/p-p-balancer-project/data/all_mdf_{}.csv'.format(datetime.now().date())
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
        '/home/jhsiao_fuelx_com/Projects/p-p-balancer-project/data/all_mdf_{}.csv'.format(datetime.now().date())
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


def synthesize_feature_data(df, m):
    """
    :type df: pandas.dataframe
    :type m: int
    update on 5/29
    """
#     df_tail = df.tail(m)
    if any(ele for ele in df.perf_clf.values) == 1 and any(ele for ele in df.pace_clf.values) == 1:
        df_tail = df[(df.perf_clf==1)&(df.pace_clf==1)].tail(3) ### update on 5/25
    elif any(ele for ele in df.pace_clf.values) == 1 and all(ele for ele in df.perf_clf.values) == 0:
        df_tail = df[df.pace_clf==1].tail(3)
    elif any(ele for ele in df.perf_clf.values) == 1 and all(ele for ele in df.pace_clf.values) == 0:
        df_tail = df[df.perf_clf==1].tail(3)
    else:
        df_tail = df.tail(3)
    np_row1 = df_tail.mean().values.reshape(1,-1)
#     np_row1[:,-2], np_row1[:,-1] = round(np_row1[:,-2]), round(np_row1[:,-1])
    np_row1[:,-3], np_row1[:,-2] = 1, 1
#    np_row1[:,-2], np_row1[:,-1] = 1, 1
    np_mrows = np.array([np_row1 for i in range(m)]).reshape(m, len(df.columns))
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
    k = 7
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
    df = synthesize_feature_data(df, m)
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

if __name__ == '__main__':
    t0 = datetime.now()
    
    data = clean_raw_data()
    print('#CIDs in total: ',data.cid.nunique())
    data.to_csv('/home/jhsiao_fuelx_com/Projects/p-p-balancer-project/data/all_mdf_{}.csv'.format(datetime.now().date()),index=False)
    print('write the cleaned, merged df to local')
    CIDs = cids_list_all(data)
    print('#CIDs: ', len(CIDs))

    _, _, CIDs = get_fully_trained_bids()
    print('# CIDs running for training: ', len(CIDs))
#     CIDs = cids_list(data, 2085) # Leesa:1932, Thrive: 2085, Olivers: 1798, Chubbies: 1774, Glossier: 1885
#     print('#CIDs for bid=2085: ', len(CIDs))
    
#    CIDs = sorted(CIDs, reverse=True)
#    CIDs = [4393]
#     CIDs = sorted(data[data.bid==1932].cid.unique().tolist()) # Leesa
#     print('#CIDs for bid=1932', len(CIDs))
    
#     CIDs = [2276,2277,4449,4486]
#    CIDs = [4391, 4392, 4393]
#    CIDs = [2393, 2394]

    df_lst = []
    cnt = 0
    for CID in CIDs[:]:
#       cnt = cnt + 1
        print('CID: ', CID)
#         m, n = 32, 1
        m, n = 4, 1
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
    print('total train sets: ', cnt)



            
    print(len(df_lst))
    assert len(df_lst) == len(CIDs)
    dic = dict(zip(CIDs, df_lst))
    with open('/home/jhsiao_fuelx_com/Projects/p-p-balancer-project/data/spd_pred_v2_{}.pkl'.format(datetime.now().date()),'wb') as fh:
    #with open('spd_pred_v2_{}.pkl'.format(datetime.now().date()),'wb') as fh:
    #with open('data/spd_pred_v2_{}.pkl'.format(datetime.now().date()), 'wb') as fh:
        pickle.dump(dic, fh)
    print('write predicitons to local')
    print('running time: ', datetime.now() - t0)

