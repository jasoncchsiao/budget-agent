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

from utils.database import get_multiple_tbs

__author__ = 'jhsiao'


def clean_raw_data():

    path1 = 'adx_funnel_combo_{}.csv'.format(datetime.now().date())
    path2 = 'objective_{}.csv'.format(datetime.now().date())
    path3 = 'roi_goal_{}.csv'.format(datetime.now().date())

    if os.path.isfile(path1) and os.path.isfile(path2) and os.path.isfile(path3):
        df1, df2, df3 = pd.read_csv(path1).drop_duplicates(), \
                        pd.read_csv(path2), \
                        pd.read_csv(path3)
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
    costs, tcs, clks, imps, tc_ovs, goals, models = \
    df.cost.values, df.tc.values, df.clk.values, df.imp.values, \
    df.tc_ov.values, df.roi_goal.values, df.price_model.values
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

def get_fully_trained_bids():
    """
    """

    data = pd.read_csv('data/all_mdf_{}.csv'.format(datetime.now().date()))
    from collections import defaultdict
    bid_cid_dic = defaultdict(list)
    for k, v in zip(data.bid.values, data.cid.values):
        bid_cid_dic[k].append(v)
    bid_cid_dic2 = {
        k: list(set(v)) for k, v in bid_cid_dic.items()
    }
    cids_lst_all = cids_list_all(data)
    print('cids running/lasting for 30 days: ', len(cids_lst_all))

    bids_lst_update = []
    for k, v in bid_cid_dic2.items():
        for ele in v:
            if ele in cids_lst_all:
                bids_lst_update.append(k)
    new_bids_lst = list(set(bids_lst_update))

    bids_ls = new_bids_lst # 92 bids currently running up to date

    #######################################################
    bids_boo = [cids_list(bid)[-1] for bid in bids_ls]

    #######################################################
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

def get_data():

    path = 'all_mdf3_2018-05-14.csv'
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

def cids_list(df, BID):
    """
    rtype: CIDs_lst: cid list for TRAINING given a BID
    """
    final_date = df.date.max()
    df2 = df[df.bid == BID]
    CIDs_lst = []
    for CID in list(set(df2.cid.values)):

        df3 = df2[df2.cid == CID]
        df4 = df3.sort_values(by='date', ascending=True)
        if df4.date.max() == final_date and \
          df4.shape[0] >= 30:
            CIDs_lst.append(CID)
    CIDs_lst = sorted(CIDs_lst)
    return CIDs_lst

def cids_list_all(df):
    """
    rtype: CIDs_lst: a full cid list for TRAINING
    """
    final_date = df.date.max()
    CIDs_lst = []
    for CID in df.cid.unique().tolist():
        df2 = df[df.cid==CID]
        df3 = df2.sort_values(by='date', ascending=True)
        if df3.date.max() == final_date and \
            df3.shape[0] >= 30:
            CIDs_lst.append(CID)
    CIDs_lst = sorted(CIDs_lst)
    return CIDs_lst

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
    np_row1[:,-2], np_row1[:,-1] = 1, 1
    np_mrows = np.array([np_row1 for i in range(m)]).reshape(m, len(df.columns))
    df_mrows = pd.DataFrame(np_mrows, columns=df.columns)
    df_total = pd.concat([df, df_mrows], axis=0)
    df = df_total
    return df

def transformer_data(df):
    train_test_split_rat = 0.70

    ### update on 6/3/18
    assert isinstance(df, pd.DataFrame)
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    df = df[indices_to_keep].astype(np.float64)

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
    k = 15
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
