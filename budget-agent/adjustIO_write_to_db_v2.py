#!/usr/bin/env python
# adjustIO_write_to_db_v2.py

from utils.db_conn import MySQLConnection
import numpy as np
import pandas as pd
import pymysql
from datetime import datetime, date, timedelta
from calendar import monthrange
from functools import reduce
import pickle
import yaml
import os.path
#########################################################################

def cids_list(BID):
    """
    rtype: CIDs_lst: cid list for TRAINING given a BID
    """

    df = pd.read_csv('data/all_mdf_{}.csv'.format(datetime.now().date()))

    final_date = df.date.max()
    df2 = df[df.bid == BID]
    CIDs_lst = []
    CIDs_lst_wo = []
    for CID in list(set(df2.cid.values)):

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
    rtype: CIDs_lst: a full cid list currently running up to date
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

#############################################################################
#############################################################################
#############################################################################
def consume3(dailylimit, days_for_pred, CID, mode='fixed'):
    """
    :type dailylimit: float
    :type days_for_pred: int
    :type CID: int
    :type mode: str
    :rtype:
    """
    global spd_lst

    amt = dailylimit*days_for_pred
    cnt = 0
    if mode == 'fixed':
        spd_lst = [ dailylimit for _ in range(days_for_pred)]
    elif mode == 'pred':

        with open('data/cid_pred_dic_{}.pkl'.format(datetime.now().date()), 'rb') as fh:

            cid_pred_dic = pickle.load(fh)
            spd_lst = [cid_pred_dic[CID] for _ in range(days_for_pred)]

    else:
        spd_lst = np.random.choice(np.arange(dailylimit*0.9,dailylimit*1.1), days_for_pred).tolist()
    print(sum(spd_lst))

    if amt <= sum(spd_lst):
        actual_spd = []
        for spd in spd_lst:
            if amt >= 0 and amt >= spd:
                amt -= spd
                cnt += 1
                print('spending for', cnt, 'day(s)')
                print('amount left for spending:', amt, 'after day', cnt)
                actual_spd.append(spd)
            elif amt >= 0 and amt < spd:
                spd = amt
                amt -= spd
                cnt += 1
                print('spending for', cnt, 'day(s)')
                print('amount left for spending:', amt, 'after day', cnt)
                actual_spd.append(spd)
            else:
                print('no more left')
                break
        return actual_spd, amt==0
    else:
        print('amount exceeds sum of spendings')
        amt_exceeds = amt - sum(spd_lst)
        print('more to spent:', amt_exceeds)
    return amt_exceeds, False


def get_params(BID):

    data = pd.read_csv('data/all_mdf_{}.csv'.format(datetime.now().date()))


    _, cid_lst_all, boo = cids_list(BID) # for a given BID ONLY
    print(len(cid_lst_all), boo)

    data2 = data[['cid','date','dailylimit']] \
            .iloc[[idx for idx, v in enumerate(data.cid.values)
                    if v in cid_lst_all],:]
    test_gp = data2.sort_values(
                    by=['cid','date'], ascending=[1,0])[['cid','dailylimit']] \
                    .groupby(by='cid', as_index=False).first()
    test_gp = test_gp.sort_values(by='dailylimit', ascending=False)
    daily_limit_dic = {k:v for k, v in zip(test_gp.cid.values, test_gp.dailylimit.values) }
    daily_limit_sum = test_gp.dailylimit.sum()

    with open('data/cid_pred_dic_{}.pkl'.format(datetime.now().date()), 'rb') as fh:



        cid_pred_dic = pickle.load(fh)
    trained_cids = [cid for cid in cid_lst_all
                    if cid in [cid for cid in cid_pred_dic.keys()]]
    not_trained_cids = [cid for cid in cid_lst_all
                        if cid not in trained_cids]

    daily_pred_spd_dic = {
	k: v for k, v in zip([k for k in trained_cids+not_trained_cids],
			     ([cid_pred_dic[k] for k in trained_cids]
			     + [0 for k in not_trained_cids]))
    }

    return cid_lst_all==trained_cids, \
            trained_cids, not_trained_cids, \
            daily_limit_dic, daily_limit_sum, \
            daily_pred_spd_dic


def consume_dailyIO(BID, mode='ml_based'):
    """
    :type mode: str ['ml_based','rand_based','fixed', 'hybrid']
    """
    boo, _, _, daily_limit_dic, _, daily_pred_spd_dic = get_params(BID)

    # keep track of keys(=cids) of daily_limit_dic for spd in descending order
    cids = [k for k in daily_limit_dic.keys()]
    if boo:

        dailyIO_lst = [v for v in daily_limit_dic.values()]
        dailyIO_sum = sum(dailyIO_lst)

        if mode == 'ml_based':
            pred_dailyIO_lst = [v for v in daily_pred_spd_dic.values()]
        elif mode == 'rand_based':
            pred_dailyIO_lst = [np.random.choice(np.arange(v*0.9,v*1.1, 0.02))
                                for v in daily_limit_dic.values()]
        elif mode == 'hybrid':
            pass
        else:
            pred_dailyIO_lst = dailyIO_lst

        pred_dailyIO_sum = sum(pred_dailyIO_lst)

        delta = dailyIO_sum - pred_dailyIO_sum
        print('original dailyIO: ', dailyIO_sum)
        print('original delta: ', delta)


        if delta > 0:
    #         print('more to spend for the next day: $', delta)

            print('more to spend among campaigns: $', delta)
    #         pred_dailyIO_lst.append(delta) # maybe in the future, create a campaign for spd
            pred_dailyIO_lst = [ele+(delta/len(pred_dailyIO_lst)) for ele in pred_dailyIO_lst]

        cnt = 0
        actual_spd = []
        for spd in pred_dailyIO_lst:
            if dailyIO_sum >= 0 and dailyIO_sum >= spd:
                dailyIO_sum -= spd
                cnt += 1
                actual_spd.append(spd)
            elif dailyIO_sum >= 0 and dailyIO_sum < spd:
                spd = dailyIO_sum
                dailyIO_sum -= spd
                cnt += 1
                actual_spd.append(spd)
            else:
                break
        print('cnt:', cnt)
        assert cnt == len(pred_dailyIO_lst)
        print('dailyIO after consumption: ', dailyIO_sum)
        actual_spd_dic = {k:v for k,v in zip(cids, actual_spd)}
        return actual_spd_dic, round(dailyIO_sum, 2)==0
#         return actual_spd, round(dailyIO_sum, 2) == 0

    else:
        print('not a fully trained client')
        print('please reinsert a bid:')

def consume_dailyIO2(BID):
    """
    :type BID: int
    :type mode: str ['ml_based','rand_based','fixed', 'hybrid']
    :rtype actual_spd_dic: dictionary
    """
    boo, _, _, daily_limit_dic, _, daily_pred_spd_dic = get_params(BID)

    daily_pred_spd_dic_tup_lst = sorted([(k,v) for k, v in daily_pred_spd_dic.items()],
                                        key=lambda x: x[1],
                                        reverse=True)

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
                # for example, [32,18,0]
                # insert 10% of sum into index2 which is 4 in this case
                # other non-zero elements take 10%~15% off from itself
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
                print('actual_spd: ', actual_spd)
                print('sum of actual_spd: ', sum(actual_spd))

                print('last elements spd: ', (1-pct)*dailyIO/length_zeros)
                print('last_elements_lst: ', length_zeros*[(1-pct)*dailyIO/length_zeros])
#####################################recursive call  ################################
#                 def get_actual_spd_lst(pcts=0.21):
#                     for pct in np.arange(pcts, 0, -0.03).tolist():
#                     while pct > 0:
#                         if any([ele < min([val for val in daily_pred_spd_dic.values()])
#                                 for ele in length_zeros*[(1-pct)*dailyIO/length_zeros]]):
#                             actual_spd[-length_zeros:] = [(1-pct)*dailyIO/length_zeros
#                                                           for _ in range(length_zeros)]
#                             return actual_spd
#                         pct -= 0.03
#                         return get_actual_spd_lst(pct)

#                 def get_actual_spd_lst(pcts=0.21):
#                     if any([ele < min([val for val in daily_pred_spd_dic.values()])
#                             for ele in length_zeros*[(1-pcts)*dailyIO/length_zeros]]):
#                         actual_spd[-length_zeros:] = [(1-pcts)*dailyIO/length_zeros
#                                                       for _ in range(length_zeros)]
#                         print('ACTUAL_SPD: ', actual_spd)
#                     for e in np.arange(pcts, 0, -0.03).tolist():
#                         if any([ele >= min([val for val in daily_pred_spd_dic.values()])
#                                 for ele in length_zeros*[(1-e)*dailyIO/length_zeros]]):
# #                             print()
#                             return get_actual_spd_lst(e)
#                     return actual_spd
####################################################################################
                actual_spd[-length_zeros:] = [(1-pct)*dailyIO/length_zeros
                                              for _ in range(length_zeros)]
                print('actual_spd: ', actual_spd)
#                 assert sum(actual_spd) == dailyIO
        actual_spd_dic = {k:v for k,v in zip(cids, actual_spd)}
        print('dailyIO_sum after: ',dailyIO_sum)
        print('actual_spd_sum: ', sum([v for v in actual_spd_dic.values()]))
        return actual_spd_dic, round(dailyIO_sum, 2)==0
    else:
        print('not a fully trained client')
        print('please reinsert a bid:')



def get_bid_clt_dic():

    with open('data/bid_clt_dic.pkl', 'rb') as fh:
        bid_clt_dic = pickle.load(fh)
    return bid_clt_dic

def get_daily_limit_tup_lst():



    daily_limit_dic_lst = [get_params(bid)[3]
                           for bid in bids_lst_boo_true]
    daily_limit_tup_lst = sorted(
        [(cid, dailylimit) for dic in daily_limit_dic_lst
         for cid, dailylimit in dic.items()],
        key=lambda x: x[0]
    )
    return daily_limit_tup_lst


def get_cid_pred_dic():


    with open('data/spd_pred_v2_{}.pkl'.format(datetime.now().date()), 'rb') as fh:

        dic_spd = pickle.load(fh)

    data_np_lst = []
    for k, df in dic_spd.items():
        df['cid'] = [k for i in range(df.shape[0])]
        df['date'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df['given_date_forcast'] = [datetime.now().date().strftime('%Y-%m-%d')
                                    for i in range(df.shape[0])]

        df['real_spd'] = df['real_spd'].apply(lambda x: round(x, 2))
        df.fillna('null', inplace=True)
        df['prediction_spd'] = df['prediction spd'].apply(lambda x: round(x, 2))
        df = df[['cid','given_date_forcast','date','real_spd','prediction_spd']]
        data_np = df.values
        data_np_lst.append(data_np)
    data_np_lst

    data = reduce(lambda x, y: np.concatenate((x,y), axis=0), data_np_lst).tolist()
    df_all = pd.DataFrame(data=data,
                          columns=['cid','given_date_forcast','date',
                                   'real_spd','prediction_spd'])

    df_cid_pred_dic = df_all.sort_values(by=['cid','date'], ascending=[1,0]) \
                        [['cid','prediction_spd']] \
                        .groupby(by='cid', as_index=False).first()
    cid_pred_dic = {k:v
            for k, v in zip(df_cid_pred_dic.cid.values,
            df_cid_pred_dic.prediction_spd.values)}
    cid_pred_dic = {k:v if v >= 0 else 0
            for k, v in zip(df_cid_pred_dic.cid.values,
            df_cid_pred_dic.prediction_spd.values)}


    with open('data/cid_pred_dic_{}.pkl'.format(datetime.now().date()), 'wb') as fh:

        pickle.dump(cid_pred_dic, fh)

    return cid_pred_dic



def get_adjusted_spd_dic():

    results_dic_new = {bid: consume_dailyIO2(bid)
                       for bid in bids_lst_boo_true}




    with open('data/results_dic_{}.pkl'.format(datetime.now().date()), 'wb') as fh:
        pickle.dump(results_dic_new, fh)

    adjusted_spd_tup_lst = [(cid, pred)
                            for _, (inner_dic,_) in results_dic_new.items()
                            for cid,pred in inner_dic.items()]
    adjusted_spd_dic = {cid:pred for cid, pred in adjusted_spd_tup_lst}
    return adjusted_spd_dic, results_dic_new

def get_bid_cid_reverse_map():



    with open('data/results_dic_{}.pkl'.format(datetime.now().date()), 'rb') as fh:
        results_dic_new = pickle.load(fh)
    bid_cid_map = {bid: [cid for cid in inner_dic.keys()]
                   for bid, (inner_dic,_) in results_dic_new.items()}

    len_dic = {k:len(v) for k, v in bid_cid_map.items()}

    bid_cid_reverse_map = {cid: bid for cid, bid in
                       zip([v for cid_list in bid_cid_map.values() for v in cid_list],
                           [k for k, v in len_dic.items() for _ in range(v)])}

    return bid_cid_reverse_map



t0 = datetime.now()
bids_lst_boo_true,_,_ = get_fully_trained_bids()
cid_pred_dic = get_cid_pred_dic()
adjusted_spd_dic, results_dic_new = get_adjusted_spd_dic()

bid_cid_reverse_map = get_bid_cid_reverse_map()
daily_limit_tup_lst = get_daily_limit_tup_lst()

bid_clt_dic = get_bid_clt_dic()

out_df = pd.DataFrame(data=[(datetime.now().date()+timedelta(1)).strftime('%Y%m%d')
                            for _ in range(len([k for k in cid_pred_dic.keys()]))],
                      columns=['pred_date'])



###########################################################################
out_df['cid'] = [k for k in cid_pred_dic.keys()]
out_df['bid'] = out_df['cid'].map(bid_cid_reverse_map)
out_df['client_name'] = out_df['bid'].map(bid_clt_dic)
out_df['dailylimit_date_before'] = [v2 for _,v2 in daily_limit_tup_lst]
out_df['ml_pred_spd'] = out_df['cid'].map(cid_pred_dic)

out_df['adjusted_spd'] = out_df['cid'].map(adjusted_spd_dic)
out_df = out_df[['client_name','bid','cid','pred_date',
                 'dailylimit_date_before','ml_pred_spd','adjusted_spd']]
#######################################################################

out_data = out_df.values.tolist()

with open('config/db_creds.yaml', 'r') as fh:
    db_creds = yaml.load(fh)

with MySQLConnection(db_creds['FUELASSET_DEV']['db'],
                     db_creds['FUELASSET_DEV']['host'],
                     db_creds['FUELASSET_DEV']['port'],
                     db_creds['FUELASSET_DEV']['user'],
                     db_creds['FUELASSET_DEV']['password']) as connection:
    print('connections successful')

    cursor = connection.cursor()
    create_tb_query = """
        CREATE TABLE if not exists dailyio_adjusted6 (
            ClientName VARCHAR(50) NOT NULL,
            bid INT,
            cid INT,
            pred_date DATE,
            dailylimit_date_before FLOAT,
            ml_pred_spd FLOAT,
            adjusted_spd FLOAT
        )
    """
    cursor.execute(create_tb_query)
    connection.commit()

    cnt = 0
    for row in out_data:
        insert_data_query = """
            INSERT INTO dailyio_adjusted6 VALUES("{}",{},{},{},{},{},{})
        """.format(row[0],row[1],row[2],row[3],row[4],row[5],row[6])

        cursor.execute(insert_data_query)
        connection.commit()
        cnt += 1
        print('write', cnt, 'row(s) to db')
    print(cnt)

    cursor.execute("SELECT * FROM dailyio_adjusted6")
    print(cursor.fetchone())

    connection.close()

    print('running time: ', datetime.now() - t0)
