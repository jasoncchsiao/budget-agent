#!/usr/bin/env python
# app.py

from utils.database import get_multiple_tbs
from utils import data_transformation
from utils import model
from utils import plot_metrics
from datetime import datetime
import pandas as pd
from datetime import datetime
import pickle
import yaml
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os.path

__author__ = 'jhsiao'

def main():
    global CID

    df = data_transformation.clean_raw_data()
    print('Done loading the whole data')


    df, last_day_in_df = data_transformation.get_cid_data(df, CID)
    print('Successful for a given cid')

    df = data_transformation.synthesize_feature_data(df, m)
    print('Successfully added feature sets')

    X_train, y_train, X_test, real_spd, sc, D, train_idx = \
        data_transformation.transformer_data(df)
    print('Done transferring data')

    regressor = model.model_train(X_train, y_train)
    print('Done training')

    predicted_spd = model.model_test(X_test, regressor)
    print('Done predictions')



    # update on 5/29/18
    df_comp, predicted_spd = plot_metrics.cmp_table(
        predicted_spd, real_spd, sc,
        X_test, df, n, D, train_idx, m, last_day_in_df
    )
    print('Done comparasion table')
    print(df_comp)

    plot_metrics.model_metrics(real_spd, predicted_spd)
    print('Done Metrics')



    return df_comp


if __name__ == '__main__':
    t0 = datetime.now()

    data = data_transformation.clean_raw_data()
    print('#CIDs: ',data.cid.nunique())

    data.to_csv('data/all_mdf_{}.csv'.format(datetime.now().date()),index=False)
    CIDs = data_transformation.cids_list_all(data)
    print('#CIDs: ', len(CIDs))

    _, _, CIDs = data_transformation.get_fully_trained_bids()
    print('# CIDs running for training: ', len(CIDs))


    df_lst = []
    cnt = 0
    for CID in CIDs[:]:

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
    with open('data/spd_pred_v2_{}.pkl'.format(datetime.now().date()),'wb') as fh:
        pickle.dump(dic, fh)
    print('write predicitons to local')
    print('running time: ', datetime.now() - t0)
