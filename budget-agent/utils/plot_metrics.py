#!/usr/bin/env python
# plot_metrics.py

import numpy as np
import matplotlib.pyplot as plt
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

def vis_plot(real_spd, predicted_spd):
    plt.plot(real_spd, color = 'red', label = 'Real Spd')
    plt.plot(predicted_spd[:,0:1], color = 'blue', label = 'Predicted Spd')
    plt.title('Spd Prediction')
    plt.xlabel('Time')
    plt.ylabel('Spd')
    plt.legend()
    plt.show()

def dailylimit_sum(df_lst):
    df_total = pd.concat([df for df in df_lst], axis=0)
    df_gp = df_total[['date', 'real_spd', 'prediction spd']] \
            .groupby(by='date', as_index=False).sum()
    return df_gp
