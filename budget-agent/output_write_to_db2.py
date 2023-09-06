#!/usr/bin/env python
# output_write_to_db2.py

from utils.db_conn import MySQLConnection
import numpy as np
import pandas as pd
import pymysql
from datetime import datetime
import pickle
from functools import reduce
import os.path
import yaml


filename = 'data/spd_pred_v2_{}.pkl'.format(datetime.now().date())

if os.path.isfile(filename):
    with open(filename, 'rb') as fh:
        global dic_spd
        dic_spd = pickle.load(fh)


data_np_lst = []
for k, df in dic_spd.items():
    df['cid'] = [k for i in range(df.shape[0])]
    df['date'] = df['date'].apply(lambda x: x.strftime('%Y%m%d'))
    df['given_date_forcast'] = [datetime.now().date().strftime('%Y%m%d')
                                for i in range(df.shape[0])]

    df['real_spd'] = df['real_spd'].apply(lambda x: round(x, 2))
    df.fillna('null', inplace=True)
    df['prediction spd'] = df['prediction spd'].apply(lambda x: round(x, 2))
    df = df[['cid','given_date_forcast','date','real_spd','prediction spd']]
    data_np = df.values
    data_np_lst.append(data_np)

data = reduce(lambda x, y: np.concatenate((x,y), axis=0), data_np_lst).tolist()

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
        create table if not exists spd_pred6 (
            cid INT,
            given_date_forcast DATE,
            date DATE,
            real_spd FLOAT,
            pred_spd FLOAT
        )
    """

    cursor.execute(create_tb_query)
    connection.commit()


    cnt = 0
    for row in data:
        insert_data_query = """INSERT INTO spd_pred6 VALUES({},{},{},{},{})"""\
        .format(row[0],row[1],row[2],row[3],row[4])
        cursor.execute(insert_data_query)
        connection.commit()
        cnt += 1
        print('write', cnt, 'row(s) to db')
    print(cnt)

    print(data)
    cnt = 0
    for row in data:
        if row[2] in ['20180606', '20180607', '20180609', '20180612']:

            update_data_query = \
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
            """.format(row[4], row[0], row[0])
            cursor.execute(update_data_query)
            connection.commit()
            cnt += 1
    print('updated', cnt, 'row(s) and rendered it in UI')

    print(cnt)

    cursor.execute("SELECT * FROM spd_pred6 ORDER BY given_date_forcast DESC")
    print(cursor.fetchone())


    connection.close()
