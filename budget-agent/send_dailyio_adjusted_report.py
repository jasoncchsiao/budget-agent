#!/usr/bin/env python
# send_dailyio_adjusted_report.py

from utils.db_conn import MySQLConnection
import getpass
import pandas as pd
from datetime import datetime
import yaml

import smtplib
import os.path as op
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
from email import encoders


def get_dailylimit_factor_frm_db():
    """
    """
    #Production fuelAsset DB
    with MySQLConnection(db_creds['FUELASSET']['db'],
                         db_creds['FUELASSET']['host'],
                         db_creds['FUELASSET']['port'],
                         db_creds['FUELASSET']['user'],
                         db_creds['FUELASSET']['password']) as conn1:

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
        df1.to_csv('data/dailylimit_factor_{}.csv'.format(datetime.now().date()),
                   index=False)
    return df1

def get_dailyio_adjusted_tb():
    """
    rtype: pandas dataframe
    """
    #Development fuelAsset DB
    with MySQLConnection(db_creds['FUELASSET_DEV']['db'],
                         db_creds['FUELASSET_DEV']['host'],
                         db_creds['FUELASSET_DEV']['port'],
                         db_creds['FUELASSET_DEV']['user'],
                         db_creds['FUELASSET_DEV']['password']) as conn2:
        print('connections successful')

        sql_query2 = """
                SELECT * FROM dailyio_adjusted2
        """
        df2 = pd.read_sql(sql_query2, conn2)
        df2.to_csv('data/dailyio_adjusted_{}.csv'.format(datetime.now().date()),
                   index=False)
    return df2

def send_mail(send_from, send_to, subject, message, files=[],
              server='localhost', port=587, username='', password='',
              use_tls=True):
    """Compose and send email with provided info and attachments.

    Args:
        send_from (str): from name
        send_to (str): to name
        subject (str): message title
        message (str): message body
        files (list[str]): list of file paths to be attached to email
        server (str): mail server host name
        port (int): port number
        username (str): server auth username
        password (str): server auth password
        use_tls (bool): use TLS mode
    """
    msg = MIMEMultipart()
    msg['From'] = send_from
    msg['To'] = COMMASPACE.join(send_to)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject

    msg.attach(MIMEText(message))

    for path in files:
        part = MIMEBase('application', 'octet-stream')
        with open(path, 'rb') as file:
            part.set_payload(file.read())
        encoders.encode_base64(part)
        part.add_header(
            'Content-Disposition',
            'attachment; filename="{}"'.format(op.basename(path))
        )
        msg.attach(part)
    smtp = smtplib.SMTP(server, port)
    if use_tls:
        smtp.starttls()
        smtp.login(username, password)
        smtp.sendmail(send_from, send_to, msg.as_string())
        smtp.quit()


def main():


    df_dailylimit_factor = get_dailylimit_factor_frm_db()
    cid_fac_dic = {cid:val
                   for cid, val in zip(df_dailylimit_factor.cid.values,
                                       df_dailylimit_factor.value.values)}

    df_dailyio_adjusted = get_dailyio_adjusted_tb()


    df_dailyio_adjusted['factor'] = df_dailyio_adjusted['cid'].map(cid_fac_dic)
    df_dailyio_adjusted['dailylimit_fac'] = df_dailyio_adjusted['dailylimit_date_before'] \
                                            /(1-df_dailyio_adjusted['factor'])
    df_dailyio_adjusted['ml_spd_fac'] = df_dailyio_adjusted['ml_pred_spd'] \
                                        /(1-df_dailyio_adjusted['factor'])
    df_dailyio_adjusted['adjusted_spd_fac'] = df_dailyio_adjusted['adjusted_spd'] \
                                              /(1-df_dailyio_adjusted['factor'])
    df_dailyio_adjusted = df_dailyio_adjusted.sort_values(by=['ClientName','pred_date'], ascending=[1,0])
    df_dailyio_adjusted.to_csv('data/dailyio_adjusted_mdf_{}.csv'.format(datetime.now().date()),
                               index=False)
    return df_dailyio_adjusted

if __name__ == '__main__':
    with open('config/db_creds.yaml', 'r') as fh:
        db_creds = yaml.load(fh)
    df = main()

    email = db_creds['GMAIL_API']['email']
    password = db_creds['GMAIL_API']['password']

    send_from = email
    send_to = [email]

    subject = 'DailyIO Adjuster Report'
    message = "Hi team, \n\nAttached please find the dailyio_adjusted_report for reference. \n\nTo note, the budget agent has been trained on campaign \nlevels(=cid level) on the daily basis. \n\nPlease expect to receive emails generated by the system on the daily basis. \n\nThanks, \nJason"
    files = [
            'data/dailyio_adjusted_mdf_{}.csv'.format(datetime.now().date())
            ]
    server='smtp.gmail.com'
    port = 587
    username = email
    password = password

    send_mail(send_from, send_to, subject, message, files, server, port,
              username, password)
