#!/usr/bin/env python
# database.py

from utils.db_conn import PostGreSQLConnection
from utils.db_conn import MySQLConnection
import getpass
import pandas as pd
from datetime import datetime
import yaml


with open('config/db_creds.yaml', 'r') as fh:
    db_creds = yaml.load(fh)

def get_multiple_tbs():
    """
    rtype: pandas dataframe
    """

    with PostGreSQLConnection(db_creds['REDSHIFT']['db'], db_creds['REDSHIFT']['host'], db_creds['REDSHIFT']['port'], db_creds['REDSHIFT']['user'], db_creds['REDSHIFT']['password']) as conn1:
        with MySQLConnection(db_creds['FUELASSET']['db'], db_creds['FUELASSET']['host'], db_creds['FUELASSET']['port'], db_creds['FUELASSET']['user'], db_creds['FUELASSET']['password']) as conn2:

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

            df1.to_csv('data/adx_funnel_combo_{}.csv'.format(datetime.now().date()), index=False)
            print('Done querying and save to the local')

            sql_query = """
                    SELECT /* get objective info on 5/3 */
                        bid,
                        max(objective) as objective
                    FROM campaign_conversion_attribution
                    GROUP BY bid
                """
            df2 = pd.read_sql(sql_query, conn1)
            df2.to_csv('data/objective_{}.csv'.format(datetime.now().date()), index=False)

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
            df3.to_csv('data/roi_goal_{}.csv'.format(datetime.now().date()), index=False)

    return df1, df2, df3
