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
