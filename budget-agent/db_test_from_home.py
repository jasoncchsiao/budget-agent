# from utils.db_conn import PostGreSQLConnection
# from utils.db_conn import MySQLConnection
from utils.database import get_multiple_tbs

df1, df2, df3 = get_multiple_tbs()
print(df1.date.max())
