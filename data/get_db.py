import pymysql
import logging
import json

# get config
f = open('./data/config.json', 'r')
config = json.loads(f.read())
junfan_host = config['junfan_host']
junfan_user = config['junfan_user']
junfan_passwd = config['junfan_passwd']
ncku_host = config['ncku_host']
ncku_user = config['ncku_user']
ncku_passwd = config['ncku_passwd']


# 冰水主機 database
def connect_icemachine_db(dbName):
    try:
        mysqldb = pymysql.connect(host=junfan_host, user=junfan_user, passwd=junfan_passwd, database=dbName)
        return mysqldb
    except Exception as e:
        logging.error('Fail to connection mysql {}'.format(str(e)))
    return None


# AIOT 室外資料、辦公室資料 database
def connect_aiot_db(dbName):
    try:
        conn = pymysql.connect(
            host=ncku_host,
            port=3306,
            user=ncku_user,
            password=ncku_passwd,
            database=dbName,
            charset='utf8'
        )
        return conn
    except Exception as e:
        logging.error('Fail to connection mysql {}'.format(str(e)))
    return None


def get_data(table, date_pick):
    resultall = []
    conn = None

    if table == 'CS_nor':
        conn = connect_icemachine_db('IoT')
        col_name = 'Date'
    elif table in ['outdoor', 'indoor', 'pointer', 'controller']:
        conn = connect_aiot_db('aiot')
        col_name = 'TIMESTAMP'
    else:
        return resultall

    try:
        if conn:
            cur = conn.cursor()
            sql = "SELECT * FROM " + table + " WHERE " + col_name + " LIKE %s"
            cur.execute(sql, (f'{date_pick}%',))
            resultall = cur.fetchall()
            cur.close()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()

    return resultall


def get_data_period(table, start_date, end_date):
    resultall = []
    conn = None

    if table == 'CS_nor':
        conn = connect_icemachine_db('IoT')
        col_name = 'Date'
    elif table in ['outdoor', 'indoor', 'pointer', 'controller']:
        conn = connect_aiot_db('aiot')
        col_name = 'TIMESTAMP'
    else:
        return resultall

    try:
        if conn:
            cur = conn.cursor()
            sql = "SELECT * FROM " + table + " WHERE " + col_name + " >= %s AND " + col_name + " <= %s"
            cur.execute(sql, (f'{start_date}', f'{end_date}'))
            resultall = cur.fetchall()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()

    return resultall