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


###################################################################################


def get_data(table, date_pick):
    if table == 'CS_ncku' or table == 'CS_tmp':
        conn = connect_icemachine_db('IoT')
        col_name = 'Date'
    elif table == 'outdoor':
        conn = connect_aiot_db('aiot')
        col_name = 'TIMESTAMP'
    else:
        conn = connect_aiot_db('aiot')
        col_name = 'TIME_STAMP'
    try:
        cur = conn.cursor()
        sql = "SELECT * FROM " + table + " WHERE " + col_name + " LIKE %s"    # = %s
        cur.execute(sql, (f'{date_pick}%'))
        resultall = cur.fetchall()
        cur.close()
        conn.close()
    except:
        pass
    return resultall


def get_data_period(table, start_date, end_date):
    if table == 'CS_ncku' or table == 'CS_tmp':
        conn = connect_icemachine_db('IoT')
        col_name = 'Date'
    elif table == 'outdoor':
        conn = connect_aiot_db('aiot')
        col_name = 'TIMESTAMP'
    else:
        conn = connect_aiot_db('aiot')
        col_name = 'TIME_STAMP'
    try:
        cur = conn.cursor()
        sql = "SELECT * FROM " + table + " WHERE " + col_name + " >= %s AND " + col_name + " <= %s"    # = %s
        cur.execute(sql, (f'{start_date}%', f'{end_date}%'))    #date_pick
        resultall = cur.fetchall()
        cur.close()
        conn.close()
    except:
        pass
    return resultall