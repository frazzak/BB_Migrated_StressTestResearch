# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:25:32 2019

@author: yifei
"""
import pymysql as msql
import numpy as np

def request_data(data_type):
    
    db = msql.connect(host = "localhost", db = "peng_xu", user = "root", passwd = "yifei993219")
    cursor = db.cursor()
    
    sql = "select distinct RSSD_ID from x_ij where RSSD_ID in (select distinct RSSD_ID from y_ij)"
    t = cursor.execute(sql)
    print(t)
    bank_ids = cursor.fetchall()
    
    # the temporal data starts from 1990 Q1 ends at 2016 Q4
    # that is totally 108 records per bank on each loan category
    if data_type == "Y":
        results = np.zeros([1, 108, 4])
    else:
        results = np.zeros([1, 108, 5])
    for bank_id in bank_ids:
        if data_type == "Y":
            sql = "SELECT ReportingDate, `ppnrRatio:Net interest income`, `ppnrRatio:Noninterest income`,"
            sql = sql + " `ppnrRatio:Compensation expense`, `ppnrRatio:Fixed assets expense` from y_ij where RSSD_ID='"
            sql = sql + str(bank_id[0]) + "'"
            temp_data = np.zeros([1, 108, 4])
        else:
            sql = "SELECT ReportingDate, `Commercial & industrial_Covas`, `Multifamily real estate`, "
            sql = sql + "`Nonfarm nonresidential CRE_2006Q4_on`, `Home equity lines of credit`, `Residential real estate (excl. HELOCs)_Covas`"
            sql = sql + " from x_ij where RSSD_ID='"
            sql = sql + str(bank_id[0]) + "'"
            temp_data = np.zeros([1, 108, 5])
        t = cursor.execute(sql)
        records = cursor.fetchall()
        print("BankID:%s\tNum:%d" % (bank_id[0], int(t)))
        if int(t) > 108 and records[0][0] == "1990 Q1" and records[107][0] == "2016 Q4":
            i = 0
            for record in records:
                temp_data[0, i, 0] = record[1]
                temp_data[0, i, 1] = record[2]
                temp_data[0, i, 2] = record[3]
                temp_data[0, i, 3] = record[4]
                temp_data[0, i, 4] = record[5]
                i = i + 1
                if i == 108:
                    break
            results = np.append(results, temp_data, axis=0)
    
    # quarter based records
    # in quarter based data, the 1st axis is the index of Q1,2,3,4,
    # the 2nd axis represents the number of different banks
    # and the 3nd axis is the temporal sequece, such as 1990 Q1 -> 1991 Q1 -> 1992 Q1 
    # the 4th axis is the dimension of attributes, which refers to the sql sentences
    n = results.shape[0] - 1
    results = results[1:n+1]
    results_quarter = np.zeros([4, n, int(108/4), 5])
    for i in range(0, 4):
        ids = [x for x in range(i, 108, 4)]
        results_quarter[i, :, :, :] = results[:, ids, :]
    return results, results_quarter

if __name__ == "__main__":
    
    data_X, data_X_quarter = request_data("X")
    
    
    
    