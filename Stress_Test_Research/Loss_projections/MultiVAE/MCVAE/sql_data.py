# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:25:32 2019

@author: yifei
"""
import pymysql as msql
import numpy as np
import os

def request_data(data_type):
    #data_type = "Y"
    db = msql.connect(host = "localhost", db = 'STR', user = "root", passwd = "")
    cursor = db.cursor()
    
    sql = "select distinct RSSD_ID from xij where RSSD_ID in (select distinct RSSD_ID from yij)"
    t = cursor.execute(sql)
    print("Bank Ids Count matching for X and Y variable:", t)
    #print(t)
    bank_ids = cursor.fetchall()
    
    # the temporal data starts from 1990 Q1 ends at 2016 Q4
    # that is totally 108 records per bank on each loan category
    if data_type == "Y":
        results = np.zeros([1, 108, 16])
    else:
        results = np.zeros([1, 108, 10])
    for bank_id in bank_ids:
        if data_type == "Y":
            # "`ppnrRatio:Net interest income`, `ppnrRatio:Noninterest income`,"
            sql = "SELECT `ReportingDate`, `ncoR:Commercial & industrial`, `ncoR:Construction & land development`, `ncoR:Multifamily real estate`,`ncoR:(Nonfarm) nonresidential CRE`, `ncoR:Home equity lines of credit`, " \
                  "`ncoR:Residential real estate (excl. HELOCs)`, `ncoR:Credit card`, `ncoR:Consumer (excl. credit card)`, `ppnrRatio:Net interest income`, `ppnrRatio:Noninterest income`, " \
                  "`ppnrRatio:Trading income`, `ppnrRatio:Compensation expense`, `ppnrRatio:Fixed assets expense`, `ppnrRatio:Noninterest expense`"
            #sql = sql + " `ppnrRatio:Compensation expense`, `ppnrRatio:Fixed assets expense` from y_ij where RSSD_ID='"
            sql = sql + "  from yij where SUBSTRING_INDEX(ReportingDate, ' ', 1) <> 2017 and RSSD_ID='"
            sql = sql + str(bank_id[0]) + "'"
            temp_data = np.zeros([1, 108, 16])
        else:
            sql = "SELECT `ReportingDate`, "
            #sql = sql + "`Nonfarm nonresidential CRE_2006Q4_on`, `Home equity lines of credit`, `Residential real estate (excl. HELOCs)_Covas`"
            sql = sql +  "`Loans categories:Commercial & industrial_Covas`,`Loans categories:Construction & land development`,`Loans categories:Multifamily real estate`," \
                        "`Loans categories:Nonfarm nonresidential CRE_Covas`,`Loans categories:Home equity lines of credit`,`Loans categories:Residential real estate (excl. HELOCs)_Covas`," \
                        "`Loans categories:Credit card`,`Loans categories:Consumer (excl. credit card)_Covas`"
            sql = sql + " from xij where  SUBSTRING_INDEX(ReportingDate, ' ', 1) <> 2017 and RSSD_ID='"
            sql = sql + str(bank_id[0]) + "'"
            temp_data = np.zeros([1, 108, 10])
        t = cursor.execute(sql)
        records = cursor.fetchall()
        print("BankID:%s\tNum:%d" % (bank_id[0], int(t)))

        if int(t) > 108 and records[0][0] == "1990 Q1" and records[107][0] == "2016 Q4":
            print("Trimming Data Accordingly")
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
        else:
            results = np.append(results, temp_data, axis=0)
    # quarter based records
    # in quarter based data, the 1st axis is the index of Q1,2,3,4,
    # the 2nd axis represents the number of different banks
    # and the 3nd axis is the temporal sequece, such as 1990 Q1 -> 1991 Q1 -> 1992 Q1 
    # the 4th axis is the dimension of attributes, which refers to the sql sentences
    n = results.shape[0] - 1
    results = results[1:n+1]
    if data_type == "X":
        results_quarter = np.zeros([4, n, int(108/4), 10])
    else:
        results_quarter = np.zeros([4, n, int(108 / 4), 16])
    for i in range(0, 4):
        ids = [x for x in range(i, 108, 4)]
        results_quarter[i, :, :, :] = results[:, ids, :]
    return results, results_quarter


def preprocess_moda_data():
    return


if __name__ == "__main__":


    os.chdir("./Data_PP_Output/")

    print("Querying MySQL Database for X variables")
    data_X, data_X_quarter = request_data("X")


    print("Querying MySQL Database for Y variables")
    data_Y, data_Y_quarter = request_data("Y")

    print("X:", data_X.shape, "X_quarter:", data_X_quarter.shape)
    print("Y:",data_Y.shape, "Y_quarter:", data_Y_quarter.shape)

    print("Saving X and Y objects to file")
    np.save("./data_X.npy", data_X)
    np.save("./data_X_quarter.npy", data_X_quarter)
    np.save("./data_Y.npy", data_Y)
    np.save("./data_Y_quarter.npy", data_Y_quarter)
