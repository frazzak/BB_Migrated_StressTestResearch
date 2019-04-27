# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:25:32 2019

@author: yifei
"""
import pymysql as msql
#from mysql import connector as mysqlc
import numpy as np
import os

#TODO: Get all partial data, not just banks with quarters that match window.
#TODO: Load Merged Bank Data.
#TODO: May not need to transform RSSD by RSSD for all banks. find way to transform whole fetched cursor at once
def request_data(data_type):
    #data_type = "XY_Cap"
    db = msql.connect(host = "localhost", db = 'STR', user = "root", passwd = "")
    cursor = db.cursor()

    
    #sql = "select distinct RSSD_ID from xij_2 where RSSD_ID in (select distinct RSSD_ID from yij_2)"
    if data_type in ["Y_full", "X_full","XYCap_full", "CapRatios_full"]:
        sql = "select x.RSSD_ID" \
              " FROM (SELECT DISTINCT (a.RSSD_ID) FROM" \
              " (SELECT RSSD_ID, COUNT(*) FROM STR.xij_2" \
              " GROUP BY RSSD_ID) a ) x" \
              " join (SELECT DISTINCT RSSD_ID from STR.yij_2) y on x.RSSD_ID = y.RSSD_id"
    elif data_type in ["Y_full_mergered", "X_full_mergered","XYCap_full_mergered", "CapRatios_full_mergered"]:
        sql = "select x.RSSD_ID" \
              " FROM (SELECT DISTINCT (a.RSSD_ID) FROM" \
              " (SELECT RSSD_ID, COUNT(*) FROM STR.X_merged" \
              " GROUP BY RSSD_ID) a ) x" \
              " join (SELECT DISTINCT RSSD_ID from STR.Y_merged) y on x.RSSD_ID = y.RSSD_id"
    elif data_type in ["X_mergered", "Y_mergered","CapRatios_mergered", "XYCap_mergered"]:
        sql = "select x.RSSD_ID" \
              " FROM (SELECT DISTINCT (a.RSSD_ID) FROM" \
              " (SELECT RSSD_ID, COUNT(*) FROM STR.X_merged" \
              " GROUP BY RSSD_ID HAVING COUNT(*) > 108) a ) x" \
              " join (SELECT DISTINCT RSSD_ID from STR.Y_merged) y on x.RSSD_ID = y.RSSD_id"
    else:
        sql = "select x.RSSD_ID" \
            " FROM (SELECT DISTINCT (a.RSSD_ID) FROM" \
            " (SELECT RSSD_ID, COUNT(*) FROM STR.xij_2" \
            " GROUP BY RSSD_ID HAVING COUNT(*) > 108) a ) x" \
            " join (SELECT DISTINCT RSSD_ID from STR.yij_2) y on x.RSSD_ID = y.RSSD_id"


    print("Executing Query for Bank RSSD_IDs")
    t = cursor.execute(sql)
    print("Bank Count:", t)
    #print(t)
    bank_ids = cursor.fetchall()
    
    # the temporal data starts from 1990 Q1 ends at 2016 Q4
    # that is totally 108 records per bank on each loan category
    if data_type in ["Y", "Y_full", "Y_mergered", "Y_full_mergered"]:
        results = np.zeros([1, 108, 14])
    elif data_type in ["X","XY_Cap", "X_full","XY_Cap_Full","CapRatios_full", "X_mergered", "X_full_mergered"]:
        results = np.zeros([1, 108, 8])
    elif data_type in  ["CapRatios","CapRatios_mergered", "CapRatios_full","CapRatios_full_mergered"]:
        results = np.zeros([1, 108, 1])
    else:
        print("No Data Type Found")

    # sql = '''SELECT b.ReportingDate, a.*
    #         FROM (SELECT DISTINCT ReportingDate from xij_2) b
    #         LEFT JOIN STR.sbidx a on a.ReportingDate = b.ReportingDate
    #         order by b.ReportingDate'''

    #data_type = "X"
    for bank_id in bank_ids:
        if data_type == "Y":
            # "`ppnrRatio:Net interest income`, `ppnrRatio:Noninterest income`,"
            sql = "SELECT `ReportingDate`, `ncoR:Commercial & industrial`, `ncoR:Construction & land development`, `ncoR:Multifamily real estate`," \
                  "`ncoR:(Nonfarm) nonresidential CRE`, `ncoR:Home equity lines of credit`, " \
                  "`ncoR:Residential real estate (excl. HELOCs)`, `ncoR:Credit card`, `ncoR:Consumer (excl. credit card)`, `ppnrRatio:Net interest income`, `ppnrRatio:Noninterest income`, " \
                  "`ppnrRatio:Trading income`, `ppnrRatio:Compensation expense`, `ppnrRatio:Fixed assets expense`, `ppnrRatio:Noninterest expense`"
            #sql = sql + " `ppnrRatio:Compensation expense`, `ppnrRatio:Fixed assets expense` from y_ij where RSSD_ID='"
            sql = sql + "  from yij_2 where RSSD_ID='"
            sql = sql + str(bank_id[0]) + "'"
            temp_data = np.zeros([1, 108, 14])

        elif data_type == "Y_mergered":
            # "`ppnrRatio:Net interest income`, `ppnrRatio:Noninterest income`,"
            sql = "SELECT `ReportingDate`, `ncoR:Commercial & industrial`, `ncoR:Construction & land development`, `ncoR:Multifamily real estate`," \
                  "`ncoR:(Nonfarm) nonresidential CRE`, `ncoR:Home equity lines of credit`, " \
                  "`ncoR:Residential real estate (excl. HELOCs)`, `ncoR:Credit card`, `ncoR:Consumer (excl. credit card)`, `ppnrRatio:Net interest income`, `ppnrRatio:Noninterest income`, " \
                  "`ppnrRatio:Trading income`, `ppnrRatio:Compensation expense`, `ppnrRatio:Fixed assets expense`, `ppnrRatio:Noninterest expense`"
            #sql = sql + " `ppnrRatio:Compensation expense`, `ppnrRatio:Fixed assets expense` from y_ij where RSSD_ID='"
            sql = sql + "  from Y_merged where RSSD_ID='"
            sql = sql + str(bank_id[0]) + "'"
            temp_data = np.zeros([1, 108, 14])


        elif data_type == "Y_full":
            # "`ppnrRatio:Net interest income`, `ppnrRatio:Noninterest income`,"
            sql = "SELECT b.ReportingDate, a.`ncoR:Commercial & industrial`, a.`ncoR:Construction & land development`, a.`ncoR:Multifamily real estate`," \
                  "a.`ncoR:(Nonfarm) nonresidential CRE`, a.`ncoR:Home equity lines of credit`, " \
                  "a.`ncoR:Residential real estate (excl. HELOCs)`, `a.ncoR:Credit card`, `a.ncoR:Consumer (excl. credit card)`, `a.ppnrRatio:Net interest income`, `a.ppnrRatio:Noninterest income`, " \
                  "`ppnrRatio:Trading income`, `ppnrRatio:Compensation expense`, `ppnrRatio:Fixed assets expense`, `ppnrRatio:Noninterest expense`"
            #sql = sql + " `ppnrRatio:Compensation expense`, `ppnrRatio:Fixed assets expense` from y_ij where RSSD_ID='"
            sql = sql + " FROM (SELECT DISTINCT ReportingDate from xij_2) b LEFT JOIN STR.yij_2 a on a.ReportingDate = b.ReportingDate where a.RSSD_ID='"
            sql = sql + str(bank_id[0]) + "'"
            temp_data = np.zeros([1, 108, 14])

        elif data_type == "Y_full_mergered":
            # "`ppnrRatio:Net interest income`, `ppnrRatio:Noninterest income`,"
            sql = "SELECT b.ReportingDate, a.`ncoR:Commercial & industrial`, a.`ncoR:Construction & land development`, a.`ncoR:Multifamily real estate`," \
                  "a.`ncoR:(Nonfarm) nonresidential CRE`, a.`ncoR:Home equity lines of credit`, " \
                  "a.`ncoR:Residential real estate (excl. HELOCs)`, `a.ncoR:Credit card`, `a.ncoR:Consumer (excl. credit card)`, `a.ppnrRatio:Net interest income`, `a.ppnrRatio:Noninterest income`, " \
                  "`ppnrRatio:Trading income`, `ppnrRatio:Compensation expense`, `ppnrRatio:Fixed assets expense`, `ppnrRatio:Noninterest expense`"
            #sql = sql + " `ppnrRatio:Compensation expense`, `ppnrRatio:Fixed assets expense` from y_ij where RSSD_ID='"
            sql = sql + " FROM (SELECT DISTINCT ReportingDate from X_merged) b LEFT JOIN STR.Y_merged a on a.ReportingDate = b.ReportingDate where a.RSSD_ID='"
            sql = sql + str(bank_id[0]) + "'"
            temp_data = np.zeros([1, 108, 14])

        elif data_type == "X_full":
            #bank_id[0] = 1020180
            sql = "SELECT b.`ReportingDate`, "
            #sql = sql + "`Nonfarm nonresidential CRE_2006Q4_on`, `Home equity lines of credit`, `Residential real estate (excl. HELOCs)_Covas`"
            sql = sql +  "a.`Loans categories:Commercial & industrial_Covas`,a.`Loans categories:Construction & land development`,a.`Loans categories:Multifamily real estate`," \
                        "a.`Loans categories:Nonfarm nonresidential CRE_Covas`,a.`Loans categories:Home equity lines of credit`,a.`Loans categories:Residential real estate (excl. HELOCs)_Covas`," \
                        "a.`Loans categories:Credit card`,a.`Loans categories:Consumer (excl. credit card)_Covas`"
#            sql = sql + " from xij_2 where RSSD_ID='"
            sql = sql + " FROM (SELECT DISTINCT ReportingDate from xij_2) b LEFT JOIN STR.xij_2 a on a.ReportingDate = b.ReportingDate where a.RSSD_ID='"
            sql = sql + str(bank_id[0]) + "'"
            temp_data = np.zeros([1, 108, 8])

        elif data_type == "X_full_mergered":
            # bank_id[0] = 1020180
            sql = "SELECT b.`ReportingDate`, "
            # sql = sql + "`Nonfarm nonresidential CRE_2006Q4_on`, `Home equity lines of credit`, `Residential real estate (excl. HELOCs)_Covas`"
            sql = sql + "a.`Loans categories:Commercial & industrial_Covas`,a.`Loans categories:Construction & land development`,a.`Loans categories:Multifamily real estate`," \
                        "a.`Loans categories:Nonfarm nonresidential CRE_Covas`,a.`Loans categories:Home equity lines of credit`,a.`Loans categories:Residential real estate (excl. HELOCs)_Covas`," \
                        "a.`Loans categories:Credit card`,a.`Loans categories:Consumer (excl. credit card)_Covas`"
            #            sql = sql + " from xij_2 where RSSD_ID='"
            sql = sql + " FROM (SELECT DISTINCT ReportingDate from X_merged) b LEFT JOIN STR.X_merged a on a.ReportingDate = b.ReportingDate where a.RSSD_ID='"
            sql = sql + str(bank_id[0]) + "'"
            temp_data = np.zeros([1, 108, 8])


        elif data_type == "X":
            #bank_id[0] = 1020180
            sql = "SELECT `ReportingDate`, "
            #sql = sql + "`Nonfarm nonresidential CRE_2006Q4_on`, `Home equity lines of credit`, `Residential real estate (excl. HELOCs)_Covas`"
            sql = sql +  "`Loans categories:Commercial & industrial_Covas`,`Loans categories:Construction & land development`,`Loans categories:Multifamily real estate`," \
                        "`Loans categories:Nonfarm nonresidential CRE_Covas`,`Loans categories:Home equity lines of credit`,`Loans categories:Residential real estate (excl. HELOCs)_Covas`," \
                        "`Loans categories:Credit card`,`Loans categories:Consumer (excl. credit card)_Covas`"
            sql = sql + " from xij_2 where RSSD_ID='"
            sql = sql + str(bank_id[0]) + "'"
            temp_data = np.zeros([1, 108, 8])

        elif data_type == "X_mergered":
            #bank_id[0] = 1020180
            sql = "SELECT `ReportingDate`, "
            #sql = sql + "`Nonfarm nonresidential CRE_2006Q4_on`, `Home equity lines of credit`, `Residential real estate (excl. HELOCs)_Covas`"
            sql = sql +  "`Loans categories:Commercial & industrial_Covas`,`Loans categories:Construction & land development`,`Loans categories:Multifamily real estate`," \
                        "`Loans categories:Nonfarm nonresidential CRE_Covas`,`Loans categories:Home equity lines of credit`,`Loans categories:Residential real estate (excl. HELOCs)_Covas`," \
                        "`Loans categories:Credit card`,`Loans categories:Consumer (excl. credit card)_Covas`"
            sql = sql + " from X_merged where RSSD_ID='"
            sql = sql + str(bank_id[0]) + "'"
            temp_data = np.zeros([1, 108, 8])



        elif data_type == "XY_Cap":
            #bank_id[0] = 1020180
            sql = "SELECT `ReportingDate`, "
            #sql = sql + "`Nonfarm nonresidential CRE_2006Q4_on`, `Home equity lines of credit`, `Residential real estate (excl. HELOCs)_Covas`"
            sql = sql +  "`Chargeoffs`,`Recoveries`,`Net income(loss)`," \
                        "`Other items:Book equity`,`Other items:Risk-weighted assets`,`Other items:Stock purchases`," \
                        "`Other items:Tier 1 common equity`,`Other items:T1CR`"
            sql = sql + " from xyaltratios_1 where RSSD_ID='"
            sql = sql + str(bank_id[0]) + "'"
            temp_data = np.zeros([1, 108, 8])

        elif data_type == "XY_Cap_mergered":
            #bank_id[0] = 1020180
            sql = "SELECT `ReportingDate`, "
            #sql = sql + "`Nonfarm nonresidential CRE_2006Q4_on`, `Home equity lines of credit`, `Residential real estate (excl. HELOCs)_Covas`"
            sql = sql +  "`Chargeoffs`,`Recoveries`,`Net income(loss)`," \
                        "`Other items:Book equity`,`Other items:Risk-weighted assets`,`Other items:Stock purchases`," \
                        "`Other items:Tier 1 common equity`,`Other items:T1CR`"
            sql = sql + " from xyaltratios_mergered where RSSD_ID='"
            sql = sql + str(bank_id[0]) + "'"
            temp_data = np.zeros([1, 108, 8])


        elif data_type == "XY_Cap_full":
            #bank_id[0] = 1020180
            sql = "SELECT b.`ReportingDate`, "
            #sql = sql + "`Nonfarm nonresidential CRE_2006Q4_on`, `Home equity lines of credit`, `Residential real estate (excl. HELOCs)_Covas`"
            sql = sql +  "a.`Chargeoffs`,a.`Recoveries`,a.`Net income(loss)`," \
                        "a.`Other items:Book equity`,a.`Other items:Risk-weighted assets`,a.`Other items:Stock purchases`," \
                        "a.`Other items:Tier 1 common equity`,a.`Other items:T1CR`"
#            sql = sql + " from xyaltratios_1 where RSSD_ID='"
            sql = sql + " FROM (SELECT DISTINCT ReportingDate from xij_2) b LEFT JOIN STR.xyaltratios_1 a on a.ReportingDate = b.ReportingDate where a.RSSD_ID='"
            sql = sql + str(bank_id[0]) + "'"
            temp_data = np.zeros([1, 108, 8])

        elif data_type == "XY_Cap_full_mergered":
            # bank_id[0] = 1020180
            sql = "SELECT b.`ReportingDate`, "
            # sql = sql + "`Nonfarm nonresidential CRE_2006Q4_on`, `Home equity lines of credit`, `Residential real estate (excl. HELOCs)_Covas`"
            sql = sql + "a.`Chargeoffs`,a.`Recoveries`,a.`Net income(loss)`," \
                        "a.`Other items:Book equity`,a.`Other items:Risk-weighted assets`,a.`Other items:Stock purchases`," \
                        "a.`Other items:Tier 1 common equity`,a.`Other items:T1CR`"
            #            sql = sql + " from xyaltratios_1 where RSSD_ID='"
            sql = sql + " FROM (SELECT DISTINCT ReportingDate from X_merged) b LEFT JOIN STR.CapitalRatio_merged a on a.ReportingDate = b.ReportingDate where a.RSSD_ID='"
            sql = sql + str(bank_id[0]) + "'"
            temp_data = np.zeros([1, 108, 8])


        elif data_type == "CapRatios":
            #bank_id[0] = 1020180
            sql = "SELECT `ReportingDate`, "
            #sql = sql + "`Nonfarm nonresidential CRE_2006Q4_on`, `Home equity lines of credit`, `Residential real estate (excl. HELOCs)_Covas`"
            sql = sql + "a.`Other items:T1CR`"  # a.`Other items:Tier 1 common equity`,
            sql = sql + " from CapitalRatio a where RSSD_ID='"
            sql = sql + str(bank_id[0]) + "'"
            temp_data = np.zeros([1, 108, 1])

        elif data_type == "CapRatios_mergered":
            #bank_id[0] = 1020180
            sql = "SELECT `ReportingDate`, "
            #sql = sql + "`Nonfarm nonresidential CRE_2006Q4_on`, `Home equity lines of credit`, `Residential real estate (excl. HELOCs)_Covas`"
            sql = sql + "a.`Other items:T1CR` "  # a.`Other items:Tier 1 common equity`,
            sql = sql + " from CapitalRatio_merged a where RSSD_ID='"
            sql = sql + str(bank_id[0]) + "'"
            temp_data = np.zeros([1, 108, 1])

        elif data_type == "CapRatios_full":
            # bank_id[0] = 1020180
            sql = "SELECT b.`ReportingDate`, "
            # sql = sql + "`Nonfarm nonresidential CRE_2006Q4_on`, `Home equity lines of credit`, `Residential real estate (excl. HELOCs)_Covas`"
            sql = sql + "a.`Other items:T1CR`" #a.`Other items:Tier 1 common equity`,
#            sql = sql + " from xyaltratios_1 where RSSD_ID='"
            sql = sql + " FROM (SELECT DISTINCT ReportingDate from xij_2) b LEFT JOIN STR.CapitalRatio a on a.ReportingDate = b.ReportingDate where a.RSSD_ID='"
            sql = sql + str(bank_id[0]) + "'"
            temp_data = np.zeros([1, 108, 1])

        elif data_type == "CapRatios_full_mergered":
            # bank_id[0] = 1020180
            sql = "SELECT b.`ReportingDate`, "
            # sql = sql + "`Nonfarm nonresidential CRE_2006Q4_on`, `Home equity lines of credit`, `Residential real estate (excl. HELOCs)_Covas`"
            sql = sql + "a.`Other items:T1CR`" #a.`Other items:Tier 1 common equity`,
            #            sql = sql + " from xyaltratios_1 where RSSD_ID='"
            sql = sql + " FROM (SELECT DISTINCT ReportingDate from X_merged) b LEFT JOIN STR.CapitalRatio_merged a on a.ReportingDate = b.ReportingDate where a.RSSD_ID='"
            sql = sql + str(bank_id[0]) + "'"
            temp_data = np.zeros([1, 108, 2])

        else:
            print("No Data Type Found")
            break
        #print("Executing SQL")
        t = cursor.execute(sql)
        #print("Getting Records")
        records = cursor.fetchall()


        #TODO: We may also need to consider banks with missing data.
        #TODO: Improve to take all the cursor records into npy array at once. and then into the temp data array
        if int(t) >= 108 and records[0][0] == "1990 Q1" and records[107][0] == "2016 Q4":
            print("BankID:%s\tNum:%d" % (bank_id[0], int(t)))
            print("Bank has 108 Quarters from 1990 to 2016")
            #print("Trimming Data Accordingly")
            i = 0
            for record in records:
                #print(record)
                #print("Replacing Blank Values with NAN")
                record = tuple([np.nan if x is '' else x for x in list(record)])
                if i == 108:
                    #print("Break Loop since i = 108")
                    break
                else:
                    for dim in range(0,len(record) - 1 ):
                        record_dim = dim + 1
                        #print(dim, record_dim, i)
                        temp_data[0, i, dim] = record[record_dim]
                        i = i + 1
                        if i == 108:
                            #print("Break Loop since i = 108")
                            break
            results = np.append(results, temp_data, axis=0)

    #results.shape
    # quarter based records
    # in quarter based data, the 1st axis is the index of Q1,2,3,4,
    # the 2nd axis represents the number of different banks
    # and the 3nd axis is the temporal sequece, such as 1990 Q1 -> 1991 Q1 -> 1992 Q1 
    # the 4th axis is the dimension of attributes, which refers to the sql sentences
    n = results.shape[0] - 1
    results = results[1:n+1]
    if data_type in ["X","X_mergered","XY_Cap","XY_Cap_mergered","X_full","X_full_mergered" ,"XY_Cap_full","XY_Cap_full_mergered"]:
        results_quarter = np.zeros([4, n, int(108/4), 8])
    elif data_type in  ["Y", "Y_mergered","Y_full","Y_full_mergered"]:
        results_quarter = np.zeros([4, n, int(108 / 4), 14])
    elif data_type in  ["CapRatios", "CapRatios_full","CapRatios_mergered", "CapRatios_full_mergered"]:
        results_quarter = np.zeros([4, n, int(108 / 4), 1])

    else:
        print("No Data Type Found")

    print("Transformation for Quarterly Data")
    for i in range(0, 4):
        ids = [x for x in range(i, 108, 4)]
        results_quarter[i, :, :, :] = results[:, ids, :]

        #results_quarter.shape

    return results, results_quarter


#TODO: Get Data from 1976 onward if feasible.
def preprocess_moda_data(tbl_name = "zmicro", tble_schema = "STR",host="localhost",
                         user="root", passwd="", time_dict = {'timeslices' : 108
                                                              ,'time_str_start' : "1990 Q1"
                                                              , 'time_str_end' : "2016 Q4"
                                                              , 'subslices' : 4}):
    print("Connect to SQL Server")
    db = msql.connect(host=host, db=tble_schema, user=user, passwd=passwd)
    cursor = db.cursor()

    print("Dynamically Get Number of Dimensions")
    sql_get_dim_count = "SELECT COUNT(*) AS NUMBEROFCOLUMNS FROM INFORMATION_SCHEMA.COLUMNS WHERE table_schema = '%s' AND table_name = '%s'" % (tble_schema,tbl_name)
    t = cursor.execute(sql_get_dim_count)
    records = cursor.fetchall()
    # Subtract 1 to not include Reporting Date in Count
    dim_count = records[0][0] - 1

    print("Dynamically Get Number of Rows")
    sql_get_row_count = "SELECT COUNT(*)  from %s" % (".".join([tble_schema,tbl_name]))
    t = cursor.execute(sql_get_row_count)
    records = cursor.fetchall()
    row_count = records[0][0]
    print("Initialize Temp NP object")
    temp_data = np.zeros([1, time_dict['timeslices'], dim_count])

    print("Initialize Results NP Array")
    data_moda_tmp = np.zeros([1, time_dict['timeslices'], dim_count])

    print("Rows:",row_count,"Dims:",dim_count)
    print("Get Data From Database")
    #Match to available time slices
    # sql = "SELECT b.ReportingDate, a.* " \
    #       "FROM (SELECT DISTINCT ReportingDate from xij_2) b " \
    #       "LEFT JOIN %s a on a.ReportingDate = b.ReportingDate order by b.ReportingDate" % (".".join([tble_schema,tbl_name]))


    #sql = "SELECT * from %s" % (".".join([tble_schema,tbl_name]))

    if tbl_name == "zmicro":
        print("Zmicro Table De-duplication sql code.")
        sql = '''SELECT m.ReportingDate, n.*
                    FROM
                    (
                    SELECT b.ReportingDate, COUNT(*), max(a.GFD_BrentCrudeOil) as maxGFD_BrentCrudeOil, max(sprtrn) as maxsprtrn
                    FROM (SELECT DISTINCT ReportingDate from xij_2) b
                    LEFT JOIN STR.zmicro a on a.ReportingDate = b.ReportingDate 
                    group by b.ReportingDate
                    ) m 
                    LEFT JOIN
                    (
                    SELECT * from STR.zmicro
                    ) n on m.ReportingDate = n.ReportingDate and m.maxGFD_BrentCrudeOil = n.GFD_BrentCrudeOil and n.sprtrn = m.maxsprtrn
                    order by m.ReportingDate'''

    elif tbl_name == "sbidx":
        sql = '''SELECT b.ReportingDate, a.*
                FROM (SELECT DISTINCT ReportingDate from xij_2) b
                LEFT JOIN STR.sbidx a on a.ReportingDate = b.ReportingDate 
                order by b.ReportingDate'''

    else:
        sql = "SELECT * from %s" % (".".join([tble_schema, tbl_name]))
    t = cursor.execute(sql)
    data_moda_tmp_records  = cursor.fetchall()


    # We may need to consider the merged data.  We may also need to consider banks with missing data.
    timeslice_end_idx = time_dict['timeslices'] - 1
    print("Evaluate Modality Data Time Slices")
    if int(t) > time_dict['timeslices'] and data_moda_tmp_records[0][0] == time_dict['time_str_start'] and data_moda_tmp_records[timeslice_end_idx][0] == time_dict['time_str_end']:
        print("Modality:%s\tNum:%d" % (tbl_name, int(t)))
        print("Trimming Data Accordingly")
        i = 0
        for record in data_moda_tmp_records:
            #print(record)
            # print("Replacing Blank Values with NAN")
            record = [np.nan if x is '' else x for x in list(record)]
            if tbl_name in ["zmicro",'sbidx']:
                print("De Duplication removal of original Date field")
                record.pop(1)
            record = tuple(record)
            #print(record)
            if i == time_dict['timeslices']:
                print("Break Loop since i = %s" % time_dict['timeslices'])
                break
            else:
                for dim in range(0, len(record) - 1):
                    record_dim = dim + 1
                    # print(dim, record_dim, i)
                    temp_data[0, i, dim] = record[record_dim]
                    i = i + 1
                    if i == time_dict['timeslices']:
                        #print("Break Loop since i = 108")
                        break
        data_moda_tmp = np.append(data_moda_tmp, temp_data, axis=0)

     if data_moda_tmp.shape[0] > 1:
         n = data_moda_tmp.shape[0] - 1
     elif data_moda_tmp.shape[0] == 1:
         n = data_moda_tmp.shape[0]
     else:
         print("Error:Index arrays did not properly store")

     data_moda_tmp = data_moda_tmp[1:n + 1]


    print("Initialize Quarterly NP Object")
    data_moda_tmp_quarter = np.zeros([time_dict['subslices'], int(time_dict['timeslices']/time_dict['subslices']), dim_count])
    print("Transformation for Quarterly Data")
    for i in range(0, time_dict['subslices']):
        ids = [x for x in range(i, time_dict['timeslices'], time_dict['subslices'])]
        data_moda_tmp_quarter[i, :, :] = data_moda_tmp[:, ids, :]

    return data_moda_tmp, data_moda_tmp_quarter


if __name__ == "__main__":


    os.chdir("./Data_PP_Output/")
    #data_types_list = ['X_mergered', 'Y_mergered', "CapRatios_mergered"]
    data_types_list = ["CapRatios_mergered"]
    for data_type in data_types_list:
        print(data_type)
        data, data_quarter = request_data(data_type)
        print(data_type,":", data.shape, data_type,"_quarter:", data_quarter.shape)
        print("Saving objects to file")
        np.save("./data_" + data_type + ".npy", data)
        np.save("./data_" + data_type + "_quarter.npy", data_quarter)


#May need to consider creating a table with all the time slices, to save on join.


#Fix Raw data and automate to upload into MySQL tables.
#Format Conditional Modality
#Create Function to handle other modalities.
#Find additional relevant Modalities


#Handle missing data by filling the time slices with nans.  Create a join to populate arrays with place holder.
#Handle Duplicate data in the tables.

#Should we consider data from 1976 onward?



    for data_type in ['sectidx',"zmicro","sbidx", "zmacro_domestic","zmacro_international"]:
        print(data_type)
        data, data_quarter = preprocess_moda_data(tbl_name= data_type, tble_schema = 'STR')
        print(data_type,":", data.shape, data_type,"_quarter:", data_quarter.shape)
        print("Saving objects to file")
        np.save("./data_" + data_type + ".npy", data)
        np.save("./data_" + data_type + "_quarter.npy", data_quarter)


