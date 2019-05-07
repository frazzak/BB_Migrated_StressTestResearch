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
def request_data(data_type = "X", rssd_reportingdate_base_tble = "X_Y_Cap", ReportingDate_Start = "1990 Q1", ReportingDate_End = "2016 Q4" , qtr_count_val =108,
                 host = "localhost", db = 'STR', user = "root", passwd = "",
                 Y_tble = "X_Y_Cap", X_tble = "X_Y_Cap", XYCap_tble = "X_Y_Cap", CreateNewRefTable = False, BankLimit = 50):
    #data_type = "XY_Cap"
    db = msql.connect(host = host , db =db , user = user , passwd = passwd )
    cursor = db.cursor()
    #Create temp cross joined table for Reporting Dates and RSSD_IDs to consider all data
    #
    # if CreateNewRefTable:
    #     sql = "DROP TABLE IF EXISTS TEMPREF_ID_DATES; "
    #     t = cursor.execute(sql)
    # else:
    #     print("Using Previously Existing Ref Table")
    #
    #
    # sql = "CREATE TABLE IF NOT EXISTS TEMPREF_ID_DATES AS select * from (SELECT DISTINCT ReportingDate from %s where ReportingDate >= '%s' and ReportingDate <= '%s' ) a  CROSS JOIN (SELECT DISTINCT RSSD_ID from %s) b;" % (
    # rssd_reportingdate_base_tble, ReportingDate_Start, ReportingDate_End, rssd_reportingdate_base_tble)
    #
    # print("Executing Query to Create Ref list for RSSD_ID and Dates")
    # t = cursor.execute(sql)
    # print("TempRef_ID_Dates:", t)
    #Update to get from 1 table.


    sql = "select DISTINCT ReportingDate" \
          " FROM %s where ReportingDate >= '%s' and ReportingDate <= '%s'" % (rssd_reportingdate_base_tble, ReportingDate_Start, ReportingDate_End)
    print("Executing Query for ReportingDate Counts")
    t = cursor.execute(sql)

    print("ReportingDate Quarters:", t)
    qtr_count = t


    sql = "select DISTINCT RSSD_ID" \
        " FROM %s limit %s" % (rssd_reportingdate_base_tble, BankLimit)

    print("Executing Query for Bank RSSD_IDs")
    t = cursor.execute(sql)
    print("Bank Count:", t)
    bank_count = t

    bank_ids = cursor.fetchall()
    

    #TODO: Make dimensions part dynamic
    #TODO: Make pivot to numpy array faster
    if data_type in ["Y",]:
        print(data_type)
        results = np.zeros([1, qtr_count, 14])
    elif data_type in ["X"]:
        print(data_type)
        results = np.zeros([1, qtr_count, 8])
    elif data_type in  ["CapRatios"]:
        print(data_type)
        results = np.zeros([1, qtr_count, 2])
    else:
        print("No Data Type Found")

    bank_id_count = 1
    for bank_id in bank_ids:
        print(str(bank_id_count), "of",str(bank_count))
        if data_type == "Y":
            # "`ppnrRatio:Net interest income`, `ppnrRatio:Noninterest income`,"
            sql = "SELECT `ReportingDate`, `ncoR:Commercial & industrial`, `ncoR:Construction & land development`, `ncoR:Multifamily real estate`," \
                  "`ncoR:(Nonfarm) nonresidential CRE`, `ncoR:Home equity lines of credit`, " \
                  "`ncoR:Residential real estate (excl. HELOCs)`, `ncoR:Credit card`, `ncoR:Consumer (excl. credit card)`, `ppnrRatio:Net interest income`, `ppnrRatio:Noninterest income`, " \
                  "`ppnrRatio:Trading income`, `ppnrRatio:Compensation expense`, `ppnrRatio:Fixed assets expense`, `ppnrRatio:Noninterest expense`"
            #sql = sql + " `ppnrRatio:Compensation expense`, `ppnrRatio:Fixed assets expense` from y_ij where RSSD_ID='"
            sql = sql + " FROM  %s where ReportingDate >= '%s' and ReportingDate <= '%s' and  RSSD_ID='" % (Y_tble, ReportingDate_Start, ReportingDate_End)
            sql = sql + str(bank_id[0]) + "'"
            temp_data = np.zeros([1, qtr_count, 14])


        elif data_type == "X":
            #bank_id[0] = 1020180
            sql = "SELECT a.`ReportingDate`, "
            #sql = sql + "`Nonfarm nonresidential CRE_2006Q4_on`, `Home equity lines of credit`, `Residential real estate (excl. HELOCs)_Covas`"
            sql = sql +  "a.`Loans categories:Commercial & industrial_Covas`,a.`Loans categories:Construction & land development`,a.`Loans categories:Multifamily real estate`," \
                        "a.`Loans categories:Nonfarm nonresidential CRE_Covas`,a.`Loans categories:Home equity lines of credit`,a.`Loans categories:Residential real estate (excl. HELOCs)_Covas`," \
                        "a.`Loans categories:Credit card`,a.`Loans categories:Consumer (excl. credit card)_Covas`"

            sql = sql + " FROM  %s a where a.ReportingDate >= '%s' and a.ReportingDate <= '%s' and  RSSD_ID='" % (X_tble, ReportingDate_Start, ReportingDate_End)
            sql = sql + str(bank_id[0]) + "'"
            temp_data = np.zeros([1, qtr_count, 8])


        elif data_type == "CapRatios":
            #bank_id[0] = 1020180
            sql = "SELECT a.`ReportingDate`, "
            sql = sql + "a.`Other items:Net Charge Offs`, a.`Other items:T1CR`"  # a.`Other items:Tier 1 common equity`,
            sql = sql + " FROM  %s a where a.ReportingDate >= '%s' and a.ReportingDate <= '%s' and  RSSD_ID='" % (XYCap_tble, ReportingDate_Start, ReportingDate_End)
            sql = sql + str(bank_id[0]) + "'"
            temp_data = np.zeros([1, qtr_count, 1])

        else:
            print("No Data Type Found")
            break
        #print("Executing SQL")
        t = cursor.execute(sql)
        #print("Getting Records")
        records = cursor.fetchall()


        #TODO: We may also need to consider banks with missing data.
        #TODO: Improve to take all the cursor records into npy array at once. and then into the temp data array
        if int(t) >= qtr_count_val and records[0][0] == ReportingDate_Start and records[qtr_count_val - 1][0] == ReportingDate_End:
            print("BankID:%s\tNum:%d" % (bank_id[0], int(t)))
            print("Bank has",qtr_count_val," Quarters from", ReportingDate_Start," to ",ReportingDate_End)
            #print("Trimming Data Accordingly")
            i = 0
            for record in records:
                #print(record)
                #print("Replacing Blank Values with NAN")
                record = tuple([np.nan if x is '' else x for x in list(record)])
                if i == qtr_count_val:
                    #print("Break Loop since i = 108")
                    break
                else:
                    for dim in range(0,len(record) - 1 ):
                        record_dim = dim + 1
                        #print(dim, record_dim, i)
                        temp_data[0, i, dim] = record[record_dim]
                        i = i + 1
                        if i == qtr_count_val:
                            #print("Break Loop since i = 108")
                            break
            results = np.append(results, temp_data, axis=0)
            transform_qtr = True
        else:
            print("Did not match Transformation Criteria")
            transform_qtr = False
        bank_id_count = bank_id_count + 1
    #results.shape
    # quarter based records
    # in quarter based data, the 1st axis is the index of Q1,2,3,4,
    # the 2nd axis represents the number of different banks
    # and the 3nd axis is the temporal sequece, such as 1990 Q1 -> 1991 Q1 -> 1992 Q1 
    # the 4th axis is the dimension of attributes, which refers to the sql sentences
    if transform_qtr:
        n = results.shape[0] - 1
        results = results[1:n+1]
        if data_type in ["X"]:
            results_quarter = np.zeros([4, n, int(qtr_count/4), 8])
        elif data_type in  ["Y"]:
            results_quarter = np.zeros([4, n, int(qtr_count / 4), 14])
        elif data_type in  ["CapRatios"]:
            results_quarter = np.zeros([4, n, int(qtr_count / 4), 1])

        else:
            print("No Data Type Found")

        print("Transformation for Quarterly Data")
        for i in range(0, 4):
            ids = [x for x in range(i, qtr_count, 4)]
            results_quarter[i, :, :, :] = results[:, ids, :]
    else:
        "No Transformations Occured for Quarterly Transform"

        #results_quarter.shape
    print("Completed")
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



def pd_raw_to_numpy_data(df, fulldatescombine = True, ReportingDate_Start = "1990 Q1", ReportingDate_End = "2016 Q4"):
    if fulldatescombine:
        print("Generating Full Mesh RSSD_ID and Reporting Date to left join to.")
        RepordingDate_Df = pd.DataFrame(df["ReportingDate"].unique())
        RepordingDate_Df = RepordingDate_Df.rename({0: "ReportingDate"}, axis=1)
        RepordingDate_Df["key"] = 0

        BankIds_Df = pd.DataFrame(df["RSSD_ID"].unique())
        BankIds_Df = BankIds_Df.rename({0: "RSSD_ID"}, axis=1)
        BankIds_Df["key"] = 0

        BankID_Date_Ref = RepordingDate_Df.assign(foo=1).merge(BankIds_Df.assign(foo=1), on="foo", how="outer").drop(
            ["foo", "key_x", "key_y"], 1)

        df_pd = BankID_Date_Ref.merge(df, left_on=["ReportingDate", "RSSD_ID"],
                                        right_on=["ReportingDate", "RSSD_ID"], how="left")
    else:
        df_pd = df

    print(df_pd.shape)
    print("Subsetting Dates from Dataframe")
    df_pd = df_pd[(df_pd["ReportingDate"] >= ReportingDate_Start) & (df_pd["ReportingDate"] <= ReportingDate_End)]
    print(df_pd.shape)

    print("Pivoting Pandas Table to be Indexed by RSSD_ID and ReportingDate")
    df_pvt = pd.pivot_table(df_pd, index=["RSSD_ID", "ReportingDate"])
    print(df_pvt.shape)

    print("Preparing to Reshape and output as Numpy Array")
    dim1 = df_pvt.index.get_level_values('RSSD_ID').nunique()
    dim2 = df_pvt.index.get_level_values('ReportingDate').nunique()
    print("Reshaping into 3 dimensional NumPy array")
    result_pv = df_pvt.values.reshape((dim1, dim2, df_pvt.shape[1]))
    print(result_pv.shape)

    print("Quarterization of the Data")
    qtr_count = result_pv.shape[1]
    n = result_pv.shape[0]  # - 1
    # result_pv = result_pv[1:n + 1]
    results_quarter = np.zeros([4, n, int(qtr_count / 4), df_pvt.shape[1]])
    print("Transformation for Quarterly Data")
    for i in range(0, 4):
        ids = [x for x in range(i, qtr_count, 4)]
        results_quarter[i, :, :, :] = result_pv[:, ids, :]
    print(results_quarter.shape)

    return result_pv, results_quarter


if __name__ == "__main__":

    os.chdir("./data")
    for keyname in [x for x in Preprocess_Dict.keys() if x in ["X", "Y", "CapRatios","NCO"]]:
        print(keyname)
        data, data_quarter = pd_raw_to_numpy_data(Preprocess_Dict[keyname], fulldatescombine= False)
        print(keyname, ":", data.shape, keyname, "_quarter:", data_quarter.shape)
        print("Saving objects to file")
        np.save("./data_" + keyname + ".npy", data)
        np.save("./quarter_based/data_" + keyname + "_quarter.npy", data_quarter)

    #os.chdir("./Data_PP_Output/")
    #data_types_list = ['X_mergered', 'Y_mergered', "CapRatios_mergered"]
    # data_types_list = ["X","Y","CapRatios"]
    # for data_type in data_types_list:
    #     print(data_type)
    #     data, data_quarter = request_data(data_type = data_type, BankLimit= 11000)
    #     print(data_type,":", data.shape, data_type,"_quarter:", data_quarter.shape)
    #     print("Saving objects to file")
    #     np.save("./data_" + data_type + ".npy", data)
    #     np.save("./data_" + data_type + "_quarter.npy", data_quarter)
    #
    #

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


