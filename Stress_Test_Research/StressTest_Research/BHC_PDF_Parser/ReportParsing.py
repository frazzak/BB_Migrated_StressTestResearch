#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 16:47:58 2018

@author: phn1x
"""

import sys,os
import csv
import pandas as pd
import re
import numpy as np
import tabula
import time
#Prep Dataframe to have metadata for parsing using pandas data parsing and cleaning
# 2017: Skip 10 rows
#Get all Schedule Names and do proper mapping for each report
    #FFIEC 101 Schedules A-S 2014-2017
    #FFIEC 102 4 pages, no Schedules only categories
    #FRY15 Scheules A-F, 2013-2014, A-G 2015 -2017
    
#Find and tag rows with appropriate labels and identify collapse required rows.
def report_item_tagger_collapser(test_tmp):
    coll_idx = []
    failed_idx = []
    indexcode_idx = test_tmp.shape[1] - 1
    indexcode_ridx = test_tmp.shape[0] - 1
    for i in range(0,indexcode_ridx):
        
        #For FFIEC101
#        if i == indexcode_ridx:
#            break
        if test_tmp.iloc[i,0] is not np.nan and '..' in test_tmp.iloc[i,0] and test_tmp.iloc[i,indexcode_idx] is not np.nan and  test_tmp.iloc[i,0].endswith(test_tmp.iloc[i,indexcode_idx]) :
           
            print(test_tmp.iloc[i,0],'| Aligned Properly')
            #pass
        elif test_tmp.iloc[i,0] is not np.nan and test_tmp.iloc[i,0].endswith('(not applicable)'):
            print(test_tmp.iloc[i,0],'| Not Applicable')
            test_tmp.iloc[i,indexcode_idx] = 'Not Applicable'
            coll_idx.append(i)
        elif test_tmp.iloc[i,0] is not np.nan  and '.' not in test_tmp.iloc[i,0] and test_tmp.iloc[i,1:test_tmp.shape[1]].isnull().all():
            print(test_tmp.iloc[i,0],'| SectionInfo')
            test_tmp.iloc[i,indexcode_idx] = 'SectionInfo'
        
        
        elif test_tmp.iloc[i,[0,indexcode_idx]].isnull().all() or "Dollar Amounts in Thousands" in test_tmp.iloc[i,0] and test_tmp.iloc[i,1:indexcode_idx - 1].notnull().any(): 
            print(test_tmp.iloc[i,1],test_tmp.iloc[i,2],test_tmp.iloc[i,3],'| HeaderInfo')
            test_tmp.iloc[i,indexcode_idx] = 'HeaderInfo'
            #pass
        
        elif test_tmp.iloc[i,0] is not np.nan and '..' not in test_tmp.iloc[i,0] and test_tmp.iloc[i,1:indexcode_idx].isnull().all():
            #print(test_tmp.iloc[i,0],test_tmp.iloc[i,1],test_tmp.iloc[i,2],test_tmp.iloc[i,3],test_tmp.iloc[i,4],' Collapse Rows to Below')
            str_tmp = []
            
            #pass
            while test_tmp.iloc[i,0] is not np.nan and '..' not in test_tmp.iloc[i,0]  and test_tmp.iloc[i,1:indexcode_idx].isnull().all():
               print("Collapsing row:", i,":",test_tmp.iloc[i,0])
               test_tmp.iloc[i,indexcode_idx] = "To Be Deleted"
               str_tmp.append(test_tmp.iloc[i,0])
               coll_idx.append(i)
               i = i + 1
               if i == indexcode_ridx + 1:
                   print("End of Report Line")
                   i = i - 1
                   break
                   
            if test_tmp.iloc[i,0] is not np.nan and  '..' in test_tmp.iloc[i,0] or test_tmp.iloc[i,1:indexcode_idx].notnull().all() and '..' in test_tmp.iloc[i,1]:
                str_tmp.append(test_tmp.iloc[i,0])
                #print(" ".join(str_tmp),test_tmp.iloc[i,1],test_tmp.iloc[i,2],test_tmp.iloc[i,3],test_tmp.iloc[i,indexcode_idx],"| Collapsed")
                print(" ".join(str_tmp),test_tmp.iloc[i,indexcode_idx],"| Collapsed")
                print("Updating Row")
                test_tmp.iloc[i,0] = " ".join(str_tmp)
                #print("Deleting Collapsed Rows")
                  
                str_tmp = []
        #elif test_tmp.iloc[i,0] is not np.nan and  '..' in test_tmp.iloc[i,0] or '..' in test_tmp.iloc[i,1] and  test_tmp.iloc[i,1] is not np.nan and test_tmp.iloc[i,2] is not np.nan and test_tmp.iloc[i,3] is not np.nan and test_tmp.iloc[i,4] is not np.nan :
            #print(test_tmp.iloc[i,0],test_tmp.iloc[i,1],test_tmp.iloc[i,2],test_tmp.iloc[i,3],test_tmp.iloc[i,4],"| Collapse Above to Here")
            #pass
        elif test_tmp.iloc[i,indexcode_idx] is np.nan:
            print(test_tmp.iloc[i,:], '| FAILED')
            failed_idx.append(i)
    try:
        print("Deleting index ", coll_idx)
        test_tmp = test_tmp.drop(test_tmp.index[coll_idx])
    
    except Exception as a:
        print(a)      
        print("Delete Failed")
    print("Failed Parsing on:", failed_idx)
    return(test_tmp)


#Compatibility 5 columns 
#[1, 2, 5, 6, 18, 23, 24, 34, 35]
    
    #FFIEC101 page index [1,2,6]
        #Pages 2-5 may be most useful and easiest to parse.  
            #Dynmaic Column detection for parsing required.
    #FFIEC102
        #All 6 tables from the report should be parseable.
            #Dynamic Column detection and robust row tagging and detection required.
    #FRY15, FRY9LP Parsable
    #BHPR will need customized parsing due to some columns requiring splititng and not index column.





#Clean up columns
def column_cleanup(repattern = None, tableobj_tmp = None, fillna = '', verbose = True, nafill = False):
    for i in range(0,tableobj_tmp.shape[1]):
        if nafill:
            if verbose:
                print("Filling NaNs")
            tableobj_tmp.iloc[:,i].fillna(fillna)
        if verbose:
            print("Replace consecutive dots in columns:", i)
        
        if tableobj_tmp.iloc[:,i].isnull().all():
            if verbose:
                print("Skipping Column: all NaN")
        elif tableobj_tmp.iloc[:,i].isnull().any():
            if verbose:
                print("Found NaNs in column")
                print(tableobj_tmp.iloc[:,i])
            for y in range(0,tableobj_tmp.shape[0]):
                if tableobj_tmp.iloc[y,i] is np.nan:
                    pass
                else:
                    if verbose:
                        print("Trim whitespaces and Removing Pattern by Row")
                    #print("Removing Consecutive periods and space patterns")
                    if repattern is None:
                        print("No REGEX Pattern, will match known patterns")
                        repattern = r'\.{2,}'
                        tableobj_tmp.iloc[y,i] = re.sub(repattern,'',tableobj_tmp.iloc[y,i])
                        repattern1 = r'[ ]\.{1,}'
                        if bool(re.match(repattern1,tableobj_tmp.iloc[y,i])):
                            print("Removing spaces and period pattern")
                            tableobj_tmp.iloc[y,i] = re.sub(repattern1,'',tableobj_tmp.iloc[y,i])
                        tableobj_tmp.iloc[y,i].strip()
                    else:
                        tableobj_tmp.iloc[y,i] = re.sub(repattern,'',tableobj_tmp.iloc[y,i])
                        tableobj_tmp.iloc[y,i].strip()
        else:   

            if repattern is None:
                repattern = r'\.{2,}'
                tableobj_tmp.iloc[:,i] = tableobj_tmp.iloc[:,i].str.replace(repattern,'')
                repattern = r'[ ]\.{1,}' 
                tableobj_tmp.iloc[:,i] = tableobj_tmp.iloc[:,i].str.replace(repattern,'')
            else:
                tableobj_tmp.iloc[:,i] = tableobj_tmp.iloc[:,i].str.replace(repattern,'')
            
            
            if verbose:
                print("Trim whitespaces")
            tableobj_tmp.iloc[:,i] = tableobj_tmp.iloc[:,i].str.strip() 
            
    return(tableobj_tmp)




def report_column_alignmentstruct(tabulaList_df = None, ReportType = "FFIEC101", ReportData = None):    
    i = 0
    coll_idx = []
    result_df = tabulaList_df
    if ReportType in  ["FFIEC101", "FFIEC102","FRY15"]:
        print("Processing:",ReportType)
        #if isinstance(result_df, pd.DataFrame):
        repattern2 = r'[ ]\.{1,}'
        while i < len(result_df):
            print(i, ReportData)
            deletedflag = False
            if isinstance(result_df, pd.DataFrame) and not result_df.empty:
                print("DataFrame Passed: Alignment Struct")
                result_df_tmp = result_df
                result_df = [result_df_tmp]
                #result_df.append(pd.DataFrame(result_df_tmp))
                i = 0
                #print(pd.DataFrame(result_df[0]).shape)
                #result_df.append(result_df_tmp)
                #print(result_df[i])
                result_df_tmp = result_df[i]
                #print(result_df_tmp)
                if result_df_tmp.shape[1] > 2:
                    print("Changing Column 1 to String")
                    result_df_tmp.iloc[:,1] = result_df_tmp.iloc[:,1].astype(str)
                    breakwhile = True
                    
            elif isinstance(result_df, list) and not result_df[i].empty:
                print("List of DataFrames Passed")
                result_df_tmp = result_df[i]
                #print(result_df_tmp)
                if result_df_tmp.shape[1] > 2:
                    print("Changing Column 1 to String")
                    result_df_tmp.iloc[:,1] = result_df_tmp.iloc[:,1].astype(str)
                    breakwhile = False
            elif isinstance(result_df, list) and result_df[i].empty:
                breakwhile = True
                print("No Data in this frame")
                coll_idx.append(i)
                deletedflag = True
                #continue
                i += 1
                
                   
                
            if not breakwhile:    
                if ReportType == "FFIEC102" and result_df_tmp.shape[1] > 3 and result_df_tmp.iloc[0,0] is np.nan and result_df_tmp.iloc[0,2] == "MRRR" and result_df_tmp.iloc[0,3] in  ["Percentage","Date","Amount"]:
                    print("FFIEC102 Misalignment, 5 columns to 4")
                    #result_df_tmp.iloc[:,0] = result_df_tmp[[0,1]].astype(str).apply(lambda x: ''.join(x), axis=1)
                    #result_df_tmp.iloc[:,1] = np.nan
                    #result_df_tmp.iloc[0,0] = "Percentage"
                    #result_df_tmp.iloc[0,3] = np.nan
                    
                    result_df_tmp.iloc[:,0] = result_df_tmp[[0,1]].astype(str).apply(lambda x: ''.join(x), axis=1)
                    
                    result_df_tmp[[1]] = np.nan
                    #result_df_tmp.iloc[:,1] = result_df_tmp.iloc[:,1].astype(object)
                    result_df_tmp.iloc[0,0] = result_df_tmp.iloc[0,3]
                    result_df_tmp.iloc[0,3] = np.nan
                    result_df_tmp[[3]] = result_df_tmp[[3]].astype(str)
                    result_df_tmp[[3]] = result_df_tmp[[3]].replace("nan",np.nan)
                    
                    
                print(result_df_tmp)
            
                for y in range(0,result_df_tmp.shape[0]):    
                    #print(i,y)
                    #print(result_df_tmp.iloc[y,:])
                    #print(result_df_tmp.iloc[y,1] not in  [np.nan,"NaN","nan"])
                    
                    if result_df_tmp.shape[1] < 2:
                        print("Additional Parsing Required For this page, not including in list: Few Columns") 
                        #del result_df_tmp #After deletion of report, next i and reset or rows required
                        coll_idx.append(i)
                        deletedflag = True
                   
                    elif result_df_tmp.iloc[y,0] is np.nan and result_df_tmp.iloc[y,1] is not np.nan and ((isinstance(result_df_tmp.iloc[y,1], float) and result_df_tmp.iloc[y,1].astype(str) not in [np.nan,"nan","NaN"]) or (isinstance(result_df_tmp.iloc[y,1], str) and result_df_tmp.iloc[y,1] not in [np.nan,"nan","NaN"])) and result_df_tmp.iloc[y,1].endswith("Dollar Amounts in Thousands"): #Should we dynamically look for the Section?
                        print("Found Header Misalignment for columns, Correcting.")
                        result_df_tmp.iloc[y,0] = "Dollar Amounts in Thousands"
                        result_df_tmp.iloc[y,1] = np.nan
                        if ReportType == "FRY15" and result_df_tmp.shape[1] == 6 and result_df_tmp.iloc[y,1] is np.nan and result_df_tmp.iloc[y,2] == "RISK" and result_df_tmp.iloc[y,3] == "Amount" and result_df_tmp.iloc[y,4] is np.nan and result_df_tmp.iloc[y,5] == "HeaderInfo":
                            print("FRY15 Addiotnal Parsing and shifting: RISK AMOUNT Header Aligment, Six Columns")
                            result_df_tmp.iloc[y,4] = result_df_tmp.iloc[y,2]
                            result_df_tmp.iloc[y,3] = np.nan
                            #result_df_tmp.iloc[y,4] = "HeaderInfo"
                            
                   
                    elif ReportType == "FFIEC102" and result_df_tmp.iloc[y,0] is np.nan and  result_df_tmp.iloc[y,1] is not np.nan and ((isinstance(result_df_tmp.iloc[y,1], float) and result_df_tmp.iloc[y,1].astype(str) not in [np.nan,"nan","NaN"]) or (isinstance(result_df_tmp.iloc[y,1], str) and result_df_tmp.iloc[y,1] not in [np.nan,"nan","NaN"])) and result_df_tmp.iloc[y,1].startswith("Dollar Amounts in Thousands") and not result_df_tmp.iloc[y,1].endswith("Dollar Amounts in Thousands"):
                        print("Found Header Misalignment that needs parsing, Correcting.")
                        result_df_tmp.iloc[y,0] = "Dollar Amounts in Thousands"
                        result_df_tmp.iloc[y,1] = result_df_tmp.iloc[y,1].split("Dollar Amounts in Thousands ")[-1]
                   
                    
                    elif result_df_tmp.iloc[y,1] is not np.nan and  (( isinstance(result_df_tmp.iloc[y,1], float) and result_df_tmp.iloc[y,1].astype(str) not in [np.nan,"nan","NaN"]) or (isinstance(result_df_tmp.iloc[y,1], str) and result_df_tmp.iloc[y,1] not in [np.nan,"nan","NaN"])) and bool(re.match(repattern2,result_df_tmp.iloc[y,1])):
                        print("Found Consecutive space and periods to remove")
                        print(result_df_tmp.iloc[y,1])
                        result_df_tmp.iloc[y,1] =  re.sub(repattern2,"",result_df_tmp.iloc[y,1]).strip()
                    elif result_df_tmp.iloc[y,0] is np.nan and result_df_tmp.iloc[y,1:].str.startswith("Percentage").any() and result_df_tmp.iloc[y,result_df_tmp.shape[1] -1] == "HeaderInfo" :
                        print("Description Misalignment: Description in Amounts") 
                        result_df_tmp.iloc[y,0] =  "Percentage"
                        result_df_tmp.iloc[y,2] = np.nan
                   
                    elif result_df_tmp.iloc[y,2] == "MRRR" and result_df_tmp.iloc[y,3] in  ["Percentage","Date","Amount","Number"] and result_df_tmp.iloc[y,result_df_tmp.shape[1] -1] == "HeaderInfo":
                        print("Shifting Header Info to Column 0")
                        result_df_tmp.iloc[y,0] = result_df_tmp.iloc[y,3]
                        result_df_tmp.iloc[y,3] = np.nan
                        
                    
                   
                        
                    
                    elif result_df_tmp.iloc[y,0] is np.nan and result_df_tmp.iloc[y,1:result_df_tmp.shape[1] - 2].str.startswith("(Column ").any() and result_df_tmp.iloc[y,result_df_tmp.shape[1] -1] == "HeaderInfo" :
                        print("Additional Parsing Required For this table, not including in list: (Column [A-Z])") 
                        #del result_df_tmp #After deletion of report, next i and reset or rows required
                        coll_idx.append(i)
                        deletedflag = True
                        break
                    
                    elif result_df_tmp.iloc[y,1:result_df_tmp.shape[1] - 1].fillna("").str.endswith("Percentage").all() and result_df_tmp.iloc[y,0] is not np.nan and  result_df_tmp.iloc[y,result_df_tmp.shape[1] - 1] is np.nan:
                        print("Additional Parsing Required For this table, not including in list: Percentage") 
                        #del result_df_tmp #After deletion of report, next i and reset or rows required
                        coll_idx.append(i)
                        deletedflag = True
                        break
                    
                    
                    elif ReportType == "FFIEC102" and result_df_tmp.iloc[y,0] is not np.nan  and result_df_tmp.iloc[y,2] is not np.nan and result_df_tmp.iloc[y,result_df_tmp.shape[1] -1] is not np.nan and (result_df_tmp.iloc[y,result_df_tmp.shape[1] -1].endswith(result_df_tmp.iloc[y,0] + ".") or result_df_tmp.iloc[y,result_df_tmp.shape[1] -1].endswith(result_df_tmp.iloc[y,0])) and result_df_tmp.iloc[y,2].count(" ") == 1:
                        print("FFIEC102 Additional Parsing Required: Concatenating Column 0 and 1 and splitting column 2 addint period to index")
                        #print(result_df_tmp.iloc[y,:])
                        if result_df_tmp.iloc[y,0].count(".") > 0:
                            result_df_tmp.iloc[y,0] = result_df_tmp.iloc[y,0] + " " + result_df_tmp.iloc[y,1]
                        else: 
                            result_df_tmp.iloc[y,0] = result_df_tmp.iloc[y,0] + ". " + result_df_tmp.iloc[y,1]
                        result_df_tmp.iloc[y,1] = result_df_tmp.iloc[y,2].split(" ")[0]
                        result_df_tmp.iloc[y,2] = result_df_tmp.iloc[y,2].split(" ")[1]
                    
                        if result_df_tmp.iloc[y,0] == "nannan":      
                            print("FFIEC102 Additional Parsing and shifting of HeaderInfo: MRRR Number")
                            result_df_tmp.iloc[y,0] = result_df_tmp.iloc[y,2]
                            result_df_tmp.iloc[y,2] = np.nan
                        else: 
                            pass
                    
                    #elif (ReportType in ["FFIEC101","FFIEC102"] and result_df_tmp.shape[1] == 5 and result_df_tmp.iloc[y,3] is not np.nan) and result_df_tmp.iloc[y,3].count(".") == 2:
                        #print("Extraneous Periods Found: Percentage prefixed period")
                        #result_df_tmp.iloc[y,3] = result_df_tmp.iloc[y,3].replace(".","",1)              
                
                    elif ReportType == "FRY15" and result_df_tmp.shape[1] > 3 and result_df_tmp.iloc[y,1] == "RISK" and result_df_tmp.iloc[y,2] == "Amount" and result_df_tmp.iloc[y,3] is np.nan and result_df_tmp.iloc[y,4] == "HeaderInfo":
                        print("FRY15 Additional Parsing and shifting: RISK AMOUNT Header Aligment")
                        result_df_tmp.iloc[y,3] = result_df_tmp.iloc[y,2]
                        result_df_tmp.iloc[y,2] = np.nan
                        #result_df_tmp.iloc[y,4] = "HeaderInfo"
    
                    
                
                if not deletedflag:
                    #print("Check if Nan column is Amount Column")
                    #if result_df_tmp.shape[1] == 4 and result_df_tmp.iloc[:,2].unique  np.nan #and result_df_tmp.iloc[:,0].unique is not np.nan and result_df_tmp.iloc[:,1].unique is not np.nan and result_df_tmp.iloc[:,3].unqiue is not np.nan:
                        #print("Filling NaN Amount Column")
                        #result_df_tmp.iloc[:,2] = "nan"
                    result_df_tmp = result_df_tmp.astype(str)
                    print("Replacing Blanks with NaNs")                
                    result_df_tmp.replace("NaN", np.nan,inplace = True)
                    result_df_tmp.replace("nan", np.nan,inplace = True)
                    result_df_tmp.replace(r'^s*$', np.nan, regex=True, inplace = True)
                    #print(result_df_tmp)
                    print ("Drop Columns that are all NaN")
                    #if result_df_tmp.shape[1] > 4:
                    result_df_tmp= result_df_tmp.dropna(axis=1,how='all')            
                    #print(result_df_tmp)
                    FFIEC101_ColumnsBase = ["Description", "ReportCode","Amount","IndexInfo","Report_Type","Report_RSSD","Report_Date"]
                    print("Adding Column Names")
                    result_df_tmp["Report_Type"] = ReportData[0] 
                    result_df_tmp["Report_RSSD"] = ReportData[1]
                    result_df_tmp["Report_Date"] = ReportData[2]
                    #print(i, result_df_tmp.columns.values)
                    if result_df_tmp.shape[1] == 7:
                        result_df_tmp.columns = FFIEC101_ColumnsBase
                    else:
                        print("Columns did not match")
                        coll_idx.append(i)
                        print(result_df_tmp)
                        breakwhile = True
                        
                    #print(i, result_df_tmp.columns.values)
                    if not breakwhile:
                        print("Post Alignment Data Cleanup")
                        #print("Filling NaN Amount Column")
                        #if result_df_tmp["Amount"].null().all():
                        #    result_df_tmp["Amount"].fillna("nan")
                        print("Removing Bad OCR __ _")
                        #result_df_tmp["Amount"][result_df_tmp["Amount"].str.match("__ _").fillna(False)] = result_df_tmp["Amount"][result_df_tmp["Amount"].str.match("__ _").fillna(False)].str.replace("__ _","")
                        #if isinstance(result_df_tmp["Amount"], str):
                        result_df_tmp["Amount"] = result_df_tmp["Amount"].str.replace("__ _ ","")
                        result_df_tmp["Amount"] = result_df_tmp["Amount"].str.replace("_._ _","")
                    
                        print("Removing Artifact Decimals and percent signs")
                        result_df_tmp["Amount"][result_df_tmp["Amount"].str.match("^\.$").fillna(False)] = np.nan
                        result_df_tmp["Amount"][result_df_tmp["Amount"].str.match("^. %$").fillna(False)] = np.nan
        
                        print("Removing Extraneous Prefix Decimal:FFIEC Reports")
                        result_df_tmp["Amount"][result_df_tmp["Amount"].str.match("^\.[0-9].*$").fillna(False)] = result_df_tmp["Amount"][result_df_tmp["Amount"].str.match("^\.[0-9].*$").fillna(False)].str.replace(".","",1)
                        
                        print("Replacing String nan with np.nan")
                        result_df_tmp.replace("NaN", np.nan,inplace = True)
                        result_df_tmp.replace("nan", np.nan,inplace = True)
                        
                        print("Replacing Number with np.nan in Amount Column")
                        result_df_tmp["Amount"].replace("Number",np.nan, inplace = True)
                        result_df_tmp["IndexInfo"][result_df_tmp["Description"].str.match("Backtesting (over the most recent calendar quarter)").fillna(False)] = "HeaderInfo"
                        result_df_tmp["Amount"][result_df_tmp["Description"].str.match("Backtesting (over the most recent calendar quarter)").fillna(False)] = np.nan
                        result_df_tmp["Description"][result_df_tmp["Description"].str.match("Backtesting (over the most recent calendar quarter)").fillna(False)] = result_df_tmp["Description"][result_df_tmp["Description"].str.match("Backtesting (over the most recent calendar quarter)").fillna(False)] + " |Number"
                        
                        print("Replace ending nan description ")
                        result_df_tmp["Description"][result_df_tmp["Description"].str.endswith("nan").fillna(False)] = result_df_tmp["Description"][result_df_tmp["Description"].str.endswith("nan").fillna(False)].replace("nan","")
                        
                        print("Post Fix MRRR Number Column Issue")
                        print(result_df_tmp)
                        result_df_tmp["Amount"][result_df_tmp["Amount"].astype(str).str.match("MRRR Number").fillna(False)] = np.nan
                        result_df_tmp["ReportCode"][result_df_tmp["Amount"].astype(str).str.match("MRRR Number").fillna(False)] = "MRRR"
                        result_df_tmp["Description"][result_df_tmp["Amount"].astype(str).str.match("MRRR Number").fillna(False)] = "Number"
                        
                    else: print("Skipping Post Cleanup")
                    
                result_df[i] = result_df_tmp
                print("Cleaned Table")
                print(result_df[i])
                if breakwhile: 
                    i += 1 
                else: 
                    print("Cleaned Tabled")
                    result_df[i] = result_df_tmp
                    i += 1            
                
    elif ReportType in ["FRY9LP","FRY9C","BHCPR"]:
        print("To be developed")
    else: print("Report Type Not Found")

    try:
        print("Deleting index ", coll_idx)
        for i in sorted(coll_idx, reverse=True):
            del result_df[i]
    except Exception as a:
        print(a)      
        print("Delete Failed")     
    
    

    
    return (result_df)
    




def report_parser_dataframer(reportsourcefolder = "/Users/phn1x/icdm2018_research_BB/Stress_Test_Research/StressTest_Research/unsecured_pdf_complete/", reportfilepath = None, extension = ".PDF.pdf"):
    #preconcat = list()
    
    master_result_list = list()
   # FFIEC101_ColumnsBase = ["Description", "ReportCode","Amount","IndexInfo","Report_Type","Report_RSSD","Report_Date"]
    if isinstance(reportfilepath, list):
        print("Paths is List:", isinstance(reportfilepath, list))
        for y in reportfilepath:
            result_df = list()                 
            ReportData = os.path.basename(y).replace(extension,"").split("_")
            print("Setting Tabula Page Parameters")
            if ReportData[0] == "FFIEC101" and ReportData[2] < "20140401":
                filepages = "2-4"
            elif    ReportData[0] == "FFIEC101" and ReportData[2] < "20160701":
                filepages = "2-5"
            elif ReportData[0] == "FFIEC101" and ReportData[2] > "20160701":
                filepages = "2-6"
            elif ReportData[0] == "FFIEC102":
                filepages = "2-4"
            elif ReportData[0] == "FRY15" and ReportData[2] > "20150101":
                filepages = "2-5"
                #Additional Logic and Parsing Needed to Process Short-Term Wholesale Funding  Schedule G page 6
            elif ReportData[0] == "FRY15" and ReportData[2] < "20150101":
                filepages = "2-4"
            
            
            print("Determining Report Type to set parameters for:",y)
            result_df = list()
            if ReportData[0] in ["FFIEC101","FFIEC102","FRY15"]:
                print("Processing Tables with Tabula")
                report = tabula.read_pdf(y, pages = filepages, guess = True, multiple_tables = True)
                
                for i in range(0,len(report)):        
                    testtmp = pd.DataFrame.copy(report[i],deep=True)
                    testtmp = report_item_tagger_collapser(testtmp)
                    testtmp = column_cleanup(None,testtmp,verbose = False)
                    #print("Adding Report Reference Data")
                    testtmp = testtmp.reset_index(drop=True)
                    result_df.append(pd.DataFrame(testtmp))
                print("Aligning Columns and adding Column Names")
                result_df = report_column_alignmentstruct(result_df,ReportData[0], ReportData)
                print("Concatenating Dataframes")
                #print(type(result_df))
                if isinstance(result_df, pd.DataFrame):# May not be needed
                    print("DataFrame to Concat") #May not be needed
                    master_result_list.append(result_df) #May not be needed.
                if isinstance(result_df, list):
                    print("List to Concat")
                    #master_result = pd.concat(result_df)
                    #print(result_df)
                    result_df = pd.concat(result_df,ignore_index = True)
                    #preconcat.append((result_df.Amount[result_df["ReportCode"].str.match("P911").fillna(False)]))
                    #print(type(result_df))
                    master_result_list.append(result_df)
                    #print(type(master_result_list))
        #print(type(master_result_list))
        master_result = pd.concat(master_result_list)
        #print(type(master_result))
        #print(preconcat)
        #print("Proper Appending of Unique Data Check")
        for j in master_result["ReportCode"].unique():
            if j is not np.nan and len(master_result.Amount[master_result["ReportCode"].str.match(j).fillna(False)].unique()) == 1:
                print("Potential Error|ReportCode:",i," all values are same")
           
        print("Spot Check P911")
        print(master_result[["ReportCode","Amount","Report_RSSD"]][master_result["ReportCode"].str.match("P911").fillna(False)])
        returnobj = master_result.dropna(axis=1,how='all')  
        #print(type(returnobj))
        print("Final Clean up of string nans")
        returnobj = returnobj.replace("nan",np.nan)
        returnobj = returnobj.reset_index(drop = True)
    else:
        ReportData = os.path.basename(reportfilepath).replace(extension,"").split("_")
        
        print("Setting Tabula Page Parameters")
        if ReportData[0] == "FFIEC101" and ReportData[2] < "20140401":
            filepages = "2-4"
        elif    ReportData[0] == "FFIEC101" and ReportData[2] < "20160701":
            filepages = "2-5"
        elif ReportData[0] == "FFIEC101" and ReportData[2] > "20160701":
            filepages = "2-6"
        elif ReportData[0] == "FFIEC102":
            filepages = "2-6"
        elif ReportData[0] == "FRY15" and ReportData[2] > "20150101":
            filepages = "2-5"
            #Additional Logic and Parsing Needed to Process Short-Term Wholesale Funding  Schedule G page 6
        elif ReportData[0] == "FRY15" and ReportData[2] < "20150101":
            filepages = "2-4"
        
        #if ReportData[0] == "FFIEC101": #Report pages change after 2016
        print("Processing:",ReportData[0])
        print("Processing Tables with Tabula")
        report = tabula.read_pdf(reportfilepath, pages = filepages, guess = True, multiple_tables = True)
        result_df = list()
        for i in range(0,len(report)):        
            testtmp = pd.DataFrame.copy(report[i],deep=True)
            testtmp = report_item_tagger_collapser(testtmp)
            testtmp = column_cleanup(None,testtmp,verbose = False)
            testtmp = testtmp.reset_index(drop=True)
            print("Adding Report Reference Data")
            result_df.append(pd.DataFrame(testtmp))
        print("Aligning Columns and adding Column Names")
        result_df = report_column_alignmentstruct(result_df,ReportData[0], ReportData)
        print("Concatenating Dataframes")
        #print(type(result_df))
        if isinstance(result_df, pd.DataFrame):
            print("DataFrame to Concat")
            master_result = result_df
        if isinstance(result_df, list): # May not be needed.
            print("List to Concat") #May not be needed
            master_result = pd.concat(result_df, ignore_index = True) # May not be needed
        returnobj = master_result.dropna(axis=1,how='all')
        print("Final Clean up of string nans")
        returnobj = returnobj.replace("nan",np.nan)
        
        
        
        
        returnobj = returnobj.reset_index(drop = True)
        
    
    
    DescriptionReportCodes = ["2170","FS88","FS90","M336","M341","Y832","Y831","M335"]
    
    print("Post Parsing Fixes")

    #Bad Parse post fix
    print("Fixing Description\ReportCode bad parses for :", DescriptionReportCodes)
    for i in DescriptionReportCodes:
        returnobj["ReportCode"][returnobj["Description"].astype(str).str.endswith(i).fillna(False)] = i
        returnobj["Description"][returnobj["Description"].astype(str).str.endswith(i).fillna(False)] = returnobj["Description"][returnobj["Description"].astype(str).str.endswith(i).fillna(False)].str.replace(i,"").str.strip()
    
    
    #NO report Code post fix
    print("Creating ReportCode Placeholders for missing ReportCodes")
    #returnobj["ReportCode"][returnobj["ReportCode"].isnull()][~returnobj["IndexInfo"].isin(["SectionInfo","HeaderInfo"])] = returnobj["Report_Type"][returnobj["ReportCode"].isnull()][~returnobj["IndexInfo"].isin(["SectionInfo","HeaderInfo"])] + "_" + returnobj["IndexInfo"][returnobj["ReportCode"].isnull()][~returnobj["IndexInfo"].isin(["SectionInfo","HeaderInfo"])]
    
    print('Generating Placeholder ReportCodes')
    updatedReportsCodes = returnobj["Report_Type"][returnobj["ReportCode"].isnull()][~returnobj["IndexInfo"].isin(["SectionInfo","HeaderInfo"])] + "_" + returnobj["IndexInfo"][returnobj["ReportCode"].isnull()][~returnobj["IndexInfo"].isin(["SectionInfo","HeaderInfo"])].str.strip(".").str.strip()    
    print('Generating Index for Update')
    index1 = returnobj["ReportCode"][~returnobj["IndexInfo"].isin(["SectionInfo","HeaderInfo"])][returnobj["ReportCode"].isnull()].index
    print('Updating with Placeholders')
    returnobj.loc[index1,'ReportCode'] = updatedReportsCodes
    
    #Fill in Report Code for "1=Yes"
    #Remove 1=Yes
    print('Generating Placeholder ReportCodes for 1=Yes')
    updatedReportsCodes = returnobj["Report_Type"][returnobj["ReportCode"].astype(str).str.match("1=Yes")] + "_" + returnobj["IndexInfo"][returnobj["ReportCode"].astype(str).str.match("1=Yes")].str.strip(".").str.strip()    
    print('Generating Index for Update')
    index1 = returnobj["ReportCode"][returnobj["ReportCode"].astype(str).str.match("1=Yes")].index
    print('Updating with Placeholders')
    returnobj.loc[index1,'ReportCode'] = updatedReportsCodes
    


    print("Fill in Blank Descriptions")
    test1 = returnobj["ReportCode"][returnobj["Description"].isnull()][~returnobj["IndexInfo"].isin(["SectionInfo","HeaderInfo"])]

    #Remove ending nan
    
    
    print('Replacing Ending Nans')
    updatedReportsCodes = returnobj["Description"][returnobj["Description"].astype(str).str.endswith("nan")][~returnobj["IndexInfo"].isin(["SectionInfo","HeaderInfo"])].str.replace("nan","").str.strip()    
    print('Generating Index for Update')
    index1 = returnobj["Description"][returnobj["Description"].astype(str).str.endswith("nan")][~returnobj["IndexInfo"].isin(["SectionInfo","HeaderInfo"])].index
    print('Updating with Placeholders')
    returnobj.loc[index1,'Description'] = updatedReportsCodes


    #Missing Description Names
    test2 = test1.unique()    
    for i in test2:
        
        test3 = returnobj["Description"][returnobj["ReportCode"].astype(str).str.match(i)].unique()
                
        if len(test3) > 1:
            print(i)
            #Drop Null
            print(test3)
            test4 = [x for x in test3 if str(x) != 'nan']    
            print(test4)
            test5 = test4[0]    
            returnobj["Description"][returnobj["ReportCode"].astype(str).str.match(i)] = test5
            print(returnobj["Description"][returnobj["ReportCode"].astype(str).str.match(i)].unique())
        

            print("Fill Standard Descriptions for same ReportCodes")
            test1 = returnobj["ReportCode"][~returnobj["IndexInfo"].isin(["SectionInfo","HeaderInfo"])]
            
            #Missing Description Names
            test2 = test1.unique()    
            for i in test2:
                test3 = returnobj["Description"][returnobj["ReportCode"].astype(str).str.match(i)].unique()
                        
                if len(test3) > 1:
                    print(i)
                    print(test3)
                    #Drop Null
                    test4 = [x for x in test3 if str(x) != 'nan']    
                    print(test4)
                    test5 = test4[-1] #Give last description in array    
                    returnobj["Description"][returnobj["ReportCode"].astype(str).str.match(i)] = test5
                    print(returnobj["Description"][returnobj["ReportCode"].astype(str).str.match(i)].unique())


    
    
    #print('Find in Amount Column Indexes with multiple periods')
    index1 = returnobj["Description"][returnobj["Description"].astype(str).str.endswith(". . . . . .")].index
    returnobj.loc[index1,"Description"]= returnobj.loc[index1,"Description"].str.replace(". .","").str.strip()
    
    
    #print('Find in Amount Column Commas to remove')
    
    
    # print('Remove First Period if multiple periods in Amount Column')
    
    
    # Merge Similar Columns
    
    
    print('Update Descriptions and Report Code to trip white spaces' )
    returnobj["Description"] = returnobj["Description"].str.strip()
    returnobj["ReportCode"] = returnobj["ReportCode"].str.strip()

    return(returnobj)


#Build Transpose and rearrange Create Function

#Column Names
def reporttransposerarrangerbycell(result_output2, verbose = False):
    start_time = time.time()
    sincelast = 0
    print("Creating Dataframe Structure")
    print("Adding Standard Identifier Columns")
    d1 = pd.DataFrame([],columns = ["Report_Type","Report_RSSD","Report_Date"])
    ReportCodes = pd.DataFrame([],columns = pd.Series(result_output2["ReportCode"][~result_output2["IndexInfo"].isin(["SectionInfo","HeaderInfo"])].unique()))
    if verbose:
        print("Adding ReportCodes from Raw File and Binding to DataFrame")
    d2 = pd.concat([d1,ReportCodes], axis=1)


    #Get initial Data for First Rows
    print("Creating Identity Rows")    
    d3 = result_output2[["Report_Type","Report_RSSD","Report_Date"]][~result_output2["IndexInfo"].isin(["SectionInfo","HeaderInfo"])]
    if verbose:
        print("Dropping duplicates")
    d4 = d3.drop_duplicates(subset = ["Report_Type","Report_RSSD","Report_Date"])
    if verbose:
        print("Inserting Rows to Dataframe")
    d2[["Report_Type","Report_RSSD","Report_Date"]] = d4

    print("Transposing and strucutring data from raw object to new structure")    
    for i in range(0,len(d2)):
        tmp_ReportType = d2.iloc[i,0]    
        tmp_ReportRSSD = d2.iloc[i,1].astype(str)
        tmp_ReportDate = d2.iloc[i,2].astype(str)
        tmp_colnames = result_output2["ReportCode"][~result_output2["IndexInfo"].isin(["SectionInfo","HeaderInfo"])][result_output2["Report_Type"] == tmp_ReportType].unique()
        print("Populating Row: ", i, "of ", len(d2), "(",(i/len(d2)) * 100, "% Completed)")
        elapsed_time = time.time() - start_time
        sincelast = elapsed_time - sincelast
        print("Since Last: ", sincelast, " Time Elapsed: ", elapsed_time)
        for y in tmp_colnames:
            if verbose:
                print(y)
            tmp_ColNum = d2.columns.get_loc(y)
            if verbose:
                print(i," ",tmp_ColNum)
            #Matching to other data frame.
            tmp_value = result_output2["Amount"][result_output2["ReportCode"].astype(str).str.match(y)][result_output2["Report_RSSD"].astype(str).str.match(tmp_ReportRSSD)][result_output2["Report_Date"].astype(str).str.match(tmp_ReportDate)][result_output2["Report_Type"].astype(str).str.match(tmp_ReportType)]
            if verbose:
                print(tmp_value, len(tmp_value))
            
            if tmp_value.empty:
                if verbose:
                    print("Series is empty, setting NAN")
                d2.iloc[i,tmp_ColNum] = np.nan
            elif not tmp_value.isnull().all() and len(tmp_value) == 1:
                d2.iloc[i,tmp_ColNum] = tmp_value.values[0]
            elif tmp_value.isnull().all():
                if verbose:
                    print("Series is NULL, setting NAN")
                d2.iloc[i,tmp_ColNum] = np.nan
                
            else:
                if verbose:
                    print(" has multiple values")
                tmp_value2 =  [x for x in tmp_value if str(x) != 'nan']    
                if verbose:
                    print("Dropping Nans and Duplicates from array of multiple values")
                #tmp_value3 = tmp_value2.unique
               # print("Value being set: ", list(set(tmp_value2))[0])
                d2.iloc[i,tmp_ColNum] = list(set(tmp_value2))[0]
    return(d2)

def reporttransposerarrangerbycolumn(result_output2, verbose = False):
    start_time = time.time()
    sincelast = 0
    print("Creating Dataframe Structure")
    print("Adding Standard Identifier Columns")
    d1 = pd.DataFrame([],columns = ["Report_Type","Report_RSSD","Report_Date"])
    ReportCodes = pd.DataFrame([],columns = pd.Series(result_output2["ReportCode"][~result_output2["IndexInfo"].isin(["SectionInfo","HeaderInfo"])].unique()))
    ReportCodes.sort_index(axis=1, inplace=True)
    
    if verbose:
        print("Adding ReportCodes from Raw File and Binding to DataFrame")
    d2 = pd.concat([d1,ReportCodes], axis=1)

    #Get initial Data for First Rows
    print("Creating Identity Rows")    
    d3 = result_output2[["Report_Type","Report_RSSD","Report_Date"]][~result_output2["IndexInfo"].isin(["SectionInfo","HeaderInfo"])]
    if verbose:
        print("Dropping duplicates")
    d4 = d3.drop_duplicates(subset = ["Report_Type","Report_RSSD","Report_Date"])
    if verbose:
        print("Inserting Rows to Dataframe")
    d2[["Report_Type","Report_RSSD","Report_Date"]] = d4[["Report_Type","Report_RSSD","Report_Date"]]

    print("Transposing and strucutring data from raw object to new structure")    
    
    a = d2[['Report_Type', 'Report_RSSD', 'Report_Date']].reset_index(drop = True)
    d2 = d2.reset_index(drop=True)
    #for i in range(0,len(d2)):
    #    tmp_ReportType = d2.iloc[i,0]    
    #    tmp_ReportRSSD = d2.iloc[i,1].astype(str)
    #    tmp_ReportDate = d2.iloc[i,2].astype(str)
    #tmp_colnames = result_output2["ReportCode"][~result_output2["IndexInfo"].isin(["SectionInfo","HeaderInfo"])][result_output2["Report_Type"] == tmp_ReportType].unique()
    tmp_colnames = [column for column in d2.columns.values if column not in ['index','Report_Type', 'Report_RSSD', 'Report_Date']]
        #print("Populating Row: ", i, "of ", len(d2), "(",(i/len(d2)) * 100, "% Completed)")
        #elapsed_time = time.time() - start_time
        #sincelast = elapsed_time - sincelast
        #print("Since Last: ", sincelast, " Time Elapsed: ", elapsed_time)
    for y in tmp_colnames:
        tmp_ColNum = d2.columns.get_loc(y)
        elapsed_time = time.time() - start_time
        sincelast = elapsed_time - sincelast

        
        print("Since Last: ", sincelast, " Time Elapsed: ", elapsed_time)
        print("Populating Column: ", y, "of ", len(tmp_colnames), "(",(tmp_ColNum/len(tmp_colnames)) * 100, "% Completed)")
        print("Getting Values from Raw DataFrame for ReportCode")
        b = result_output2[['Report_Type', 'Report_RSSD', 'Report_Date','Amount']][result_output2["ReportCode"] == y].reset_index(drop = True)
        b.drop_duplicates(subset = ['Report_Type', 'Report_RSSD', 'Report_Date','Amount'])
        print("Merging on Identifiers to New Dataframe")
        d2[y] = a.merge(b,how = 'left', on = ['Report_Type', 'Report_RSSD', 'Report_Date'])['Amount']
        print("Cleaning Up Data (Two periods)")
        d2[y][d2[y].str.count("\.") > 1] = d2[y][d2[y].str.count("\.") > 1].str.replace("\.","",n = 1)
        print("Cleaning Up Data (AMOUNT)  ")
        d2[y][d2[y].str.match("Amount").fillna(False)] = np.nan
    print("Setting ReportDate to DateTime")
    d2["Report_Date"] = pd.to_datetime(d2["Report_Date"],format = "%Y%m%d")
    print("Cleaning Up Identifier Rows")
    d2["Report_RSSD"] = d2["Report_RSSD"].astype(str).str.strip()    
    return(d2)




    
#FFIEC_101
###File path variables
# File naming and renameing for input.
homepath = os.environ['HOME']
basepath = os.path.join(homepath,'icdm2018_research_BB/Stress_Test_Research/StressTest_Research/')
sourcefolder = os.path.join(basepath,"unsecured_pdf_complete")
os.listdir(sourcefolder)
ReportName_prefix = 'FFIEC101_'
ReportName_suffix = '.PDF.pdf'
paths = [ os.path.join(sourcefolder,fn) for fn in os.listdir(sourcefolder) if fn.startswith(ReportName_prefix) & fn.endswith(ReportName_suffix)]
########
len(paths)
del(result_ffiec101)
result_ffiec101 = report_parser_dataframer(reportsourcefolder = "/Users/phn1x/icdm2018_research_BB/Stress_Test_Research/StressTest_Research/unsecured_pdf_complete/", reportfilepath = paths, extension = ".PDF.pdf")
result_ffiec101.shape
#Output to CSV
result_ffiec101.to_csv(os.path.join(basepath,"ParsedFiles/ffiec101_out_2.csv"),sep = ",",encoding = "utf-8", index= False)




#FFIEC_102




###File path variables
# File naming and renameing for input.
homepath = os.environ['HOME']
basepath = os.path.join(homepath,'icdm2018_research_BB/Stress_Test_Research/StressTest_Research/')
sourcefolder = os.path.join(basepath,"unsecured_pdf_complete")
os.listdir(sourcefolder)
ReportName_prefix = 'FFIEC102_'
ReportName_suffix = '.PDF.pdf'
paths = [ os.path.join(sourcefolder,fn) for fn in os.listdir(sourcefolder) if fn.startswith(ReportName_prefix) & fn.endswith(ReportName_suffix)]
########
len(paths)
del(result_ffiec102)
result_ffiec102 = report_parser_dataframer(reportsourcefolder = "/Users/phn1x/icdm2018_research_BB/Stress_Test_Research/StressTest_Research/unsecured_pdf_complete/", reportfilepath = paths, extension = ".PDF.pdf")
result_ffiec102.shape

#result_ffiec102["Amount"].str.replace("__ _ ","")

#Output to CSV
result_ffiec102.to_csv(os.path.join(basepath,"ParsedFiles/ffiec102_out_1.csv"),sep = ",",encoding = "utf-8", index= False)

#Fix bugs with alignment on FFIEC102


#FRY15
###File path variables
# File naming and renameing for input.
homepath = os.environ['HOME']
basepath = os.path.join(homepath,'icdm2018_research_BB/Stress_Test_Research/StressTest_Research/')
sourcefolder = os.path.join(basepath,"unsecured_pdf_complete")
os.listdir(sourcefolder)
ReportName_prefix = 'FRY15'
ReportName_suffix = '.PDF.pdf'
paths = [ os.path.join(sourcefolder,fn) for fn in os.listdir(sourcefolder) if fn.startswith(ReportName_prefix) & fn.endswith(ReportName_suffix)]
########
len(paths)


#report = tabula.read_pdf(paths[0], pages = "all", guess = True, multiple_tables = True)
#report[0]
del(result_fry15)
result_fry15 = report_parser_dataframer(reportsourcefolder = "/Users/phn1x/icdm2018_research_BB/Stress_Test_Research/StressTest_Research/unsecured_pdf_complete/", reportfilepath = paths, extension = ".PDF.pdf")
result_fry15.shape




#Output to CSV
result_fry15.to_csv(os.path.join(basepath,"ParsedFiles/fry15_out_1.csv"),sep = ",",encoding = "utf-8", index= False)




##
result_ffiec101.shape
result_ffiec102.shape
result_fry15.shape


len(result_ffiec101["ReportCode"].unique())
len(result_ffiec102["ReportCode"].unique())
len(result_fry15["ReportCode"].unique())

result_ffiec101 = result_ffiec101.replace("nan",np.nan)

result_ffiec101[["ReportCode","Amount","Report_RSSD"]][result_ffiec101["ReportCode"].str.match("AAAB").fillna(False)]




ffiec_report = pd.concat([ffiec101,result_ffiec102,result_fry15])
#May need to add reseting of index
ffiec_report.reset_index(inplace=True,drop = True)

ffiec_report.to_csv(os.path.join(basepath,"ParsedFiles/ffiec_result.csv"),sep = ",",encoding = "utf-8", index= False)






#All Files
###File path variables
# File naming and renameing for input.
homepath = os.environ['HOME']
basepath = os.path.join(homepath,'icdm2018_research_BB/Stress_Test_Research/StressTest_Research/')
sourcefolder = os.path.join(basepath,"unsecured_pdf_complete")
os.listdir(sourcefolder)
ReportName_prefix = ''
ReportName_suffix = '.PDF.pdf'
paths = [ os.path.join(sourcefolder,fn) for fn in os.listdir(sourcefolder) if fn.startswith(ReportName_prefix) & fn.endswith(ReportName_suffix)]
########
len(paths)


#report = tabula.read_pdf(paths[0], pages = "all", guess = True, multiple_tables = True)
#report[0]
#del(result_output)
result_output = report_parser_dataframer(reportsourcefolder = "/Users/phn1x/icdm2018_research_BB/Stress_Test_Research/StressTest_Research/unsecured_pdf_complete/", reportfilepath = paths, extension = ".PDF.pdf")
result_output.shape



#Output to CSV
result_output.to_csv(os.path.join(basepath,"ParsedFiles/result_output_2.csv"),sep = ",",encoding = "utf-8", index= False)
#Backup
result_output2 = pd.read_csv(os.path.join(basepath,"ParsedFiles/result_output_2.csv"))




test1 = result_output2["ReportCode"][result_output2["Description"].isnull()][~result_output2["IndexInfo"].isin(["SectionInfo","HeaderInfo"])]




#Removes Ending Nans
print('Replacing Ending Nans')
updatedReportsCodes = result_output2["Description"][result_output2["Description"].astype(str).str.endswith("nan")][~result_output2["IndexInfo"].isin(["SectionInfo","HeaderInfo"])].str.replace("nan","").str.strip()    
print('Generating Index for Update')
index1 = result_output2["Description"][result_output2["Description"].astype(str).str.endswith("nan")][~result_output2["IndexInfo"].isin(["SectionInfo","HeaderInfo"])].index
print('Updating with Placeholders')
result_output2.loc[index1,'Description'] = updatedReportsCodes


#Missing Description Names
test2 = test1.unique()


for i in test2:
    print(i)
    test3 = result_output2["Description"][result_output2["ReportCode"].astype(str).str.match(i)].unique()
    
    
    if len(test3) > 1:
        print(test3)
        #Drop Null
        test4 = [x for x in test3 if str(x) != 'nan']    
        print(test4)
        test5 = test4[-1]
        result_output2["Description"][result_output2["ReportCode"].astype(str).str.match(i)] = test5
        print(result_output2["Description"][result_output2["ReportCode"].astype(str).str.match(i)].unique())
        

#result_output2.to_csv(os.path.join(basepath,"ParsedFiles/result_output2.csv"),sep = ",",encoding = "utf-8", index= False)



print(result_output2["Description"][result_output2["ReportCode"].astype(str).str.match("Y832")].str.strip().unique())

len(result_output2["Description"].str.strip().unique())
    


ReportCodes = result_output2["ReportCode"][~result_output2["IndexInfo"].isin(["SectionInfo","HeaderInfo"])]


#Provide Standard Description for same report codes

print("Fill Standard Descriptions for same ReportCodes")
test1 = result_output2["ReportCode"][~result_output2["IndexInfo"].isin(["SectionInfo","HeaderInfo"])]

#Missing Description Names
test2 = test1.unique()    
for i in test2:
    test3 = result_output2["Description"][result_output2["ReportCode"].astype(str).str.match(i)].unique()
            
    if len(test3) > 1:
        print(i)
        print(test3)
        #Drop Null
        test4 = [x for x in test3 if str(x) != 'nan']    
        print(test4)
        test5 = test4[-1] #Give last description in array    
        result_output2["Description"][result_output2["ReportCode"].astype(str).str.match(i)] = test5
        print(result_output2["Description"][result_output2["ReportCode"].astype(str).str.match(i)].unique())
    



result_output2.to_csv(os.path.join(basepath,"ParsedFiles/result_output3.csv"),sep = ",",encoding = "utf-8", index= False)

result_output2 = pd.read_csv(os.path.join(basepath,"ParsedFiles/result_output3.csv"))

index1 = result_output2["Description"][result_output2["Description"].astype(str).str.endswith(". . . . . .")].index

result_output2.loc[index1,"Description"]= result_output2.loc[index1,"Description"].str.replace("\. \.","").str.strip()


#Remove 1=Yes
print('Generating Placeholder ReportCodes for 1=Yes')
updatedReportsCodes = result_output2["Report_Type"][result_output2["ReportCode"].astype(str).str.match("1=Yes")] + "_" + result_output2["IndexInfo"][result_output2["ReportCode"].astype(str).str.match("1=Yes")].str.strip(".").str.strip()    
print('Generating Index for Update')
index1 = result_output2["ReportCode"][result_output2["ReportCode"].astype(str).str.match("1=Yes")].index
print('Updating with Placeholders')
result_output2.loc[index1,'ReportCode'] = updatedReportsCodes




result_output2.to_csv(os.path.join(basepath,"ParsedFiles/result_output4.csv"),sep = ",",encoding = "utf-8", index= False)

###Transpose and rearrange


result_output3 = reporttransposerarrangerbycolumn(result_output2, verbose = False)


ReportCodes = pd.DataFrame(result_output2["ReportCode"][~result_output2["IndexInfo"].isin(["SectionInfo","HeaderInfo"])].unique(), columns = ["ReportCode"])

FFIEC101_Transformed = result_output3.loc[result_output3["Report_Type"] == "FFIEC101"].dropna(axis="columns", how = "all")
FFIEC102_Transformed = result_output3.loc[result_output3["Report_Type"] == "FFIEC102"].dropna(axis="columns", how = "all")
FRY15_Transformed = result_output3.loc[result_output3["Report_Type"] == "FRY15"].dropna(axis="columns", how = "all")
DataDictionary = result_output2[["ReportCode","Description","IndexInfo"]].drop_duplicates(subset = ["ReportCode","Description","IndexInfo"])


FFIEC101_Transformed.to_csv(os.path.join(basepath,"ParsedFiles/FFIEC101_Transformed.csv"),sep = ",",encoding = "utf-8", index= False)
FFIEC102_Transformed.to_csv(os.path.join(basepath,"ParsedFiles/FFIEC102_Transformed.csv"),sep = ",",encoding = "utf-8", index= False)
FRY15_Transformed.to_csv(os.path.join(basepath,"ParsedFiles/FRY15_Transformed.csv"),sep = ",",encoding = "utf-8", index= False)
DataDictionary.to_csv(os.path.join(basepath,"ParsedFiles/DataDictionary.csv"),sep = ",",encoding = "utf-8", index= False)


FFIEC102_Transformed['S363'][FFIEC102_Transformed['S363'].str.count("\.") > 1].str.replace("\.","",n = 1)

FFIEC102_Transformed['MRRR'][FFIEC102_Transformed['MRRR'].str.match("Amount").fillna(False)]


pd.to_datetime(FFIEC102_Transformed['Report_Date'],format = "%Y%m%d")

