import pandas as pd
import numpy as np
import os

class StressTestData():
    def __init__(self,varpath_dict_args = {
                    "Z_macro_dir" : "Z_macro/",
                    "Z_micro_dir" : "Z_micro/",
                    "BankPerf_dir" : "BankPerf/",
                    "Data_dir" : "/Users/phn1x/Google Drive/Spring 2019/LossProjection_Research/Data"
                    }):
        print("Initialize Objects")
        print("Set Data Working Directories")

        for k,v in varpath_dict_args.items():
            print(k,v)
            if k == "Data_dir":
                exec('''self.''' + k + ''' = "''' + v + '''"''')
                exec("print(self." + k + ")")
                exec("os.chdir(self." + k + ")")
            else:
                exec('''self.'''+ k + ''' = "''' + os.path.join(varpath_dict_args["Data_dir"],v) + '''"''')
                exec("print(self."+ k + ")")

        print("Initalizing Return Dict Object")
        self.final_df = {}

    def file_dict_read(self, rootdir,  filetype = ".csv"):
        print("Initializing Raw Data Frame Dict")
        tmp_dict = {}
        print("Searching path:", rootdir)
        for dirName, subdirList, fileList in os.walk(rootdir):
            print('Found directory: %s' % dirName)
            for fname in fileList:
                if fname.endswith(filetype):
                    print("Reading File and Adding to Dataframe Dictionary")
                    print(os.path.join(dirName, fname))
                    print('\t%s' % fname)
                    print(fname.split(filetype)[0])
                    exec('''tmp_dict[fname.split(filetype)[0]] = pd.read_csv("''' + os.path.join(dirName, fname) + '''")''')
        return(tmp_dict)


    def Z_macro_process(self, filetype = ".csv"):
        Z_macro_raw_data_dict = self.file_dict_read(self.Z_macro_dir,filetype = ".csv")

        for k in Z_macro_raw_data_dict.keys():
            print(k)
            Z_macro_raw_data_dict[k]["pdDate"] = pd.to_datetime(Z_macro_raw_data_dict[k]["Date"].apply(lambda x: "-".join(x.split(" "))).str.replace(r'(Q\d) (\d+)', r'\2-\1'), errors = 'coerce')
            print("Dropping Original Date Column")
            Z_macro_raw_data_dict[k] = Z_macro_raw_data_dict[k].drop("Date", axis = 1)

        return(Z_macro_raw_data_dict)

    def Z_micro_process(self, filetype = ".csv"):
        Z_micro_raw_data_dict = self.file_dict_read(self.Z_micro_dir, filetype=".csv")

        Z_micro_raw_data_dict_final = {}
        for key in Z_micro_raw_data_dict.keys():
            tmp_df = Z_micro_raw_data_dict[key]

            if key.startswith("GFD_"):
                print("GFD data source:", key)
                tmp_subset = ["Date", "Ticker", "Close"]
                tmp_df = tmp_df[tmp_subset]

                print(tmp_df.shape)
                print("Formatting Date")
                tmp_df["Date"] = pd.to_datetime(tmp_df.Date, errors='coerce')

                print("Subsetting for 1970-2020")
                tmp_df = tmp_df.loc[(tmp_df["Date"] >= "1969-11-01") & (tmp_df["Date"] <= "2020-01-01")]

                print(tmp_df.shape)
                print("Appending to Final Dictionary")
                Z_micro_raw_data_dict_final[key] = tmp_df.sort_values("Date")

            if key.startswith("WRDS_"):
                print("WRDS data source:", key)

                if key == "WRDS_SnP500_Returns":
                    tmp_subset = ["caldt", "spindx", "sprtrn"]
                    tmp_df = tmp_df[tmp_subset]
                    tmp_df["caldt"] = pd.to_datetime(tmp_df.caldt, format="%Y%M%d", errors='coerce')
                    tmp_df = tmp_df.rename({"caldt": "Date"}, axis=1)
                    print("Subsetting for 1970-2020")
                    tmp_df = tmp_df.loc[(tmp_df["Date"] >= "1969-11-01") & (tmp_df["Date"] <= "2020-01-01")]

                #test = Z_micro["WRDS_government_bonds"]
                    print(tmp_df.shape)
                    print("Appending to Final Dictionary")
                    Z_micro_raw_data_dict_final[key] = tmp_df.sort_values("Date")
            Z_micro_raw_data_dict_final[key] = tmp_df
            #Once all scenarios are accounted for.
            #print(tmp_df.shape)
            #print("Appending to Final Dictionary")
            #Z_micro_raw_data_dict_final[key] = tmp_df.sort_values("Date")
        return (Z_micro_raw_data_dict_final)

    def X_Y_bankingchar_perf_process(self,filetype = ".csv",
                                     params_dict = {"Calc_df" : "NCO_PPNR_Loan_Calcs",
                                                    "BankPerf_raw" : "WRDS_Covas_BankPerf",
                                                    "MergerInfo " : "merger_info_frb"}
                                     ):
        BankCharPerf_raw_data_dict = self.file_dict_read(self.BankPerf_dir, filetype=".csv")
        print(BankCharPerf_raw_data_dict.keys())
        print("Fixing Column Names with Errors")

        print("Preparing Calculated Fields")
        print("Fixing Errors")
        print("Fixing BHCK892 to BHCKC892")
        print("Creating VARLIST Column")
        BankCharPerf_raw_data_dict[params_dict["Calc_df"]]["VarList"] = BankCharPerf_raw_data_dict[params_dict["Calc_df"]]["Report: FR Y-9C"].astype(str).apply(lambda x : [i.strip() for i in  x.replace("BHCK892","BHCKC892").replace("+",",").replace("-",",").replace("(","").replace(")","").replace(". . .", "").replace(",  ,",",").split(",")])

        print("Additional Clean up for Varlist")
        BankCharPerf_raw_data_dict[params_dict["Calc_df"]]["VarList"] =  BankCharPerf_raw_data_dict[params_dict["Calc_df"]]["VarList"].apply(lambda x: [i for i in x if i not in ["",np.nan,"nan"]])

        print("Checking for Column name Discrepancy")
        miss_col = self.check_missing_BHCCodes(BankCharPerf_raw_data_dict,params_dict)

        print("Creating Derived Columns")
        BankCharPerf_raw_data_dict["BankPerf_Calculated"] = self.BHC_loan_nco_ppnr_create(BankCharPerf_raw_data_dict,params_dict)


        return(BankCharPerf_raw_data_dict)
    def check_missing_BHCCodes(self, BankPerf, params_dict):
        print("Creating List of all Variables from Regulatory and BankPerf dataframes.")
        varlist_tmp = []
        for i in range(BankPerf[params_dict["Calc_df"]]["VarList"].shape[0]):
            # print(BankPerf[params_dict["Calc_df"]]["VarList"][i])
            varlist_tmp = varlist_tmp + BankPerf[params_dict["Calc_df"]]["VarList"][i]
            varlist_tmp = [x for x in varlist_tmp if x not in ['nan', ""]]

        bankperf = list(BankPerf[params_dict["BankPerf_raw"]].columns)

        mismatch_result = list(set(varlist_tmp) - set(bankperf))

        if mismatch_result.__len__() > 0:
            print("Columns not Matching from VarList to BankPerf Data:", mismatch_result)
        else:
            print("All Columns Exist in dataframes.")
        return (mismatch_result)
    def BHC_loan_nco_ppnr_create(self, BankPerf, params_dict):
        print("Initialize Result DF")
        loan_nco_ppnr_df = pd.DataFrame()
        print("Initialize Calculation Column")
        calc_tmp = BankPerf[params_dict["Calc_df"]]["Report: FR Y-9C"].astype(str).apply(
            lambda x: x.replace("BHCK892", "BHCKC892").replace("(", "").replace(")", "").replace(". . .", "").replace(
                "nan", "").replace("+  +", "+").strip())
        print("Create String Column with usage calculations")
        BankPerf[params_dict["Calc_df"]]["Calc_varstr"] = calc_tmp
        print("Generate Calculated Columns")
        for i in range(0, BankPerf[params_dict["Calc_df"]].shape[0]):
            tmp_subset = BankPerf[params_dict["Calc_df"]].loc[i, "VarList"]
            tmp_df = BankPerf[params_dict["BankPerf_raw"]][tmp_subset]
            tmp_varname = BankPerf[params_dict["Calc_df"]].loc[i, "Variable"]
            tmp_varcat = BankPerf[params_dict["Calc_df"]].loc[i, "Variable Category"]
            tmp_calc_str = BankPerf[params_dict["Calc_df"]].loc[i, "Calc_varstr"]
            print("Get Column Vectors")
            for col in tmp_subset:
                print(col)
                exec(col + ''' = tmp_df["''' + col + '''"]''')
                print(tmp_df[col].shape)

            print("Calculate Derived Filed for:", tmp_varname, tmp_varcat, tmp_calc_str)
            if tmp_calc_str:
                tmp_result_obj = eval(tmp_calc_str)
            else:
                tmp_result_obj = np.nan
            print("Add Derived Field to Result DataFrame")
            tmp_col_name = tmp_varcat + ":" + tmp_varname
            loan_nco_ppnr_df[tmp_col_name] = tmp_result_obj
            print(loan_nco_ppnr_df[tmp_col_name].shape)

        print("Combine Calculated Columns to Original Dataframe")
        identifier_columns = ["RSSD9001","RSSD9999","RSSD9010","RSSD9017","RSSD9161","BHSP8519","RSSD9005","RSSD9007","RSSD9008","RSSD9010","RSSD9016","RSSD9045","RSSD9052","RSSD9053","RSSD9101","RSSD9130","RSSD9200","RSSD9950"]
        BankPerf_result = pd.concat([BankPerf[params_dict["BankPerf_raw"]][identifier_columns], loan_nco_ppnr_df], axis=1)
        #TODO: Some issue here with data structure, does not allow to view the object.
        #print("Resetting Index")
        #BankPerf_result = BankPerf_result.reset_index(drop = True)
        print("Rename Identifier Columns")
        BankPerf_result = BankPerf_result.rename({"RSSD9001":"RSSD_ID","RSSD9999":"ReportingDate","RSSD9161":"CUSIP"}, axis = 1)
        print("Format ReportingDate column")
        BankPerf_result["ReportingDate"] = pd.to_datetime(BankPerf_result["ReportingDate"], format = "%Y%m%d")
        #BankPerf_result["DateofOpening"] = pd.to_datetime(BankPerf_result["DateofOpening"], format="%Y%m%d")
        #BankPerf_result["FinalDay"] = pd.to_datetime(BankPerf_result["FinalDay"], format="%Y%m%d")
        #BankPerf_result["DateStart"] = pd.to_datetime(BankPerf_result["DateStart"], format="%Y%m%d")
        #BankPerf_result["DateEnd"] = pd.to_datetime(BankPerf_result["DateEnd"], format="%Y%m%d")

        #TODO: Incorporate Merger Information.

        return (BankPerf_result)

    def bankperf_rates_ratios(self, BankPerf):
        print("Net Charge-Off rates by type of loan calculations")
        for nco, loan in zip(
                list(BankPerf.columns[pd.Series(BankPerf.columns).str.startswith("Net charge-offs by type of loan:")]),
                list(BankPerf.columns[pd.Series(BankPerf.columns).str.startswith("Loans categories:")])):
            tmp_str = nco.replace("Net charge-offs by type of loan:", "ncoR:")
            print(tmp_str)
            BankPerf[tmp_str] = 400 * ((BankPerf[nco].fillna(0)) / (BankPerf[loan].fillna(0)))

        print(BankPerf[BankPerf.columns[pd.Series(BankPerf.columns).str.startswith("ncoR:")]].describe().transpose())

        for ppnr_comp in list(BankPerf.columns[pd.Series(BankPerf.columns).str.startswith(
                "Components of pre-provision net revenue:")]):
            tmp_str = ppnr_comp.replace("Components of pre-provision net revenue:", "ppnrRatio:")
            print(tmp_str)
            BankPerf[tmp_str] = 400 * (BankPerf[ppnr_comp].fillna(0)) / (BankPerf['Other items:Consolidated assets'])
        print(
            BankPerf[BankPerf.columns[pd.Series(BankPerf.columns).str.startswith("ppnrRatio:")]].describe().transpose())

        # Balance sheet composition indicators
        # Requires interest earning assets calculations.
        return(BankPerf)


#Entry Point
init_ST = StressTestData()

#Z_Macro complate and distributions match to Malik 2018
Z_macro = init_ST.Z_macro_process()
#Should pring out Summary Statistics

Z_micro = init_ST.Z_micro_process()
#Need Additional Subsetting and formatting logic for the WRDS data
#Also need additional interest rate swaps data.
#May need to calcualte the returns for some of the indices and prices.
#Should output Summary Statistics



#May need to subset BankPerf to get Xi and Yi

BankPerf = init_ST.X_Y_bankingchar_perf_process()


BankPerf.keys()

merger_df = BankPerf["merger_info_frb"]


#BankPerf_3 = BankPerf

#Must find some way to reduce the # of banks in dataset to closer to 1000.
#Can base it only on bank RSSIDs as BHCs from FFIEC topBHCs.
#Based on top Consolidated Total assets

test = BankPerf_Consecutive_Merger_Aggregation_process(BankPerf)
#Incorporate merger info

def BankPerf_Consecutive_Merger_Aggregation_process(BankPerf,concecutiveqtrs = 8, merger_df_name = "merger_info_frb", BankPerf_Calc_df_name = "BankPerf_Calculated"):
    if merger_df_name in BankPerf.keys():
        print("Getting Merger Info from Dictionary")
        merger_info_df = BankPerf[merger_df_name]
        merger_info_df =  merger_info_df[["MERGE_DT","NON_ID","SURV_ID"]]
        merger_info_df["MERGE_DT"] = pd.to_datetime(merger_info_df["MERGE_DT"], format = "%Y%M%d")
        merger_info_df["NON_ID"] = merger_info_df["NON_ID"].astype(int)
    else:
        print("Merger Information not found in dictionary")

    if BankPerf_Calc_df_name in BankPerf.keys():
        print("Getting BankPerf Calculated rows")
        BankPerf_Calc_df = BankPerf[BankPerf_Calc_df_name]
    else:
        print("BankPerf Calculated DF not found")


    print("Filter Banks for those that are above::",concecutiveqtrs,"consecutive quarters.")

    BankPerf_Calc_df = BankPerf_Calc_df.sort_values(['RSSD_ID', 'ReportingDate'])
    BankPerf_Calc_df["ConsecutivePeriods"] = BankPerf_Calc_df.sort_values(['RSSD_ID', 'ReportingDate']).groupby(
        ['RSSD_ID']).cumcount() + 1
    BankPerf_Calc_df_gb_tmp = BankPerf_Calc_df.groupby('RSSD_ID')
    BankPerf_Calc_df_gb_tmp = BankPerf_Calc_df_gb_tmp.agg({"ConsecutivePeriods": ['min', 'max', 'count']})
    RSSD_ID_consecutive = list(BankPerf_Calc_df_gb_tmp[BankPerf_Calc_df_gb_tmp["ConsecutivePeriods"]["max"] >= concecutiveqtrs].index)
    BankPerf_Calc_df = BankPerf_Calc_df[BankPerf_Calc_df["RSSD_ID"].isin(RSSD_ID_consecutive)]


    print("Loop/merge through merger information and update RSSD of non survivor with survivor ID.")
    orig_columns = list(BankPerf_Calc_df.columns)
    BankPerf_Merger_df = BankPerf_Calc_df.merge(merger_info_df, left_on = "RSSD_ID", right_on = "NON_ID", how = "left")
    print("Found",BankPerf_Merger_df["RSSD_ID"][pd.notnull(BankPerf_Merger_df["NON_ID"])].unique().__len__(),' Non surviving RSSD_IDs')
    print("Updating Non-Surviving RSSD_IDs with Surviving IDs")
    BankPerf_Merger_df["RSSD_ID"][pd.notnull(BankPerf_Merger_df["NON_ID"])] = BankPerf_Merger_df["SURV_ID"][pd.notnull(BankPerf_Merger_df["NON_ID"])]
    #BankPerf_Merger_df = BankPerf_Calc_df.merge(merger_info_df, left_on = "RSSD_ID", right_on = "NON_ID", how = "left")


    print("Aggregate all the updated RSSD_ID columns")
    BankPerf_Merger_df_gbsum = BankPerf_Merger_df.groupby(["RSSD_ID","ReportingDate"]).sum()
    BankPerf_Merger_df_gbsum = BankPerf_Merger_df_gbsum.reset_index()
#    gb_agg = gb.agg({
#        'Loans categories:Commercial & industrial': np.sum,
#        'Loans categories:Construction & land development': np.sum,
#        'Loans categories:Multifamily real estate': np.sum,
#        'Loans categories:Nonfarm nonresidential CRE': np.sum,
#        'Loans categories:Home equity lines of credit': np.sum,
#        'Loans categories:Residential real estate (excl. HELOCs)': np.sum,
#        'Loans categories:Credit card': np.sum,
#        'Loans categories:Consumer (excl. credit card)': np.sum
#    })
    print(BankPerf_Merger_df_gbsum.describe().transpose())
    return(BankPerf_Merger_df_gbsum)



def bankperf_rates_ratios(BankPerf):
    #Need to handle NAN rows and outliers.
    print("Replacing nans, infs, and -infs in dataframe")
    #BankPerf = BankPerf[~BankPerf.isin([np.nan, np.inf, -np.inf]).any(1)]

    print("Loans categories Descriptive Statistics")
    #BankPerf = BankPerf[~BankPerf[BankPerf.columns[pd.Series(BankPerf.columns).str.startswith("Loans categories:")]].isin([np.nan, np.inf, -np.inf]).any(1)]
    #print(BankPerf[BankPerf.columns[pd.Series(BankPerf.columns).str.startswith("Loans categories:")]].describe().transpose())

    print("Net Charge-Off rates by type of loan calculations")
    for nco, loan in zip(list(BankPerf.columns[pd.Series(BankPerf.columns).str.startswith("Net charge-offs by type of loan:")]),list(BankPerf.columns[pd.Series(BankPerf.columns).str.startswith("Loans categories:")])):
        tmp_str = nco.replace("Net charge-offs by type of loan:","ncoR:")
        print(tmp_str)
        BankPerf[tmp_str] = 400 * ((BankPerf[nco]) / (BankPerf[loan]))
        #BankPerf = pd.concat([BankPerf,BankPerf_tmp[tmp_str]], ignore_index=True, axis = 1)
    print(BankPerf[BankPerf.columns[pd.Series(BankPerf.columns).str.startswith("ncoR:")]].describe().transpose())

    print("PPNR Ratios calculations")
    for ppnr_comp in list(BankPerf.columns[pd.Series(BankPerf.columns).str.startswith("Components of pre-provision net revenue:")]):
        tmp_str = ppnr_comp.replace("Components of pre-provision net revenue:", "ppnrRatio:")
        print(tmp_str)
        BankPerf[tmp_str] = 400 * (BankPerf[ppnr_comp])/(BankPerf['Other items:Consolidated assets'])

    print(BankPerf[BankPerf.columns[pd.Series(BankPerf.columns).str.startswith("ppnrRatio:")]].describe().transpose())

    #Balance sheet composition indicators
    #Requires interest earning assets calculations.

    return(BankPerf)









#Filter BHC Indicator Yes,

#Filter Domestic Indicator?
#Date Range
BankPerf_2 = BankPerf[(BankPerf["ReportingDate"] >= "1989-12-30") & (BankPerf["ReportingDate"] <= "2017-12-31")] #& (BankPerf["RSSD9045"] == 1) & (BankPerf["RSSD9101"] == 'Y')]




#May need to consider merger inforation and suvivorship information first.
#Count consecutive dates
BankPerf_2 = BankPerf_2.sort_values(['RSSD_ID','ReportingDate'])
BankPerf_2["ConsecutivePeriods"] = BankPerf_2.sort_values(['RSSD_ID','ReportingDate']).groupby(['RSSD_ID']).cumcount() + 1

gb = BankPerf_2.groupby('RSSD_ID')
gb = gb.agg({"ConsecutivePeriods":['min','max','count']})
RSSD_ID_consecutive = list(gb[gb["ConsecutivePeriods"]["max"] >= 8].index)

BankPerf_2['Other items:Consolidated assets'].min()
BankPerf_2['Other items:Consolidated assets'].max()


BankPerf_2 = BankPerf_2[BankPerf_2["RSSD_ID"].isin(RSSD_ID_consecutive)]


BankPerf_2[['Loans categories:Commercial & industrial',
           'Loans categories:Construction & land development',
           'Loans categories:Multifamily real estate',
           'Loans categories:Nonfarm nonresidential CRE',
           'Loans categories:Home equity lines of credit',
           'Loans categories:Residential real estate (excl. HELOCs)',
           'Loans categories:Credit card',
           'Loans categories:Consumer (excl. credit card)']].describe().transpose()




BankPerf_2.RSSD_ID.unique().__len__()


#Filter out for bhc indicator.
#Filter out for above 50 bln at end of sample period.
#To match Malik we need around 1000 banks.
#Filter to banks that have 8 consecutive quarters













identifier_cols = ["RSSD_ID","ReportingDate","CUSIP","RSSD9045"]
Xi_Yi_cols = ["Loans categories:", "Net charge-offs by type of loan","Components of pre-provision net revenue:","Other items:"]

tmp_dict = {}
for colcat in Xi_Yi_cols:
    tmp_col_names = list(test.columns[pd.Series(test.columns).str.startswith(colcat)])
    tmp_col_names = identifier_cols + tmp_col_names

    print("Subset DataFrame and append to Dict")
    exec('''tmp_dict["'''+colcat.replace(":","")+'''"] = test[tmp_col_names]''')













#Find best way to subset the data for Yi and Xi
#A Wrapper function can handle these additional tasks to produce a Dict with the Dataframes necessary




#VAE-CVAE=MNIST
#import utils, models, train
