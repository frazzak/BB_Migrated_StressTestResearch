import pandas as pd
import numpy as np
import os
import gc
from functools import reduce
from sqlalchemy import create_engine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import inspect, re
import matplotlib.pyplot as plt
import seaborn as sns
'''
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
   '''



class StressTestData():
    def __init__(self,varpath_dict_args = {
                    "Z_macro_dir" : "Z_macro/",
                    "Z_micro_dir" : "Z_micro/",
                    "BankPerf_dir" : "BankPerf/",
                    "SBidx_dir":"ShadowBanking_Proxies/",
                    "Sectoridx_dir":"Sector_Indices/",
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

    def file_dict_read(self, rootdir,  filetype = ".csv", skip_prefix = " "):
        print("Initializing Raw Data Frame Dict")
        tmp_dict = {}
        print("Searching path:", rootdir)
        for dirName, subdirList, fileList in os.walk(rootdir):
            print('Found directory: %s' % dirName)
            for fname in fileList:
                if not fname.startswith(skip_prefix):
                    if fname.endswith(filetype):
                        print("Reading File and Adding to Dataframe Dictionary")
                        print(os.path.join(dirName, fname))
                        print('\t%s' % fname)
                        print(fname.split(filetype)[0])
                        exec('''tmp_dict[fname.split(filetype)[0]] = pd.read_csv("''' + os.path.join(dirName, fname) + '''")''')
        return(tmp_dict)

    def sectoridx_process(self, Sectoridx_dir = None , filetype = ".csv"):
        if Sectoridx_dir is None:
            Sectoridx_dir = self.Sectoridx_dir

        #Sectoridx_dir = "Sector_Indices/"

        sectoridx_dict = self.file_dict_read(Sectoridx_dir, filetype=filetype)
        #sectoridx_dict = file_dict_read(Sectoridx_dir, filetype=filetype)

        dfs = list()
        for keyname in sectoridx_dict.keys():
            print("Processing:", keyname)
            tmp_df = sectoridx_dict[keyname]
            tmp_df["datadate"] = pd.to_datetime(tmp_df["datadate"], format="%Y%m%d").dt.date
            tmp_df = tmp_df[["datadate", "tic", "prccm"]]
            tmp_df = tmp_df.pivot_table(index="datadate", columns="tic", values="prccm")
            tmp_df = tmp_df.reset_index()
            print(tmp_df.describe().transpose())
            dfs.append(tmp_df)



        print("Combine List with Left Merge")
        final_raw_df = reduce(lambda left, right: pd.merge(left, right, on="datadate", how="left"), dfs)
        final_raw_df["datadate"]
        final_raw_df = final_raw_df.rename({"datadate": "Date"}, axis=1)
        print(final_raw_df.describe().transpose())
        sectoridx_dict["sectoridx"] = final_raw_df
        return (sectoridx_dict)


    def SBidx_process(self, filetype = ".csv"):
        SBidx_dict = self.file_dict_read(self.SBidx_dir,filetype = ".csv")
        dfs = list()
        for keyname in SBidx_dict.keys():
            print(keyname)
            tmp_df = SBidx_dict[keyname]
            tmp_df["DATE"] = pd.to_datetime(tmp_df["DATE"]).dt.date
            dfs.append(tmp_df)

        print("Combine List with Left Merge")
        final_raw_df = reduce(lambda left, right: pd.merge(left, right, on="DATE", how="left"), dfs)
        final_raw_df = final_raw_df.rename({"DATE":"Date"}, axis = 1)
        print(final_raw_df.describe().transpose())
        SBidx_dict["SB_idx_prox"] = final_raw_df
        return(SBidx_dict)

    def Z_macro_process(self, filetype = ".csv"):
        Z_macro_raw_data_dict = self.file_dict_read(self.Z_macro_dir,filetype = ".csv")
        dfs = list()
        for k in Z_macro_raw_data_dict.keys():
            print(k)
            Z_macro_raw_data_dict[k]["Date"] = pd.to_datetime(Z_macro_raw_data_dict[k]["Date"].apply(lambda x: "-".join(x.split(" "))).str.replace(r'(Q\d) (\d+)', r'\2-\1'), errors = 'coerce')
            #print("Dropping Original Date Column")
            #Z_macro_raw_data_dict[k] = Z_macro_raw_data_dict[k].drop("Date", axis = 1)
            print(Z_macro_raw_data_dict[k].describe().transpose())
        for keyname in [v for v in Z_macro_raw_data_dict.keys() if v.startswith("Historic_")]:
            print(keyname)
            tmp_df = Z_macro_raw_data_dict[keyname]
            tmp_df["Date"] = pd.to_datetime(tmp_df["Date"]).dt.date
            dfs.append(tmp_df)

        print("Combine List with Left Merge")
        final_raw_df = reduce(lambda left, right: pd.merge(left, right, on="Date", how="left"), dfs)
        print(final_raw_df.describe().transpose())
        Z_macro_raw_data_dict["Z_macro"] = final_raw_df
        return(Z_macro_raw_data_dict)

    def Z_micro_process(self, filetype = ".csv"):
        Z_micro_raw_data_dict = self.file_dict_read(self.Z_micro_dir, filetype=".csv")


        dfs = list()
        for keyname in [v for v in Z_micro_raw_data_dict.keys() if v.startswith("GFD_")]:
            print(keyname)
            tmp_df = Z_micro_raw_data_dict[keyname]
            tmp_df["Date"] = pd.to_datetime(tmp_df["Date"]).dt.date
            dfs.append(tmp_df[["Date", "Close"]].rename({"Close": keyname}, axis=1))

        for keyname in [v for v in Z_micro_raw_data_dict.keys() if v.startswith("WRDS_")]:
            # keyname = "WRDS_SnP500_Returns"
            print(keyname)
            if "caldt" in Z_micro_raw_data_dict[keyname].keys():
                print("Filter from 1970")
                tmp_df = Z_micro_raw_data_dict[keyname][Z_micro_raw_data_dict[keyname]["caldt"] > 19691231]
                tmp_df["caldt"] = pd.to_datetime(tmp_df["caldt"], format="%Y%m%d").dt.date
                dfs.append(tmp_df.rename({"caldt": "Date"}, axis=1))

            if "date" in Z_micro_raw_data_dict[keyname].keys():
                print("Filter from 1970")
                tmp_df = Z_micro_raw_data_dict[keyname][Z_micro_raw_data_dict[keyname]["date"] > 19691231]
                tmp_df["date"] = pd.to_datetime(tmp_df["date"], format="%Y%m%d").dt.date
                dfs.append(tmp_df.rename({"date": "Date"}, axis=1))

        print("Combine List with Left Merge")
        final_raw_df = reduce(lambda left, right: pd.merge(left, right, on="Date", how="left"), dfs)
        print(final_raw_df.describe().transpose())

        Z_micro_raw_data_dict["Z_Micro"] = final_raw_df
        return (Z_micro_raw_data_dict)

    def X_Y_bankingchar_perf_process(self, filetype=".csv",
                                     params_dict={"Calc_df": "NCO_PPNR_Loan_Calcs",
                                                  "BankPerf_raw": "WRDS_Covas_BankPerf",
                                                  "MergerInfo": "merger_info_frb",
                                                  "CallReport_prefix": "WRDS_Covas_BankPerf_CallReport"},

                                     replace_dict={"BankPerf_FR9YC_varlist": {
                                         "df_keyname": "WRDS_Covas_BankPerf",
                                         "calcol": "Report: FR Y-9C",
                                         "BHCK892": "BHCKC892",
                                         "+": ",",
                                         "-": ",",
                                         "(": "",
                                         ")": "",
                                         ". . .": "",
                                         ",  ,": ",",
                                         "+  +": "+",
                                         "-  -": "-"},
                                         "BankPerf_FFIEC_varlist": {
                                             "df_keyname": "WRDS_Covas_CallReport_Combined",
                                             "calcol": "Report: FFIEC 031/041",
                                             "RCFD560": "RCFDB560",
                                             "RIADK129": "RIAD4639",
                                             "RIADK205": "RIAD4657",
                                             "RIADK133": "RIAD4609",
                                             "RIADK206": "RIAD4667",
                                             "+": ",",
                                             "-": ",",
                                             "(": "",
                                             ")": "",
                                             ". . .": "",
                                             ",  ,": ",",
                                             "+  +": "+",
                                             "-  -": "-"
                                         },


                                     },
                                     keyname=None, groupfunction=np.mean, groupby=["RSSD_ID", "ReportingDate"],
                                     groupagg_col='Other items:Consolidated assets',
                                     RSSD_DateParam=["1989-12-31", "2018-01-01"],
                                     ReportingDateParam=["1989-12-31", "2017-12-31"], RSSDList_len=1000, dropdup=False,
                                     replace_nan=False, merge = True, Consqtr = True, combinecol = True, reducedf = True, skip_prefix = None, Y_calc = True, RSSD_Subset = True
                                     , calc_CR_hist = True,get_prev_time_cols = True
                                     , concecutiveqtrs = 8
                                     ):

        print("Reading in CSV files from", self.BankPerf_dir)
        BankCharPerf_raw_data_dict = self.file_dict_read(self.BankPerf_dir, filetype=filetype, skip_prefix = skip_prefix)
        print(BankCharPerf_raw_data_dict.keys())

        if reducedf:
            print("Combining DataFrames")
            BankCharPerf_raw_data_dict = self.BHC_DF_Reducer(BankCharPerf_raw_data_dict, keyprefix = params_dict["CallReport_prefix"],merge_on =  ["RSSD9001", "RSSD9999"], outputkeyname = "WRDS_Covas_CallReport_Combined")

        print("Preparing Calculated Fields")
        replace_dict_tmp = replace_dict
        print("Creating temp VARLISTs Column")
        for k, v in replace_dict_tmp.items():
            print(k,v)
            print("Replacing characters from dict for", k)
            BankCharPerf_raw_data_dict[params_dict["Calc_df"]][k] = BankCharPerf_raw_data_dict[params_dict["Calc_df"]][v["calcol"]].astype(str).apply(lambda x: [i.strip() for i in self.replace_all(x, replace_dict_tmp[k]).split(",")])
            print("Replacing blanks, nans and nan strings")
            BankCharPerf_raw_data_dict[params_dict["Calc_df"]][k] = BankCharPerf_raw_data_dict[params_dict["Calc_df"]][k].apply(lambda x: [i for i in x if i not in ["", np.nan, "nan"]])
            print(BankCharPerf_raw_data_dict[params_dict["Calc_df"]][k])

        print("Checking for Column name Discrepancy")
        miss_col = self.check_missing_BHCCodes(BankCharPerf_raw_data_dict, params_dict, replace_dict_tmp)
        print(miss_col)
        # Iterate through Objects, combine and concat
        print("Combining and Creating Derived Columns")
        BankCharPerf_raw_data_dict["BankPerf_Calculated"] = self.BHC_loan_nco_ppnr_create(BankCharPerf_raw_data_dict,params_dict, replace_dict_tmp)



        if Consqtr:
            print("Applying Consecutive Quarters Rule")
            BankCharPerf_raw_data_dict = self.BankPerf_ConsecutiveQtrs_Reduce(BankCharPerf_raw_data_dict, concecutiveqtrs=concecutiveqtrs,
                                                                 BankPerf_Calc_df_name="BankPerf_Calculated",
                                                                 sincedt=ReportingDateParam[0], outputkeyname = "BankPerf_MainCalc")

        #BankPerfConRedcued.keys()
        if combinecol:
            print("Updating Calculated Columns based on Date Varying Calculations")
            BankCharPerf_raw_data_dict = self.BankPerf_Combine_Cols(BankCharPerf_raw_data_dict, src_keyname="BankPerf_MainCalc",
                                                            output_keyname="BankPerf_MainCalc")
        if merge:
            print("Applying Mergers File")
            BankCharPerf_raw_data_dict = self.BankPerf_Merger_process(BankCharPerf_raw_data_dict
                                                          , merger_df_name="merger_info_frb_new"
                                                          ,BankPerf_Calc_df_name="BankPerf_MainCalc"
                                                          , merger_df_subset=["#ID_RSSD_PREDECESSOR", "ID_RSSD_SUCCESSOR", "DT_TRANS"], merger_df_datecol="DT_TRANS",
                                                            merger_df_predecessor="#ID_RSSD_PREDECESSOR", merger_df_successor="ID_RSSD_SUCCESSOR", outputkeyname = "BankPerf_MainCalc")

            #BankPerf_Merger_process(self, BankPerf, merger_df_name="merger_info_frb",
            #                        merger_df_subset=["MERGE_DT", "NON_ID", "SURV_ID"], merger_df_datecol="MERGE_DT",
            #                        merger_df_predecessor="NON_ID", merger_df_successor="SURV_ID"
            #BankPerf_Calc_df_name = None)

            print("Aggregating Based on Mergers")
            BankCharPerf_raw_data_dict = self.BankPerf_Aggregation_process(BankCharPerf_raw_data_dict, BankPerf_ToAgg_df_name="BankPerf_MainCalc", outputkeyname = "BankPerf_MainCalc")

        if Y_calc:
            for keyname in [v for v in BankCharPerf_raw_data_dict.keys() if v.startswith("BankPerf_")]:
                print("Calculating Y_i NCO Ratios and PPNR Ratios")
                keyname_tmp = keyname + "_XYcalc"
                print(keyname_tmp)
                BankCharPerf_raw_data_dict[keyname_tmp] = self.bankperf_rates_ratios(BankCharPerf_raw_data_dict[keyname])
                #print("Merging Yi to original dataframe.")
                #BankCharPerf_raw_data_dict[keyname] = BankCharPerf_raw_data_dict[keyname].merge(BankCharPerf_raw_data_dict[keyname_tmp], on = ["RSSD_ID","ReportingDate"])

        if calc_CR_hist:
            for keyname in [v for v in BankCharPerf_raw_data_dict.keys() if v.startswith("BankPerf_")]:
                BankCharPerf_raw_data_dict[keyname] =  self.calc_TCR1_hist(BankCharPerf_raw_data_dict[keyname], fillna= False)


        if get_prev_time_cols:
            for keyname in [v for v in BankCharPerf_raw_data_dict.keys() if v.startswith("BankPerf_")]:
                BankCharPerf_raw_data_dict[keyname] =  self.get_prev_timeperiod(BankCharPerf_raw_data_dict[keyname], shift = 1, fillna= False)


        if RSSD_Subset:
            for keyname in [v for v in BankCharPerf_raw_data_dict.keys() if v.startswith("BankPerf_")]:
                print("Getting Subset of Each Data Frame based on top", RSSDList_len, "RSSD's based on ", groupagg_col)
                Rssd_tmp = self.RSSD_Subset(BankCharPerf_raw_data_dict, keyname=keyname, groupfunction=groupfunction, groupby=groupby,
                                       groupagg_col=groupagg_col, RSSD_DateParam=RSSD_DateParam,
                                       ReportingDateParam=ReportingDateParam, RSSDList_len=RSSDList_len, dropdup=dropdup,
                                       replace_nan=replace_nan)
                for k, v in Rssd_tmp.items():
                    keyname_tmp = keyname + "_Subset" + "_" + k
                    print(keyname_tmp)
                    BankCharPerf_raw_data_dict[keyname_tmp] = v


        gc.collect()
        return (BankCharPerf_raw_data_dict)

    def BankPerf_Combine_Cols(self, BankPerf, output_keyname=None, src_keyname="BankPerf_Calculated",
                              calc_keyname="NCO_PPNR_Loan_Calcs"):
        agg_exceptions = BankPerf["NCO_PPNR_Loan_Calcs"][
            pd.notnull(BankPerf[calc_keyname]["DT_Since"]) | pd.notnull(BankPerf[calc_keyname]["DT_Till"])][
            ["Variable Category", "Variable", "DT_Since", "DT_Till"]]

        print("Source DF", src_keyname)
        Master_DF_tmp = BankPerf[src_keyname]

        for row in list(agg_exceptions.index):
            # print(agg_exceptions.iloc[row,])
            # tmp_result_obj = pd.DataFrame()
            tmp_col_name = agg_exceptions.loc[row, "Variable Category"] + ":" + agg_exceptions.loc[row, "Variable"]
            tmp_agg_col = tmp_col_name.split("_")[0]
            tmp_DT_Since = agg_exceptions.loc[row, "DT_Since"]
            tmp_DT_Till = agg_exceptions.loc[row, "DT_Till"]
            print("Fixing out of bounds dates")
            if tmp_DT_Till == "9999-12-31":
                tmp_DT_Till = "2100-12-31"
            else:
                tmp_DT_Till = agg_exceptions.loc[row, "DT_Till"]

            if tmp_DT_Since == "9999-12-31":
                tmp_DT_Since = "2100-12-31"
            else:
                tmp_DT_Since = agg_exceptions.loc[row, "DT_Since"]

            print(tmp_col_name, tmp_DT_Since, tmp_DT_Till, tmp_agg_col)
            print("Getting Subset Index for Reportign Dates with Different calculations")
            tmp_subset_idx = Master_DF_tmp[
                (Master_DF_tmp["ReportingDate"] > tmp_DT_Since) & (Master_DF_tmp["ReportingDate"] < tmp_DT_Till)].index
            if tmp_agg_col in Master_DF_tmp.columns:
                print("Appending to New Vector")
                # Master_DF_tmp[tmp_agg_col] = pd.concat([Master_DF_tmp[tmp_agg_col], col_subset_df])
                Master_DF_tmp.loc[tmp_subset_idx, tmp_agg_col] = Master_DF_tmp.loc[tmp_subset_idx, tmp_col_name]

            else:
                print("Initializing Column")
                Master_DF_tmp[tmp_agg_col] = np.nan
                print("Appending to New Vector")
                Master_DF_tmp.loc[tmp_subset_idx, tmp_agg_col] = Master_DF_tmp.loc[tmp_subset_idx, tmp_col_name]
                # Master_DF_tmp[tmp_agg_col] = pd.concat([Master_DF_tmp[tmp_agg_col], col_subset_df])

        LoanList = [x for x in Master_DF_tmp.columns[Master_DF_tmp.columns.str.startswith("Loans categories:")] if
                    "_" not in x]
        print(Master_DF_tmp[LoanList].describe().transpose())

        print("Saving updated frame to Dictionary")
        BankPerf["BankPerf_CombinedCols"] = Master_DF_tmp
        if output_keyname is None:
            tmp_name = src_keyname + "_Exceptions"
            BankPerf[tmp_name] = Master_DF_tmp
        else:
            BankPerf[output_keyname] = Master_DF_tmp
        return (BankPerf)

    def BHC_DF_Reducer(self, BankPerf, keyprefix=None, merge_on=["RSSD9001", "RSSD9999"],
                       outputkeyname="WRDS_Covas_BankPerf_CallReport"):
        print("Create list of dataframes to combine")
        DF_raw_list = [v for v in BankPerf.keys() if v.startswith(keyprefix)]

        print("Initialize DFS list")
        dfs = list()
        for keyname in DF_raw_list:
            dfs.append(BankPerf[keyname])
        print("Reduce merging the Data Frames.")
        final_raw_df = reduce(lambda left, right: pd.merge(left, right, on=merge_on), dfs)
        print("Renaming Columns")
        final_raw_df.columns = [v.replace("_x", "").replace("_y", "") for v in final_raw_df.columns]
        print("Drop Duplicated Columns")
        final_raw_df = final_raw_df.loc[:, ~final_raw_df.columns.duplicated()]

        BankPerf[outputkeyname] = final_raw_df
        return (BankPerf)

    def check_missing_BHCCodes(self, BankPerf, params_dict, replace_dict):
        print("Creating List of all Variables from Regulatory and BankPerf dataframes.")

        for k, v in replace_dict.items():
            varlist_tmp = []
            print(k)
            # print("Replacing characters from dict for",k)

            for i in range(BankPerf[params_dict["Calc_df"]][k].shape[0]):
                # print(BankPerf[params_dict["Calc_df"]]["VarList"][i])
                varlist_tmp = varlist_tmp + BankPerf[params_dict["Calc_df"]][k][i]
                varlist_tmp = [x for x in varlist_tmp if x not in ['nan', ""]]

            if v["df_keyname"] in BankPerf.keys():
                bankperf = list(BankPerf[v["df_keyname"]].columns)

                mismatch_result = list(set(varlist_tmp) - set(bankperf))

            if mismatch_result.__len__() > 0:
                print("Columns not Matching from VarList to BankPerf Data:", mismatch_result)
            else:
                print("All Columns Exist in dataframes.")
        return (mismatch_result)

    def replace_all(self, text, dic):
        for i, j in dic.items():
            text = text.replace(i, j)
        return text

    def BHC_loan_nco_ppnr_create(self, BankPerf, params_dict, replace_dict,
                                 identifier_columns=["RSSD9001", "RSSD9999", "RSSD9010", "RSSD9017", "RSSD9161",
                                                     "RSSD9045", "RSSD9016", "RSSD9101","BHCK4635","BHCK4605","BHCK4340","BHCK4598","BHCK3519","BHCK3210","BHCK3368","BHCKC245","BHCKC249"],
                                 rename_col_dict={"RSSD9001": "RSSD_ID", "RSSD9999": "ReportingDate",
                                                  "RSSD9161": "CUSIP", "BHCK2170": "TotalAssets",
                                                  "BHDM3516": "QtrAvgLoansLeases_1", "BHBC3516": "QtrAvgLoansLeases_2",
                                                  "BHCK3516": "QtrAvgLoansLeases_3",
                                                  "BHFN3360": "QtrAvgTotalLoans", "BHCK7206": "CapRatios_T1RiskCR_1",
                                                  "BHCW7206": "CapRatios_T1RiskCR_2",
                                                  "BHCA7206": "CapRatios_T1RiskCR_3",
                                                  "BHCK7205": "CapRatios_TotalRiskCR_1",
                                                  "BHCW7205": "CapRatios_TotalRiskCR_2",
                                                  "BHCA7205": "CapRatios_TotalRiskCR_3",
                                                  "BHCK7204": "CapRatios_T1LR_1", "BHCW7204": "CapRatios_T1LR_2",
                                                  "BHCA7204": "CapRatios_T1LR_3",
                                                  "BHCWP793": "CapRatios_CET1CR_1", "BHCAP7204": "CapRatios_CET1CR_2",
                                                  "BHBC6061": "Net Charge Offs",
                                                  "RSSD9045": "BHC_Indicator", "RSSD9016": "FHC_Indicator",
                                                  "RSSD9101": "Domestic_Indicator"
                                                 , "RSSD9138": "Financial_Sub_Indicator", "RSSD9397": "LargestEntityBHC",
                                                  "RSSD9375": "HeadOffice_RSSD_ID", "RCFD2170" : "TotalAssets","BHCK3368":"QrtAvgTotalAssets",
                                                  "BHCK4635":"Chargeoffs","BHCK4605":"Recoveries","BHCK3210":"Total Equity Capital",
                                                  "BHCKC245":"Total Equity_1","BHCKC249":"Total Equity_2", "BHCK3519":"QrtAvgEqCap",
                                                  "BHCK4340":"Net income(loss)","BHCK4598":"Less:Cash dividends on perp perf stock",
                                                  "RSSD9010":"Entity short name","RSSD9017":"Legal name"

                                                  }):
        print("Initialize Result DF")
        BankPerf_result = pd.DataFrame()
        loan_nco_ppnr_df = pd.DataFrame()
        for k, v in replace_dict.items():
            #print(k,v)
            replace_dict_tmp = replace_dict
            if v["df_keyname"] in BankPerf.keys():
                print(k,v)
                print("Replacing characters from dict for", k,"['+','-']")
                if "+" in replace_dict_tmp[k].keys():
                    del replace_dict_tmp[k]["+"]
                if "-" in replace_dict_tmp[k].keys():
                    del replace_dict_tmp[k]["-"]
                #        del v[")"]
                #        del v["("]
                print("Initialize Calculation Column")
                calc_tmp = BankPerf[params_dict["Calc_df"]][v["calcol"]].astype(str).apply(
                    lambda x: self.replace_all(x, replace_dict_tmp[k]).strip())

                print("Create String Column with usage calculations")
                BankPerf[params_dict["Calc_df"]]["Calc_varstr"] = calc_tmp

                print("Generate Calculated Columns")
                for i in range(0, BankPerf[params_dict["Calc_df"]].shape[0]):

            #    if v["df_keyname"] in BankPerf.keys():

                    tmp_subset = BankPerf[params_dict["Calc_df"]].loc[i, k]
                    tmp_df = BankPerf[v["df_keyname"]][tmp_subset]

                    tmp_date_str = (BankPerf[params_dict["Calc_df"]].loc[i, "DT_Since"] == np.nan) & (
                                BankPerf[params_dict["Calc_df"]].loc[i, "DT_Till"] == np.nan)

                    tmp_varname = BankPerf[params_dict["Calc_df"]].loc[i, "Variable"]
                    tmp_varcat = BankPerf[params_dict["Calc_df"]].loc[i, "Variable Category"]
                    tmp_calc_str = BankPerf[params_dict["Calc_df"]].loc[i, "Calc_varstr"]


                    print("Get Column Vectors")
                    for col in tmp_subset:
                        print(col,tmp_date_str)
                        exec(col + ''' = tmp_df["''' + col + '''"]''')

                    print("Calculate Derived Filed for:", tmp_varname, tmp_varcat, tmp_calc_str)

                    tmp_col_name = tmp_varcat + ":" + tmp_varname

                    if tmp_calc_str not in ["", "nan", np.nan]:
                        #if not tmp_date_str:
                        tmp_result_obj = eval(tmp_calc_str)
                    else:
                        tmp_result_obj = np.nan
                    print("Add Derived Field to Result DataFrame")

                    loan_nco_ppnr_df[tmp_col_name] = tmp_result_obj
                    print(loan_nco_ppnr_df[tmp_col_name].shape)
                # Combine
                #if v["df_keyname"] in BankPerf.keys():
                print("Combine Calculated Columns to Original Dataframe")
                BankPerf_result_tmp = pd.concat([BankPerf[v["df_keyname"]][identifier_columns], loan_nco_ppnr_df], axis=1)
                print("Rename Identifier Columns")
                BankPerf_result_tmp = BankPerf_result_tmp.rename(rename_col_dict, axis=1)
                print("Format ReportingDate column")
                BankPerf_result_tmp["ReportingDate"] = pd.to_datetime(BankPerf_result_tmp["ReportingDate"], format="%Y%m%d")
                BankPerf_result = pd.concat([BankPerf_result, BankPerf_result_tmp])

            return (BankPerf_result)

    def RSSD_Subset(self, BankPerf, keyname=None, groupfunction=np.mean, groupby=["RSSD_ID", "ReportingDate"],
                    groupagg_col="'Other items:Consolidated assets'", RSSD_DateParam=["1989-12-31", "2017-12-31"],
                    ReportingDateParam=["1989-12-31", "2017-12-31"], RSSDList_len=1000, dropdup=False,
                    replace_nan=True):
        if keyname in BankPerf.keys():
            print("Getting BankPerf  rows")
            BankPerf_Calc_df = BankPerf[keyname]
            if replace_nan:
                print("Replaceing 0.0 with NAN")
                BankPerf_Calc_df = BankPerf_Calc_df.replace(0.0, np.nan)
        else:
            return ("BankPerf key DF not found")

        print("First Group By:", groupby, "And Aggregate using SUM")
        test1_groupby = BankPerf_Calc_df.groupby(groupby).agg({groupagg_col: np.sum})
        print("Sorting by:", groupagg_col)
        test1_groupby = test1_groupby.reset_index().sort_values([groupagg_col], ascending=[0])
        if "TotalAssets" in groupagg_col:
            print("Updating Total Assets by increasing values by 1e3")
            test1_groupby["TotalAssets"] = test1_groupby["TotalAssets"] * 1e3

        print("Filtering out Zero values and Reporting Date => criteria:", ReportingDateParam)
        test1_groupby = test1_groupby[
            (test1_groupby[groupagg_col] != 0) & (test1_groupby['ReportingDate'] >= RSSD_DateParam[0]) & (
                        test1_groupby['ReportingDate'] <= RSSD_DateParam[1])]
        if dropdup:
            test1_groupby = test1_groupby.drop_duplicates(subset=['RSSD_ID'], keep='first')
            test1_groupby = test1_groupby[(test1_groupby[groupagg_col] != 0)].reset_index().sort_values([groupagg_col],
                                                                                                        ascending=[0])
        else:
            print("Grouping across all dates using:", str(groupfunction))
            test1_groupby = test1_groupby.groupby(["RSSD_ID"]).agg({groupagg_col: groupfunction})
            test1_groupby = test1_groupby[(test1_groupby[groupagg_col] != 0)].reset_index().sort_values([groupagg_col],
                                                                                                        ascending=[0])

        print("Getting Top", RSSDList_len, "From RSSD List")
        RSSD_ID_1k = test1_groupby['RSSD_ID'][0:RSSDList_len]

        print("Getting Descriptive Statistics of RSSD_List on Dataframe")
        BankPerf_tmp = BankPerf_Calc_df
        BankPerf_tmp = BankPerf_tmp[
            (BankPerf_tmp["RSSD_ID"].isin(RSSD_ID_1k)) & (BankPerf_tmp['ReportingDate'] >= ReportingDateParam[0]) & (
                        BankPerf_tmp['ReportingDate'] <= ReportingDateParam[1])]
        results_df = BankPerf_tmp[(BankPerf_tmp.columns[
            pd.Series(BankPerf_tmp.columns).str.startswith("Loans categories:")])].describe().transpose()
        print(results_df)

        return ({"RSSD_List": RSSD_ID_1k, "DescriptiveStats": results_df, "BankPerf": BankPerf_tmp})

    def BankPerf_Year_RSSD_Subset(self, BankPerf, keyname="BankPerfAgg", year=None, between=["1990-01-01", "2017-12-31"]):
        if year is not None:
            RSSD_ID_tmp = list(BankPerf[keyname]["RSSD_ID"][BankPerf[keyname]["ReportingDate"].isin(year)])
        else:
            RSSD_ID_tmp = list(BankPerf[keyname]["RSSD_ID"].unique())
        BankPerf_tmp = BankPerf[keyname][(BankPerfAgg[keyname]["ReportingDate"] >= between[0]) & (
                    BankPerfAgg[keyname]["ReportingDate"] <= between[1]) & (
                                             BankPerf[keyname]["RSSD_ID"].isin(RSSD_ID_tmp))]

        print_tmp = BankPerf_tmp[BankPerf_tmp.columns[
            pd.Series(BankPerf_tmp.columns).str.startswith("Loans categories:")]].describe().transpose()
        print(print_tmp)
        print_tmp = print_tmp.reset_index().rename({"index": "varname"}, axis=1)
        return (print_tmp)

    def BankPerf_ConsecutiveQtrs_Reduce(self, BankPerf, concecutiveqtrs=8, BankPerf_Calc_df_name=None, sincedt="1989-12-31", outputkeyname = None):
        if BankPerf_Calc_df_name in BankPerf.keys():
            print("Getting BankPerf Calculated rows")
            BankPerf_Calc_df = BankPerf[BankPerf_Calc_df_name]
        else:
            return ("BankPerf Calculated DF not found")

        print("Filter Banks for those that are above::", concecutiveqtrs, "consecutive quarters.")
        BankPerf_Merger_df_gbsum = BankPerf_Calc_df
        BankPerf_Merger_df_gbsum = BankPerf_Merger_df_gbsum.sort_values(['RSSD_ID', 'ReportingDate'])

        print("Filtering Since Date", sincedt)
        BankPerf_Merger_df_gbsum_tmp = BankPerf_Merger_df_gbsum[BankPerf_Merger_df_gbsum["ReportingDate"] > sincedt]
        BankPerf_Merger_df_gbsum_tmp["ConsecutivePeriods"] = BankPerf_Merger_df_gbsum_tmp.sort_values(
            ['RSSD_ID', 'ReportingDate']).groupby(['RSSD_ID']).cumcount() + 1

        BankPerf_Merger_df_gb_tmp = BankPerf_Merger_df_gbsum_tmp.groupby('RSSD_ID')
        BankPerf_Merger_df_gb_tmp = BankPerf_Merger_df_gb_tmp.agg({"ConsecutivePeriods": ['min', 'max', 'count']})

        RSSD_ID_consecutive = list(
            BankPerf_Merger_df_gb_tmp[BankPerf_Merger_df_gb_tmp["ConsecutivePeriods"]["max"] >= concecutiveqtrs].index)
        BankPerf_Merger_df_gbsum = BankPerf_Merger_df_gbsum[
            BankPerf_Merger_df_gbsum["RSSD_ID"].isin(RSSD_ID_consecutive)]


        BankPerf["BankPerf_ConsecutiveReduced"] = BankPerf_Merger_df_gbsum
        print(BankPerf["BankPerf_ConsecutiveReduced"].describe().transpose())

        if outputkeyname is not None:
            BankPerf[outputkeyname] = BankPerf_Merger_df_gbsum
            print(BankPerf[outputkeyname].describe().transpose())

        return (BankPerf)

    def BankPerf_Merger_process(self, BankPerf, merger_df_name="merger_info_frb", merger_df_subset = ["MERGE_DT", "NON_ID", "SURV_ID"], merger_df_datecol = "MERGE_DT",merger_df_predecessor = "NON_ID",merger_df_successor = "SURV_ID",
                                BankPerf_Calc_df_name=None, outputkeyname = None):
        if merger_df_name in BankPerf.keys():
            print("Getting Merger Info from Dictionary")
            merger_info_df = BankPerf[merger_df_name]
            merger_info_df = merger_info_df[merger_df_subset]
            merger_info_df[merger_df_datecol] = pd.to_datetime(merger_info_df[merger_df_datecol], format="%Y%m%d")
            merger_info_df[merger_df_predecessor] = merger_info_df[merger_df_predecessor].astype(int)
        # print(merger_info_df.info())
        # print(merger_info_df.describe())
        else:
            return ("Merger Information not found in dictionary")

        if BankPerf_Calc_df_name in BankPerf.keys():
            print("Getting BankPerf Calculated rows")
            BankPerf_Calc_df = BankPerf[BankPerf_Calc_df_name]
        else:
            return ("BankPerf Calculated DF not found")

        print("Loop/merge through merger information and update RSSD of non survivor with survivor ID.")
        #    orig_columns = list(BankPerf_Calc_df.columns)
        BankPerf_Merger_df = BankPerf_Calc_df.merge(merger_info_df, left_on="RSSD_ID", right_on=merger_df_predecessor, how="left")
        print("Found", BankPerf_Merger_df["RSSD_ID"][pd.notnull(BankPerf_Merger_df[merger_df_predecessor])].unique().__len__(),
              ' Non surviving RSSD_IDs')
        print("Updating Non-Surviving RSSD_IDs with Surviving IDs")
        BankPerf_Merger_df.loc[pd.notnull(BankPerf_Merger_df[merger_df_predecessor]), ["RSSD_ID"]] = BankPerf_Merger_df.loc[
            pd.notnull(BankPerf_Merger_df[merger_df_predecessor]), [merger_df_successor]]
        print("Dropping Merge Columns")
        BankPerf_Merger_df = BankPerf_Merger_df.drop(list(merger_info_df.columns), axis=1)
        print("Re Merging to check for remaining mergers")
        BankPerf_Merger_df = BankPerf_Merger_df.merge(merger_info_df, left_on="RSSD_ID", right_on=merger_df_predecessor, how="left")
        print("Number of NON_ID matching with RSSD_ID:",
              BankPerf_Merger_df[pd.notnull(BankPerf_Merger_df[merger_df_predecessor])].shape)
        # if BankPerf_Merger_df[pd.notnull(BankPerf_Merger_df["NON_ID"])].shape == 0:

        BankPerf["BankPerf_Mergered"] = BankPerf_Merger_df
        print(BankPerf["BankPerf_Mergered"].describe().transpose())

        if outputkeyname is not None:
            BankPerf[outputkeyname] = BankPerf_Merger_df
            print(BankPerf[outputkeyname].describe().transpose())

        return (BankPerf)

    def BankPerf_Aggregation_process(self, BankPerf, BankPerf_ToAgg_df_name=None, outputkeyname = None):

        if BankPerf_ToAgg_df_name in BankPerf.keys():
            print("Getting BankPerf Calculated rows")
            BankPerf_Merger_df_gbsum = BankPerf[BankPerf_ToAgg_df_name]
        else:
            return ("BankPerf Calculated DF not found")

        print("Aggregate all the updated RSSD_ID columns")

        # BankPerf_Merger_df_gbsum.columns
        BankPerf_Merger_df_gbsum = BankPerf_Merger_df_gbsum.groupby(["RSSD_ID", "ReportingDate","BHC_Indicator","FHC_Indicator"]).sum()
        #Find a way to make this part dynamic.
        #BankPerf_Merger_df_gbsum = BankPerf_Merger_df_gbsum.agg({
            # 'TotalAssets' : np.sum,
            #'Net charge-offs by type of loan:Commercial & industrial': np.sum,
            #'Net charge-offs by type of loan:Construction & land development': np.sum,
            #'Net charge-offs by type of loan:Multifamily real estate': np.sum,
            #'Net charge-offs by type of loan:(Nonfarm) nonresidential CRE': np.sum,
            #'Net charge-offs by type of loan:Home equity lines of credit': np.sum,
            #'Net charge-offs by type of loan:Residential real estate (excl. HELOCs)': np.sum,
            #'Net charge-offs by type of loan:Credit card': np.sum,
            #'Net charge-offs by type of loan:Consumer (excl. credit card)': np.sum,
            #'Loans categories:Commercial & industrial': np.sum,
            #'Loans categories:Construction & land development': np.sum,
            #'Loans categories:Multifamily real estate': np.sum,
            #'Loans categories:Nonfarm nonresidential CRE': np.sum,
            #'Loans categories:Home equity lines of credit': np.sum,
            #'Loans categories:Residential real estate (excl. HELOCs)': np.sum,
            #'Loans categories:Credit card': np.sum,
            #'Loans categories:Consumer (excl. credit card)': np.sum,
            #'Components of pre-provision net revenue:Net interest income': np.sum,
            #'Components of pre-provision net revenue:Noninterest income': np.sum,
            #'Components of pre-provision net revenue:Trading income': np.sum,
            #'Components of pre-provision net revenue:Compensation expense': np.sum,
            #'Components of pre-provision net revenue:Fixed assets expense': np.sum,
            #'Components of pre-provision net revenue:Noninterest expense': np.sum,
            #'Other items:Consolidated assets': np.sum,
            #'Other items:Interest-earning assets': np.sum,
            #'Other items:Trading assets': np.sum,
            #'Other items:Book equity': np.sum,
            #'Other items:Risk-weighted assets': np.sum,
            #'Other items:Dividends ': np.sum,
            #'Other items:Stock purchases': np.sum,
            #'Other items:Tier 1 common equity': np.sum,
            #'Other items:= Tier 1 capital': np.sum,
            #'Other items: = - Perpetual preferred stock': np.sum,
            #'Other items: = + Nonqual. Perpetual preferred stock': np.sum,
            #'Other items: = - Qual. class A minority interests': np.sum,
            #'Other items: = - Qual. restricted core capital': np.sum,
            #'Other items: = - Qual. mandatory convert. pref. sec.': np.sum,
            #'Other items:Total Assets' : np.sum
        #})
        # May need to add manual column sums.
        BankPerf_Merger_df_gbsum = BankPerf_Merger_df_gbsum.reset_index()

        BankPerf["BankPerf_Agg"] = BankPerf_Merger_df_gbsum
        print(BankPerf["BankPerf_Agg"].describe().transpose())

        if outputkeyname is not None:
            BankPerf[outputkeyname] = BankPerf_Merger_df_gbsum
            print(BankPerf[outputkeyname].describe().transpose())


        return (BankPerf)

    def bankperf_rates_ratios(self,BankPerf, replace_nan=True, fillna = True
                              , ncoR_dict={
                'Net charge-offs by type of loan:Commercial & industrial': 'Loans categories:Commercial & industrial_Covas'
                ,
                'Net charge-offs by type of loan:Construction & land development': 'Loans categories:Construction & land development'
                ,
                'Net charge-offs by type of loan:Multifamily real estate':'Loans categories:Multifamily real estate'
                ,
                'Net charge-offs by type of loan:(Nonfarm) nonresidential CRE': 'Loans categories:Nonfarm nonresidential CRE_Covas'
                ,
                'Net charge-offs by type of loan:Home equity lines of credit': 'Loans categories:Home equity lines of credit'
                ,
                'Net charge-offs by type of loan:Residential real estate (excl. HELOCs)': 'Loans categories:Residential real estate (excl. HELOCs)_Covas'
                ,
                'Net charge-offs by type of loan:Credit card': 'Loans categories:Credit card'
                ,
                'Net charge-offs by type of loan:Consumer (excl. credit card)': 'Loans categories:Consumer (excl. credit card)_Covas'
                }

                              ):
        # Need to handle NAN rows and outliers.



        if fillna:
            print("Interpolating and Filling Nans")
            BankPerf = BankPerf.interpolate(method = 'polynomial',order = 2).fillna(method = 'ffill').fillna(method = 'bfill')
        if replace_nan:
            print("Replace 0.0 with np.nan")
            BankPerf = BankPerf.replace(0.0, np.nan)


        print("Loans categories Descriptive Statistics")
        # BankPerf = BankPerf[~BankPerf[BankPerf.columns[pd.Series(BankPerf.columns).str.startswith("Loans categories:")]].isin([np.nan, np.inf, -np.inf]).any(1)]
        # print(BankPerf[BankPerf.columns[pd.Series(BankPerf.columns).str.startswith("Loans categories:")]].describe().transpose())

        # BankPerf_tmp = BankPerf
        BankPerf_tmp = BankPerf
        print("Net Charge-Off rates by type of loan calculations")

        for nco, loan in ncoR_dict.items():
            print("Charge-Off", nco, "Loan Category:", loan)
            BankPerf_subset_idx = BankPerf_tmp[
                (BankPerf_tmp[nco] >= 0) & (BankPerf_tmp[nco] <= BankPerf_tmp[loan])].index

            tmp_str = nco.replace("Net charge-offs by type of loan:", "ncoR:")
            print(tmp_str)
            BankPerf_tmp[tmp_str] = np.nan
            BankPerf_tmp.loc[BankPerf_subset_idx, tmp_str] = 100 * (
                        (BankPerf_tmp.loc[BankPerf_subset_idx, nco]) / (BankPerf_tmp.loc[BankPerf_subset_idx, loan]))
            BankPerf_clean_tmp_idx = BankPerf_tmp.loc[BankPerf_subset_idx, tmp_str][
                BankPerf_tmp[tmp_str].isin([np.inf, -np.inf])].index
            BankPerf_tmp.loc[BankPerf_clean_tmp_idx, tmp_str] = np.nan
        #        BankPerf[tmp_str] = BankPerf[tmp_str] * 400

        # BankPerf_tmp = pd.concat([BankPerf_tmp,BankPerf_tmp2[tmp_str]], ignore_index=True, axis = 1)
        print(BankPerf_tmp[
                  BankPerf_tmp.columns[pd.Series(BankPerf_tmp.columns).str.startswith("ncoR:")]].describe().transpose())

        print("PPNR Ratios calculations")
        for ppnr_comp in list(BankPerf_tmp.columns[pd.Series(BankPerf_tmp.columns).str.startswith(
                "Components of pre-provision net revenue:")]):
            print(ppnr_comp)
            BankPerf_subset_idx = BankPerf_tmp[
                (BankPerf_tmp[ppnr_comp] <= BankPerf_tmp['Other items:Consolidated assets'])].index

            tmp_str = ppnr_comp.replace("Components of pre-provision net revenue:", "ppnrRatio:")
            print(tmp_str)

            BankPerf_tmp[tmp_str] = np.nan

            BankPerf_tmp.loc[BankPerf_subset_idx, tmp_str] = 100 * (
                        (BankPerf_tmp.loc[BankPerf_subset_idx, ppnr_comp]) / (
                BankPerf_tmp.loc[BankPerf_subset_idx, 'Other items:Consolidated assets']))
            BankPerf_clean_tmp_idx = BankPerf_tmp.loc[BankPerf_subset_idx, tmp_str][
                BankPerf_tmp[tmp_str].isin([np.inf, -np.inf])].index
            BankPerf_tmp.loc[BankPerf_clean_tmp_idx, tmp_str] = np.nan

        print(BankPerf_tmp[BankPerf_tmp.columns[
            pd.Series(BankPerf_tmp.columns).str.startswith("ppnrRatio:")]].describe().transpose())

        # Balance sheet composition indicators
        # Requires interest earning assets calculations.

        return (BankPerf_tmp)

    def calc_TCR1_hist(self, BankPerf_df, T1C_col = "Other items:= Tier 1 capital", RWA_col = "Other items:Risk-weighted assets", fillna = False):
        if pd.Series([T1C_col, RWA_col]).isin(BankPerf_df.columns).all():
            print("Initialize T1CR column")
            tmp_T1CR = "Other items:T1CR"
            if fillna:
                BankPerf_df = BankPerf_df.fillna(method = "ffill")
                tmp_idx = BankPerf_df[[T1C_col,RWA_col]].index
            else:
                tmp_idx = BankPerf_df[[T1C_col,RWA_col]][pd.notnull(BankPerf_df[T1C_col]) & pd.notnull(BankPerf_df[RWA_col])].index

            print("Calculating T1CR based on historical columns")
            BankPerf_df.loc[tmp_idx,tmp_T1CR] = BankPerf_df.loc[tmp_idx,T1C_col]/BankPerf_df.loc[tmp_idx,RWA_col]
            print(BankPerf_df[BankPerf_df.columns[pd.Series(BankPerf_df.columns).str.startswith(tmp_T1CR)]].describe().transpose())
        else:
            print("Columns to calculate T1CR does not exist")

        return(BankPerf_df)

    #Get previous time periods
    def get_prev_timeperiod(self, Bankperf_df, shift = 1, colnames = None, groupby = "RSSD_ID", fillna = None):
        if fillna:
            Bankperf_df = Bankperf_df.fillna(method = "ffill")
        if colnames == None:
            for colname in [v for v in Bankperf_df.keys() if v not in [groupby, "ReportingDate"]]:

                tmp_colname = colname + "_t-" + str(shift)
                print(colname, tmp_colname)
                Bankperf_df[tmp_colname] = Bankperf_df.groupby(groupby)[colname].shift(shift)
        else:
            for colname in [v for v in colnames if v not in [groupby, "ReportingDate"]]:
                tmp_colname = colname + "_t-" + str(shift)
                print(colname, tmp_colname)
                Bankperf_df[tmp_colname] = Bankperf_df.groupby(groupby)[colname].shift(shift)
        print(Bankperf_df.shape)
        print(Bankperf_df.describe().transpose())
        return(Bankperf_df)


def DateValues_Extractor(DF, DateColumn="ReportingDate", day=[30, 31], month=[3, 6, 9, 12], year=None):
    if day != None and len(day) > 0:
        print("Subsetting Day")
        DF = DF[(pd.to_datetime(DF[DateColumn]).dt.day.isin(day))]

    if month != None and len(month) > 0:
        print("Subsetting Month")
        DF = DF[(pd.to_datetime(DF[DateColumn]).dt.month.isin(month))]

    if year != None and len(year) > 0:
        print("Subsetting Year")
        DF = DF[(pd.to_datetime(DF[DateColumn]).dt.year.isin(year))]

    return (DF)


def outlier_preprocess( df, col_exclude = ["RSSD_ID","ReportingDate"], quantiles = [.15,.80], interpolate = True, interpolate_method = "linear", makezero = None, verbose = False):
    down_quantiles = df[df.columns.difference(col_exclude)].quantile(quantiles[0])
    if verbose:
        print(down_quantiles)

    outliers_low = (df[df.columns.difference(col_exclude)] < down_quantiles)

    up_quantiles = df[df.columns.difference(col_exclude)].quantile(quantiles[1])

    if verbose:
        print(up_quantiles)
    outliers_high = (df[df.columns.difference(col_exclude)] > up_quantiles)


    df_tmp = df[df.columns.difference(col_exclude)].mask(outliers_low, down_quantiles, axis=1).mask(outliers_high, up_quantiles, axis=1)

    if makezero is not None:
        if isinstance(makezero,list):
            zero_mask = (df_tmp < makezero[0]) & (df_tmp < makezero[1])
            df_tmp = df_tmp.mask(zero_mask, 0, axis=1)
        else:
            zero_mask = (df_tmp < makezero)
            df_tmp = df_tmp.mask(zero_mask,0, axis = 1)

    if interpolate:
        df_tmp = df_tmp.interpolate(method=interpolate_method, order=2)

    df_tmp = pd.concat([df[col_exclude], df_tmp], axis = 1)

    if verbose:
        print(df_tmp.describe())

    return(df_tmp)



def process_raw_bankdat_PCA_Norm(df_raw, dates = ["1990 Q1", "2016 Q4"], raw_date_col = "ReportingDate",
                                 keyname = None, interpolatefill = True, n_components = 10,
                                 visualizations = False, target_list = ['Other items: CapRatios_T1RiskCR_coalesced']):
    #df_raw = Z_macro
    #keyname = "Z_macro"
    df_raw_dict = dict()
    #df_raw = Preprocess_Dict['PCA_DF'][pca_colist]

    if keyname is not None:
        print("Preproecessing Raw DataFrame:%s" % keyname)
        df_raw = df_raw[keyname]
    else:
        keyname = 'DF'

    if n_components < 10 and n_components >  df_raw.shape[1]:
        n_components = df_raw.shape[1] - 2
    elif df_raw.shape[1] > 10 and n_components < df_raw.shape[1]:
        n_components = 10
    else:
        n_components = df_raw.shape[1] - 2


    if "ReportingDate" not in df_raw.columns and raw_date_col in df_raw.columns:
        print("Renaming Date Column: %s to ReportingDate" % raw_date_col)
        df_raw =df_raw.rename({raw_date_col: "ReportingDate"}, axis=1)

    print("Filtering for indicated dates %s to %s" % tuple(dates))
    df_raw = df_raw[(df_raw["ReportingDate"] >= dates[0]) & (df_raw["ReportingDate"] <= dates[1])]

    print("Dropping all Columns that are all NAN")
    df_raw = df_raw.dropna(axis=1, how='all')


    if interpolatefill:
        print("Interpolating Data")
        df_raw = df_raw.interpolate(method="polynomial", order=2).fillna(method="bfill").fillna(method="ffill").dropna(axis=1, how='all')

     print("Normalizing, Transforming to PCA, Correlating, and Generating Plots")

    for target in target_list:
        exclude = ["ReportingDate","RSSD_ID"]
        print("Target:", target)
        non_target_tmp =  exclude #+ [x for x in target_list if not x == target]
        print("Exlcuding:", non_target_tmp)
        print("First Remove Exclusions:",non_target_tmp)
        df_raw_tmp = df_raw[df_raw.columns.difference(non_target_tmp)]

        print('Normalize without target')
        scale = StandardScaler()
        print("Preparing Normalization DataSet")
        x_train = df_raw_tmp
        print("Dropping Target column:%s" % target)
        columns = x_train.drop(target, axis=1).columns
        #columns = x_train.columns
        print("Normalizing DataFrame")
        x_train = scale.fit_transform(x_train.drop(target, axis=1))  # drop the label and normalizing
        x_train = scale.fit_transform(x_train)
        print("Creating Normalized DataFrame")
        x_train_normalized = pd.DataFrame(x_train)  # It is required to have X converted into a data frame to later plotting need
        print("Reapplying Column Names")
        x_train_normalized.columns = columns
        print("Re-Ordering Column Names")
        x_train_normalized = x_train_normalized.sort_index(axis=1)

        print("PCA Dim Reduction")
        print("Setting Y target:%s" % target)
        #y_train = df_raw[target].reset_index(drop=True)  # .apply(lambda x: x.split(' ')[0]).reset_index(drop = True)
        y_train = df_raw["ReportingDate"].reset_index(drop = True)


        print("Getting Classes and Labels from: %s" % target)
        # VeryHigh = 1.96 * y_train.mean()
        # Average  =  y_train.mean()
        # VeryBelow = y_train.mean() / 1.96

        #labels = ["Very High" if x > VeryHigh else "Above Average" if x > Average else 'Below Average' if x < Average else 'Very Below Average' if x < VeryBelow else "Average" for x in y_train.tolist()]
        #classes = np.sort(np.unique(labels))
        #y_train_discrete = labels
        labels = y_train
        classes = np.sort(np.unique(labels))



        print("Initializing PCA and transforming Nomarlized DataFrame with Componets:%s" % str(n_components))
        # Run PCA
        pca = PCA(n_components=n_components)
        x_train_projected = pca.fit_transform(x_train_normalized)

        print("Correlation Heatmap Calculations for first two componets")
        pca_cols = ['PCA_' + str(i) for i in range(0, x_train_projected.shape[1])]
        pca_df = pd.DataFrame(x_train_projected, columns=pca_cols).reset_index(drop=True)
        pca_normalized = pd.concat([pca_df[["PCA_0", "PCA_1"]], x_train_normalized.reset_index(drop=True)], axis=1)
        # pca_normalized = pca_normalized.rename({0:"PCA_1", 1:"PCA_2"}, axis = 1)
        pca_normalized_corr = pca_normalized.corr()

        if visualizations:
            print("Generating Correlation Heatmap")
            print("PCA Correlation Heatmap")
            fig = plt.figure(figsize=(18, 18))
            fig.set_size_inches(18.5, 18.5, forward=True)
            sns.heatmap(pca_normalized_corr,
                        # xticklabels=[x for x in pca_normalized_corr.columns if x.startswith("PCA")],
                        # yticklabels=[x for x in pca_normalized_corr.columns if not x.startswith("PCA")]
                        xticklabels=pca_normalized_corr.columns,
                        yticklabels=pca_normalized_corr.columns
                        )
            plt.savefig("./Images/PCA_Normalized_Correlation_Heatmap_%s.png" % keyname, format='png')
            plt.savefig("./Images/PCA_Normalized_Correlation_Heatmap_%s.pdf" % keyname, format='pdf')
            plt.close()

            print("Scatter of PCA BiPlot Visualization")
            fig = plt.figure(figsize=(18, 18))
            fig.set_size_inches(18.5, 18.5, forward=True)
            ax = fig.add_subplot(1, 1, 1)

            # markers = ["o", "D"]

            for class_ix in zip(
                    classes):
                ax.scatter(x_train_projected[np.where(y_train == class_ix[0]), 0],
                           x_train_projected[np.where(y_train == class_ix[0]), 1],
                           # marker=marker, color=color, edgecolor='whitesmoke',
                           linewidth='1', alpha=0.9, label=class_ix[0])
                # ax.legend(loc='best')

            plt.title(
                "Scatter plot projections on the "
                "2 first principal components")
            plt.xlabel("Principal axis 1 - Explains %.1f %% of the variance" % (
                    pca.explained_variance_ratio_[0] * 100.0))
            plt.ylabel("Principal axis 2 - Explains %.1f %% of the variance" % (
                    pca.explained_variance_ratio_[1] * 100.0))

            print("Annotations for Datapoints")
            for i, txt in enumerate(labels):
                ax.annotate(txt, (x_train_projected[i, 0], x_train_projected[i, 1]))

            plt.savefig("./Images/pca_biplot_%s.pdf" % "temp", format='pdf')
            plt.savefig("Images/pca_biplot_%s.png" % "temp", format='png')
            # plt.show()
            plt.close()

        print("Reattaching Columns:", non_target_tmp)
        df_raw_dict["_".join([keyname, "normalized"])] = pd.concat([df_raw[non_target_tmp].reset_index(drop = True),x_train_normalized.reset_index(drop = True)], axis = 1)
        df_raw_dict["_".join([keyname, "normalized_pca"])] = pd.concat([df_raw[non_target_tmp].reset_index(drop = True), pca_df.reset_index(drop = True)], axis=1)
        df_raw_dict["_".join([keyname, "normalized_2_pca"])]  = pd.concat([df_raw[non_target_tmp].reset_index(drop = True), pca_df[["PCA_0","PCA_1"]].reset_index(drop = True)], axis=1)
        df_raw_dict["_".join([keyname,'original_processed'])] = df_raw
        df_raw_dict["_".join([keyname, 'target'])] = df_raw[non_target_tmp + [target]]
        return(df_raw_dict)


def Lineplot(df, x = "ReportingDate", y = "'Other items:Loan Charge Offs_Ratio'"):
    #plt.figure(figsize=(20, 8))
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5, forward=True)
    ax.plot(df[x], df[y])
    plt.xticks(df[x], df[x], rotation='vertical')
    ax.set(xlabel=x, ylabel=y, title='Evolution of %s' % y)
    plt.show()
    plt.savefig("./Images/%s_LinePlot.png" % y, format="png")
    plt.savefig("./Images/%s_LinePlot.pdf" % y, format="pdf")
    plt.close()

#ax.grid()

def process_raw_modality_PCA_Norm(df_raw, dates = ["1976-01-01", "2017-12-31"], raw_date_col = "Date", keyname = None, extractquarters = True
                                  , interpolatefill = True,Norm_PCA_Corr_Plot = True, removecols = None, rootdir = "/Users/phn1x/icdm2018_research_BB/Stress_Test_Research/Loss_projections/", normalize = True
                                  ,n_components = 12):
    os.chdir(rootdir)
    #df_raw = Z_macro
    #keyname = "Z_macro"
    df_raw_dict = dict()

    if keyname is not None:
        print("Preproecessing Raw DataFrame:%s" % keyname)
        df_raw = df_raw[keyname]
    else:
        keyname = 'DF'
    print("Renaming Date Column: %s to ReportingDate" % raw_date_col)
    df_raw =df_raw.rename({raw_date_col: "ReportingDate"}, axis=1)
    #removecols = [x for x in df_raw.columns if  x.startswith('Scenario Name')]
    if removecols is not None and len(removecols) > 0:
        df_raw = df_raw[df_raw.columns.difference(removecols)]

    if n_components >= df_raw.shape[1] and n_components >= df_raw.shape[1] - 2 and df_raw.shape[1] > 9:
        print("Setting n_componets to 10")
        n_components = 10

    if extractquarters:
        print("Extracting Quarterly Dates")
        df_raw = DateValues_Extractor(df_raw)
    print("Filtering for indicated dates %s to %s" % tuple(dates))
    df_raw = df_raw[(pd.to_datetime(df_raw["ReportingDate"]) >= dates[0]) & (pd.to_datetime(df_raw["ReportingDate"]) <= dates[1])]

    print("Dropping all Columns that are all NAN")
    df_raw = df_raw.dropna(axis=1, how='all')

    print("Converting Dates into YYYY Q# format")
    df_raw["ReportingDate"] = pd.to_datetime(df_raw.ReportingDate).dt.year.astype(str) + " Q" + pd.to_datetime(df_raw.ReportingDate).dt.quarter.astype(str)

    if interpolatefill:
        print("Interpolating Data")
        df_raw = df_raw.interpolate(method="polynomial", order=2).fillna(method="bfill").fillna(method="ffill").dropna(axis=1, how='all')

    #
    if Norm_PCA_Corr_Plot:
        print("Normalizing, Transforming to PCA, Correlating, and Generating Plots")
        df_raw_dict = Normalize_PCA_Correlations_Plots(df_raw, Target="ReportingDate",n_components= n_components, filename = keyname, normalize = normalize)

    df_raw_dict["_".join(["df_raw", keyname, 'processed'])] = df_raw

    return(df_raw_dict)


def varname(p):
  for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
    m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
    if m:
      return m.group(1)
def Normalize_PCA_Correlations_Plots(raw_df,Target = "ReportingDate",n_components = 12, filename = "None", normalize = True):
    if n_components >= raw_df.shape[1] and n_components >= raw_df.shape[1] - 2 and raw_df.shape[1] > 9 :
        print("Setting n_componets to 10")
        n_components = 10


    print("PCA , Normalization and Figure, Correlation Heatmap")
    scale = StandardScaler()
    print("Preparing Normalization DataSet")
    x_train = raw_df
    print("Dropping Target column:%s" % Target)
    columns = x_train.drop(Target, axis=1).columns

    if normalize:
        print("Normalizing DataFrame")
        x_train = scale.fit_transform(x_train.drop(Target, axis=1))  # drop the label and normalizing
        print("Creating Normalized DataFrame")
        x_train_normalized = pd.DataFrame(x_train)  # It is required to have X converted into a data frame to later plotting need
        print("Reapplying Column Names")
        x_train_normalized.columns = columns
        print("Re-Ordering Column Names")
        x_train_normalized = x_train_normalized.sort_index(axis = 1)
    else:
        x_train_normalized = x_train.drop(Target, axis=1)
        x_train_normalized.columns = columns

    print("Setting Y target:%s" % Target)
    y_train = raw_df[Target].reset_index(drop = True)#.apply(lambda x: x.split(' ')[0]).reset_index(drop = True)

    print("Getting Classes and Labels from: %s" % Target)
    classes = np.sort(np.unique(y_train))
    labels = y_train.tolist()

    print("Initializing PCA and transforming Normalized DataFrame with Components:%s" % str(n_components))
    # Run PCA
    pca = PCA(n_components=n_components)
    x_train_projected = pca.fit_transform(x_train_normalized)


    print("Correlation Heatmap Calculations for first two componets")
    pca_cols = ['PCA_' + str(i) for i in range(0,x_train_projected.shape[1])]
    pca_df = pd.DataFrame(x_train_projected, columns=pca_cols).reset_index(drop=True)
    pca_normalized  = pd.concat([pca_df,x_train_normalized.reset_index(drop = True)], axis = 1)
    #pca_normalized = pca_normalized.rename({0:"PCA_1", 1:"PCA_2"}, axis = 1)
    pca_normalized_corr = pca_normalized.corr()



    print("Generating Correlation Heatmap")
    print("PCA Correlation Heatmap")
    fig = plt.figure(figsize=(20, 20))
    fig.set_size_inches(20.5, 20.5, forward=True)
    sns.heatmap(pca_normalized_corr,
            # xticklabels=[x for x in pca_normalized_corr.columns if x.startswith("PCA")],
            # yticklabels=[x for x in pca_normalized_corr.columns if not x.startswith("PCA")]
            xticklabels=pca_normalized_corr.columns,
            yticklabels=pca_normalized_corr.columns
                 )
    plt.savefig("./Images/PCA_Normalized_Correlation_Heatmap_%s.png" % filename, format='png')
    plt.savefig("./Images/PCA_Normalized_Correlation_Heatmap_%s.pdf"  % filename, format='pdf')
    plt.close()

    if normalize:
        print("Scatter of PCA BiPlot Visualization")
        fig = plt.figure(figsize=(20, 20))
        fig.set_size_inches(20.5, 20.5, forward=True)
        ax = fig.add_subplot(1, 1, 1)

    #markers = ["o", "D"]

        for class_ix in zip(
                    classes):
                ax.scatter(x_train_projected[np.where(y_train == class_ix[0]), 0],
                           x_train_projected[np.where(y_train == class_ix[0]), 1],
                           #marker=marker, color=color, edgecolor='whitesmoke',
                           linewidth='1', alpha=0.9, label= class_ix[0])
                #ax.legend(loc='best')

        plt.title(
            "Scatter plot projections on the "
            "2 first principal components")
        plt.xlabel("Principal axis 1 - Explains %.1f %% of the variance" % (
                pca.explained_variance_ratio_[0] * 100.0))
        plt.ylabel("Principal axis 2 - Explains %.1f %% of the variance" % (
                pca.explained_variance_ratio_[1] * 100.0))

        print("Annotations for Datapoints")
        for i, txt in enumerate(labels):
            ax.annotate(txt, (x_train_projected[i,0], x_train_projected[i,1]))

        plt.savefig("./Images/pca_biplot_%s.pdf" % filename, format='pdf')
        plt.savefig("Images/pca_biplot_%s.png" % filename, format='png')
        #plt.show()
        plt.close()
    print("Recombine DataFrame with Target and PCA")
    results_dict = dict()
    results_dict["Orig_DF"] = raw_df
    results_dict["Norm_DF"] = pd.concat([y_train.reset_index(drop = True), x_train_normalized], axis = 1)
    results_dict["PCA_1_2_DF"] =  pd.concat([y_train.reset_index(drop = True),pd.DataFrame(x_train_projected).iloc[:,0:2].reset_index(drop = True),x_train_normalized.reset_index(drop = True)], axis = 1)
    results_dict["PCA_DF"] = pd.concat(
        [y_train.reset_index(drop=True), pd.DataFrame(x_train_projected).reset_index(drop=True),
         x_train_normalized.reset_index(drop=True)], axis=1)
    results_dict["PCA_Corr"] = pca_normalized_corr
    return(results_dict)





def preprocess_load_wrapper(df,keyname = None,keyname_out = None,datatype = None,collist = None,
                            server = "mysql+pymysql://{user}:{pw}@localhost/{db}", user = "root", pw="", db="STR",
                            interpolate = True,interpolatemethod = 'linear',if_exists="replace", fillnan = False, upload_df_db = False):
    if datatype not in ["X","X_tminus1","Y","Y_tminus1","CapRatios","CapRatios_tminus1","XYCap", "XYCap_tminus1", "X_Y_Cap","PCA_DF"] + ["zmacro_domestic", "zmacro_international", "zmacro","zmicro","sectoridx", "sbidx"]:
        print("Datatype:", datatype,"Not Found in Preprocess List")
        return
    if keyname is not None and isinstance(df,dict):
        df = df[keyname]
    elif keyname is None and isinstance(df,dict):
        print("Need Keyname for Dictionary")
        return
    else:
        print("No Dict Found")

    if keyname_out is None:
        tbl_name = keyname
        keyname_out = keyname

    if collist is not None and isinstance(collist,list):
        df = df[collist]
    elif collist == "PCA":
        collist = ['ReportingDate'] + [x for x in df.keys() if isinstance(x,int) ]
        df = df[collist]
    else:
        print("No Collist")
    print("Running Preprocessing")
    preprocess_tmp =      preprocess_loadMySQL(df, datatype= datatype
                             , server= server, user= user, pw=pw, db=db
                             , tbl_name=tbl_name, if_exists=if_exists, upload_df_db = upload_df_db)
    if interpolate:
        if interpolatemethod is None:
            interpolatemethod = "linear"

        preprocess_tmp =  preprocess_tmp.interpolate(method = interpolatemethod, order = 2)
    if fillnan:
        preprocess_tmp  = preprocess_tmp.fillna(method = "ffill").fillna(method = "bfill")

    return preprocess_tmp




def PandasNormalize(df, exclude_columns = ["Date"], type = "minmax"):
    #df = Z_micro_tmp
    #exclude_columns = ["ReportingDate"]
    print('Create subset without excluded columns')
    df_tmp = df[df.columns.difference(exclude_columns)]
    df_tmp = df_tmp.astype(float)
    print('Normalization Process')
    if type == "minmax":
        df_norm = (df_tmp - df_tmp.mean()) / (df_tmp.max() - df_tmp.min())
    elif type == "std":
        df_norm=(df_tmp-df_tmp.mean())/df_tmp.std()
    elif type == "scilearn":
        scale = StandardScaler()
        # print("Preparing Normalization DataSet")
        x_train = df_tmp
        # #print("Dropping Target column:%s" % target)
        # columns = x_train.columns
        columns = x_train.columns
        # print("Normalizing DataFrame")
        x_train = scale.fit_transform(x_train)
        # #x_train = scale.fit_transform(x_train)
        # print("Creating Normalized DataFrame")
        x_train_normalized = pd.DataFrame(x_train)  # It is required to have X converted into a data frame to later plotting need
        # print("Reapplying Column Names")
        x_train_normalized.columns = columns
        df_norm = x_train_normalized
    else:
        return("No type Selected")
    print("Recombining Excluded Columns")
    df_norm = pd.concat([df[exclude_columns].reset_index(drop = True),df_norm.reset_index(drop=True)], axis = 1)
    return(df_norm)





def preprocess_loadMySQL(BankPerf, datatype = "X"
                         ,server = "mysql+pymysql://{user}:{pw}@localhost/{db}", user="root", pw="", db="STR"
                         , tbl_name = "TempDF", if_exists = "replace", upload_df_db = False):

        if datatype in ["X","X_tminus1","Y","Y_tminus1","CapRatios","CapRatios_tminus1","XYCap", "XYCap_tminus1", "X_Y_Cap","PCA_DF"]:
            if not BankPerf.ReportingDate.apply(lambda x: x.split(" ")[1][0]).unique() == "Q":
                print("Converting Date to YYYY Q# format")
                BankPerf.ReportingDate = BankPerf.ReportingDate.dt.year.astype(str) + " Q" + BankPerf.ReportingDate.dt.quarter.astype(str)

            if "Other items:Dividends " in BankPerf.keys():
                print("Renmaing Column Other items:Dividends ")
                BankPerf = BankPerf.rename({"Other items:Dividends ": "Other items:Dividends"})
            print("Subsetting Columns for ", datatype)

            if datatype == "X":
                BankPerf = BankPerf[BankPerf.columns[pd.Series(BankPerf.columns).str.endswith(("RSSD_ID",
                                                               "ReportingDate",
                                                               "Loans categories:Commercial & industrial_Covas",
                                                               "Loans categories:Construction & land development",
                                                               "Loans categories:Multifamily real estate",
                                                               "Loans categories:Nonfarm nonresidential CRE_Covas",
                                                               "Loans categories:Home equity lines of credit",
                                                               "Loans categories:Residential real estate (excl. HELOCs)_Covas",
                                                               "Loans categories:Credit card",
                                                               "Loans categories:Consumer (excl. credit card)_Covas"))]]

            elif datatype == "X_tminus1":
                BankPerf = BankPerf[
                    BankPerf.columns[
                        pd.Series(BankPerf.columns).str.startswith((
                                                                     "RSSD_ID",
                                                                     "ReportingDate",
                                                                     "Loans categories:Commercial & industrial_Covas_t-1",
                                                                     "Loans categories:Construction & land development_t-1",
                                                                     "Loans categories:Multifamily real estate_t-1",
                                                                     "Loans categories:Nonfarm nonresidential CRE_Covas_t-1",
                                                                     "Loans categories:Home equity lines of credit_t-1",
                                                                     "Loans categories:Residential real estate (excl. HELOCs)_Covas_t-1",
                                                                     "Loans categories:Credit card_t-1",
                                                                             "Loans categories:Consumer (excl. credit card)_Covas_t-1"))]]
            elif datatype == "XYCap":
                BankPerf = BankPerf[
                    BankPerf.columns[
                        pd.Series(BankPerf.columns).str.endswith((
                                                                                           "RSSD_ID",
                                                                                           "ReportingDate",
                                                                                           "Chargeoffs",
                                                                                           "Recoveries",
                                                                                           "Net income(loss)",
                                                                                           "Less:Cash dividends on perp perf stock",
                                                                                           "Total Equity_1",
                                                                                           "Total Equity_2",
                                                                                           "Other items:Book equity",
                                                                                           "Other items:Dividends",
                                                                                           "Other items:Stock purchases",
                                                                                           "Other items:Risk-weighted assets",
                                                                                           "Other items:Tier 1 common equity",
                                                                                           "Other items:T1CR",
                                                                                           'Other items: CapRatios_T1RiskCR_1',
                                                                                           'Other items: CapRatios_T1RiskCR_2',
                                                                                           'Other items: CapRatios_T1RiskCR_3',
                                                                                           'Other items: CapRatios_TotalRiskCR_1',
                                                                                           'Other items: CapRatios_TotalRiskCR_2',
                                                                                           'Other items: CapRatios_TotalRiskCR_3',
                                                                                           'Other items: CapRatios_T1LR_1 ',
                                                                                           'Other items: CapRatios_T1LR_2',
                                                                                           'Other items: CapRatios_CET1CR_1 ',
                                                                                           'Other items: CapRatios_CET1CR_2'
                                                                                                                            ))]]
            elif datatype == "XYCap_tminus1":
                BankPerf = BankPerf[
                    BankPerf.columns[pd.Series(
                        BankPerf.columns).str.startswith((
                                                           "RSSD_ID",
                                                           "ReportingDate",
                                                           "Chargeoffs_t-1",
                                                           "Recoveries_t-1",
                                                           "Net income(loss)_t-1",
                                                           "Less:Cash dividends on perp perf stock_t-1",
                                                           "Total Equity_1_t-1",
                                                           "Total Equity_2_t-1",
                                                           "Other items:Book equity_t-1",
                                                           "Other items:Dividends _t-1",
                                                           "Other items:Stock purchases_t-1",
                                                           "Other items:Risk-weighted assets_t-1",
                                                           "Other items:Tier 1 common equity_t-1",
                                                           "Other items:T1CR_t-1"))]]
            elif datatype == "Y":
                BankPerf = BankPerf[BankPerf.columns[(pd.Series(BankPerf.columns).str.startswith(
                    ("RSSD_ID", "ReportingDate", "ncoR:", "ppnrRatio:"))) & (~pd.Series(BankPerf.columns).str.endswith("_t-1"))]]
            elif datatype == "Y_minus1":
                BankPerf = BankPerf[BankPerf.columns[(pd.Series(BankPerf.columns).str.startswith(
                    ("RSSD_ID", "ReportingDate", "ncoR:", "ppnrRatio:"))) & (pd.Series(BankPerf.columns).str.endswith("_t-1"))]]
            elif datatype == "X_Y_Cap":
                BankPerf =BankPerf[BankPerf.columns[pd.Series(BankPerf.columns).str.endswith(("RSSD_ID",
                                                                                               "ReportingDate",
                                                                                               "Loans categories:Commercial & industrial_Covas",
                                                                                               "Loans categories:Construction & land development",
                                                                                               "Loans categories:Multifamily real estate",
                                                                                               "Loans categories:Nonfarm nonresidential CRE_Covas",
                                                                                               "Loans categories:Home equity lines of credit",
                                                                                               "Loans categories:Residential real estate (excl. HELOCs)_Covas",
                                                                                               "Loans categories:Credit card",
                                                                                               "Loans categories:Consumer (excl. credit card)_Covas"))]
                                                                                            | BankPerf.columns[(pd.Series(BankPerf.columns).str.startswith(
                                                                                            ("RSSD_ID", "ReportingDate", "ncoR:", "ppnrRatio:"))) & (~pd.Series(BankPerf.columns).str.endswith("_t-1"))]

                                                                                            | BankPerf.columns[pd.Series(BankPerf.columns).str.endswith(("Chargeoffs",
                                                                                                                                                     "Recoveries",
                                                                                                                                                     "Net income(loss)",
                                                                                                                                                     "Less:Cash dividends on perp perf stock",
                                                                                                                                                     "Total Equity_1",
                                                                                                                                                     "Total Equity_2",
                                                                                                                                                     "Other items:Book equity",
                                                                                                                                                     "Other items:Dividends",
                                                                                                                                                     "Other items:Stock purchases",
                                                                                                                                                     "Other items:Risk-weighted assets",
                                                                                                                                                     "Other items:Tier 1 common equity",
                                                                                                                                                     "Other items:T1CR",
                                                                                                                                                     'Other items:Net Charge Offs',
                                                                                                                                                     'Other items:QtrAvgTotalLoans',
                                                                                                                                                     'Other items: QtrAvgLoansLeases_1 ',
                                                                                                                                                     'Other items: QtrAvgLoansLeases_2',
                                                                                                                                                     'Other items: QtrAvgLoansLeases_3',
                                                                                                                                                     'Other items: QtrAvgTotalLoans ',
                                                                                                                                                     'Other items: CapRatios_T1RiskCR_1',
                                                                                                                                                     'Other items: CapRatios_T1RiskCR_2',
                                                                                                                                                     'Other items: CapRatios_T1RiskCR_3',
                                                                                                                                                     'Other items: CapRatios_TotalRiskCR_1',
                                                                                                                                                     'Other items: CapRatios_TotalRiskCR_2',
                                                                                                                                                     'Other items: CapRatios_TotalRiskCR_3',
                                                                                                                                                     'Other items: CapRatios_T1LR_1 ',
                                                                                                                                                     'Other items: CapRatios_T1LR_2',
                                                                                                                                                     'Other items: CapRatios_CET1CR_1 ',
                                                                                                                                                    'Other items: CapRatios_CET1CR_2',
                                                                                                                                                    'Other items:Chargeoffs on Commercial and Industrial Loans',
                                                                                                                                                     'Other items:Chargeoffs on Construction and Land Development Loans',
                                                                                                                                                     'Other items:Chargeoffs on Residential Real Estate_1_4_Family Loans',
                                                                                                                                                     'Other items:Chargeoffs on Finance CRE_C&L Loans',
                                                                                                                                                     'Other items:Chargeoffs on HELOC Loans',
                                                                                                                                                     'Other items:Chargeoffs on (Nonfarm) nonresidential CRE Loans',
                                                                                                                                                     'Other items:Chargeoffs on Residential Real Estate_1_4_Family_Other_Loans',
                                                                                                                                                     'Other items:Chargeoffs on Credit Card Loans',
                                                                                                                                                     'Other items:Chargeoffs on Consumer Loans',
                                                                                                                                                     'Other items:Chargeoffs on Consumer_Other_Loans'
                                                                                                                                                         ))]]


                print("Coalescing Captial Ratios")
                print("T1 Capital Ratio")
                BankPerf['Other items: CapRatios_T1RiskCR_coalesced'] =  BankPerf[
                    'Other items: CapRatios_T1RiskCR_3'].combine_first(
                    BankPerf['Other items: CapRatios_T1RiskCR_1']).combine_first(
                    BankPerf['Other items: CapRatios_T1RiskCR_2'])

                print("T1 Capital Ratio")
                BankPerf['Other items: CapRatios_TotalRiskCR_coalesced'] = BankPerf[
                    'Other items: CapRatios_TotalRiskCR_3'].combine_first(
                    BankPerf['Other items: CapRatios_TotalRiskCR_1']).combine_first(
                    BankPerf['Other items: CapRatios_TotalRiskCR_2'])

                print("T1 Leverage Ratio")
                BankPerf['Other items: CapRatios_T1LR_coalesced'] = BankPerf[
                    'Other items: CapRatios_T1LR_2'].combine_first(
                    BankPerf['Other items: CapRatios_T1LR_1 '])

                print("T1 Common Equity Ratio")
                BankPerf['Other items: CapRatios_CET1CR_coalesced'] = BankPerf[
                    'Other items: CapRatios_CET1CR_2'].combine_first(
                    BankPerf['Other items: CapRatios_CET1CR_1 '])

                print("Calculation Net Charge Offs")
                BankPerf["Other items:Net Charge Offs_Calced"] = BankPerf['Chargeoffs'] - BankPerf['Recoveries']
                print("Coalescing Net Charge Offs")
                BankPerf["Other items:Net Charge Offs_coalesced"] = BankPerf["Other items:Net Charge Offs_Calced"].combine_first(BankPerf['Other items:Net Charge Offs'])

                print("Coalescing Quarterly average loans and leases")
                BankPerf["Other items:QtrAvgTotalLoans_coalesced"] = BankPerf["Other items:QtrAvgTotalLoans"].combine_first(BankPerf['Other items: QtrAvgLoansLeases_3']).combine_first(BankPerf['Other items: QtrAvgLoansLeases_1 ']).combine_first(BankPerf['Other items: QtrAvgLoansLeases_2']).combine_first(BankPerf['Other items: QtrAvgTotalLoans '])

                print("Calculating Net Charge Offs Ratio")
                BankPerf["Other items:Net Charge Offs_Ratio"] = BankPerf["Other items:Net Charge Offs_Calced"] / BankPerf["Other items:QtrAvgTotalLoans_coalesced"]
                print("Calculating Loan Charge Offs Ratio")
                BankPerf["Other items:Loan Charge Offs_Ratio"] = BankPerf['Chargeoffs'] / BankPerf["Other items:QtrAvgTotalLoans_coalesced"]

                print("Calculating Charge Off Rates")
                colist = [x for x in list(BankPerf.keys()) if x.startswith("Other items:Chargeoffs")]
                for col_name in colist:
                    print("_".join([col_name,"Rate"]))
                    BankPerf["_".join([col_name,"Rate"])] =  BankPerf[col_name] / BankPerf["Other items:QtrAvgTotalLoans_coalesced"]
            elif datatype == "PCA_DF":
                print("Combine X, Y, NCO, CapRatios")
                BankPerf = BankPerf[BankPerf.columns[(pd.Series(BankPerf.columns).str.startswith(
                    ("RSSD_ID", "ReportingDate", "ncoR:", "ppnrRatio:")))
                                                      |  pd.Series(BankPerf.columns).str.endswith(("Loans categories:Commercial & industrial_Covas",
                                                               "Loans categories:Construction & land development",
                                                               "Loans categories:Multifamily real estate",
                                                               "Loans categories:Nonfarm nonresidential CRE_Covas",
                                                               "Loans categories:Home equity lines of credit",
                                                               "Loans categories:Residential real estate (excl. HELOCs)_Covas",
                                                               "Loans categories:Credit card",
                                                               "Loans categories:Consumer (excl. credit card)_Covas"))
                                                      | (pd.Series(BankPerf.columns).str.endswith(("Chargeoffs",
                                                                                                   "Recoveries",
                                                                                             "Net income(loss)",
                                                                                             "Less:Cash dividends on perp perf stock",
                                                                                             "Total Equity_1",
                                                                                             "Total Equity_2",
                                                                                             "Other items:Book equity",
                                                                                             "Other items:Dividends",
                                                                                             "Other items:Stock purchases",
                                                                                             "Other items:Risk-weighted assets",
                                                                                             "Other items:Tier 1 common equity",
                                                                                             "Other items:T1CR",
                                                                                             'Other items:Net Charge Offs',
                                                                                             'Other items:QtrAvgTotalLoans',
                                                                                             'Other items: QtrAvgLoansLeases_1 ',
                                                                                             'Other items: QtrAvgLoansLeases_2',
                                                                                             'Other items: QtrAvgLoansLeases_3',
                                                                                             'Other items: QtrAvgTotalLoans ',
                                                                                             'Other items: CapRatios_T1RiskCR_1',
                                                                                             'Other items: CapRatios_T1RiskCR_2',
                                                                                             'Other items: CapRatios_T1RiskCR_3',
                                                                                             'Other items: CapRatios_TotalRiskCR_1',
                                                                                             'Other items: CapRatios_TotalRiskCR_2',
                                                                                             'Other items: CapRatios_TotalRiskCR_3',
                                                                                             'Other items: CapRatios_T1LR_1 ',
                                                                                             'Other items: CapRatios_T1LR_2',
                                                                                             'Other items: CapRatios_CET1CR_1 ',
                                                                                            'Other items: CapRatios_CET1CR_2',
                                                                                            'Other items:Chargeoffs on Commercial and Industrial Loans',
                                                                                             'Other items:Chargeoffs on Construction and Land Development Loans',
                                                                                             'Other items:Chargeoffs on Residential Real Estate_1_4_Family Loans',
                                                                                             'Other items:Chargeoffs on Finance CRE_C&L Loans',
                                                                                             'Other items:Chargeoffs on HELOC Loans',
                                                                                             'Other items:Chargeoffs on (Nonfarm) nonresidential CRE Loans',
                                                                                             'Other items:Chargeoffs on Residential Real Estate_1_4_Family_Other_Loans',
                                                                                             'Other items:Chargeoffs on Credit Card Loans',
                                                                                             'Other items:Chargeoffs on Consumer Loans',
                                                                                             'Other items:Chargeoffs on Consumer_Other_Loans')))
                                                      & (~pd.Series(BankPerf.columns).str.endswith("_t-1"))]]


                print("Coalescing Captial Ratios")
                print("T1 Capital Ratio")
                BankPerf['Other items: CapRatios_T1RiskCR_coalesced'] =  BankPerf[
                    'Other items: CapRatios_T1RiskCR_3'].combine_first(
                    BankPerf['Other items: CapRatios_T1RiskCR_1']).combine_first(
                    BankPerf['Other items: CapRatios_T1RiskCR_2'])

                print("T1 Capital Ratio")
                BankPerf['Other items: CapRatios_TotalRiskCR_coalesced'] = BankPerf[
                    'Other items: CapRatios_TotalRiskCR_3'].combine_first(
                    BankPerf['Other items: CapRatios_TotalRiskCR_1']).combine_first(
                    BankPerf['Other items: CapRatios_TotalRiskCR_2'])

                print("T1 Leverage Ratio")
                BankPerf['Other items: CapRatios_T1LR_coalesced'] = BankPerf[
                    'Other items: CapRatios_T1LR_2'].combine_first(
                    BankPerf['Other items: CapRatios_T1LR_1 '])

                print("T1 Common Equity Ratio")
                BankPerf['Other items: CapRatios_CET1CR_coalesced'] = BankPerf[
                    'Other items: CapRatios_CET1CR_2'].combine_first(
                    BankPerf['Other items: CapRatios_CET1CR_1 '])

                print("Calculation Net Charge Offs")
                BankPerf["Other items:Net Charge Offs_Calced"] = BankPerf['Chargeoffs'] - BankPerf['Recoveries']
                print("Coalescing Net Charge Offs")
                BankPerf["Other items:Net Charge Offs_coalesced"] = BankPerf["Other items:Net Charge Offs_Calced"].combine_first(BankPerf['Other items:Net Charge Offs'])

                print("Coalescing Quarterly average loans and leases")
                BankPerf["Other items:QtrAvgTotalLoans_coalesced"] = BankPerf["Other items:QtrAvgTotalLoans"].combine_first(BankPerf['Other items: QtrAvgLoansLeases_3']).combine_first(BankPerf['Other items: QtrAvgLoansLeases_1 ']).combine_first(BankPerf['Other items: QtrAvgLoansLeases_2']).combine_first(BankPerf['Other items: QtrAvgTotalLoans '])

                print("Calculating Net Charge Offs Ratio")
                BankPerf["Other items:Net Charge Offs_Ratio"] = BankPerf["Other items:Net Charge Offs_Calced"] / BankPerf["Other items:QtrAvgTotalLoans_coalesced"]
                print("Calculating Loan Charge Offs Ratio")
                BankPerf["Other items:Loan Charge Offs_Ratio"] = BankPerf['Chargeoffs'] / BankPerf["Other items:QtrAvgTotalLoans_coalesced"]

                print("Calculating Charge Off Rates")
                colist1 = [x for x in list(BankPerf.keys()) if x.startswith("Other items:Chargeoffs") if not x.endswith("_Rate")]
                print(colist1)
                for col_name1 in colist1:
                    print(col_name1)
                    print("Calculating:","_".join([col_name1,"Rate"]))
                    BankPerf["_".join([col_name1,"Rate"])] =  BankPerf[col_name1] / BankPerf["Other items:QtrAvgTotalLoans_coalesced"]
        else:

            if datatype.lower() in ["zmacro_domestic", "zmacro_international", "zmacro","zmicro","sectoridx", "sbidx"]:
                if "Scenario Name" in BankPerf.keys():
                    BankPerf = BankPerf.drop("Scenario Name", axis=1)
                if "Scenario Name_x" in BankPerf.keys():
                    BankPerf = BankPerf.drop("Scenario Name_x", axis=1)
                if "Date" in BankPerf.keys():
                    # if not BankPerf.Date.apply(lambda x: x.str.split(" ")[1][0]).unique() == "Q":
                    BankPerf["Date"] = pd.to_datetime(BankPerf.Date).dt.year.astype(str) + " Q" + pd.to_datetime(
                        BankPerf.Date).dt.quarter.astype(str)




        print("Saving Object to file")
        BankPerf.to_csv("./" + datatype +".csv" , sep=",", index=False)
        print(BankPerf.describe().transpose())
        print(BankPerf.keys())

        if upload_df_db:
            tmp_df = BankPerf
            print("Creating Connection to MySQL server")
            engine = create_engine(server
                                   .format(user=user,
                                           pw=pw,
                                           db=db))
            print("Upload dataframe to Database")
            tmp_df.to_sql(name=tbl_name, con=engine,  if_exists= if_exists, index=False, index_label=None)
        return(BankPerf)





init_ST = StressTestData()



#Bank Data Preprocesing
BankPerf = StressTestData().X_Y_bankingchar_perf_process(groupfunction=np.mean, groupby=["RSSD_ID", "ReportingDate"],
                                     groupagg_col='Other items:Total Assets',
                                     RSSD_DateParam=["1976-01-01", "2018-01-01"],
                                     ReportingDateParam=["1976-01-01", "2018-01-01"], RSSDList_len=None, dropdup=False,
                                     replace_nan=False, combinecol=False, merge = False, reducedf = False,skip_prefix = "WRDS_Covas_BankPerf_CallReport",  replace_dict={"BankPerf_FR9YC_varlist": {
                                         "df_keyname": "WRDS_Covas_BankPerf",
                                         "calcol": "Report: FR Y-9C",
                                         "BHCK892": "BHCKC892",
                                         "+": ",",
                                         "-": ",",
                                         "(": "",
                                         ")": "",
                                         ". . .": "",
                                         ",  ,": ",",
                                         "+  +": "+",
                                         "-  -": "-"},
                                         "BankPerf_FFIEC_varlist": {
                                             "df_keyname": "WRDS_Covas_CallReport_Combined",
                                             "calcol": "Report: FFIEC 031/041",
                                             "RCFD560": "RCFDB560",
                                             "RIADK129": "RIAD4639",
                                             "RIADK205": "RIAD4657",
                                             "RIADK133": "RIAD4609",
                                             "RIADK206": "RIAD4667",
                                             "+": ",",
                                             "-": ",",
                                             "(": "",
                                             ")": "",
                                             ". . .": "",
                                             ",  ,": ",",
                                             "+  +": "+",
                                             "-  -": "-"
                                         },
                                     }, RSSD_Subset = True, Y_calc= True,concecutiveqtrs = 10)


#Workspace




#TODO: Functionalize this part. Preprocessing for transformations.

#TODO:Filter for RSSD_IDs that actually exisit in 2016.
RSSD_IDS = BankPerf["BankPerf_ConsecutiveReduced_XYcalc_Subset_BankPerf"]['RSSD_ID'][BankPerf["BankPerf_ConsecutiveReduced_XYcalc_Subset_BankPerf"]['ReportingDate'].isin(['2016-12-31'])].unique()
# This way , not to consider nan or filled values, but actual values.
BankPerf["BankPerf_ConsecutiveReduced_XYcalc_Subset_BankPerf_2016Q4"] = \
    BankPerf["BankPerf_ConsecutiveReduced_XYcalc_Subset_BankPerf"][BankPerf["BankPerf_ConsecutiveReduced_XYcalc_Subset_BankPerf"]['RSSD_ID'].isin(RSSD_IDS)]
#Joins and creates dataframe to dates avaialble.

RepordingDate_Df = pd.DataFrame(BankPerf["BankPerf_ConsecutiveReduced_XYcalc_Subset_BankPerf_2016Q4"]["ReportingDate"].unique())
RepordingDate_Df = RepordingDate_Df.rename({0:"ReportingDate"}, axis = 1)
RepordingDate_Df["key"] = 0
BankIds_Df = pd.DataFrame(BankPerf["BankPerf_ConsecutiveReduced_XYcalc_Subset_BankPerf_2016Q4"]["RSSD_ID"].unique())
BankIds_Df = BankIds_Df.rename({0:"RSSD_ID"}, axis = 1)
BankIds_Df["key"] = 0
BankID_Date_Ref = RepordingDate_Df.assign(foo=1).merge(BankIds_Df.assign(foo=1), on = "foo",how = "outer" ).drop(["foo","key_x","key_y"],1)
BankPerf["BankPerf_ConsecutiveReduced_XYcalc_ReportingDated"] = BankID_Date_Ref.merge(BankPerf["BankPerf_ConsecutiveReduced_XYcalc_Subset_BankPerf_2016Q4"], left_on = ["ReportingDate","RSSD_ID"], right_on = ["ReportingDate","RSSD_ID"], how = "left")


BankPerf["BankPerf_ConsecutiveReduced_XYcalc_ReportingDated"]["ReportingDate"] =  pd.to_datetime(BankPerf["BankPerf_ConsecutiveReduced_XYcalc_ReportingDated"].ReportingDate).dt.year.astype(str) + " Q" + pd.to_datetime(BankPerf["BankPerf_ConsecutiveReduced_XYcalc_ReportingDated"].ReportingDate).dt.quarter.astype(str)


Preprocess_Dict = dict()
Preprocess_Dict['X'] = preprocess_loadMySQL(BankPerf["BankPerf_ConsecutiveReduced_XYcalc_ReportingDated"], datatype = "X"
                         ,server = "mysql+pymysql://{user}:{pw}@localhost/{db}", user="root", pw="", db="STR"
                         , tbl_name = "X", if_exists = "replace")#.interpolate(method = 'linear')




#outlier_preprocess(Preprocess_Dict['X'], quantiles=[0,1], makezero = 1e-10).round(2).describe()

Preprocess_Dict['X'] = outlier_preprocess(Preprocess_Dict['X'], quantiles=[0,1], makezero = 1e-10).round(2)

#Normalize and interpolate
Preprocess_Dict['X']  = PandasNormalize(Preprocess_Dict['X'], exclude_columns = ["RSSD_ID", "ReportingDate"], type = "scilearn").interpolate(method = 'linear', order = 2)


Preprocess_Dict['Y'] = preprocess_loadMySQL(BankPerf["BankPerf_ConsecutiveReduced_XYcalc_ReportingDated"], datatype = "Y"
                         ,server = "mysql+pymysql://{user}:{pw}@localhost/{db}", user="root", pw="", db="STR"
                         , tbl_name = "Y", if_exists = "replace")#.interpolate(method = 'linear')


# test = \
#Preprocess_Dict['Y'][["RSSD_ID","ReportingDate"] + [x for x in Preprocess_Dict['Y'].keys() if x.startswith(('ppnrRatio:','ncoR:'))]] = Preprocess_Dict['Y'][["RSSD_ID","ReportingDate"] + [x for x in Preprocess_Dict['Y'].keys() if x.startswith(('ppnrRatio:','ncoR:'))]]
#outlier_preprocess(Preprocess_Dict['Y'][["RSSD_ID","ReportingDate"] + [x for x in Preprocess_Dict['Y'].keys() if x.startswith(('ppnrRatio:','ncoR:'))]],quantiles=[0,.9736], makezero = 1e-9, verbose = False).round(4)


Preprocess_Dict['Y'][[x for x in Preprocess_Dict['Y'] if x.startswith('ppnrRatio:')]] = Preprocess_Dict['Y'][[x for x in Preprocess_Dict['Y'] if x.startswith('ppnrRatio:')]] *.01
Preprocess_Dict['Y'][[x for x in Preprocess_Dict['Y'] if x.startswith('ncoR:')]] = Preprocess_Dict['Y'][[x for x in Preprocess_Dict['Y'] if x.startswith('ncoR:')]] * .01


Preprocess_Dict['Y'][["RSSD_ID","ReportingDate"] + [x for x in Preprocess_Dict['Y'].keys() if x.startswith(('ppnrRatio:','ncoR:'))]] = outlier_preprocess(Preprocess_Dict['Y'][["RSSD_ID","ReportingDate"] + [x for x in Preprocess_Dict['Y'].keys() if x.startswith(('ppnrRatio:','ncoR:'))]],quantiles=[0,.9736], makezero = 1e-9, verbose = False)[["RSSD_ID","ReportingDate"] + [x for x in Preprocess_Dict['Y'].keys() if x.startswith(('ppnrRatio:','ncoR:'))]].round(4)

# Preprocess_Dict['Y'][[x for x in Preprocess_Dict['Y'] if x.startswith('ppnrRatio:')]].describe()
#  outlier_preprocess(Preprocess_Dict['Y'][["RSSD_ID","ReportingDate"] + [x for x in Preprocess_Dict['Y'] if x.startswith('ppnrRatio:') if not x.endswith("ppnrRatio:Noninterest expense")]]
#                                          ,quantiles = [0,.9875], makezero= 1e-30)[[x for x in Preprocess_Dict['Y'] if x.startswith('ppnrRatio:') if not x.endswith("ppnrRatio:Noninterest expense")]].round(2).describe()
#
# outlier_preprocess(Preprocess_Dict['Y'][["RSSD_ID", "ReportingDate"] + [x for x in Preprocess_Dict['Y']if x.endswith("ppnrRatio:Noninterest expense")]]
#                                         , quantiles = [0, .9735], makezero = 1e-30)[[x for x in Preprocess_Dict['Y'] if x.endswith("ppnrRatio:Noninterest expense")]].round(2).describe()


#Preprocess_Dict['Y'][[x for x in Preprocess_Dict['Y'] if x.startswith('ppnrRatio:') if not x.endswith("ppnrRatio:Noninterest expense")]] = outlier_preprocess(Preprocess_Dict['Y'][["RSSD_ID","ReportingDate"] + [x for x in Preprocess_Dict['Y'] if x.startswith('ppnrRatio:') if not x.endswith("ppnrRatio:Noninterest expense")]],quantiles = [0,.9875], makezero= 1e-30)[[x for x in Preprocess_Dict['Y'] if x.startswith('ppnrRatio:') if not x.endswith("ppnrRatio:Noninterest expense")]].round(2)


#Preprocess_Dict['Y'][[x for x in Preprocess_Dict['Y'] if x.endswith("ppnrRatio:Noninterest expense")]] = outlier_preprocess(Preprocess_Dict['Y'][["RSSD_ID","ReportingDate"] + [x for x in Preprocess_Dict['Y'] if x.endswith("ppnrRatio:Noninterest expense")]],quantiles = [0,.9735], makezero= 1e-30)[[x for x in Preprocess_Dict['Y'] if x.endswith("ppnrRatio:Noninterest expense")]].round(2)


#Preprocess_Dict['Y'][[x for x in Preprocess_Dict['Y'] if x.startswith('ncoR:')]].describe()

#Preprocess_Dict['Y'][[x for x in Preprocess_Dict['Y'] if x.startswith('ncoR:')]] = outlier_preprocess(Preprocess_Dict['Y'][["RSSD_ID","ReportingDate"] + [x for x in Preprocess_Dict['Y'] if x.startswith('ncoR:')]]
#                        ,quantiles = [0,1], makezero= 1e-15)[[x for x in Preprocess_Dict['Y'] if x.startswith('ncoR:')]].round(2)

#Preprocess_Dict['Y'].describe()




Preprocess_Dict['X_Y_Cap'] = preprocess_loadMySQL(BankPerf["BankPerf_ConsecutiveReduced_XYcalc_ReportingDated"], datatype = "X_Y_Cap"
                         ,server = "mysql+pymysql://{user}:{pw}@localhost/{db}", user="root", pw="", db="STR"
                         , tbl_name = "X_Y_Cap", if_exists = "replace")#.interpolate(method = 'linear')

collist = [ x for x in list(Preprocess_Dict['X_Y_Cap'].keys()) if x.startswith("Other items: CapRatios") if not x.endswith("_t-1") if x.endswith("_coalesced") ]
Preprocess_Dict['CapRatios'] = Preprocess_Dict['X_Y_Cap'][["RSSD_ID", "ReportingDate"] + collist]
Preprocess_Dict['CapRatios'][collist] = Preprocess_Dict['CapRatios'][collist] * .01
Preprocess_Dict['CapRatios'][collist].describe()
#outlier_preprocess(Preprocess_Dict['CapRatios'],quantiles = [.086,.832],verbose = False).round(2).describe()

Preprocess_Dict['CapRatios'] = outlier_preprocess(Preprocess_Dict['CapRatios'],quantiles = [.086,.832],verbose = False).round(2)

#Preprocess_Dict['CapRatios'] .keys()
#Preprocess_Dict['Y'].keys()
Preprocess_Dict["Y_Cap"] = Preprocess_Dict['Y'].merge(Preprocess_Dict['CapRatios'][["RSSD_ID","ReportingDate",'Other items: CapRatios_T1RiskCR_coalesced']])
#Preprocess_Dict['CapRatios']  = Preprocess_Dict['CapRatios'][["RSSD_ID","ReportingDate","Other items: CapRatios_T1RiskCR_coalesced"]]
#Preprocess_Dict["Y_Cap"].round(4).describe()


#colist  = [ x for x in list(Preprocess_Dict['X_Y_Cap'].keys()) if x.startswith("Other items:Chargeoffs") if x.endswith("_Rate") if not x.endswith("_t-1") ]
# Preprocess_Dict['NCO'] = Preprocess_Dict['X_Y_Cap'][["RSSD_ID", "ReportingDate", "Other items:Net Charge Offs_Ratio",'Other items:Loan Charge Offs_Ratio']]
# #'Other items:Net Charge Offs_coalesced'
#
# Preprocess_Dict['NCO'] = outlier_preprocess(Preprocess_Dict['NCO'],quantiles = [0,.73785], makezero = 1e-20, verbose = True).round(2)
# Preprocess_Dict['NCO'].describe()




#######Visualizations
#df = Preprocess_Dict['CapRatios']

#outlier_preprocess(df, quantiles = [.15,.72]).describe()

# Visualization Code
# Preprocess_Dict['CapRatios']["ReportingDate"] = Preprocess_Dict['CapRatios']["ReportingDate"].apply(lambda x: pd.to_datetime(str(int(12/int(x.split("Q")[1]))) + "/" + x.split(" ")[0]) )
# #Plot Mean Cap Ratio
# CapitalRatios_plt = pd.pivot_table(Preprocess_Dict['CapRatios'][Preprocess_Dict['CapRatios'].columns.difference(["RSSD_ID",'Other items: CapRatios_T1LR_coalesced'])][Preprocess_Dict['CapRatios']["ReportingDate"] >= "1986-01-01"], index=["ReportingDate"], aggfunc = np.mean)
#
# CapitalRatios_plt.plot()





os.chdir("/Users/phn1x/icdm2018_research_BB/Stress_Test_Research/Loss_projections/data")
keylist = ["X","Y_Cap"]
for keyname in [x for x in Preprocess_Dict.keys() if x in keylist]:
    print(keyname)
    data, data_quarter = pd_raw_to_numpy_data(Preprocess_Dict[keyname], fulldatescombine=False,  ReportingDate_Start = "1987 Q1", ReportingDate_End = "2017 Q4")
    print(keyname, ":", data.shape, keyname, "_quarter:", data_quarter.shape)
    print("Saving objects to file")
    np.save("./data_" + keyname + ".npy", data)
    np.save("./quarter_based/data_" + keyname + "_quarter.npy", data_quarter)



# Preprocess_Dict['PCA_DF'] = preprocess_loadMySQL(BankPerf["BankPerf_ConsecutiveReduced_XYcalc_ReportingDated"], datatype = "PCA_DF"
#                          ,server = "mysql+pymysql://{user}:{pw}@localhost/{db}", user="root", pw="", db="STR"
#                          , tbl_name = "PCA_DF", if_exists = "replace")#.interpolate(method = 'linear')
#
#
#
#
# pca_colist = [x for x in Preprocess_Dict['PCA_DF'].keys().tolist() if x.endswith(("_coalesced","Rate","Ratio"))] \
# + [x for x in Preprocess_Dict['PCA_DF'].keys().tolist() if x.startswith(("ppnrRatio","ncoR","'Other items","Loans categories:")) if not  x.endswith(("t-1"))] \
# + ['ReportingDate', 'RSSD_ID','Chargeoffs', 'Recoveries','Net income(loss)', 'Less:Cash dividends on perp perf stock','Total Equity_1', 'Total Equity_2']
#



#target = 'Other items: CapRatios_T1RiskCR_coalesced'
#interpolatefill = True
#Preprocess_Dict['PCA_DF'][pca_colist]
Preprocess_Dict['PCA_DF_1'] = Preprocess_Dict['PCA_DF'][pca_colist]

Preprocess_Dict['PCA_DF_1'] = outlier_preprocess(Preprocess_Dict['PCA_DF_1'], interpolate=True, interpolate_method='polynomial')


Preprocess_One = process_raw_bankdat_PCA_Norm(Preprocess_Dict, keyname = 'PCA_DF_1', target_list =  ['Other items: CapRatios_T1RiskCR_coalesced'], n_components = 20, visualizations = False)

Preprocess_One.keys()


Preprocess_One['CapRatios'] = Preprocess_One.pop('PCA_DF_1_target')
Preprocess_One['CapRatios'] = outlier_preprocess(Preprocess_One['CapRatios'])
#Normalize only
# df_raw = Preprocess_Dict['CapRatios']
# exclude = ["ReportingDate", "RSSD_ID"]
# #print("Target:", target)
# non_target_tmp = exclude  # + [x for x in target_list if not x == target]
# print("Excluding:", non_target_tmp)
# print("First Remove Exclusions:", non_target_tmp)
# df_raw_tmp = df_raw[df_raw.columns.difference(non_target_tmp)]
#
# print('Normalize without target')
# scale = StandardScaler()
# print("Preparing Normalization DataSet")
# x_train = df_raw_tmp
# #print("Dropping Target column:%s" % target)
# columns = x_train.columns
# # columns = x_train.columns
# print("Normalizing DataFrame")
# x_train = scale.fit_transform(x_train)
# #x_train = scale.fit_transform(x_train)
# print("Creating Normalized DataFrame")
# x_train_normalized = pd.DataFrame(x_train)  # It is required to have X converted into a data frame to later plotting need
# print("Reapplying Column Names")
# x_train_normalized.columns = columns
# print("Re-Ordering Column Names")
# x_train_normalized = x_train_normalized.sort_index(axis=1)
#
#
#
#
# df_norm_df = pd.concat([df_raw[non_target_tmp].reset_index(drop=True), x_train_normalized.reset_index(drop=True)], axis=1)
#
# df_norm_df.describe()

#df_norm_df = outlier_preprocess(df_norm_df, quantiles = [.21,1])

Preprocess_Dict['CapRatios'] = df_norm_df
#Preprocess_One['CapRatios'] = df_norm_df

Preprocess_One['X_Y_NCO_pca'] = Preprocess_One.pop('PCA_DF_1_normalized_2_pca')
Preprocess_One['X_Y_NCO_all_pca'] = Preprocess_One.pop('PCA_DF_1_normalized_pca')
Preprocess_One['X_Y_NCO_norm'] = Preprocess_One.pop('PCA_DF_1_normalized')




# Preprocess_Dict['zmicro_pca_all'] = preprocess_loadMySQL(Z_micro_dict_1['PCA_DF'].loc[:,['ReportingDate',0,1]], datatype = "z_micro"
#                          ,server = "mysql+pymysql://{user}:{pw}@localhost/{db}", user="root", pw="", db="STR"
#                          , tbl_name = "Z_micro_pca", if_exists = "replace").interpolate(method = 'linear')

#df = Z_micro_dict_1['PCA_DF']

#Z_micro_dict_1['PCA_DF'][[0]].plot.density(fill = True)

#collist = ['ReportingDate'] + [x for x in df.keys() if isinstance(x,int) ]
#Preprocess_Dict['zmicro_pca_all'] = preprocess_loadMySQL(df[collist], datatype = "z_micro"
                         # ,server = "mysql+pymysql://{user}:{pw}@localhost/{db}", user="root", pw="", db="STR"
                         # , tbl_name = "Z_micro_pca", if_exists = "replace").interpolate(method = 'linear')








####Process Modalities
Z_micro = init_ST.Z_micro_process()
Z_micro["Z_Micro"]
Z_micro_dict_1 = process_raw_modality_PCA_Norm(Z_micro, keyname = "Z_Micro")
Preprocess_Dict['zmicro_pca_all'] = preprocess_load_wrapper(Z_micro_dict_1, keyname = 'PCA_DF', collist = 'PCA', datatype = "zmicro")



Z_macro = init_ST.Z_macro_process()
excludeZmacro = ['Nominal GDP growth','Nominal disposable income growth','5-year Treasury yield','House Price Index (Level)']
Z_macro_subset = Z_macro["Historic_Domestic"].columns.difference(excludeZmacro)
Z_macro["Historic_Domestic_subset"] = Z_macro["Historic_Domestic"][Z_macro_subset]

Z_macro_domestic_dict_1 = process_raw_modality_PCA_Norm(Z_macro, keyname = "Historic_Domestic_subset", raw_date_col="Date", removecols= [x for x in Z_macro["Historic_Domestic"].columns if  x.startswith('Scenario Name')],extractquarters = False, normalize = True, n_components = 12)
Preprocess_Dict['zmacro_domestic_pca_all'] = preprocess_load_wrapper(Z_macro_domestic_dict_1, keyname = 'PCA_DF', collist = 'PCA', datatype = "zmacro_domestic")



Preprocess_Dict['NCO'][["RSSD_ID","ReportingDate","Other items:Net Charge Offs_Ratio",'Other items:Loan Charge Offs_Ratio']]
df_raw = Preprocess_Dict['NCO']
#NCO_test = process_raw_bankdat_PCA_Norm(Preprocess_Dict, keyname = 'NCO',target_list='Other items:Loan Charge Offs_Ratio', n_components = 12, visualizations = False)
exclude = ["ReportingDate","RSSD_ID"]
#print("Target:", target)
non_target_tmp =  exclude #+ [x for x in target_list if not x == target]
print("Exlcuding:", non_target_tmp)
print("First Remove Exclusions:",non_target_tmp)
df_raw_tmp = df_raw[df_raw.columns.difference(non_target_tmp)]

# print('Normalize without target')
# scale = StandardScaler()
# print("Preparing Normalization DataSet")
# x_train = df_raw_tmp
# #print("Dropping Target column:%s" % target)
# #columns = x_train.drop(target, axis=1).columns
# columns = x_train.columns
# print("Normalizing DataFrame")
# #x_train = scale.fit_transform(x_train.drop(target, axis=1))  # drop the label and normalizing
# x_train = scale.fit_transform(x_train)
# print("Creating Normalized DataFrame")
# x_train_normalized = pd.DataFrame(x_train)  # It is required to have X converted into a data frame to later plotting need
# print("Reapplying Column Names")
# x_train_normalized.columns = columns
# #print("Re-Ordering Column Names")
# #x_train_normalized = x_train_normalized.sort_index(axis=1)
#
# print("PCA Dim Reduction")
# #print("Setting Y target:%s" % target)
# #y_train = df_raw[target].reset_index(drop=True)  # .apply(lambda x: x.split(' ')[0]).reset_index(drop = True)
# #y_train = df_raw["ReportingDate"].reset_index(drop = True)
#
# #print("Initializing PCA and transforming Nomarlized DataFrame with Componets:%s" % str(n_components))
# # Run PCA
# n_components = 1
# pca = PCA(n_components=n_components)
# x_train_projected = pca.fit_transform(x_train_normalized)
#
# #print("Correlation Heatmap Calculations for first two componets")
# pca_cols = ['PCA_' + str(i) for i in range(0, x_train_projected.shape[1])]
# pca_df = pd.DataFrame(x_train_projected, columns=pca_cols).reset_index(drop=True)
# pca_normalized = pd.concat([pca_df, df_raw_tmp.reset_index(drop=True)], axis=1)
#
# NCO_test = pca_normalized
# # pca_normalized = pca_normalized.rename({0:"PCA_1", 1:"PCA_2"}, axis = 1)
# #The plot is done with predicted distributions.
# # import seaborn as sns
# # #from importlib import reload
# # #reload(sns)
#  fig, ax = plt.subplots(figsize=(11, 8.5))
# #
#
# #
# # #Shaded Density Plot
# #
# desnityplot_df = Z_macro_domestic_dict_1['PCA_DF'][Z_macro_domestic_dict_1['PCA_DF']["ReportingDate"].str.startswith("2016")]
#  g = sns.distplot(desnityplot_df[0],hist = False, kde = True,
#                   kde_kws = {'shade': True, 'linewidth': 3},
#                    label = "PCA"
#                          )
# #
#
# #Scatter plot
# #
# NCO_test = Preprocess_Dict['NCO'][["RSSD_ID","ReportingDate","Other items:Net Charge Offs_Ratio",'Other items:Loan Charge Offs_Ratio']]
# #Merge PCA with NCO on dates or create own PCA
#
# #NCO_test[['Other items:Net Charge Offs_Ratio', 'Other items:Loan Charge Offs_Ratio']]  = NCO_test[['Other items:Net Charge Offs_Ratio','Other items:Loan Charge Offs_Ratio']] * .01
# NCO_test = NCO_test[NCO_test["ReportingDate"].str.startswith("2016")]
# scatterNCOtest = NCO_test.merge(desnityplot_df[["ReportingDate",0]], on = "ReportingDate")
#
#  s = sns.scatterplot(x= 0, y = "Other items:Loan Charge Offs_Ratio", data = scatterNCOtest, s = 5, color = 'b')
# #
#
# # #Draw Vertical Lines Across Plots
# hlines = desnityplot_df[["ReportingDate",0]]
#
#
# for i in  hlines.index:
#     x_bounds = ax.get_xlim()
#     print(hlines.loc[i,["ReportingDate",0]])
#     vline_value = hlines.loc[i,0].astype(float)
#     label = hlines.loc[i,"ReportingDate"]
#     ax.axvline(x=vline_value, color="r", linewidth=1, label= label)
#     ax.annotate(s=label, xy=(((vline_value - x_bounds[0]) / (x_bounds[1] - x_bounds[0])), .5),
#                 xycoords='axes fraction', verticalalignment='left', horizontalalignment='left bottom', rotation=270)
#



#scatterNCOtest.keys()




# plt.close()

Z_macro = init_ST.Z_macro_process()
Z_macro_international_dict_1 = process_raw_modality_PCA_Norm(Z_macro, keyname = "Historic_International", raw_date_col="Date", removecols= [x for x in Z_macro["Historic_International"].columns if  x.startswith('Scenario Name')],extractquarters = False, n_components= 12)
Preprocess_Dict['zmacro_international_pca_all'] = preprocess_load_wrapper(Z_macro_international_dict_1, keyname = 'PCA_DF', collist = 'PCA', datatype = "zmacro_international")


SectorIdx = init_ST.sectoridx_process()
SectorIdx_dict_1 = process_raw_modality_PCA_Norm(SectorIdx, keyname = "sectoridx", raw_date_col="Date",extractquarters = True)
Preprocess_Dict['sectidx_pca_all'] = preprocess_load_wrapper(SectorIdx_dict_1, keyname = 'PCA_DF', collist = 'PCA', datatype = "sectoridx")


SBidx = init_ST.SBidx_process()
SBidx_dict_1 = process_raw_modality_PCA_Norm(SBidx, keyname = "SB_idx_prox", raw_date_col="Date",extractquarters = True, n_components= 7)
Preprocess_Dict['sbidx_pca_all'] = preprocess_load_wrapper(SBidx_dict_1, keyname = 'PCA_DF', collist = 'PCA', datatype = "sbidx")
#TODO: Improve n_componets logic


#Create Transformational Cubes
os.chdir("/Users/phn1x/icdm2018_research_BB/Stress_Test_Research/Loss_projections/data")
keylist = ['zmacro_domestic_pca_all', 'zmacro_international_pca_all', "sectidx_pca_all", "sbidx_pca_all",
           "zmicro_pca_all"]
for keyname in [x for x in Preprocess_Dict.keys() if x in keylist]:
    print(keyname)
    data, data_quarter = pd_raw_to_numpy_moda(Preprocess_Dict[keyname], fulldatescombine=False,ReportingDate_Start = "1976 Q1", ReportingDate_End = "2016 Q4")
    print(keyname, ":", data.shape, keyname, "_quarter:", data_quarter.shape)
    print("Saving objects to file")
    np.save("./data_moda_" + keyname + ".npy", data)
    np.save("./quarter_based/data_moda_" + keyname + "_quarter.npy", data_quarter)



Preprocess_Dict['sectidx_pca_all']["ReportingDate"].iloc[160:168,]

pd.Series(Preprocess_Dict['zmacro_domestic_pca_all']["ReportingDate"].unique())[pd.Series(Preprocess_Dict['zmacro_domestic_pca_all']["ReportingDate"].unique()).isin(['1986 Q3','2016 Q4'])]

list(pd.Series(Preprocess_Dict['X']["ReportingDate"].unique())[pd.Series(Preprocess_Dict['X']["ReportingDate"].unique()).isin(datamoda_test_date)].index)
122 - 86


Preprocess_Dict['X'][Preprocess_Dict['X']["RSSD_ID"].isin([1020201])].iloc[86:122,:].describe()


moda_Reporting_Ref = pd.Series(Preprocess_Dict['zmacro_domestic_pca_all']["ReportingDate"].unique())
Preprocess_Dict['moda_DateIdx']  = moda_Reporting_Ref
data_Reporting_Ref = pd.Series(Preprocess_Dict['X']["ReportingDate"][Preprocess_Dict['X']["ReportingDate"] >= "1987 Q1"].unique())
Preprocess_Dict['data_DateIdx']  = data_Reporting_Ref

keylist = ['zmacro_domestic_pca_all','zmacro_international_pca_all',"sectidx_pca_all","sbidx_pca_all","zmicro_pca_all"]
for keyname in [x for x in Preprocess_Dict.keys() if x in keylist]:
    os.chdir("/Users/phn1x/icdm2018_research_BB/Stress_Test_Research/Loss_projections/data")
    print(keyname)
    data, data_quarter = pd_raw_to_numpy_moda(Preprocess_Dict[keyname], fulldatescombine= False,  ReportingDate_Start = "1976 Q1", ReportingDate_End = "2017 Q4")
    print(keyname, ":", data.shape, keyname, "_quarter:", data_quarter.shape)
    print("Saving objects to file")
    np.save("./data_moda_" + keyname + ".npy", data)
    np.save("./quarter_based/data_moda_" + keyname + "_quarter.npy", data_quarter)




