import pandas as pd
import numpy as np
import os
import gc
from functools import reduce

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


CDS_Swaps_Full = pd.read_csv("~/Downloads/CDS_WRDS.csv")
del CDS_Swaps_Full

CDS_Swaps_Full.describe().transpose()

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

        sectoridx_dict = self.file_dict_read(Sectoridx_dir, filetype=filetype)
        dfs = list()
        for keyname in sectoridx_dict.keys():
            print("Processing:", keyname)
            tmp_df = sectoridx_dict[keyname]
            tmp_df["datadate"] = pd.to_datetime(tmp_df["datadate"], format="%Y%M%d").dt.date
            tmp_df = tmp_df[["datadate", "tic", "prccm"]]
            tmp_df = tmp_df.pivot_table(index="datadate", columns="tic", values="prccm")
            tmp_df = tmp_df.reset_index()
            print(tmp_df.describe().transpose())
            dfs.append(tmp_df)

        print("Combine List with Left Merge")
        final_raw_df = reduce(lambda left, right: pd.merge(left, right, on="datadate", how="left"), dfs)
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
        for keyname in [v for v in Z_micro.keys() if v.startswith("GFD_")]:
            print(keyname)
            tmp_df = Z_micro[keyname]
            tmp_df["Date"] = pd.to_datetime(tmp_df["Date"]).dt.date
            dfs.append(tmp_df[["Date", "Close"]].rename({"Close": keyname}, axis=1))

        for keyname in [v for v in Z_micro.keys() if v.startswith("WRDS_")]:
            # keyname = "WRDS_SnP500_Returns"
            print(keyname)
            if "caldt" in Z_micro[keyname].keys():
                print("Filter from 1970")
                tmp_df = Z_micro[keyname][Z_micro[keyname]["caldt"] > 19691231]
                tmp_df["caldt"] = pd.to_datetime(tmp_df["caldt"], format="%Y%M%d").dt.date
                dfs.append(tmp_df.rename({"caldt": "Date"}, axis=1))

            if "date" in Z_micro[keyname].keys():
                print("Filter from 1970")
                tmp_df = Z_micro[keyname][Z_micro[keyname]["date"] > 19691231]
                tmp_df["date"] = pd.to_datetime(tmp_df["date"], format="%Y%M%d").dt.date
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
                                     replace_nan=False, merge = True, Consqtr = True, combinecol = True, reducedf = True, skip_prefix = None, Y_calc = True, RSSD_Subset = True, calc_CR_hist = True,get_prev_time_cols = True
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
            BankCharPerf_raw_data_dict = self.BankPerf_ConsecutiveQtrs_Reduce(BankCharPerf_raw_data_dict, concecutiveqtrs=8,
                                                                 BankPerf_Calc_df_name="BankPerf_Calculated",
                                                                 sincedt="1990-01-30", outputkeyname = "BankPerf_MainCalc")

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
            merger_info_df[merger_df_datecol] = pd.to_datetime(merger_info_df[merger_df_datecol], format="%Y%M%d")
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
            print("Forward Filling Nans")
            BankPerf = BankPerf.fillna(method = 'ffill')
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







#Entry Point
init_ST = StressTestData()

#Z_Macro complate and distributions match to Malik 2018
Z_macro = init_ST.Z_macro_process()


Z_macro.keys()

#Z_macro["Historic_Domestic"].keys()
pd.to_datetime(Z_macro["Historic_Domestic"]["Date"]).min()

#Z_macro.keys()

#Z_macro_Domestic_colsuse = [x for x in Z_macro["Historic_Domestic"].columns if x not in ["Scenario Name"]]
Z_macro["Historic_Domestic"][Z_macro_Domestic_colsuse][pd.to_datetime(Z_macro["Historic_Domestic"]["Date"]) <= "2017-12-31"].describe().transpose()
#Z_macro["Historic_Domestic"][Z_macro_Domestic_colsuse].to_csv("../Data_Output/Z_Macro_Domestic.csv", sep = ",", index = False)
#Z_macro_International_colsuse = [x for x in Z_macro["Historic_International"].columns if x not in ["Scenario Name"]]
Z_macro["Historic_International"][Z_macro_International_colsuse][pd.to_datetime(Z_macro["Historic_International"]["Date"]) <= "2017-12-31"].describe().transpose()
#Z_macro["Historic_International"][Z_macro_International_colsuse].to_csv("../Data_Output/Z_Macro_International.csv", sep = ",", index = False)
#Z_macro_combined = Z_macro["Historic_Domestic"][Z_macro_Domestic_colsuse].merge(Z_macro["Historic_International"][Z_macro_International_colsuse], on = "pdDate")
#Z_macro_combined.to_csv("../Data_Output/Z_Macro.csv", sep = ",", index = False)

#Shadow Banking Proxies
SBidx = init_ST.SBidx_process()



SBidx["SB_idx_prox"] = SBidx["SB_idx_prox"].rename({"Date":"ReportingDate"},axis = 1)
SBidx["SB_idx_prox"]["ReportingDate"] =  pd.to_datetime(SBidx["SB_idx_prox"].ReportingDate).dt.year.astype(str) + " Q" + pd.to_datetime(SBidx["SB_idx_prox"].ReportingDate).dt.quarter.astype(str)

SBidx["SB_idx_prox"].keys()


SBidx["SB_idx_prox"]["ReportingDate"]

SBidx["SB_idx_prox"][(pd.to_datetime(SBidx["SB_idx_prox"]["ReportingDate"]) >= "1976-01-01") & (pd.to_datetime(SBidx["SB_idx_prox"]["ReportingDate"]) <= "2017-12-31")].describe().transpose()


SBidx["SB_idx_prox"].to_csv("../Data_Output/SBidx.csv", sep = ",", index = False)



#Sector Indices
SectorIdx = init_ST.sectoridx_process()
#
SectorIdx.keys()
SectorIdx["sectoridx"].keys()

SectorIdx["sectoridx"].to_csv("../Data_Output/Sectidx.csv", sep = ",", index = False)

SectorIdx["sectoridx"].Date

SectorIdx["sectoridx"] = SectorIdx["sectoridx"].rename({"Date":"ReportingDate"},axis = 1)
SectorIdx["sectoridx"]["ReportingDate"] =  pd.to_datetime(SectorIdx["sectoridx"].ReportingDate).dt.year.astype(str) + " Q" + pd.to_datetime(SectorIdx["sectoridx"].ReportingDate).dt.quarter.astype(str)
SectorIdx["sectoridx"].to_csv("../Data_Output/Sectidx.csv", sep = ",", index = False)


SectorIdx["sectoridx"][(pd.to_datetime(SectorIdx["sectoridx"]["ReportingDate"]) >= "1976-01-01") & (pd.to_datetime(SectorIdx["sectoridx"]["ReportingDate"]) <= "2017-12-31")].describe().transpose()


SectorIdx["WRDS_SP500_RealEstate_Indicies"]["tic"].unique()


SectorIdx["sectoridx"].keys()
SectorIdx.keys()
#Should pring out Summary Statistics

Z_micro = init_ST.Z_micro_process()

Z_micro.keys()
Z_micro["WRDS_government_bonds"].keys()

Z_micro["WRDS_US Treasury and Inflation Indexes"].keys()


Z_micro["Z_Micro"] = Z_micro["Z_Micro"].rename({"Date":"ReportingDate"},axis = 1)



Z_micro["WRDS_currency_swaps"].keys()

Z_micro["WRDS_government_bonds"].keys()

Z_micro["Z_Micro"][(pd.to_datetime(Z_micro["Z_Micro"]["ReportingDate"]) >= "1976-01-01") & (pd.to_datetime(Z_micro["Z_Micro"]["ReportingDate"]) <= "2017-12-31")].describe().transpose()

CurrencySwapsList = ('exusal',
'exalus',
'exbzus',
'excaus',
'exchus',
'exdnus',
'exhkus',
'exinus',
'exjpus',
'exkous',
'exmaus',
'exmxus',
'exusnz',
'exnzus',
'exnous',
'exsius',
'exsfus',
'exslus',
'exsdus',
'exszus',
'extaus',
'exthus',
'exusuk',
'exukus',
'exvzus',
'exusir',
'exusec',
'execus',
'exuseu',
'exeuus',
'twexb',
'twexm',
'twexo',
'indexgx',
'ReportingDate')

list(Z_micro["Z_Micro"].columns)


z_micro_currsqp = Z_micro["Z_Micro"][Z_micro["Z_Micro"].columns[(pd.Series(Z_micro["Z_Micro"].columns).str.startswith(CurrencySwapsList))]]

z_micro_currsqp[(pd.to_datetime(z_micro_currsqp["ReportingDate"]) >= "1976-01-01") & (pd.to_datetime(z_micro_currsqp["ReportingDate"]) <= "2017-12-31")].describe().transpose()



#
GovernmentBondsList = ('D_AH_M3',
 'D_AH_M6',
 'D_AH_Y1',
 'D_COMP_Y10P',
 'D_LTNOM_Y25P',
 'D_TCMNOM_Y20',
 'LTAVG_Y10P',
 'TB_M3',
 'TB_M6',
 'TB_WK4',
 'TB_Y1',
 'TCMII_Y10',
 'TCMII_Y20',
 'TCMII_Y30',
 'TCMII_Y5',
 'TCMII_Y7',
 'TCMNOM_M1',
 'TCMNOM_M3',
 'TCMNOM_M6',
 'TCMNOM_Y10',
 'TCMNOM_Y1',
 'TCMNOM_Y20',
 'TCMNOM_Y2',
 'TCMNOM_Y30',
 'TCMNOM_Y3',
 'TCMNOM_Y5',
 'TCMNOM_Y7',
  'ReportingDate')

z_micro_govtbonds = Z_micro["Z_Micro"][Z_micro["Z_Micro"].columns[(pd.Series(Z_micro["Z_Micro"].columns).str.startswith(GovernmentBondsList))]]

z_micro_govtbonds.keys().__len__()

z_micro_govtbonds[(pd.to_datetime(z_micro_govtbonds["ReportingDate"]) >= "1976-01-01") & (pd.to_datetime(z_micro_govtbonds["ReportingDate"]) <= "2017-12-31")].describe().transpose()[["count","mean","std","min","50%","max"]]




Z_micro["Z_Micro"]["ReportingDate"] =  pd.to_datetime(Z_micro["Z_Micro"].ReportingDate).dt.year.astype(str) + " Q" + pd.to_datetime(Z_micro["Z_Micro"].ReportingDate).dt.quarter.astype(str)

#Foward fill, then backward fill
list(Z_micro["Z_Micro"].keys()).__len__()

Z_micro["Z_Micro"] = Z_micro["Z_Micro"].fillna(method = "ffill").fillna(method = "bfill")

Z_micro["Z_Micro"].to_csv("../Data_Output/Z_micro.csv", sep = ",", index = False)


Z_micro["Z_Micro"][(pd.to_datetime(Z_micro["Z_Micro"]["ReportingDate"]) >= "1976-01-01") & (pd.to_datetime(Z_micro["Z_Micro"]["ReportingDate"]) <= "2017-12-31")].describe().transpose()


#Need Additional Subsetting and formatting logic for the WRDS data
#Also need additional interest rate swaps data.
#May need to calcualte the returns for some of the indices and prices.
#Should output Summary Statistics


#gc.collect()
#May need to subset BankPerf to get Xi and Yi

BankPerf = StressTestData().X_Y_bankingchar_perf_process(groupfunction=np.mean, groupby=["RSSD_ID", "ReportingDate"],
                                     groupagg_col='Other items:Total Assets',
                                     RSSD_DateParam=["1990-03-30", "2018-01-01"],
                                     ReportingDateParam=["1990-03-30", "2018-01-01"], RSSDList_len=1000, dropdup=False,
                                     replace_nan=False, combinecol=True, merge = True, reducedf = False,skip_prefix = "WRDS_Covas_BankPerf_CallReport",  replace_dict={"BankPerf_FR9YC_varlist": {
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

                                     }, RSSD_Subset = True, Y_calc= True)


#Workspace
[v for v in BankPerf.keys() if v.startswith("BankPerf_ConsecutiveReduced")]

BankPerf["BankPerf_ConsecutiveReduced_Subset_DescriptiveStats"]




#Create TCR1 Column
identifiers = ["RSSD_ID","ReportingDate"]

loans = ["Loans categories:Commercial & industrial_Covas"
        ,"Loans categories:Construction & land development"
        ,"Loans categories:Multifamily real estate"

        ,"Loans categories:Nonfarm nonresidential CRE"
        ,"Loans categories:Home equity lines of credit"
        ,"Loans categories:Residential real estate (excl. HELOCs)_Covas"
        ,"Loans categories:Credit card"
        ,"Loans categories:Consumer (excl. credit card)_Covas"]

loans_t1 = [v + "_t-1" for v in loans]

#Need to re-run
lossrates = [v for v in BankPerf["BankPerf_ConsecutiveReduced_XYcalc_Subset_BankPerf"].columns[pd.Series(BankPerf["BankPerf_ConsecutiveReduced_XYcalc_Subset_BankPerf"].columns).str.startswith("Net charge-offs by type of loan:")] if not v.endswith("_t-1")]

df_tmp = BankPerf["BankPerf_ConsecutiveReduced_XYcalc_Subset_BankPerf"]
for i,y in zip(loans_t1,lossrates):
    print(i,y)
    df_tmp[] = df_tmp[i] * df_tmp[y]



#Conditional Vector
#Net charge offs.
#
list(BankPerf["BankPerf_ConsecutiveReduced_XYcalc_Subset_BankPerf"].columns)
collist = ["RSSD_ID","ReportingDate","Chargeoffs","Recoveries"]

Conditional_ij = BankPerf["BankPerf_ConsecutiveReduced_XYcalc_Subset_BankPerf"][collist]
Conditional_ij["NetChargeOffs"] = Conditional_ij["Chargeoffs"] - Conditional_ij["Recoveries"]


Conditional_ij["ReportingDate"] =  Conditional_ij.ReportingDate.dt.year.astype(str) + " Q" + Conditional_ij.ReportingDate.dt.quarter.astype(str)
Conditional_ij.to_csv("../Data_Output/Conditional_ij.csv", sep = ",", index= False)


#['RSSD_ID',"ReportingDate','Other items:T1CR','Other items:T1CR_t-1']
BankPerf["BankPerf_ConsecutiveReduced_Subset_BankPerf"][['RSSD_ID','ReportingDate','Other items:T1CR','Other items:T1CR_t-1']]

X_i.ReportingDate.dt.year.astype(str) + " Q" + X_i.ReportingDate.dt.quarter.astype(str)

list(BankPerf["BankPerf_ConsecutiveReduced_Subset_BankPerf"].columns)

X_i = BankPerf["BankPerf_ConsecutiveReduced_Subset_BankPerf"][BankPerf["BankPerf_ConsecutiveReduced_Subset_BankPerf"].columns[pd.Series(BankPerf["BankPerf_ConsecutiveReduced_Subset_BankPerf"].columns).str.startswith(("RSSD_ID","ReportingDate","Loans categories:Commercial & industrial_Covas","Loans categories:Construction & land development","Loans categories:Multifamily real estate","Loans categories:Nonfarm nonresidential CRE_Covas","Loans categories:Nonfarm nonresidential CRE","Loans categories:Home equity lines of credit","Loans categories:Residential real estate (excl. HELOCs)_Covas","Loans categories:Credit card","Loans categories:Consumer (excl. credit card)_Covas",'LABEL:T1CR','LABEL:T1CR_t-1'))]]

X_i["ReportingDate"] =  X_i.ReportingDate.dt.year.astype(str) + " Q" + X_i.ReportingDate.dt.quarter.astype(str)



X_i.to_csv("../Data_Output/X_ij.csv", sep = ",", index= False)

X_i.describe().transpose()

X_i.keys()



Y_i = BankPerf["BankPerf_ConsecutiveReduced_XYcalc_Subset_BankPerf"][BankPerf["BankPerf_ConsecutiveReduced_XYcalc_Subset_BankPerf"].columns[(pd.Series(BankPerf["BankPerf_ConsecutiveReduced_XYcalc_Subset_BankPerf"].columns).str.startswith(("RSSD_ID","ReportingDate","ncoR:","ppnrRatio:")))
& (~pd.Series(BankPerf["BankPerf_ConsecutiveReduced_XYcalc_Subset_BankPerf"].columns).str.endswith("_t-1"))]]

Y_i.describe().transpose()
Y_i["ReportingDate"] =  Y_i.ReportingDate.dt.year.astype(str) + " Q" + Y_i.ReportingDate.dt.quarter.astype(str)
Y_i.to_csv("../Data_Output/Y_ij.csv", sep = ",", index= False)


Y_i.keys()



#Create calculated Net Charge Off and Book Equity.
#Create calculated Capital Ratio.
#Get Net Charge off amount via Data Source
#Get PPNR




#Calc NCO





#Need to develop a Capital Ratio Calculator.

#This will be needed for the predicted values to compare with the Ground Truth.

#Need to calculate the measured CR_t,i
#Other items:Book equity #t
#Other items:Risk-weighted assets #t-1

#May need to calculate  the Net Charge Off, Book Equirty and CRi.
#Have to get time period lags.


#Calculate PPNR as PPNR componet ratio * consolidated assets of previous period.

#calculate net charge off
#Sum of associated loan of previous period * current period charge off rate.
#Book Equity
#Book Equity of previous period + .65 * (ppnr - nco) - dividends_t-1 - Stock Repurchases t-1
#CR - Book Equity - Reg Capital Deductions of previous period/ Risk weighted Assets


#Should we just update each row with columns with t-1




#Discrimitative to Macro-Economic and Bank.
#Get from source data,
#Charge-offs : BHCK4635
#Total Equity : BHCK310
#Net Charge-offs : BHCK4635 - BHCK4605
#Return on Equity : BHCK4340-BHCK4598/Average(BHCK310)



#VAE-CVAE=MNIST
#import utils, models, train


