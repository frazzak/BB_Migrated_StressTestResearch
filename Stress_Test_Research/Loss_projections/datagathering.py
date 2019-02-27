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
                                                    "MergerInfo" : "merger_info_frb",
                                                    "CallReport_prefix" : "WRDS_Covas_BankPerf_CallReport"}
                                     ):
        BankCharPerf_raw_data_dict = self.file_dict_read(self.BankPerf_dir, filetype=".csv")
        print(BankCharPerf_raw_data_dict.keys())
        print("Fixing Column Names with Errors")

        print("Preparing Calculated Fields")
        print("Fixing Errors")
        print("Fixing BHCK892 to BHCKC892")
        print("Creating VARLIST Column")
        BankCharPerf_raw_data_dict[params_dict["Calc_df"]]["VarList"] = BankCharPerf_raw_data_dict[params_dict["Calc_df"]]["Report: FR Y-9C"].astype(str).apply(lambda x : [i.strip() for i in  x.replace("BHCK892","BHCKC892").replace("+",",").replace("-",",").replace("(","").replace(")","").replace(". . .", "").replace(",  ,",",").split(",")])

        BankCharPerf_raw_data_dict[params_dict["Calc_df"]]["VarList2"] = \
        BankCharPerf_raw_data_dict[params_dict["Calc_df"]]["Report: FR Y-9C"].astype(str).apply(
            lambda x: [i.strip() for i in
                       x.replace("BHCK892", "BHCKC892").replace("+", ",").replace("-", ",").replace("(", "").replace(
                           ")", "").replace(". . .", "").replace(",  ,", ",").split(",")])

        print("Additional Clean up for Varlist")
        BankCharPerf_raw_data_dict[params_dict["Calc_df"]]["VarList"] =  BankCharPerf_raw_data_dict[params_dict["Calc_df"]]["VarList"].apply(lambda x: [i for i in x if i not in ["",np.nan,"nan"]])
        BankCharPerf_raw_data_dict[params_dict["Calc_df"]]["VarList2"] =  BankCharPerf_raw_data_dict[params_dict["Calc_df"]]["VarList"].apply(lambda x: [i for i in x if i not in ["",np.nan,"nan"]])

        print("Checking for Column name Discrepancy")
        miss_col = self.check_missing_BHCCodes(BankCharPerf_raw_data_dict,params_dict)

        print("Creating Derived Columns")
        BankCharPerf_raw_data_dict["BankPerf_Calculated"] = self.BHC_loan_nco_ppnr_create(BankCharPerf_raw_data_dict,params_dict)
        print("Subsetting for BHC and FHC banks")
        BankCharPerf_raw_data_dict["BankPerf_Calculated_BHCFHC"] = BankCharPerf_raw_data_dict["BankPerf_Calculated"][(BankCharPerf_raw_data_dict["BankPerf_Calculated"]["BHC_Indicator"] == 1) | (BankCharPerf_raw_data_dict["BankPerf_Calculated"]["FHC_Indicator"] == 1)]

        print("Subsetting out Banks that do not have appropirate Consecutive Quarters")
        BankPerf_ConsReduce = BankPerf_ConsecutiveQtrs_Reduce(BankCharPerf_raw_data_dict, BankPerf_Calc_df_name='BankPerf_Calculated_BHCFHC')
        print("Mergering based on Merger Data the Banks")
        BankPerfMerger = BankPerf_Merger_process(BankPerf_ConsReduce,BankPerf_Calc_df_name='BankPerf_ConsecutiveReduced')
        BankPerfAgg = BankPerf_Aggregation_process(BankPerfMerger, BankPerf_ToAgg_df_name='BankPerf_Mergered')

        return(BankPerfAgg)
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
        identifier_columns = ["RSSD9001","RSSD9375","RSSD9999","RSSD9010","RSSD9017","RSSD9161",
                              "BHCK2170",'RSSD9045','RSSD9016','RSSD9101','RSSD9138',"RSSD9397","PRIM_FED_REG"]
        BankPerf_result = pd.concat([BankPerf[params_dict["BankPerf_raw"]][identifier_columns], loan_nco_ppnr_df], axis=1)

        print("Rename Identifier Columns")
        BankPerf_result = BankPerf_result.rename({"RSSD9001":"RSSD_ID" , "RSSD9999" : "ReportingDate" , "RSSD9161" : "CUSIP" , "BHCK2170" : "TotalAssets",
                                                  "RSSD9045" : "BHC_Indicator", "RSSD9016" : "FHC_Indicator", "RSSD9101" : "Domestic_Indicator"
                                                  ,"RSSD9138" : "Financial_Sub_Indicator", "RSSD9397" : "LargestEntityBHC", "RSSD9375" : "HeadOffice_RSSD_ID"}, axis = 1)
        print("Format ReportingDate column")
        BankPerf_result["ReportingDate"] = pd.to_datetime(BankPerf_result["ReportingDate"], format = "%Y%m%d")

        return (BankPerf_result)

    def BankPerf_ConsecutiveQtrs_Reduce(self, BankPerf, concecutiveqtrs=8, BankPerf_Calc_df_name=None, sincedt="1989-12-31"):
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

        return (BankPerf)

    def BankPerf_Merger_process(self, BankPerf, merger_df_name="merger_info_frb", BankPerf_Calc_df_name=None):
        if merger_df_name in BankPerf.keys():
            print("Getting Merger Info from Dictionary")
            merger_info_df = BankPerf[merger_df_name]
            merger_info_df = merger_info_df[["MERGE_DT", "NON_ID", "SURV_ID"]]
            merger_info_df["MERGE_DT"] = pd.to_datetime(merger_info_df["MERGE_DT"], format="%Y%M%d")
            merger_info_df["NON_ID"] = merger_info_df["NON_ID"].astype(int)
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
        BankPerf_Merger_df = BankPerf_Calc_df.merge(merger_info_df, left_on="RSSD_ID", right_on="NON_ID", how="left")
        print("Found", BankPerf_Merger_df["RSSD_ID"][pd.notnull(BankPerf_Merger_df["NON_ID"])].unique().__len__(),
              ' Non surviving RSSD_IDs')
        print("Updating Non-Surviving RSSD_IDs with Surviving IDs")
        BankPerf_Merger_df.loc[pd.notnull(BankPerf_Merger_df["NON_ID"]), ["RSSD_ID"]] = BankPerf_Merger_df.loc[
            pd.notnull(BankPerf_Merger_df["NON_ID"]), ["SURV_ID"]]
        print("Dropping Merge Columns")
        BankPerf_Merger_df = BankPerf_Merger_df.drop(list(merger_info_df.columns), axis=1)
        print("Re Merging to check for remaining mergers")
        BankPerf_Merger_df = BankPerf_Merger_df.merge(merger_info_df, left_on="RSSD_ID", right_on="NON_ID", how="left")
        print("Number of NON_ID matching with RSSD_ID:",
              BankPerf_Merger_df[pd.notnull(BankPerf_Merger_df["NON_ID"])].shape)
        # if BankPerf_Merger_df[pd.notnull(BankPerf_Merger_df["NON_ID"])].shape == 0:
        BankPerf["BankPerf_Mergered"] = BankPerf_Merger_df
        print(BankPerf["BankPerf_Mergered"].describe().transpose())

        return (BankPerf)

    def BankPerf_Aggregation_process(self, BankPerf, BankPerf_ToAgg_df_name=None):

        if BankPerf_ToAgg_df_name in BankPerf.keys():
            print("Getting BankPerf Calculated rows")
            BankPerf_Merger_df_gbsum = BankPerf[BankPerf_ToAgg_df_name]
        else:
            return ("BankPerf Calculated DF not found")

        print("Aggregate all the updated RSSD_ID columns")

        # BankPerf_Merger_df_gbsum.columns
        BankPerf_Merger_df_gbsum = BankPerf_Merger_df_gbsum.groupby(["RSSD_ID", "ReportingDate"])
        BankPerf_Merger_df_gbsum = BankPerf_Merger_df_gbsum.agg({
            'TotalAssets': np.sum,
            'Net charge-offs by type of loan:Commercial & industrial': np.sum,
            'Net charge-offs by type of loan:Construction & land development': np.sum,
            'Net charge-offs by type of loan:Multifamily real estate': np.sum,
            'Net charge-offs by type of loan:(Nonfarm) nonresidential CRE': np.sum,
            'Net charge-offs by type of loan:Home equity lines of credit': np.sum,
            'Net charge-offs by type of loan:Residential real estate (excl. HELOCs)': np.sum,
            'Net charge-offs by type of loan:Credit card': np.sum,
            'Net charge-offs by type of loan:Consumer (excl. credit card)': np.sum,
            'Loans categories:Commercial & industrial': np.sum,
            'Loans categories:Construction & land development': np.sum,
            'Loans categories:Multifamily real estate': np.sum,
            'Loans categories:Nonfarm nonresidential CRE': np.sum,
            'Loans categories:Home equity lines of credit': np.sum,
            'Loans categories:Residential real estate (excl. HELOCs)': np.sum,
            'Loans categories:Credit card': np.sum,
            'Loans categories:Consumer (excl. credit card)': np.sum,
            'Components of pre-provision net revenue:Net interest income': np.sum,
            'Components of pre-provision net revenue:Noninterest income': np.sum,
            'Components of pre-provision net revenue:Trading income': np.sum,
            'Components of pre-provision net revenue:Compensation expense': np.sum,
            'Components of pre-provision net revenue:Fixed assets expense': np.sum,
            'Components of pre-provision net revenue:Noninterest expense': np.sum,
            'Other items:Consolidated assets': np.sum,
            'Other items:Interest-earning assets': np.sum,
            'Other items:Trading assets': np.sum,
            'Other items:Book equity': np.sum,
            'Other items:Risk-weighted assets': np.sum,
            'Other items:Dividends ': np.sum,
            'Other items:Stock purchases': np.sum,
            'Other items:Tier 1 common equity': np.sum,
            'Other items:= Tier 1 capital': np.sum,
            'Other items: = - Perpetual preferred stock': np.sum,
            'Other items: = + Nonqual. Perpetual preferred stock': np.sum,
            'Other items: = - Qual. class A minority interests': np.sum,
            'Other items: = - Qual. restricted core capital': np.sum,
            'Other items: = - Qual. mandatory convert. pref. sec.': np.sum
        })
        # May need to add manual column sums.
        BankPerf_Merger_df_gbsum = BankPerf_Merger_df_gbsum.reset_index()

        BankPerf["BankPerfAgg"] = BankPerf_Merger_df_gbsum
        print(BankPerf["BankPerfAgg"].describe().transpose())

        return (BankPerf)

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


#gc.collect()
#May need to subset BankPerf to get Xi and Yi

BankPerf = init_ST.X_Y_bankingchar_perf_process()

params_dict = {"Calc_df": "NCO_PPNR_Loan_Calcs",
               "BankPerf_raw": "WRDS_Covas_BankPerf",
               "MergerInfo": "merger_info_frb",
               "CallReport_prefix": "WRDS_Covas_BankPerf_CallReport"}


CallReport_raw_list = [v for v in BankPerf.keys() if v.startswith(params_dict["CallReport_prefix"])]
#create list of dataframes to combine
#Initialize DF list
dfs = list()
for keyname in CallReport_raw_list:
    dfs.append(BankPerf[keyname])

dfs.__len__()

from functools import reduce
final_raw_df = reduce(lambda left, right: pd.merge(left,right, on =["RSSD9001","RSSD9999"]), dfs)

final_raw_df.columns = [v.replace("_x","").replace("_y","") for v in final_raw_df.columns]
print("Drop Duplicated Columns")
final_raw_df = final_raw_df.loc[:,~final_raw_df.columns.duplicated()]


replace_dict = {"RCFD560":"RCFDB560",
                "RIADK129":"RIAD4639",
                "RIADK205": "RIAD4657",
                "RIADK133": "RIAD4609",
                "RIADK206": "RIAD4667",
                "+" : ",",
                "-" : ",",
                "(" : "",
                ")" : "",
                ". . .": "",
                ",  ,": ",",}

def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text





print("Initialize VarList2")
BankPerf[params_dict["Calc_df"]]["VarList2"] = BankPerf[params_dict["Calc_df"]]["Report: FFIEC 031/041"].astype(str).apply(
        lambda x: [i.strip() for i in  replace_all(x,replace_dict).split(",")])
    #BankPerf[params_dict["Calc_df"]]["Report: FFIEC 031/041"].astype(str).apply(
    #    lambda x: [i.strip() for i in  x.replace.replace("RIADK129", "RIAD4639").replace("RIADK205","RIAD4657").replace("RIADK133","RIAD4609").replace("RIADK206","RIAD4667").replace("+", ",").replace("-", ",").replace("(", "").replace(")", "").replace(". . .", "").replace(",  ,", ",").split(",")])

BankPerf[params_dict["Calc_df"]]["VarList2"] =  BankPerf[params_dict["Calc_df"]]["VarList2"].apply(lambda x: [i for i in x if i not in ["",np.nan,"nan"]])

print("Initialize Result DF")
        loan_nco_ppnr_df = pd.DataFrame()
print("Initialize Calculation Column")

replace_dict = {"RCFD560":"RCFDB560",
                "RIADK129":"RIAD4639",
                "RIADK205": "RIAD4657",
                "RIADK133": "RIAD4609",
                "RIADK206": "RIAD4667",
                "(" : "",
                ")" : "",
                ". . .": "",
                "nan": "",
                "+  +" : "+"}


calc_tmp = BankPerf[params_dict["Calc_df"]]["Report: FFIEC 031/041"].astype(str).apply( lambda x: replace_all(x,replace_dict).strip())
    #BankPerf[params_dict["Calc_df"]]["Report: FFIEC 031/041"].astype(str).apply(
    #lambda x: x.replace("RIADK129", "RIAD4639").replace("RIADK205","RIAD4657").replace("RIADK133","RIAD4609").replace("RIADK206","RIAD4667").replace("(", "").replace(")", "").replace(". . .", "").replace(
    #    "nan", "").replace("+  +", "+").strip())
print("Create String Column with usage calculations")
BankPerf[params_dict["Calc_df"]]["Calc_varstr"] = calc_tmp
print("Generate Calculated Columns")
for i in range(0, BankPerf[params_dict["Calc_df"]].shape[0]):
    tmp_subset = BankPerf[params_dict["Calc_df"]].loc[i, "VarList2"]
    tmp_df = final_raw_df[tmp_subset]
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
identifier_columns = ["RSSD9001","RSSD9999", "RSSD9010", "RSSD9017",
                      "RSSD9161","RSSD9045", "RSSD9016", "RSSD9101"]

final_raw_df[identifier_columns]

BankPerf_CallReport = pd.concat([final_raw_df[identifier_columns], loan_nco_ppnr_df], axis=1)

print("Rename Identifier Columns")
BankPerf_CallReport = BankPerf_CallReport.rename(
    {"RSSD9001": "RSSD_ID", "RSSD9999": "ReportingDate", "RSSD9161": "CUSIP", "BHCK2170": "TotalAssets",
     "RSSD9045": "BHC_Indicator", "RSSD9016": "FHC_Indicator", "RSSD9101": "Domestic_Indicator"
        , "RSSD9138": "Financial_Sub_Indicator", "RSSD9397": "LargestEntityBHC", "RSSD9375": "HeadOffice_RSSD_ID"},
    axis=1)
print("Format ReportingDate column")
BankPerf_CallReport["ReportingDate"] = pd.to_datetime(BankPerf_CallReport["ReportingDate"], format = "%Y%M%d")

BankPerf_FR9yc = BankPerf["BankPerf_Calculated"][list(BankPerf_CallReport.columns)]

BankPerf_CallReport_FR9yc = pd.concat([BankPerf_FR9yc,BankPerf_CallReport])


BankPerf["BankPerf_CallReport_FR9yc_raw"] = BankPerf_CallReport_FR9yc
BankPerfConRedcued = BankPerf_ConsecutiveQtrs_Reduce(BankPerf,concecutiveqtrs = 8,BankPerf_Calc_df_name = "BankPerf_CallReport_FR9yc_raw", sincedt = "1989-12-31")
BankPerfMerged =  BankPerf_Merger_process(BankPerfConRedcued, merger_df_name = "merger_info_frb", BankPerf_Calc_df_name = "BankPerf_ConsecutiveReduced")
BankPerfMerged["BankPerf_Mergered"].shape
BankAgg = BankPerf_Aggregation_process(BankPerfMerged, BankPerf_ToAgg_df_name= "BankPerf_Mergered")

BankPerfConRedcued = BankPerf_ConsecutiveQtrs_Reduce(BankAgg,concecutiveqtrs = 8,BankPerf_Calc_df_name = "BankPerfAgg", sincedt = "1989-12-31")

BankAgg["BankPerfAgg"].keys()
BankAgg["BankPerfAgg"]["T1CR"] = (BankAgg["BankPerfAgg"]["Other items:= Tier 1 capital"])/(BankAgg["BankPerfAgg"]['Other items:Risk-weighted assets'])
BankAgg["BankPerfAgg"]["T1CR"].sort_values(ascending = False)

RSSD_Subset_test = RSSD_Subset(BankAgg, keyname = "BankPerfAgg", groupfunction = np.mean, groupby = ["RSSD_ID", "ReportingDate"], groupagg_col =  'Other items:Consolidated assets', RSSD_DateParam = ["1989-12-31","2018-01-01"], ReportingDateParam = ["1989-12-31","2017-12-31"], RSSDList_len = 1000, dropdup =False, replace_nan = False)

NCO_Test = bankperf_rates_ratios(RSSD_Subset_test["BankPerf"], replace_nan = True)
RSSD_Subset_test.keys()


def RSSD_Subset(BankPerf, keyname = None, groupfunction = np.mean, groupby = ["RSSD_ID", "ReportingDate"], groupagg_col = "TotalAssets", RSSD_DateParam = ["1989-12-31","2017-12-31"], ReportingDateParam = ["1989-12-31","2017-12-31"], RSSDList_len = 1000, dropdup = False, replace_nan = True):
    if keyname in BankPerf.keys():
        print("Getting BankPerf  rows")
        BankPerf_Calc_df = BankPerf[keyname]
        if replace_nan:
            print("Replaceing 0.0 with NAN")
            BankPerf_Calc_df = BankPerf_Calc_df.replace(0.0,np.nan)
    else:
        return("BankPerf key DF not found")

    print("First Group By:",groupby,"And Aggregate using SUM")
    test1_groupby = BankPerf_Calc_df.groupby(groupby).agg({groupagg_col: np.sum})
    print("Sorting by:", groupagg_col)
    test1_groupby = test1_groupby.reset_index().sort_values([groupagg_col], ascending=[0])
    if "TotalAssets" in groupagg_col:
        print("Updating Total Assets by increasing values by 1e3")
        test1_groupby["TotalAssets"] = test1_groupby["TotalAssets"] * 1e3

    print("Filtering out Zero values and Reporting Date => criteria:", ReportingDateParam)
    test1_groupby = test1_groupby[(test1_groupby[groupagg_col] != 0) & (test1_groupby['ReportingDate'] >= RSSD_DateParam[0])  & (test1_groupby['ReportingDate'] <= RSSD_DateParam[1])]
    if dropdup:
        test1_groupby = test1_groupby.drop_duplicates(subset=['RSSD_ID'], keep= 'first')
        test1_groupby = test1_groupby[(test1_groupby[groupagg_col] != 0)].reset_index().sort_values([groupagg_col],ascending=[0])
    else:
        print("Grouping across all dates using:", str(groupfunction))
        test1_groupby = test1_groupby.groupby(["RSSD_ID"]).agg({groupagg_col: groupfunction})
        test1_groupby = test1_groupby[(test1_groupby[groupagg_col] != 0)].reset_index().sort_values([groupagg_col], ascending=[0])

    print("Getting Top",RSSDList_len,"From RSSD List")
    RSSD_ID_1k = test1_groupby['RSSD_ID'][0:RSSDList_len]

    print("Getting Descriptive Statistics of RSSD_List on Dataframe")
    BankPerf_tmp = BankPerf_Calc_df
    BankPerf_tmp = BankPerf_tmp[(BankPerf_tmp["RSSD_ID"].isin(RSSD_ID_1k)) &  (BankPerf_tmp['ReportingDate'] >= ReportingDateParam[0])  & (BankPerf_tmp['ReportingDate'] <= ReportingDateParam[1])]
    results_df = BankPerf_tmp[(BankPerf_tmp.columns[pd.Series(BankPerf_tmp.columns).str.startswith("Loans categories:")]) ].describe().transpose()
    print(results_df)

    return({"RSSD_List": RSSD_ID_1k, "DescriptiveStats" : results_df,"BankPerf" : BankPerf_tmp })

def BankPerf_Year_RSSD_Subset(BankPerf,keyname = "BankPerfAgg", year = None, between = ["1990-01-01","2017-12-31"]):
    if year is not None:
        RSSD_ID_tmp = list(BankPerf[keyname]["RSSD_ID"][BankPerf[keyname]["ReportingDate"].isin(year)])
    else:
        RSSD_ID_tmp = list(BankPerf[keyname]["RSSD_ID"].unique())
    BankPerf_tmp = BankPerf[keyname][(BankPerfAgg[keyname]["ReportingDate"] >= between[0]) & (BankPerfAgg[keyname]["ReportingDate"] <= between[1]) & (BankPerf[keyname]["RSSD_ID"].isin(RSSD_ID_tmp))]

    print_tmp = BankPerf_tmp[BankPerf_tmp.columns[pd.Series(BankPerf_tmp.columns).str.startswith("Loans categories:")]].describe().transpose()
    print(print_tmp)
    print_tmp = print_tmp.reset_index().rename({"index":"varname"}, axis = 1)
    return(print_tmp)

def BankPerf_ConsecutiveQtrs_Reduce(BankPerf,concecutiveqtrs = 8,BankPerf_Calc_df_name = None, sincedt = "1989-12-31"):
    if BankPerf_Calc_df_name in BankPerf.keys():
        print("Getting BankPerf Calculated rows")
        BankPerf_Calc_df = BankPerf[BankPerf_Calc_df_name]
    else:
        return("BankPerf Calculated DF not found")


    print("Filter Banks for those that are above::",concecutiveqtrs,"consecutive quarters.")
    BankPerf_Merger_df_gbsum = BankPerf_Calc_df
    BankPerf_Merger_df_gbsum = BankPerf_Merger_df_gbsum.sort_values(['RSSD_ID', 'ReportingDate'])

    print("Filtering Since Date", sincedt)
    BankPerf_Merger_df_gbsum_tmp = BankPerf_Merger_df_gbsum[BankPerf_Merger_df_gbsum["ReportingDate"] > sincedt]
    BankPerf_Merger_df_gbsum_tmp["ConsecutivePeriods"] = BankPerf_Merger_df_gbsum_tmp.sort_values(['RSSD_ID', 'ReportingDate']).groupby(['RSSD_ID']).cumcount() + 1



    BankPerf_Merger_df_gb_tmp = BankPerf_Merger_df_gbsum_tmp.groupby('RSSD_ID')
    BankPerf_Merger_df_gb_tmp = BankPerf_Merger_df_gb_tmp.agg({"ConsecutivePeriods": ['min', 'max', 'count']})

    RSSD_ID_consecutive = list(BankPerf_Merger_df_gb_tmp[BankPerf_Merger_df_gb_tmp["ConsecutivePeriods"]["max"] >= concecutiveqtrs].index)
    BankPerf_Merger_df_gbsum = BankPerf_Merger_df_gbsum[BankPerf_Merger_df_gbsum["RSSD_ID"].isin(RSSD_ID_consecutive)]

    BankPerf["BankPerf_ConsecutiveReduced"] = BankPerf_Merger_df_gbsum
    print(BankPerf["BankPerf_ConsecutiveReduced"].describe().transpose())

    return(BankPerf)

def BankPerf_Merger_process(BankPerf, merger_df_name = "merger_info_frb", BankPerf_Calc_df_name = None):
    if merger_df_name in BankPerf.keys():
        print("Getting Merger Info from Dictionary")
        merger_info_df = BankPerf[merger_df_name]
        merger_info_df = merger_info_df[["MERGE_DT","NON_ID","SURV_ID"]]
        merger_info_df["MERGE_DT"] = pd.to_datetime(merger_info_df["MERGE_DT"], format = "%Y%M%d")
        merger_info_df["NON_ID"] = merger_info_df["NON_ID"].astype(int)
       # print(merger_info_df.info())
       # print(merger_info_df.describe())
    else:
        return("Merger Information not found in dictionary")

    if BankPerf_Calc_df_name in BankPerf.keys():
        print("Getting BankPerf Calculated rows")
        BankPerf_Calc_df = BankPerf[BankPerf_Calc_df_name]
    else:
        return("BankPerf Calculated DF not found")

    print("Loop/merge through merger information and update RSSD of non survivor with survivor ID.")
#    orig_columns = list(BankPerf_Calc_df.columns)
    BankPerf_Merger_df = BankPerf_Calc_df.merge(merger_info_df, left_on = "RSSD_ID", right_on = "NON_ID", how = "left")
    print("Found",BankPerf_Merger_df["RSSD_ID"][pd.notnull(BankPerf_Merger_df["NON_ID"])].unique().__len__(),' Non surviving RSSD_IDs')
    print("Updating Non-Surviving RSSD_IDs with Surviving IDs")
    BankPerf_Merger_df.loc[pd.notnull(BankPerf_Merger_df["NON_ID"]), ["RSSD_ID"]] = BankPerf_Merger_df.loc[pd.notnull(BankPerf_Merger_df["NON_ID"]), ["SURV_ID"]]
    print("Dropping Merge Columns")
    BankPerf_Merger_df = BankPerf_Merger_df.drop(list(merger_info_df.columns), axis = 1 )
    print("Re Merging to check for remaining mergers")
    BankPerf_Merger_df = BankPerf_Merger_df.merge(merger_info_df, left_on = "RSSD_ID", right_on = "NON_ID", how = "left")
    print("Number of NON_ID matching with RSSD_ID:", BankPerf_Merger_df[pd.notnull(BankPerf_Merger_df["NON_ID"])].shape)
    #if BankPerf_Merger_df[pd.notnull(BankPerf_Merger_df["NON_ID"])].shape == 0:
    BankPerf["BankPerf_Mergered"] = BankPerf_Merger_df
    print(BankPerf["BankPerf_Mergered"].describe().transpose())

    return(BankPerf)

def BankPerf_Aggregation_process(BankPerf, BankPerf_ToAgg_df_name= None):

    if BankPerf_ToAgg_df_name in BankPerf.keys():
        print("Getting BankPerf Calculated rows")
        BankPerf_Merger_df_gbsum = BankPerf[BankPerf_ToAgg_df_name]
    else:
        return("BankPerf Calculated DF not found")


    print("Aggregate all the updated RSSD_ID columns")

    #BankPerf_Merger_df_gbsum.columns
    BankPerf_Merger_df_gbsum = BankPerf_Merger_df_gbsum.groupby(["RSSD_ID","ReportingDate"])
    BankPerf_Merger_df_gbsum = BankPerf_Merger_df_gbsum.agg({
        #'TotalAssets' : np.sum,
        'Net charge-offs by type of loan:Commercial & industrial' : np.sum,
        'Net charge-offs by type of loan:Construction & land development' : np.sum,
        'Net charge-offs by type of loan:Multifamily real estate' : np.sum,
        'Net charge-offs by type of loan:(Nonfarm) nonresidential CRE' : np.sum,
        'Net charge-offs by type of loan:Home equity lines of credit' : np.sum,
        'Net charge-offs by type of loan:Residential real estate (excl. HELOCs)' : np.sum,
        'Net charge-offs by type of loan:Credit card' : np.sum,
        'Net charge-offs by type of loan:Consumer (excl. credit card)' : np.sum,
        'Loans categories:Commercial & industrial' : np.sum,
        'Loans categories:Construction & land development' : np.sum,
        'Loans categories:Multifamily real estate' : np.sum,
        'Loans categories:Nonfarm nonresidential CRE' : np.sum,
        'Loans categories:Home equity lines of credit' : np.sum,
        'Loans categories:Residential real estate (excl. HELOCs)' : np.sum,
        'Loans categories:Credit card' : np.sum,
        'Loans categories:Consumer (excl. credit card)' : np.sum,
        'Components of pre-provision net revenue:Net interest income' : np.sum,
        'Components of pre-provision net revenue:Noninterest income' : np.sum,
        'Components of pre-provision net revenue:Trading income' : np.sum,
        'Components of pre-provision net revenue:Compensation expense' : np.sum,
        'Components of pre-provision net revenue:Fixed assets expense' : np.sum,
        'Components of pre-provision net revenue:Noninterest expense' : np.sum,
        'Other items:Consolidated assets' : np.sum,
        'Other items:Interest-earning assets' : np.sum,
        'Other items:Trading assets' : np.sum,
        'Other items:Book equity' : np.sum,
        'Other items:Risk-weighted assets' : np.sum,
        'Other items:Dividends ' : np.sum,
        'Other items:Stock purchases' : np.sum,
        'Other items:Tier 1 common equity' : np.sum,
        'Other items:= Tier 1 capital' : np.sum,
        'Other items: = - Perpetual preferred stock' : np.sum,
        'Other items: = + Nonqual. Perpetual preferred stock' : np.sum,
        'Other items: = - Qual. class A minority interests' : np.sum,
        'Other items: = - Qual. restricted core capital' : np.sum,
        'Other items: = - Qual. mandatory convert. pref. sec.' : np.sum
    })
    #May need to add manual column sums.
    BankPerf_Merger_df_gbsum = BankPerf_Merger_df_gbsum.reset_index()

    BankPerf["BankPerfAgg"] = BankPerf_Merger_df_gbsum
    print(BankPerf["BankPerfAgg"].describe().transpose())

    return(BankPerf)

def bankperf_rates_ratios(BankPerf, replace_nan = True):
    #Need to handle NAN rows and outliers.
    if replace_nan:
        print("Replacing nans, infs, and -infs in dataframe")
        BankPerf = BankPerf[~BankPerf.isin([np.nan, np.inf, -np.inf]).any(1)]

    print("Loans categories Descriptive Statistics")
    #BankPerf = BankPerf[~BankPerf[BankPerf.columns[pd.Series(BankPerf.columns).str.startswith("Loans categories:")]].isin([np.nan, np.inf, -np.inf]).any(1)]
    #print(BankPerf[BankPerf.columns[pd.Series(BankPerf.columns).str.startswith("Loans categories:")]].describe().transpose())

    print("Net Charge-Off rates by type of loan calculations")
    for nco, loan in zip(list(BankPerf.columns[pd.Series(BankPerf.columns).str.startswith("Net charge-offs by type of loan:")]),list(BankPerf.columns[pd.Series(BankPerf.columns).str.startswith("Loans categories:")])):
        tmp_str = nco.replace("Net charge-offs by type of loan:","ncoR:")
        print(tmp_str)
        BankPerf[tmp_str] = ((BankPerf[nco].astype(float)) / (BankPerf[loan].astype(float)))
        BankPerf[tmp_str] = BankPerf[tmp_str] * 400
        #BankPerf = pd.concat([BankPerf,BankPerf_tmp[tmp_str]], ignore_index=True, axis = 1)
    print(BankPerf[BankPerf.columns[pd.Series(BankPerf.columns).str.startswith("ncoR:")]].describe().transpose())

    print("PPNR Ratios calculations")
    for ppnr_comp in list(BankPerf.columns[pd.Series(BankPerf.columns).str.startswith("Components of pre-provision net revenue:")]):
        tmp_str = ppnr_comp.replace("Components of pre-provision net revenue:", "ppnrRatio:")
        print(tmp_str)
        BankPerf[tmp_str] = ((BankPerf[ppnr_comp].astype(float))/(BankPerf['Other items:Consolidated assets'].astype(float)))
        BankPerf[tmp_str] = BankPerf[tmp_str]  * 400
    print(BankPerf[BankPerf.columns[pd.Series(BankPerf.columns).str.startswith("ppnrRatio:")]].describe().transpose())

    #Balance sheet composition indicators
    #Requires interest earning assets calculations.

    return(BankPerf)


#Find best way to subset the data for Yi and Xi
#A Wrapper function can handle these additional tasks to produce a Dict with the Dataframes necessary




#VAE-CVAE=MNIST
#import utils, models, train
