import pandas as pd
import numpy as np
import os

class StressTestData():
    def __init__(self,varpath_dict):
        print("Initialize Objects")
        print("Set Data Working Directories")

        for k,v in varpath_dict_args.items():
            print(k,v)
            if k != "Data_dir":
                exec('''self.'''+ k + ''' = "''' + os.path.join(varpath_dict_args["Data_dir"],v) + '''"''')
                exec("print("+ k + ")")
            else:
                exec('''self.''' + k + ''' = "''' + v + '''"''')
                exec("print(" + k + ")")
                exec("os.chdir("+k+")")

        print("Initalizing Return Dict Object")
        self.final_df = {}

    def file_dict_read(self, rootdir,  filetype = ".csv"):
        print("Initializing Raw Data Frame Dict")
        tmp_dict = {}
        print("Searching path for Z_Macro:", rootdir)
        for dirName, subdirList, fileList in os.walk(rootdir):
            print('Found directory: %s' % dirName)
            for fname in fileList:
                if fname.endswith(filetype):
                    print("Reading File and Adding to Dataframe Dictionary")
                    print('\t%s' % fname)
                    exec('''tmp_dict[fname.split(filetype)[0]] = pd.read_csv("''' + os.path.join(
                        rootdir, fname) + '''")''')
        return(tmp_dict)


    def Z_macro_process(self, filetype = ".csv"):
        Z_macro_raw_data_dict = self.file_dict_read(self.Z_macro_dir,filetype = ".csv")

        for k in Z_macro_raw_data_dict.keys():
            print(k)
            Z_macro_raw_data_dict[k]["pdDate"] = pd.to_datetime(Z_macro_raw_data_dict[k]["Date"].apply(lambda x: "-".join(x.split(" "))).str.replace(r'(Q\d) (\d+)', r'\2-\1'), errors = 'coerce')
            print("Dropping Original Date Column")
            Z_macro_raw_data_dict[k] = Z_macro_raw_data_dict[k].drop("Date", axis = 1)

        return(Z_macro_raw_data_dict)

    def X_Y_bankingchar_perf_process(self,filetype = ".csv",
                                     params_dict = {"Calc_df" : "NCO_PPNR_Loan_Calcs",
                                                    "BankPerf_raw" : "WRDS_Covas_BankPerf",
                                                    "MergerInfo " : "merger_info_frb"}
                                     ):
        BankCharPerf_raw_data_dict = self.file_dict_read(self.BankPerf_dir, filetype=".csv")

        print("Fixing Column Names with Errors")

        print("Preparing Calculated Fields")
        print("Fixing Errors")
        print("Fixing BHCK892 to BHCKC892")
        print("Creating VARLIST Column")
        BankCharPerf_raw_data_dict[params_dict["Calc_df"]]["VarList"] = BankPerf[params_dict["Calc_df"]]["Report: FR Y-9C"].astype(str).apply(lambda x : [i.strip() for i in  x.replace("BHCK892","BHCKC892").replace("+",",").replace("-",",").replace("(","").replace(")","").replace(". . .", "").replace(",  ,",",").split(",")])

        print("Additional Clean up for Varlist")
        BankCharPerf_raw_data_dict[params_dict["Calc_df"]]["VarList"] =  BankCharPerf_raw_data_dict[params_dict["Calc_df"]]["VarList"].apply(lambda x: [i for i in x if i not in ["",np.nan,"nan"]])

        print("Checking for Column name Discrepancy")
        miss_col = self.check_missing_BHCCodes(BankCharPerf_raw_data_dict,params_dict)

        print("Creating Derived Columns")
        BankCharPerf_raw_data_dict = self.BHC_loan_nco_ppnr_create(BankCharPerf_raw_data_dict,params_dict)
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

    def create_calculated_fields(self, BankPerf_Raw_df, Calc_df):
        return("Hello, Work In progress")

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
        BankPerf_result = pd.concat([BankPerf[params_dict["BankPerf_raw"]], loan_nco_ppnr_df], axis=1)
        return (BankPerf_result)


#Entry Point
varpath_dict_args = {
                    "Z_macro_dir" : "Z_macro/",
                    "Z_micro_dir" : "Z_micro/",
                    "BankPerf_dir" : "BankPerf/",
                    "Data_dir" : "/Users/phn1x/Google Drive/Spring 2019/LossProjection_Research/Data"
                    }




Z_macro = StressTestData(varpath_dict_args).Z_macro_process()

BankPerf = StressTestData(varpath_dict_args).X_Y_bankingchar_perf_process()

