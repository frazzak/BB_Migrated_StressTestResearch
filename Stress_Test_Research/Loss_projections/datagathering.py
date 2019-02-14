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


        print("Preparing Calculated Fields")
        BankCharPerf_raw_data_dict[params_dict["Calc_df"]]["VarList"] = BankPerf[params_dict["Calc_df"]]["Report: FR Y-9C"].astype(str).apply(
                                                                    lambda x: x.replace("+", ",").replace("-", ",").replace("(", "").replace(")", "").replace(". . .","").replace(
                                                                     ",  ,", ",").split(","))
        return(BankCharPerf_raw_data_dict)








#Entry Point
varpath_dict_args = {
                    "Z_macro_dir" : "Z_macro/",
                    "Z_micro_dir" : "Z_micro/",
                    "BankPerf_dir" : "BankPerf/",
                    "Data_dir" : "/Users/phn1x/Google Drive/Spring 2019/LossProjection_Research/Data"
                    }




Z_macro = StressTestData(varpath_dict_args).Z_macro_process()

BankPerf = StressTestData(varpath_dict_args).X_Y_bankingchar_perf_process()


BankPerf[params_dict["Calc_df"]]["VarList"] = BankPerf[params_dict["Calc_df"]]["Report: FR Y-9C"].astype(str).apply(lambda x :  x.replace("+",",").replace("-",",").replace("(","").replace(")","").replace(". . .", "").replace(",  ,",",").split(","))


BankPerf[params_dict["Calc_df"]][["Variable","Report: FR Y-9C","VarList"]]


BankPerf[params_dict["BankPerf_raw"]]