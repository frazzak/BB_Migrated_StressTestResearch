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
        #print("Initializing Raw Data Frame Dict")
        #Z_macro_raw_data_dict = {}
        #print("Searching path for Z_Macro:", self.Z_macro_dir)
        #for dirName, subdirList, fileList in os.walk(self.Z_macro_dir):
        #    print('Found directory: %s' % dirName)
        #        for fname in fileList:
        #            if fname.endswith(filetype):
        #                print("Reading File and Adding to Dataframe Dictionary")
        #                print('\t%s' % fname)
        #                exec('''Z_macro_raw_data_dict[fname.split(filetype)[0]] = pd.read_csv("'''+ os.path.join(self.Z_macro_dir,fname) + '''")''')
        #                print("Updating Date Column to be Date type and handle Quarters.")
        Z_macro_raw_data_dict = self.file_dict_read(self.Z_macro_dir,filetype = ".csv")

        for k in Z_macro_raw_data_dict.keys():
            print(k)
            Z_macro_raw_data_dict[k]["pdDate"] = pd.to_datetime(Z_macro_raw_data_dict[k]["Date"].apply(lambda x: "-".join(x.split(" "))).str.replace(r'(Q\d) (\d+)', r'\2-\1'), errors = 'coerce')
            print("Dropping Original Date Column")
            Z_macro_raw_data_dict[k] = Z_macro_raw_data_dict[k].drop("Date", axis = 1)

        return(Z_macro_raw_data_dict)

    def X_Y_bankingchar_perf_process(self,filetype = ".csv"):


        return("Banking Characterisitcs and Performance Indicators")








#Entry Point
varpath_dict_args = {
                    "Z_macro_dir" : "Z_macro/",
                    "Z_micro_dir" : "Z_micro/",
                    "BankPerf_dir" : "BankPerf/",
                    "Data_dir" : "/Users/phn1x/Google Drive/Spring 2019/LossProjection_Research/Data"
                    }




InputVars_ST = StressTestData(varpath_dict_args).Z_macro_process()

Z_macro = test["Historic_Domestic"]

Z_macro.info()



Z_macro[Z_macro["pdDate"] <= "2017"].describe().transpose()