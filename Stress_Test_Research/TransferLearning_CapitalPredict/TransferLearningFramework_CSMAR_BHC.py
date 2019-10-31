import logging
import torch
from functools import reduce
import os,sys,gc
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import itertools

#Models
from MultiVAE import m_LSTM,m_GRU, m_DA_RNN, m_LinReg
from statsmodels.tsa.vector_ar.var_model import VAR
from pmdarima.arima import auto_arima
from math import sqrt
from sklearn.metrics import mean_squared_error

# from statsmodels.tsa.api import VAR
# from statsmodels.tsa.stattools import adfuller
# from statsmodels.tools.eval_measures import rmse, aic

def setup_log(tag='VOC_TOPICS'):
    # create logger
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = logging.getLogger(tag)
    logger.handlers = []
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    # logger.handlers = []
    logger.addHandler(ch)
    return logger




class BHC_CSMAR_TLF():
    def __init__(self, varpath_dict_args={
        "rootdir": "/Users/phn1x/icdm2018_research_BB/Stress_Test_Research/TransferLearning_CapitalPredict/",
        "Indcol": "Indnme",
        "Industry": "Finance",
        "datadir": "data",
        "CSMAR_CompProfile": "CSMAR_CompayProfiles.csv",
        "CSMAR_BalanceSheet": "CSMAR_BalanceSheet_All.csv",
        "CSMAR_IncomeStatement": "CSMAR_all_IncomeStatement.csv",
        "CSMAR_BHC_Codes": "CSMAR_BHC_Codes.csv",
        "CSMAR_MacroData": "China_MacroEconomicData.csv",
        "BHC_Full_Data": "BHC_Full_Data.csv",
        "BHC_MacroData_Dom": "BHC_MacroEconomicData_Domestic.csv",
        "BHC_MacroData_Intl": "BHC_MacroEconomicData_International.csv",
        "CSMAR_MacroData": "CSMAR_MacroEconomicData.csv"
    }):
        print("Initialize Objects")
        self.file_dict = {}
        self.logger = setup_log(tag='BHCCSMAR_CapitalPredict')
        self.logger.info(f"Initialize Class CSMAR Capital Predict: Started")
        print("Set CSMAR Data Working Directories")
        for k, v in varpath_dict_args.items():
            # print(k,v)
            if k == "rootdir":
                # Changing to Root Directory
                exec('''self.''' + k + ''' = "''' + v + '''"''')
                # exec("print(self." + k + ")")
                exec("os.chdir(self." + k + ")")
            if not k.startswith(("CSMAR", "BHC")):
                # Setting variable names to class variables.
                exec('''self.''' + k + ''' = "''' + v + '''"''')
                # exec("print(self." + k + ")")
            if k.startswith(("CSMAR", "BHC")) and v.endswith(".csv"):
                self.file_dict[k] = os.path.join(varpath_dict_args["rootdir"], varpath_dict_args["datadir"], v)

            # print(self.file_dict)
        self.logger.info(f"Initialize Class TFL: Completed")

    def load_BHC_CSMAR_Codes(self):
        self.logger.info(f"load_BHC_CSMAR_Codes File Loader: Started")
        tmp_dict = {}

        for k, v in self.file_dict.items():
            if k.startswith("CSMAR_BHC_Codes"):
                print("Initializing:", k)
                print("Loading File:", v)
                tmp = pd.read_csv(v)
                print("Loading Complete")

        varname_tmp = [x for x in tmp.columns if "Variable Name" in x]
        datatype_tmp = [x for x in tmp.columns if "Data Type" in x]

        print("Filter only NUM Data Types")
        tmp_test = tmp[(tmp[datatype_tmp[0]] == "NUM") & (tmp[datatype_tmp[1]] == "NUM")]

        # CSMAR
        # tmp_dict = {}
        tmp_dict["CSMAR_Codes"] = list(tmp_test[varname_tmp[0]].unique())

        # BHC
        tmp_dict["BHC_Codes"] = list(tmp_test[varname_tmp[1]].unique())

        self.logger.info(f"CSMAR Data File Loader: Completed")
        self.Codes_Dict = tmp_dict
        return

    def CSMAR_Data_MergeLoad(self, CompanyProfile="CSMAR_CompProfile", ref_dict="CSMAR_BalanceSheet", idcol="Stkcd",
                             mergecols=["Stkcd", "Accper"], checkfileexists=True, existspath="./data/CSMAR_Subset.csv",
                             normalize=True):

        if checkfileexists and os.path.exists(existspath):  # TODO: Add logic to check for file
            print(existspath, "File Exists, Loading now")
            self.CSMAR_Data_Raw = pd.read_csv(existspath, sep=',')

        else:
            self.logger.info(f"CSMAR Data Filtering and Object Merging: Started")
            print("Loading CSMAR Files")
            tmp_dict = {}

            for k, v in self.file_dict.items():
                if k.startswith("CSMAR_") and not k.endswith("_BHC_Codes"):
                    print("Initializing:", k)
                    print("Loading File:", v)
                    tmp_dict[k] = pd.read_csv(v)
                    print("Loading Complete")

            tmp_ids = tmp_dict[CompanyProfile][tmp_dict[CompanyProfile][test.Indcol] == test.Industry][idcol].unique()
            # tmp_ids = dict[CompanyProfile][dict[CompanyProfile]["Indnme"] == "Finance"]["Stkcd"].unique()
            print(test.Industry, "count:", tmp_ids.__len__())

            print("Filtering to Industry Ids")
            tmp_dict[ref_dict] = tmp_dict[ref_dict][tmp_dict[ref_dict][idcol].isin(tmp_ids)]

            dfList = [tmp_dict[x] for x in tmp_dict.keys() if not x.endswith((CompanyProfile, "MacroData"))]
            # print("Preparing to merge:", dfList, "by:",idcol)
            # print("Converting idCol to INT")

            df = reduce(lambda df1, df2: pd.merge(df1, df2, on=mergecols), dfList)

            print("Applying Column Rules")
            df = df.drop([x for x in df.keys() if x.endswith("_y")], 1)
            df.columns = [x.replace("_x", "") for x in df.keys()]
            print("Applying Manual Rule to select Company Level Statements")
            df = df[df["Typrep"] == 'A'].drop("Typrep", 1)

            df.columns = df.columns.str.upper()

            df = df[[idcol.upper(), "ACCPER"] + test.Codes_Dict["CSMAR_Codes"]]
            self.logger.info(f"CSMAR Data Filtering and Object Merging: Completed")

            df['ACCPER'] = pd.to_datetime(df['ACCPER'], format='%Y%m%d')
            df['ACCPER'][(pd.to_datetime(df['ACCPER']).dt.day == 1) & (pd.to_datetime(df['ACCPER']).dt.month == 1)] = \
            df['ACCPER'][
                (pd.to_datetime(df['ACCPER']).dt.day == 1) & (pd.to_datetime(df['ACCPER']).dt.month == 1)].apply(
                lambda x: pd.to_datetime(str(pd.to_datetime(x).year - 1) + '-12-31'))

            gc.collect()
            self.CSMAR_Data_Raw = df
            self.CSMAR_Data_Raw.to_csv(existspath, sep=',', index=False)
        if normalize:
            self.CSMAR_Data_Raw = self.nomalize_bankdf(self.CSMAR_Data_Raw, exclude_columns=['STKCD', 'ACCPER'])

        return

    def BHC_Data_Load(self, checkfileexists=True, existspath="./data/BHC_Subset.csv",
                      id_cols=["RSSD9001", "RSSD9999", "RSSD9017", "BHCK3368"], consqtr=True, concecutiveqtrs=8,
                      sincedt='1989-12-31', topbanks=True, topbanksorderby='BHCK3368', topbankslimit=1000,
                      normalize=True):
        if checkfileexists and os.path.exists(existspath):  # TODO: Add logic to check for file
            print(existspath, "File Exists, Loading now")
            self.BHC_Data_Raw = pd.read_csv(existspath, sep=',')
        else:
            # Load BHC Files and Subset appropriately
            self.logger.info(f"BHC Data Filtering and Object Merging: Started")
            print("Loading BHC Files")
            tmp_dict = {}

            for k, v in test.file_dict.items():
                if k.startswith("BHC_") and not k.endswith("_BHC_Codes"):
                    print("Initializing:", k)
                    print("Loading File:", v)
                    tmp_dict[k] = pd.read_csv(v, engine='python')
                    print("Loading Complete")
                    gc.collect()

            if "BHC_Full_Data" in tmp_dict.keys():
                print("Capitalize Column names in BHC_Full_Data")
                tmp_dict["BHC_Full_Data"].columns = tmp_dict["BHC_Full_Data"].columns.str.upper()

                print("Remove duplicated columns")
                tmp_dict["BHC_Full_Data"] = tmp_dict["BHC_Full_Data"].loc[:,
                                            ~tmp_dict["BHC_Full_Data"].columns.duplicated()]

                print("Subsetting Dataset with ID Cols and BHC_Codes")
                tmp_subset = tmp_dict["BHC_Full_Data"][id_cols + self.Codes_Dict["BHC_Codes"]]

                print("Formatting Date Column to Datetime data type")
                tmp_subset['RSSD9999'] = pd.to_datetime(tmp_subset['RSSD9999'], format="%Y%m%d")

                if consqtr:
                    print("Determining RSSD9001 Ids that have at least", concecutiveqtrs, "quarters of data.")
                    BankPerf_Merger_df_gbsum = tmp_subset
                    BankPerf_Merger_df_gbsum = BankPerf_Merger_df_gbsum.sort_values(['RSSD9001', 'RSSD9999'])

                    print("Filtering Since Date", sincedt)
                    BankPerf_Merger_df_gbsum_tmp = BankPerf_Merger_df_gbsum[
                        BankPerf_Merger_df_gbsum["RSSD9999"] > sincedt]
                    BankPerf_Merger_df_gbsum_tmp["ConsecutivePeriods"] = BankPerf_Merger_df_gbsum_tmp.sort_values(
                        ['RSSD9001', 'RSSD9999']).groupby(['RSSD9001']).cumcount() + 1
                    BankPerf_Merger_df_gb_tmp = BankPerf_Merger_df_gbsum_tmp.groupby('RSSD9001')
                    BankPerf_Merger_df_gb_tmp = BankPerf_Merger_df_gb_tmp.agg(
                        {"ConsecutivePeriods": ['min', 'max', 'count']})
                    RSSD_ID_consecutive = list(BankPerf_Merger_df_gb_tmp[
                                                   BankPerf_Merger_df_gb_tmp["ConsecutivePeriods"][
                                                       "count"] >= concecutiveqtrs].index)
                    print("Subsetting Banks with Consecutive Quarters requirement from the data.")
                    BankPerf_Merger_df_gbsum = BankPerf_Merger_df_gbsum[
                        BankPerf_Merger_df_gbsum["RSSD9001"].isin(RSSD_ID_consecutive)]
                else:
                    BankPerf_Merger_df_gbsum = tmp_subset

                if topbanks:
                    print("Filtering TopBanks based on", topbanksorderby)
                    topbanks_filter = BankPerf_Merger_df_gbsum[["RSSD9001", "RSSD9999", topbanksorderby]]

                    RSSD_IDS = (topbanks_filter.groupby('RSSD9001')
                                .agg({'RSSD9999': 'count', topbanksorderby: 'mean'})
                                .reset_index()
                                .rename(
                        columns={'RSSD9999': 'ReportingDate_count', topbanksorderby: topbanksorderby + '_avg'})
                                ).sort_values([topbanksorderby + '_avg', 'ReportingDate_count'], ascending=False).head(
                        topbankslimit).reset_index(drop=True)["RSSD9001"]
                    BankPerf_Merger_df_gbsum = BankPerf_Merger_df_gbsum[
                        BankPerf_Merger_df_gbsum['RSSD9001'].isin(RSSD_IDS)]

                print("Assigning BHC data to class object")
                self.BHC_Data_Raw = BankPerf_Merger_df_gbsum
                self.BHC_Data_Raw.to_csv(existspath, sep=',', index=False)
        if normalize:
            self.BHC_Data_Raw = self.nomalize_bankdf(self.BHC_Data_Raw,
                                                     exclude_columns=['RSSD9001', 'RSSD9999', 'RSSD9017'])

        return

    def nomalize_bankdf(self, df_tmp, exclude_columns):

        print('Normalizing DataFrame')
        # Drop Sparse Columns TODO: Consider this later with PCA maybe
        # df_tmp.shape
        # df_tmp.dropna(thresh=10000, axis='columns').shape

        print('Interpolate missing cvalues and fill NAN')

        # df = tmp_dict['CSMAR_MacroData'].fillna(method= 'bfill').fillna(method= 'ffill')
        df = df_tmp.interpolate(method='linear')
        df = df.fillna(method='bfill').fillna(method='ffill')
        # Normalize

        # df = tmp_dict['CSMAR_MacroData']
        print('Columns to be Excluded:', exclude_columns)
        df_tmp = df[df.columns.difference(exclude_columns)]
        df_tmp = df_tmp.astype(float)
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
        x_train_normalized = pd.DataFrame(
            x_train)  # It is required to have X converted into a data frame to later plotting need
        # print("Reapplying Column Names")
        x_train_normalized.columns = columns
        df_norm = x_train_normalized

        print("Recombining Excluded Columns")
        df_norm = pd.concat([df[exclude_columns].reset_index(drop=True), df_norm.reset_index(drop=True)], axis=1)

        print('Returning Normalized DataFrame')
        # Set to df_norm to test.BHC_Data_Raw then run subset and 3d again.
        return (df_norm)

    def BHC_MarcoEcon_Load(self, exclude_columns=["Date"], n_components=12):
        # Load EconData BHC
        # Load Files
        self.logger.info(f"BHC EconData Loading and Cleaning: Started")
        tmp_dict = {}
        for k, v in test.file_dict.items():
            if k.startswith("BHC_MacroData"):
                print("Initializing:", k)
                print("Loading File:", v)
                tmp_dict[k] = pd.read_csv(v)
                print("Loading Complete")

        if len(tmp_dict.keys()) > 0:
            # Clean both, alter date formatting, subset, combine, normalize, dim reduction.
            print("Combine and Merge")
            dfs = list()
            for k in tmp_dict.keys():
                print(k)
                tmp_dict[k]["Date"] = pd.to_datetime(tmp_dict[k]["Date"].apply(
                    lambda x: x.replace(' Q1', "-03-31").replace(" Q2", '-06-30').replace(" Q3", '-09-30').replace(
                        " Q4", "-12-31")), errors='coerce')
                dfs.append(tmp_dict[k])

            print("Combine List with Left Merge")
            tmp_final_df = reduce(lambda left, right: pd.merge(left, right, on="Date", how="left"), dfs)

            print("Drop Columns")
            tmp_final_df = tmp_final_df[
                tmp_final_df.columns.difference([x for x in tmp_final_df.columns if x.startswith("Scenario Name")])]

            print("Normalize DF")

            print("Interpolate")
            # df = tmp_final_df.fillna(method= 'bfill').fillna(method= 'ffill')
            df = tmp_final_df.interpolate(method='linear')
            df = df.fillna(method='bfill').fillna(method='ffill')

            print("Normalizing Data Frame")
            df_tmp = df[tmp_final_df.columns.difference(exclude_columns)]
            df_tmp = df_tmp.astype(float)
            scale = StandardScaler()
            x_train = df_tmp
            columns = x_train.columns
            x_train = scale.fit_transform(x_train)
            x_train_normalized = pd.DataFrame(
                x_train)  # It is required to have X converted into a data frame to later plotting need
            # print("Reapplying Column Names")
            x_train_normalized.columns = columns
            df_norm = x_train_normalized
            print("Recombining Excluded Columns")
            df_norm = pd.concat([tmp_final_df[exclude_columns].reset_index(drop=True), df_norm.reset_index(drop=True)],
                                axis=1)

            # PCA
            print("Performing PCA dimension reduction")
            x_train_normalized = df_norm[df_norm.columns.difference(exclude_columns)]
            pca = PCA(n_components=n_components)
            x_train_projected = pca.fit_transform(x_train_normalized)
            pca_cols = ['PCA_' + str(i) for i in range(0, x_train_projected.shape[1])]
            pca_df = pd.DataFrame(x_train_projected, columns=pca_cols).reset_index(drop=True)
            pca_normalized = pd.concat([df_norm[exclude_columns].reset_index(drop=True), pca_df], axis=1)
            self.BHC_EconData_PCA = pca_normalized

        return

    def CSMAR_MacroEcon_Load(self, n_components=12, exclude_columns=['q_dates_data']):  # TODO: Add arguments
        # Load EconData CSMAR
        self.logger.info(f"CSMAR EconData Loading and Cleaning: Started")
        print("Loading CSMAR Files")
        tmp_dict = {}

        for k, v in self.file_dict.items():
            if k.startswith("CSMAR_MacroData"):
                print("Initializing:", k)
                print("Loading File:", v)
                tmp_dict[k] = pd.read_csv(v, skiprows=1)
                print("Loading Complete")

        if "CSMAR_MacroData" in tmp_dict.keys():
            print("Convert Date Column Accordingly")
            tmp_dict['CSMAR_MacroData']['q_dates_data'] = tmp_dict['CSMAR_MacroData']['q_dates_data'].astype(str).apply(
                lambda x: x.replace('.0', '-03-31').replace('.25', '-06-30').replace('.5', '-09-30').replace('.75',
                                                                                                             '-12-31'))
            tmp_dict['CSMAR_MacroData']['q_dates_data'] = pd.to_datetime(tmp_dict['CSMAR_MacroData']['q_dates_data'])

            # Filter to 1990 onward
            tmp_dict['CSMAR_MacroData'] = tmp_dict['CSMAR_MacroData'][
                tmp_dict['CSMAR_MacroData']['q_dates_data'] >= '1990-01-01']

            # Drop Sparse Columns
            tmp_dict['CSMAR_MacroData'] = tmp_dict['CSMAR_MacroData'].dropna(thresh=100, axis='columns')

            # Interpolate

            # df = tmp_dict['CSMAR_MacroData'].fillna(method= 'bfill').fillna(method= 'ffill')
            df = tmp_dict['CSMAR_MacroData'].interpolate(method='linear')
            df = df.fillna(method='bfill').fillna(method='ffill')
            # Normalize
            # df = tmp_dict['CSMAR_MacroData']

            df_tmp = df[df.columns.difference(exclude_columns)]
            df_tmp = df_tmp.astype(float)
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
            x_train_normalized = pd.DataFrame(
                x_train)  # It is required to have X converted into a data frame to later plotting need
            # print("Reapplying Column Names")
            x_train_normalized.columns = columns
            df_norm = x_train_normalized

            print("Recombining Excluded Columns")
            df_norm = pd.concat([df[exclude_columns].reset_index(drop=True), df_norm.reset_index(drop=True)], axis=1)

            # df_norm.describe()

            # PCA
            print("Performing PCA dimension reduction")
            x_train_normalized = df_norm[df_norm.columns.difference(exclude_columns)]
            pca = PCA(n_components=n_components)
            x_train_projected = pca.fit_transform(x_train_normalized)
            # print("Correlation Heatmap Calculations for first two componets")
            pca_cols = ['PCA_' + str(i) for i in range(0, x_train_projected.shape[1])]
            pca_df = pd.DataFrame(x_train_projected, columns=pca_cols).reset_index(drop=True)
            # pca_normalized = pd.concat([pca_df, x_train_normalized.reset_index(drop=True)], axis=1)
            pca_normalized = pd.concat([df_norm[exclude_columns].reset_index(drop=True), pca_df], axis=1)
            self.CSMAR_EconData_PCA = pca_normalized
        return

    def Bank_Data_Subsetting(self, bhc_id_cols=['RSSD9001', 'RSSD9999'], bhc_target_cols=['BHCT3247'],
                             bhc_exclude_cols=['RSSD9017'], csmar_id_cols=['STKCD', 'ACCPER'],
                             csmar_target_cols=['A003105000'], csmar_exclude_cols=[],
                             bank_dates=['1991-12-31', '2018-12-31'], econ_dates=['1991-12-31', '2018-12-31']):
        self.logger.info(f"Subsetting Bank Input and Target Data: Started")
        print("Subsetting Inputs and Targets for BHC")
        # inputs
        BHC_X = self.BHC_Data_Raw[self.BHC_Data_Raw.columns.difference(bhc_target_cols + bhc_exclude_cols)]
        BHC_X = BHC_X.rename({bhc_id_cols[0]: "ID", bhc_id_cols[1]: "DATE"}, axis=1)

        # target
        BHC_Y = self.BHC_Data_Raw[bhc_id_cols + bhc_target_cols]
        BHC_Y = BHC_Y.rename({bhc_id_cols[0]: "ID", bhc_id_cols[1]: "DATE"}, axis=1)

        print("Subsetting Inputs and Targets for CSMAR")
        CSMAR_X = self.CSMAR_Data_Raw[self.CSMAR_Data_Raw.columns.difference(csmar_target_cols + csmar_exclude_cols)]
        CSMAR_X = CSMAR_X.rename({csmar_id_cols[0]: "ID", csmar_id_cols[1]: "DATE"}, axis=1)

        CSMAR_Y = self.CSMAR_Data_Raw[csmar_id_cols + csmar_target_cols]
        CSMAR_Y = CSMAR_Y.rename({csmar_id_cols[0]: "ID", csmar_id_cols[1]: "DATE"}, axis=1)

        print("Saving to Data Dict")
        self.Data_Dict = dict()
        self.Data_Dict['BHC_X'] = BHC_X
        self.Data_Dict['BHC_Y'] = BHC_Y
        self.Data_Dict['BHC_ECON_X'] = self.BHC_EconData_PCA.rename({'Date': "DATE"}, axis=1)

        self.Data_Dict['CSMAR_X'] = CSMAR_X
        self.Data_Dict['CSMAR_Y'] = CSMAR_Y
        self.Data_Dict['CSMAR_ECON_X'] = self.CSMAR_EconData_PCA.rename({'q_dates_data': "DATE"}, axis=1)

        print('Subsetting Dataset Dates')

        for k in self.Data_Dict.keys():
            print(k, 'Processing')
            if 'ECON' in k:

                self.Data_Dict[k] = self.Data_Dict[k][
                    (self.Data_Dict[k]['DATE'] >= econ_dates[0]) & (self.Data_Dict[k]['DATE'] <= econ_dates[1])]
                print(k, 'Completed')
            else:
                self.Data_Dict[k] = self.Data_Dict[k][
                    (self.Data_Dict[k]['DATE'] >= bank_dates[0]) & (self.Data_Dict[k]['DATE'] <= bank_dates[1])]
                print(k, 'Completed')
        return

    def Data_Process_PD_Numpy3d(self, fulldatescombine=True, qtr_data=True, datecol='DATE', idcol='ID',
                                dates=["2002-12-31", "2018-09-30"], econ_dates=["2002-12-31", "2018-09-30"],
                                datefreq='3M'):

        self.logger.info(f"Tranform to 3 dimension array for EconData and Banking Data: Started")

        self.Data_Dict_NPY = dict()

        self.Data_Dict_NPY['DateIDX_Ref'] = pd.DataFrame(
            pd.date_range(start=dates[0], end=dates[1], freq=datefreq, normalize=True), columns=['DATE']).reset_index()

        self.Data_Dict_NPY['ECONDateIDX_Ref'] = pd.DataFrame(
            pd.date_range(start=econ_dates[0], end=econ_dates[1], freq=datefreq, normalize=True),
            columns=['DATE']).reset_index()

        for k in self.Data_Dict.keys():
            # print(k)
            # k = 'BHC_ECON_X'
            if 'ECON' not in k:
                df = self.Data_Dict[k]

                if fulldatescombine:
                    print("Generating Full Mesh", idcol, " and ", datecol, " to left join to.")
                    RepordingDate_Df = pd.DataFrame(df[datecol].unique())
                    RepordingDate_Df = RepordingDate_Df.rename({0: datecol}, axis=1)
                    RepordingDate_Df["key"] = 0

                    BankIds_Df = pd.DataFrame(df[idcol].unique())
                    BankIds_Df = BankIds_Df.rename({0: idcol}, axis=1)
                    BankIds_Df["key"] = 0

                    BankID_Date_Ref = RepordingDate_Df.assign(foo=1).merge(BankIds_Df.assign(foo=1), on="foo",
                                                                           how="outer").drop(["foo", "key_x", "key_y"],
                                                                                             1)

                    df_pd = BankID_Date_Ref.merge(df, left_on=[datecol, idcol],
                                                  right_on=[datecol, idcol], how="left")
                    print("Merge Completed")
            else:
                print("No Dates Merging needed for ECON data")
                df_pd = test.Data_Dict[k]

            if 'ECON' not in k:
                print("Subsetting Dates from Dataframe")
                df_pd = df_pd[(df_pd[datecol] >= dates[0]) & (df_pd[datecol] <= dates[1])]
                print(df_pd.shape)

                print("Pivoting Pandas Table to be Indexed by RSSD_ID and ReportingDate")
                df_pvt = pd.pivot_table(df_pd, index=[idcol, datecol], dropna=False)
                print(df_pvt.shape)

                print("Preparing to Reshape and output as Numpy Array")
                dim1 = df_pvt.index.get_level_values(idcol).nunique()
                dim2 = df_pvt.index.get_level_values(datecol).nunique()
                print("Reshaping into 3 dimensional NumPy array")

                result_pv = df_pvt.values.reshape((dim1, dim2, df_pvt.shape[1]))
                print(result_pv.shape)

                if qtr_data:
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

            else:
                print(k, df_pd.shape)
                print("Subsetting Dates from Dataframe")
                df_pd = df_pd[(df_pd[datecol] >= econ_dates[0]) & (df_pd[datecol] <= econ_dates[1])]
                print(df_pd.shape)

                print("Pivoting Pandas Table to be Indexed by ReportingDate")
                df_pvt = pd.pivot_table(df_pd, index=[datecol], dropna=False)
                print(df_pvt.shape)

                print("Preparing to Reshape and output as Numpy Array")
                # dim1 = df_pvt.index.get_level_values('RSSD_ID').nunique()
                dim1 = df_pvt.index.get_level_values('DATE').nunique()
                print("Reshaping into 3 dimensional NumPy array")
                result_pv = df_pvt.values.reshape((1, dim1, df_pvt.shape[1]))
                print(result_pv.shape)

                if qtr_data:
                    print("Quarterization of the Data")
                    qtr_count = result_pv.shape[1]
                    n = result_pv.shape[0]  # - 1
                    # result_pv = result_pv[1:n + 1]
                    results_quarter = np.zeros([4, int(qtr_count / 4), df_pvt.shape[1]])
                    print("Transformation for Quarterly Data")
                    for i in range(0, 4):
                        ids = [x for x in range(i, qtr_count, 4)]
                        results_quarter[i, :, :] = result_pv[:, ids, :]
                    print(results_quarter.shape)

            print("Exporting and Saving :", k)
            # Save them to a Dictionary to be used for splitting and modeling later.
            print("Saving objects to file")
            np.save(k + ".npy", result_pv)
            if qtr_data:
                np.save(k + "_qtr.npy", results_quarter)

            print("Saving to class dict")
            self.Data_Dict_NPY[k] = result_pv
            if qtr_data:
                self.Data_Dict_NPY[k + '_qtr'] = results_quarter

        return

    def Preprocess_Y_PD_Numpy3d(self):
        print('Removing banks with no Y targets')

        for banktargetkey in [x for x in self.Data_Dict_NPY.keys() if x.endswith('_Y')]:
            count = 0
            idx_list = []
            for bank in range(0, self.Data_Dict_NPY[banktargetkey].shape[0]):
                # for dateidx in range(0, test.Data_Dict_NPY['BHC_Y'].shape[1]):
                if pd.DataFrame(self.Data_Dict_NPY[banktargetkey][bank]).isnull().values.all(axis=0):
                    # print(bank)
                    count = count + 1
                    idx_list.append(bank)
                    # Remove Entries with NAN Y targets
            self.Data_Dict_NPY[banktargetkey] = np.delete(self.Data_Dict_NPY[banktargetkey], idx_list, 0)
            self.Data_Dict_NPY[banktargetkey.replace('_Y', '_X')] = np.delete(
                self.Data_Dict_NPY[banktargetkey.replace('_Y', '_X')], idx_list, 0)
            print(count, 'out of ', self.Data_Dict_NPY[banktargetkey].shape[0], 'have NAN in all Y values for ',
                  banktargetkey)

            print('Interpolate the partial Y targets')
            count = 0
            idx_list = []
            for bank in range(0, self.Data_Dict_NPY[banktargetkey].shape[0]):
                # for dateidx in range(0, test.Data_Dict_NPY['BHC_Y'].shape[1]):
                if pd.DataFrame(self.Data_Dict_NPY[banktargetkey][bank]).isnull().values.any(axis=0):
                    # print(bank)
                    count = count + 1
                    idx_list.append(bank)
                    self.Data_Dict_NPY[banktargetkey][bank] = pd.DataFrame(
                        self.Data_Dict_NPY[banktargetkey][bank]).interpolate(
                        method='linear').fillna(method='bfill').values
            print(count, 'out of ', self.Data_Dict_NPY[banktargetkey].shape[0], 'have NAN in some Y values for ',
                  banktargetkey)
        return
    def Bank_Data_TestTrain_Splitting(self, training_dates = ["2002-12-31", "2016-12-31"], testing_dates = ['2017-03-31', "2018-09-30"], econ_training_dates = ["2002-12-31", "2016-12-31"], econ_testing_dates = ['2017-03-31', "2018-09-30"]):
        print('Data Train Test splitting')
        print("Set Training and Testing Date Intervals")
        print('Training')
        train_start_idx = int(self.Data_Dict_NPY['DateIDX_Ref']['index'][self.Data_Dict_NPY['DateIDX_Ref']['DATE'] == training_dates[0]])
        train_end_idx = int(self.Data_Dict_NPY['DateIDX_Ref']['index'][self.Data_Dict_NPY['DateIDX_Ref']['DATE'] == training_dates[1]])
        # train_start_qtr_idx = int(test.Data_Dict_NPY['DateIDX_Ref']['index'][test.Data_Dict_NPY['DateIDX_Ref']['DATE'] == training_window[0]])
        # train_end_qtr_idx = int(test.Data_Dict_NPY['DateIDX_Ref']['index'][test.Data_Dict_NPY['DateIDX_Ref']['DATE'] == training_window[1]])
        train_start_econ_idx = int(self.Data_Dict_NPY['ECONDateIDX_Ref']['index'][self.Data_Dict_NPY['ECONDateIDX_Ref']['DATE'] == econ_training_dates[0]])
        train_end_econ_idx = int(self.Data_Dict_NPY['ECONDateIDX_Ref']['index'][self.Data_Dict_NPY['ECONDateIDX_Ref']['DATE'] == econ_training_dates[1]])

        # train_start_econ_qtr_idx = int(test.Data_Dict_NPY['ECONDateIDX_Ref']['index'][test.Data_Dict_NPY['ECONDateIDX_Ref']['DATE'] == training_window[0]])
        # train_end_econ_qtr_idx = int(test.Data_Dict_NPY['ECONDateIDX_Ref']['index'][test.Data_Dict_NPY['ECONDateIDX_Ref']['DATE'] == training_window[1]])

        print("Testing")
        test_start_idx = int(self.Data_Dict_NPY['DateIDX_Ref']['index'][self.Data_Dict_NPY['DateIDX_Ref']['DATE'] == testing_dates[0]])
        test_end_idx = int(self.Data_Dict_NPY['DateIDX_Ref']['index'][self.Data_Dict_NPY['DateIDX_Ref']['DATE'] == testing_dates[1]])
        test_start_econ_idx = int(self.Data_Dict_NPY['ECONDateIDX_Ref']['index'][self.Data_Dict_NPY['ECONDateIDX_Ref']['DATE'] == econ_testing_dates[0]])
        test_end_econ_idx = int(self.Data_Dict_NPY['ECONDateIDX_Ref']['index'][self.Data_Dict_NPY['ECONDateIDX_Ref']['DATE'] == econ_testing_dates[1]])

        print('Subset each non yearly key')
        for qtrnonecon_key in [x for x in test.Data_Dict_NPY.keys() if not x.endswith('_qtr') if 'ECON' not in x if
                               'Ref' not in x if not 'Tminus' in x if not 'train_' in x if not 'test_' in x]:

            self.Data_Dict_NPY['_'.join(['train', qtrnonecon_key])] = self.Data_Dict_NPY[qtrnonecon_key][:,train_start_idx:train_end_idx, :]
            self.Data_Dict_NPY['_'.join(['test', qtrnonecon_key])] = self.Data_Dict_NPY[qtrnonecon_key][:, test_start_idx:test_end_idx, :]
            for lag in range(1, 5):
                print("Make Objects upto T minus",lag)
                tmp_3d_train = self.Data_Dict_NPY['_'.join(['train', qtrnonecon_key])]
                tmp_3d_test = self.Data_Dict_NPY['_'.join(['test', qtrnonecon_key])]
                 # print(lag)
                shiftlag = lag * -1
                for b in range(0, tmp_3d_train.shape[0]):
                # print(b)
                    tmp_3d_train[b, :, :] = pd.DataFrame(tmp_3d_train[b, :, :]).shift(shiftlag).values
                    tmp_3d_test[b, :, :] = pd.DataFrame(tmp_3d_test[b, :, :]).shift(shiftlag).values
                self.Data_Dict_NPY['_'.join(['train', 'Tminus' + str(lag), qtrnonecon_key])] = tmp_3d_train
                self.Data_Dict_NPY['_'.join(['test', 'Tminus' + str(lag), qtrnonecon_key])] = tmp_3d_test

        print('Subset ECON data')
        for qtrnonecon_key in [x for x in test.Data_Dict_NPY.keys() if not x.endswith('_qtr') if 'ECON' in x if 'Ref' not in x]:
            self.Data_Dict_NPY['_'.join(['train', qtrnonecon_key])] = self.Data_Dict_NPY[qtrnonecon_key][:,train_start_econ_idx:train_end_econ_idx, :]
            self.Data_Dict_NPY['_'.join(['test', qtrnonecon_key])] = self.Data_Dict_NPY[qtrnonecon_key][:,test_start_econ_idx:test_end_econ_idx, :]
            for lag in range(1, 5):
                print("Make Objects upto T minus", lag)
                tmp_3d_train = self.Data_Dict_NPY['_'.join(['train', qtrnonecon_key])]
                tmp_3d_test = self.Data_Dict_NPY['_'.join(['test', qtrnonecon_key])]
            # print(lag)
                shiftlag = lag * -1
                for b in range(0, tmp_3d_train.shape[0]):
                    # print(b)
                    tmp_3d_train[b, :, :] = pd.DataFrame(tmp_3d_train[b, :, :]).shift(shiftlag).values
                    tmp_3d_test[b, :, :] = pd.DataFrame(tmp_3d_test[b, :, :]).shift(shiftlag).values
                self.Data_Dict_NPY['_'.join(['train', 'Tminus' + str(lag), qtrnonecon_key])] = tmp_3d_train
                self.Data_Dict_NPY['_'.join(['test', 'Tminus' + str(lag), qtrnonecon_key])] = tmp_3d_test

        # for qtrnonecon_key in [x for x in test.Data_Dict_NPY.keys() if x.endswith('_qtr') if 'ECON' not in x if 'Ref' not in x]:
        #    test.Data_Dict_NPY['_'.join(['train',qtrnonecon_key])] = test.Data_Dict_NPY[qtrnonecon_key][:,train_start_idx:train_end_idx,:]
        #    test.Data_Dict_NPY['_'.join(['test', qtrnonecon_key])] = test.Data_Dict_NPY[qtrnonecon_key][:,test_start_idx:test_end_idx, :]

        # for qtrnonecon_key in [x for x in test.Data_Dict_NPY.keys() if x.endswith('_qtr') if 'ECON' in x if 'Ref' not in x]:
        #    test.Data_Dict_NPY['_'.join(['train',qtrnonecon_key])] = test.Data_Dict_NPY[qtrnonecon_key][:,train_start_econ_idx:train_end_econ_idx,:]
        #    test.Data_Dict_NPY['_'.join(['test', qtrnonecon_key])] = test.Data_Dict_NPY[qtrnonecon_key][:,test_start_econ_idx:test_end_econ_idx, :]
        # Create yearly version
        return
    def Data_Modeling_Preprocess(self, banktypes = ['BHC','CSMAR']):
        print("Preprocess ECON DataFrames")
        for banktype in banktypes:
            print("Bank Type:", banktype)
            for testtrain_type in ['train_', 'test_']:
                condkey = testtrain_type + banktype + '_X'
                econ_tmp_list = [x for x in self.Data_Dict_NPY.keys() if x.startswith((testtrain_type)) if 'ECON' in x if banktype in x]
                for econ_keyname in econ_tmp_list:
                    print("Econ Keyname:", econ_keyname)
                    # Get number of firms to be used.
                    n = self.Data_Dict_NPY[condkey].shape[0]
                    dim1 = self.Data_Dict_NPY[econ_keyname].shape[1]
                    dim2 = self.Data_Dict_NPY[econ_keyname].shape[2]
                    # Make Temp Array
                    tmp_econ = np.zeros([n, dim1, dim2])
                    if not self.Data_Dict_NPY[econ_keyname].shape[0] == n:
                        print("Generate Copies for Array")
                        for i in range(0, n):
                            tmp_econ[i, :, :] = np.expand_dims(self.Data_Dict_NPY[econ_keyname][0, :, :], axis=0)
                            print("Save New Object into Dict with 3d Appended")
                    self.Data_Dict_NPY[econ_keyname] = tmp_econ
        print("Convert all Dataframes to Tensor object")
        self.traintest_sets_dict = self.Data_Dict_NPY
        # COnvert all to Tensor
        for keyname in [x for x in self.traintest_sets_dict.keys() if x.startswith(('train', 'test'))]:
            print(keyname)
            self.traintest_sets_dict[keyname] = torch.from_numpy(self.traintest_sets_dict[keyname]).float()
            print(self.traintest_sets_dict[keyname].shape)
        return
    def Modeling_DatasetSubsets(self, modelTarget = 'BHC_Y', exclude = ['BHC_X', 'Tminus2_BHC_X', 'Tminus3_BHC_X', 'Tminus4_BHC_X', 'Tminus2_BHC_Y', 'Tminus2_BHC_Y', 'Tminus3_BHC_Y', 'Tminus4_BHC_Y', 'Tminus2_BHC_ECON_X', 'Tminus3_BHC_ECON_X', 'Tminus4_BHC_ECON_X'], includes = []):
        self.modelTarget = modelTarget
        print("Preparing Inclusion and Exclusions for Inputs and Outputs for Modeling")
        banktype = modelTarget.split('_')[0]

        print("Updating TrainTest Exclusion list: Adding ModelTarget:", modelTarget)
        exclude.append(modelTarget)

        exclude_train = ['train_' + x for x in exclude]
        # exclude_test = ['test_' + x for x in exclude]
        print("Exclusion List:", exclude)
        #trainsets_tmp = [x for x in test.traintest_sets_dict.keys() if x.startswith(("train_")) if banktype in x if not any(z in x for z in exclude_train)]
        trainsets_tmp = [x for x in self.traintest_sets_dict.keys() if x.startswith(("train_")) if banktype in x if not any(z in x for z in exclude_train)]
        trainsets = list(filter(None, trainsets_tmp))
        print("Determining all Object Permutations for Experiments")
        dataset_subsets = list()
        for L in range(0, len(trainsets) + 1):
            for subset in itertools.combinations(trainsets, L):
                dataset_subsets.append(list(subset))
        dataset_subsets = [x for x in dataset_subsets if x != []]

        print("Experiment Object Permutations:", dataset_subsets.__len__())

        print("Must Include Following Inputs", include)
        if len(includes) > 0:
            print("Filtering out Datasubsets based on include logic:", include)
            dataset_subsets = [x for x in dataset_subsets if set(include) <= set(x)]
            print("With required inclusions: ", dataset_subsets.__len__())

        self.dataset_subsets = dataset_subsets
        return
    def Modeling_DataFormattingIO(self, subset):
        #for subset in self.dataset_subsets:
        self.tmp_dict_name = "&".join(["_".join(x.split("train_")[1:]) for x in subset])
        # tmp_dict_name = tmp_dict_name + "&GenModel_" + genmodel
        # print("Features to be used:", tmp_dict_name)
        # print("Current Subset:", subset)
        subset_test_tmp = [x.replace("train_", "test_") for x in subset]
        #    print("Create testset names from subset", subset_test_tmp)

        if len(subset) == 1:
            print("Condition 1", subset)
            raw_inputs = self.traintest_sets_dict[subset[0]]
            print("Raw Training Set Input Shape:", raw_inputs.shape)
            print("Setting Testing\Eval Input")
            raw_eval_inputs = self.traintest_sets_dict[subset_test_tmp[0]]

        elif len(subset) > 1:
            print("Condition 2", subset)
            print("Setting Initial Training Input")
            raw_inputs = self.traintest_sets_dict[subset[0]]
            print("Setting Initial Testing\Eval Input")
            raw_eval_inputs = self.traintest_sets_dict[subset_test_tmp[0]]
            for subcnt in range(1, len(subset)):
                print("Setting Training Input:", subcnt)
                raw_inputs = torch.cat((raw_inputs, self.traintest_sets_dict[subset[subcnt]]), dim=2)
                print("Setting Testing\Eval Input:", subcnt)
                raw_eval_inputs = torch.cat((raw_eval_inputs, self.traintest_sets_dict[subset_test_tmp[subcnt]]), dim=2)
        else:
            print("Subset not found or some other issue")

        print("Setting up inputs for Models")
        print("Setting Inputs and Target Parameters for Training")
        model_raw_target_keyname =  [x for x in self.traintest_sets_dict.keys() if x.startswith("train") if x.endswith(self.modelTarget)][0]
        model_raw_eval_target_keyname = [x for x in self.traintest_sets_dict.keys() if x.startswith("test") if x.endswith(self.modelTarget)][0]
        raw_targets = self.traintest_sets_dict[model_raw_target_keyname]
        raw_eval_targets = self.traintest_sets_dict[model_raw_eval_target_keyname]
        n, t, m1 = raw_inputs.shape
        m2 = raw_targets.shape[2]

        inputs_train = torch.zeros([t, m1, n]).float()
        targets_train = torch.zeros([t, n, m2]).float()
        for i in range(0, n):
            inputs_train[:, :, i] = raw_inputs[i, :, :]
            targets_train[:, i, :] = raw_targets[i, :, :]
        inputs_train[torch.isnan(inputs_train)] = 0
        targets_train[torch.isnan(targets_train)] = 0

        print("Setting up dimensions for testing")
        n, t, m1 = raw_eval_inputs.shape
        m2 = raw_eval_targets.shape[2]
        inputs_test = torch.zeros([t, m1, n]).float()
        targets_test = torch.zeros([t, n, m2]).float()
        for i in range(0, n):
            inputs_test[:, :, i] = raw_eval_inputs[i, :, :]
            targets_test[:, i, :] = raw_eval_targets[i, :, :]
        inputs_test[torch.isnan(inputs_test)] = 0
        targets_test[torch.isnan(targets_test)] = 0

        print("Returning Inputs and Targets for Modeling")
        return(inputs_train, targets_train, inputs_test, targets_test)
    def Modeling_PredictModel(self,inputs_train, targets_train, inputs_test, targets_test, predictmodel_list = ['gru'], epoch = 100, lstm_lr = 1e-3, threshold = 1e-4   ):
        print("Comparison on Prediction models")
        rmse_train_list = []
        rmse_lst = []
        mse_train_list = []
        mse_lst = []
        learn_types_list = []
        train_trainhist_dict = dict()

        for predictmodel in predictmodel_list:
            print("Training:", predictmodel.lower())
            if predictmodel.lower() in ['lstm']:
                model, train_loss, train_rmse, train_trainhist = m_LSTM.train(inputs_train, targets_train, epoch, lstm_lr, threshold)
                train_trainhist_dict[ "_".join([tmp_dict_name, "trainhist", predictmodel, self.modelTarget])] = train_trainhist
                model_trained = m_LSTM
            elif predictmodel.lower() in ['lr', 'linearregression', 'linreg']:
                model, train_loss, train_rmse, train_trainhist = m_LinReg.train(inputs_train, targets_train, epoch,
                                                                                lstm_lr,
                                                                                threshold)
                # train_trainhist_dict["_".join([tmp_dict_name, "trainhist", predictmodel])] = train_trainhist
                model_trained = m_LinReg


            elif predictmodel.lower() in ['gru']:

                model, train_loss, train_rmse, train_trainhist = m_GRU.train(inputs_train, targets_train, epoch,
                                                                             lstm_lr, threshold)
                #     train_trainhist_dict["_".join([tmp_dict_name, "trainhist", predictmodel, modelTarget])] = train_trainhist
                model_trained = m_GRU

            #
            elif predictmodel.lower() in ['darnn', "da-rnn", "dualstagelstm", "dual-stage-lstm"]:
                #     # importlib.reload(m_DA_RNN)
                model, train_loss, train_rmse, train_trainhist = m_DA_RNN.train(inputs_train, targets_train, epoch,
                                                                                lstm_lr,
                                                                                threshold)
                #     train_trainhist_dict["_".join([tmp_dict_name, "trainhist", predictmodel, modelTarget])] = train_trainhist
                model_trained = m_DA_RNN

            else:
                print("No Deep Learning Prediction Model Code Found:", predictmodel)
                train_loss = torch.tensor(np.nan)
                train_rmse = torch.tensor(np.nan)
                # return

            print("Calculating Training RMSE:\t", self.modelTarget)
            mse_train_list.append(train_loss)
            rmse_train_list.append(train_rmse)
            print("%s\tMSEerror:\t%.5f\tRMSEerror:\t%.5f" % (self.tmp_dict_name, train_loss, train_rmse))

            if predictmodel.lower() in ['darnn', "da-rnn", "dualstagelstm", "dual-stage-lstm"] + ['gru', 'lstm'] + [
                'lr', 'linearregression', 'linreg']:
                if predictmodel.lower() in ['darnn', "da-rnn", "dualstagelstm", "dual-stage-lstm"]:
                    pred = model_trained.predict(model, inputs=inputs_test, targets=targets_test)

                else:
                    pred = model_trained.predict(model, inputs_test)
                mse = torch.nn.functional.mse_loss(pred, targets_test)
                rmse = torch.sqrt(torch.nn.functional.mse_loss(pred, targets_test))


            elif predictmodel.lower() in ["arima"]:
                # Setup Auto Arima
                # from pmdarima.arima import auto_arima
                # from math import sqrt
                # from sklearn.metrics import mean_squared_error
                tmp_mse_list = list()
                tmp_rmse_list = list()
                tmp_var_mse = list()
                tmp_var_rmse = list()
                print('Fitting ARIMA model')
                if targets_train.shape[2] == 1:
                    for b in range(0, targets_train.shape[1]):
                        train_tmp = pd.DataFrame(targets_train[:, b, :].numpy())
                        model = auto_arima(train_tmp, trace=True, error_action='ignore', suppress_warnings=True)
                        model.fit(train_tmp)

                        valid_tmp = pd.DataFrame(targets_test[:, b, :].numpy())
                        forecast = model.predict(n_periods=len(valid_tmp))
                        test_pred = pd.DataFrame(forecast, index=valid_tmp.index, columns=['Prediction'])

                        mse_tmp = mean_squared_error(valid_tmp, test_pred)
                        tmp_mse_list.append(mse_tmp)
                        rmse_tmp = sqrt(mse_tmp)
                        tmp_rmse_list.append(rmse_tmp)

                    mse = torch.tensor(np.mean(tmp_mse_list))
                    rmse = torch.tensor(np.mean(tmp_rmse_list))
                    print(self.tmp_dict_name, 'MSE:', mse.item(), 'RMSE:', rmse.item())

                else:
                    for j in range(0, targets_train.shape[2]):
                        for b in range(0, targets_train.shape[1]):
                            train_tmp = pd.DataFrame(targets_train[:, b, j].numpy())
                            model = auto_arima(train_tmp, trace=True, error_action='ignore', suppress_warnings=True)
                            model.fit(train_tmp)

                            valid_tmp = pd.DataFrame(targets_test[:, b, j].numpy())
                            forecast = model.predict(n_periods=len(valid_tmp))
                            test_pred = pd.DataFrame(forecast, index=valid_tmp.index, columns=['Prediction'])
                            mse_tmp = mean_squared_error(valid_tmp, test_pred)
                            tmp_mse_list.append(mse_tmp)
                            rmse_tmp = sqrt(mse_tmp)
                            tmp_rmse_list.append(rmse_tmp)

                        mse = np.mean(tmp_mse_list)
                        tmp_var_mse.append(mse)
                        rmse = np.mean(tmp_rmse_list)
                        tmp_var_rmse.append(rmse)

                    mse = torch.tensor(np.mean(tmp_var_mse))
                    rmse = torch.tensor(np.mean(tmp_var_rmse))
                    print(self.tmp_dict_name, 'MSE:', mse.item(), 'RMSE:', rmse.item())


            elif predictmodel.lower() in ["var"] and targets_train.shape[2] > 1:
                tmp_mse_list = list()
                tmp_rmse_list = list()
                for b in range(0, targets_train.shape[1]):
                    train_tmp = pd.DataFrame(targets_train[:, b, :].numpy())
                    cols = train_tmp.columns
                    model = VAR(endog=train_tmp)
                    # train_tmp.shape
                    model_fit = model.fit()
                    valid_tmp = pd.DataFrame(targets_test[:, b, :].numpy())
                    forecast = model_fit.forecast(model_fit.y, steps=len(valid_tmp))
                    test_pred = pd.DataFrame(forecast, index=valid_tmp.index)
                    # pred = pd.DataFrame(index=range(0, len(test_pred)), columns=[cols])
                    mse_tmp = mean_squared_error(valid_tmp, test_pred)
                    tmp_mse_list.append(mse_tmp)
                    rmse_tmp = sqrt(mse_tmp)
                    tmp_rmse_list.append(rmse_tmp)
                mse = torch.tensor(np.mean(tmp_mse_list))
                rmse = torch.tensor(np.mean(tmp_rmse_list))
                print(self.tmp_dict_name, 'MSE:', mse.item(), 'RMSE:', rmse.item())
            else:
                print("no code developed for", predictmodel.lower())
                mse = np.nan
                rmse = np.nan
                train_loss = np.nan
                train_rmse = np.nan

            print("Calculating Testing RMSE:\t", self.modelTarget)
            # print(pred.shape, targes_test.shape)
            rmse_lst.append(rmse)
            mse_lst.append(mse)
            learn_types_list.append(predictmodel + '_' + self.tmp_dict_name)
            print("%s\tMSEerror:\t%.5f\tRMSEerror:\t%.5f" % (predictmodel + '_' + self.tmp_dict_name, mse, rmse))
            gc.collect()
        rmse_train_lst_sk = torch.stack(rmse_train_list)
        rmse_lst_sk = torch.stack(rmse_lst)
        rmse_list_final = [rmse_train_lst_sk, rmse_lst_sk.data]
        result_obj = pd.DataFrame(rmse_list_final, columns=learn_types_list, index=["TrainErr", "TestErr"])
        print("Target:", modelTarget, "\nMinimum Training Error:", result_obj.astype(float).idxmin(axis=1)[0]
              , result_obj.min(axis=1)[0], "\nMinimum Testing Error:", result_obj.astype(float).idxmin(axis=1)[1],
              result_obj.min(axis=1)[1])
        return(result_obj.astype(float))
#Loading Initi Class
test = BHC_CSMAR_TLF()

#Configure BHC Codes Method to then subset CSMAR and BHC
test.load_BHC_CSMAR_Codes()

#Load CSMAR Files , Merge (Comp Profile, Balance Sheet and Income Statement), Subset.
test.CSMAR_Data_MergeLoad()

#Load BHC files, Filter based on consecutive qtrs, top banksm data formatting and subsetting.
test.BHC_Data_Load()

#Load EconData CSMAR
test.CSMAR_MacroEcon_Load()

#Load EconData BHC #TODO: Get newest US Macroeconomics data
test.BHC_MarcoEcon_Load()

#Transform Data into appropirate Tensor Formatting

test.Bank_Data_Subsetting(bhc_target_cols = ['BHCK3247'])

test.Data_Process_PD_Numpy3d()

test.Preprocess_Y_PD_Numpy3d()

test.Bank_Data_TestTrain_Splitting()

test.Data_Modeling_Preprocess()

test.Modeling_DatasetSubsets(modelTarget='CSMAR_Y')


for subset in test.dataset_subsets:
    inputs_train, targets_train, inputs_test, targets_test = test.Modeling_DataFormattingIO(subset)
    results_tmp = test.Modeling_PredictModel(inputs_train, targets_train, inputs_test, targets_test, predictmodel_list = ['lr', 'gru'], epoch = 20)
    results = pd.concat([results, results_tmp], axis = 1)
    print(results_tmp.transpose()['TestErr'].idxmin(), results_tmp.transpose()['TestErr'].min())

# #Data Train Test splitting
#     #Consider Indexer Logic to Dates for splitting testing and training intelligently.
#     #Consider Percentage ratio splitter.
#
# training_dates = ["2002-12-31", "2016-12-31"]
# testing_dates = ['2017-03-31', "2018-09-30"]
# econ_training_dates = ["2002-12-31", "2016-12-31"]
# econ_testing_dates = ['2017-03-31', "2018-09-30"]
# #training_window = dates
# #testing_window = econ_dates
# #Quarterly
# #Set Training and Testing Date Intervals
# #Training
# train_start_idx = int(test.Data_Dict_NPY['DateIDX_Ref']['index'][test.Data_Dict_NPY['DateIDX_Ref']['DATE'] == training_dates[0]])
# train_end_idx = int(test.Data_Dict_NPY['DateIDX_Ref']['index'][test.Data_Dict_NPY['DateIDX_Ref']['DATE'] == training_dates[1]])
#
# #train_start_qtr_idx = int(test.Data_Dict_NPY['DateIDX_Ref']['index'][test.Data_Dict_NPY['DateIDX_Ref']['DATE'] == training_window[0]])
# #train_end_qtr_idx = int(test.Data_Dict_NPY['DateIDX_Ref']['index'][test.Data_Dict_NPY['DateIDX_Ref']['DATE'] == training_window[1]])
#
# train_start_econ_idx = int(test.Data_Dict_NPY['ECONDateIDX_Ref']['index'][test.Data_Dict_NPY['ECONDateIDX_Ref']['DATE'] == econ_training_dates[0]])
# train_end_econ_idx = int(test.Data_Dict_NPY['ECONDateIDX_Ref']['index'][test.Data_Dict_NPY['ECONDateIDX_Ref']['DATE'] == econ_training_dates[1]])
#
# #train_start_econ_qtr_idx = int(test.Data_Dict_NPY['ECONDateIDX_Ref']['index'][test.Data_Dict_NPY['ECONDateIDX_Ref']['DATE'] == training_window[0]])
# #train_end_econ_qtr_idx = int(test.Data_Dict_NPY['ECONDateIDX_Ref']['index'][test.Data_Dict_NPY['ECONDateIDX_Ref']['DATE'] == training_window[1]])
#
#
#
# #Testing
# test_start_idx = int(test.Data_Dict_NPY['DateIDX_Ref']['index'][test.Data_Dict_NPY['DateIDX_Ref']['DATE'] == testing_dates[0]])
# test_end_idx = int(test.Data_Dict_NPY['DateIDX_Ref']['index'][test.Data_Dict_NPY['DateIDX_Ref']['DATE'] == testing_dates[1]])
#
# test_start_econ_idx = int(test.Data_Dict_NPY['ECONDateIDX_Ref']['index'][test.Data_Dict_NPY['ECONDateIDX_Ref']['DATE'] == econ_testing_dates[0]])
# test_end_econ_idx = int(test.Data_Dict_NPY['ECONDateIDX_Ref']['index'][test.Data_Dict_NPY['ECONDateIDX_Ref']['DATE'] == econ_testing_dates[1]])
#
#
#
# #Subset each non yearly key
# for qtrnonecon_key in [x for x in test.Data_Dict_NPY.keys() if not x.endswith('_qtr') if 'ECON' not in x if 'Ref' not in x if not 'Tminus' in x if not 'train_' in x if not 'test_' in x]:
#     #Make Objects upto minus 4
#     test.Data_Dict_NPY['_'.join(['train',qtrnonecon_key])] = test.Data_Dict_NPY[qtrnonecon_key][:,train_start_idx:train_end_idx,:]
#     test.Data_Dict_NPY['_'.join(['test', qtrnonecon_key])] = test.Data_Dict_NPY[qtrnonecon_key][:,test_start_idx:test_end_idx, :]
#     for lag in range(1,5):
#         tmp_3d_train = test.Data_Dict_NPY['_'.join(['train', qtrnonecon_key])]
#         tmp_3d_test = test.Data_Dict_NPY['_'.join(['test', qtrnonecon_key])]
#         #print(lag)
#         shiftlag = lag * -1
#         for b in range(0,tmp_3d_train.shape[0]):
#             #print(b)
#             tmp_3d_train[b,:,:] = pd.DataFrame(tmp_3d_train[b,:,:]).shift(shiftlag).values
#             tmp_3d_test[b, :, :] = pd.DataFrame(tmp_3d_test[b, :, :]).shift(shiftlag).values
#         test.Data_Dict_NPY['_'.join(['train', 'Tminus' + str(lag), qtrnonecon_key])] = tmp_3d_train
#         test.Data_Dict_NPY['_'.join(['test', 'Tminus' + str(lag), qtrnonecon_key])] = tmp_3d_test
#
#
#
#
# for qtrnonecon_key in [x for x in test.Data_Dict_NPY.keys() if not x.endswith('_qtr') if 'ECON' in x if 'Ref' not in x]:
#     test.Data_Dict_NPY['_'.join(['train',qtrnonecon_key])] = test.Data_Dict_NPY[qtrnonecon_key][:,train_start_econ_idx:train_end_econ_idx,:]
#     test.Data_Dict_NPY['_'.join(['test',qtrnonecon_key])] = test.Data_Dict_NPY[qtrnonecon_key][:,test_start_econ_idx:test_end_econ_idx,:]
#     for lag in range(1,5):
#         tmp_3d_train = test.Data_Dict_NPY['_'.join(['train', qtrnonecon_key])]
#         tmp_3d_test = test.Data_Dict_NPY['_'.join(['test', qtrnonecon_key])]
#         #print(lag)
#         shiftlag = lag * -1
#         for b in range(0,tmp_3d_train.shape[0]):
#             #print(b)
#             tmp_3d_train[b,:,:] = pd.DataFrame(tmp_3d_train[b,:,:]).shift(shiftlag).values
#             tmp_3d_test[b, :, :] = pd.DataFrame(tmp_3d_test[b, :, :]).shift(shiftlag).values
#         test.Data_Dict_NPY['_'.join(['train', 'Tminus' + str(lag), qtrnonecon_key])] = tmp_3d_train
#         test.Data_Dict_NPY['_'.join(['test', 'Tminus' + str(lag), qtrnonecon_key])] = tmp_3d_test
#
#
#
#
#
# #for qtrnonecon_key in [x for x in test.Data_Dict_NPY.keys() if x.endswith('_qtr') if 'ECON' not in x if 'Ref' not in x]:
# #    test.Data_Dict_NPY['_'.join(['train',qtrnonecon_key])] = test.Data_Dict_NPY[qtrnonecon_key][:,train_start_idx:train_end_idx,:]
# #    test.Data_Dict_NPY['_'.join(['test', qtrnonecon_key])] = test.Data_Dict_NPY[qtrnonecon_key][:,test_start_idx:test_end_idx, :]
#
#
#
# #for qtrnonecon_key in [x for x in test.Data_Dict_NPY.keys() if x.endswith('_qtr') if 'ECON' in x if 'Ref' not in x]:
# #    test.Data_Dict_NPY['_'.join(['train',qtrnonecon_key])] = test.Data_Dict_NPY[qtrnonecon_key][:,train_start_econ_idx:train_end_econ_idx,:]
# #    test.Data_Dict_NPY['_'.join(['test', qtrnonecon_key])] = test.Data_Dict_NPY[qtrnonecon_key][:,test_start_econ_idx:test_end_econ_idx, :]
#
#
# #Create yearly version




#Convert ECON to have same dimensions as inputs


# for banktype in ['BHC', 'CSMAR']:
#     print("Bank Type:", banktype)
#     for testtrain_type in ['train_', 'test_']:
#         condkey = testtrain_type + banktype + '_X'
#         econ_tmp_list = [x for x in test.Data_Dict_NPY.keys() if x.startswith((testtrain_type)) if 'ECON' in x if banktype in x]
#         for econ_keyname in econ_tmp_list:
#             print("Econ Keyname:", econ_keyname)
#             #Get number of firms to be used.
#             n = test.Data_Dict_NPY[condkey].shape[0]
#             dim1 = test.Data_Dict_NPY[econ_keyname].shape[1]
#             dim2 = test.Data_Dict_NPY[econ_keyname].shape[2]
#             #Make Temp Array
#             tmp_econ = np.zeros([n, dim1, dim2])
#             if not test.Data_Dict_NPY[econ_keyname].shape[0] == n:
#                 print("Generate Copies for Array")
#                 for i in range(0, n):
#                     tmp_econ[i, :, :] = np.expand_dims(test.Data_Dict_NPY[econ_keyname][0,:,:], axis=0)
#                     print("Save New Object into Dict with 3d Appended")
#             test.Data_Dict_NPY[econ_keyname] = tmp_econ



# [x for x in test.Data_Dict_NPY.keys() if x.endswith('ECON_X')]
#
# test.Data_Dict_NPY['BHC_ECON_X'].shape
# test.Data_Dict_NPY['CSMAR_ECON_X'].shape


# for keyname in [x for x in test.Data_Dict_NPY.keys() if x.startswith(('train','test')) if 'ECON' in x]:
#     print(keyname)
#     print(test.Data_Dict_NPY[keyname].shape)

##torch.from_numpy(cond_tmp).float()
#
# traintest_sets_dict = test.Data_Dict_NPY
#
# #COnvert all to Tensor
# for keyname in [x for x in traintest_sets_dict.keys() if x.startswith(('train','test'))]:
#     print(keyname)
#     traintest_sets_dict[keyname] = torch.from_numpy(traintest_sets_dict[keyname]).float()
#     print(traintest_sets_dict[keyname].shape)

# modelTarget = 'BHC_Y'
# banktype = modelTarget.split('_')[0]
#
# #Get combinations of all the inputs
# includes = []
# exclude = ['BHC_X','Tminus2_BHC_X','Tminus3_BHC_X','Tminus4_BHC_X','Tminus2_BHC_Y','Tminus2_BHC_Y','Tminus3_BHC_Y','Tminus4_BHC_Y', 'Tminus2_BHC_ECON_X','Tminus3_BHC_ECON_X','Tminus4_BHC_ECON_X']
# print("Updating TrainTest Exclusion list: Adding ModelTarget:", modelTarget)
# exclude.append(modelTarget)
# #print("Converting Exclusion list to tuple")
# #exclude = tuple(exclude)
# exclude_train = ['train_' + x for x in exclude]
# #exclude_test = ['test_' + x for x in exclude]
# print("Exclusion List:", exclude)
# trainsets_tmp = [x for x in traintest_sets_dict.keys() if x.startswith(("train_")) if banktype in x if not any(z in x for z in exclude_train)]
#
# trainsets = list(filter(None,trainsets_tmp))
# dataset_subsets = list()
# for L in range(0, len(trainsets) + 1):
#     for subset in itertools.combinations(trainsets, L):
#         dataset_subsets.append(list(subset))
# dataset_subsets = [x for x in dataset_subsets if x !=[]]
#
# dataset_subsets.__len__()
#
# #Must Include Following Inputs
# include = None
# if include is not None:
#     print("Filtering out Datasubsets based on include logic:", include)
#     dataset_subsets = [x for x in dataset_subsets if set(include) <= set(x)]

#Concatenate the Inputs


#Normalize the inputs?


# for subset in dataset_subsets:
#     tmp_dict_name = "&".join(["_".join(x.split("train_")[1:]) for x in subset])
#     #tmp_dict_name = tmp_dict_name + "&GenModel_" + genmodel
#     #print("Features to be used:", tmp_dict_name)
#     #print("Current Subset:", subset)
#     subset_test_tmp = [x.replace("train_", "test_") for x in subset]
# #    print("Create testset names from subset", subset_test_tmp)
#
#     if len(subset) == 1 and not any("ECON" in x for x in subset):
#         print("Condition 1", subset)
#         raw_inputs = traintest_sets_dict[subset[0]]
#         print("Raw Training Set Input Shape:", raw_inputs.shape)
#         print("Setting Testing\Eval Input")
#         raw_eval_inputs = traintest_sets_dict[subset_test_tmp[0]]
#
#     elif len(subset) > 1 :
#         print("Condition 2", subset)
#         print("Setting Initial Training Input")
#         raw_inputs = traintest_sets_dict[subset[0]]
#         print("Setting Initial Testing\Eval Input")
#         raw_eval_inputs = traintest_sets_dict[subset_test_tmp[0]]
#         for subcnt in range(1, len(subset)):
#             print("Setting Training Input:", subcnt)
#             raw_inputs = torch.cat((raw_inputs, traintest_sets_dict[subset[subcnt]]), dim=2)
#             print("Setting Testing\Eval Input:", subcnt)
#             raw_eval_inputs = torch.cat((raw_eval_inputs, traintest_sets_dict[subset_test_tmp[subcnt]]), dim=2)
#
#
#
# #Setting up inputs for Models
# print("Setting Inputs and Target Parameters for Training")
# model_raw_target_keyname = [x for x in traintest_sets_dict.keys() if x.startswith("train") if x.endswith(modelTarget)][0]
# model_raw_eval_target_keyname = [x for x in traintest_sets_dict.keys() if x.startswith("test") if x.endswith(modelTarget)][0]
# raw_targets = traintest_sets_dict[model_raw_target_keyname]
# raw_eval_targets = traintest_sets_dict[model_raw_eval_target_keyname]
#
# #print("Setting up Dimensions for inputs and targets for Training model:", predictmodel)
# n, t, m1 = raw_inputs.shape
# m2 = raw_targets.shape[2]
#
# inputs_train = torch.zeros([t, m1, n]).float()
# targets_train = torch.zeros([t, n, m2]).float()
#
# #TODO: Resolve uneven dimensions for lagged objects.
#
# #TODO: Resolve Target missing data.
#
# for i in range(0, n):
#     inputs_train[:, :, i] = raw_inputs[i, :, :]
#     targets_train[:, i, :] = raw_targets[i, :, :]
#
# #Set nan to values?
# inputs_train[torch.isnan(inputs_train)] = 0
# targets_train[torch.isnan(targets_train)] = 0
#
# # Setting up dimensions for testing
# n, t, m1 = raw_eval_inputs.shape
# m2 = raw_eval_targets.shape[2]
# inputs_test = torch.zeros([t, m1, n]).float()
# targets_test = torch.zeros([t, n, m2]).float()
# for i in range(0, n):
#     inputs_test[:, :, i] = raw_eval_inputs[i, :, :]
#     targets_test[:, i, :] = raw_eval_targets[i, :, :]
# inputs_test[torch.isnan(inputs_test)] = 0
# targets_test[torch.isnan(targets_test)] = 0

#predictmodel_list = ['lr','lstm','gru', 'arima', 'darnn']
# predictmodel_list = ['gru']
# epoch = 100
# lstm_lr = 1e-3
# threshold = 1e-4
#
# print("Comparison on Prediction models")
# rmse_train_list = []
# rmse_lst = []
# mse_train_list = []
# mse_lst = []
# learn_types_list = []
# train_trainhist_dict = dict()
# #Model training for output.
#
# for predictmodel in predictmodel_list:
#     print("Training:", predictmodel.lower())
#     if predictmodel.lower() in ['lstm']:
#         model, train_loss, train_rmse, train_trainhist = m_LSTM.train(inputs_train, targets_train, epoch, lstm_lr, threshold)
#         train_trainhist_dict["_".join([tmp_dict_name, "trainhist", predictmodel, modelTarget])] = train_trainhist
#         model_trained = m_LSTM
#     elif predictmodel.lower() in ['lr', 'linearregression', 'linreg']:
#          model, train_loss, train_rmse, train_trainhist = m_LinReg.train(inputs_train, targets_train, epoch, lstm_lr,
#                                                                          threshold)
#          # train_trainhist_dict["_".join([tmp_dict_name, "trainhist", predictmodel])] = train_trainhist
#          model_trained = m_LinReg
#
#
#     elif predictmodel.lower() in ['gru']:
#
#          model, train_loss, train_rmse, train_trainhist = m_GRU.train(inputs_train, targets_train, epoch,
#                                                                       lstm_lr, threshold)
#     #     train_trainhist_dict["_".join([tmp_dict_name, "trainhist", predictmodel, modelTarget])] = train_trainhist
#          model_trained = m_GRU
#
#     #
#     elif predictmodel.lower() in ['darnn', "da-rnn", "dualstagelstm", "dual-stage-lstm"]:
#     #     # importlib.reload(m_DA_RNN)
#          model, train_loss, train_rmse, train_trainhist = m_DA_RNN.train(inputs_train, targets_train, epoch, lstm_lr,
#                                                                          threshold)
#     #     train_trainhist_dict["_".join([tmp_dict_name, "trainhist", predictmodel, modelTarget])] = train_trainhist
#          model_trained = m_DA_RNN
#
#     else:
#         print("No Deep Learning Prediction Model Code Found:", predictmodel)
#         train_loss = torch.tensor(np.nan)
#         train_rmse = torch.tensor(np.nan)
#         #return
#
#     print("Calculating Training RMSE:\t", modelTarget)
#     mse_train_list.append(train_loss)
#     rmse_train_list.append(train_rmse)
#     print("%s\tMSEerror:\t%.5f\tRMSEerror:\t%.5f" % (tmp_dict_name, train_loss, train_rmse))
#
#
#
#
#
#     if predictmodel.lower() in ['darnn', "da-rnn", "dualstagelstm", "dual-stage-lstm"] + ['gru', 'lstm'] + ['lr', 'linearregression', 'linreg']:
#         if predictmodel.lower() in ['darnn', "da-rnn", "dualstagelstm", "dual-stage-lstm"]:
#             pred = model_trained.predict(model, inputs=inputs_test, targets=targets_test)
#
#         else:
#             pred = model_trained.predict(model, inputs_test)
#         mse = torch.nn.functional.mse_loss(pred, targets_test)
#         rmse = torch.sqrt(torch.nn.functional.mse_loss(pred, targets_test))
#
#
#     elif predictmodel.lower() in ["arima"]:
#         # Setup Auto Arima
#         # from pmdarima.arima import auto_arima
#         # from math import sqrt
#         # from sklearn.metrics import mean_squared_error
#         tmp_mse_list = list()
#         tmp_rmse_list = list()
#         tmp_var_mse = list()
#         tmp_var_rmse = list()
#         print('Fitting ARIMA model')
#         if targets_train.shape[2] == 1:
#             for b in range(0, targets_train.shape[1]):
#                 train_tmp = pd.DataFrame(targets_train[:, b, :].numpy())
#                 model = auto_arima(train_tmp, trace=True, error_action='ignore', suppress_warnings=True)
#                 model.fit(train_tmp)
#
#                 valid_tmp = pd.DataFrame(targets_test[:, b, :].numpy())
#                 forecast = model.predict(n_periods=len(valid_tmp))
#                 test_pred = pd.DataFrame(forecast, index=valid_tmp.index, columns=['Prediction'])
#
#                 mse_tmp = mean_squared_error(valid_tmp, test_pred)
#                 tmp_mse_list.append(mse_tmp)
#                 rmse_tmp = sqrt(mse_tmp)
#                 tmp_rmse_list.append(rmse_tmp)
#
#             mse = torch.tensor(np.mean(tmp_mse_list))
#             rmse = torch.tensor(np.mean(tmp_rmse_list))
#             print(tmp_dict_name, 'MSE:', mse.item(), 'RMSE:', rmse.item())
#
#         else:
#             for j in range(0,targets_train.shape[2]):
#                 for b in range(0, targets_train.shape[1]):
#                     train_tmp = pd.DataFrame(targets_train[:, b, j].numpy())
#                     model = auto_arima(train_tmp, trace=True, error_action='ignore', suppress_warnings=True)
#                     model.fit(train_tmp)
#
#                     valid_tmp = pd.DataFrame(targets_test[:, b, j].numpy())
#                     forecast = model.predict(n_periods=len(valid_tmp))
#                     test_pred = pd.DataFrame(forecast, index=valid_tmp.index, columns=['Prediction'])
#                     mse_tmp = mean_squared_error(valid_tmp, test_pred)
#                     tmp_mse_list.append(mse_tmp)
#                     rmse_tmp = sqrt(mse_tmp)
#                     tmp_rmse_list.append(rmse_tmp)
#
#
#                 mse = np.mean(tmp_mse_list)
#                 tmp_var_mse.append(mse)
#                 rmse = np.mean(tmp_rmse_list)
#                 tmp_var_rmse.append(rmse)
#
#
#             mse = torch.tensor(np.mean(tmp_var_mse))
#             rmse = torch.tensor(np.mean(tmp_var_rmse))
#             print(tmp_dict_name, 'MSE:', mse.item(), 'RMSE:', rmse.item())
#
#
#     elif predictmodel.lower() in ["var"] and targets_train.shape[2] > 1:
#         tmp_mse_list = list()
#         tmp_rmse_list = list()
#         for b in range(0, targets_train.shape[1]):
#             train_tmp = pd.DataFrame(targets_train[:, b, :].numpy())
#             cols = train_tmp.columns
#             model = VAR(endog=train_tmp)
#             #train_tmp.shape
#             model_fit = model.fit()
#             valid_tmp = pd.DataFrame(targets_test[:, b, :].numpy())
#             forecast = model_fit.forecast(model_fit.y, steps=len(valid_tmp))
#             test_pred = pd.DataFrame(forecast, index=valid_tmp.index)
#             #pred = pd.DataFrame(index=range(0, len(test_pred)), columns=[cols])
#             mse_tmp = mean_squared_error(valid_tmp, test_pred)
#             tmp_mse_list.append(mse_tmp)
#             rmse_tmp = sqrt(mse_tmp)
#             tmp_rmse_list.append(rmse_tmp)
#         mse = torch.tensor(np.mean(tmp_mse_list))
#         rmse = torch.tensor(np.mean(tmp_rmse_list))
#         print(tmp_dict_name, 'MSE:', mse.item(), 'RMSE:', rmse.item())
#     else:
#         print("no code developed for", predictmodel.lower())
#         mse = np.nan
#         rmse = np.nan
#         train_loss = np.nan
#         train_rmse = np.nan
#
#
#     print("Calculating Testing RMSE:\t", modelTarget)
#     # print(pred.shape, targes_test.shape)
#     rmse_lst.append(rmse)
#     mse_lst.append(mse)
#     learn_types_list.append(predictmodel + '_' + tmp_dict_name)
#     print("%s\tMSEerror:\t%.5f\tRMSEerror:\t%.5f" % (predictmodel + '_' + tmp_dict_name, mse, rmse))
#     gc.collect()
#
#
# rmse_train_lst_sk = torch.stack(rmse_train_list)
# rmse_lst_sk = torch.stack(rmse_lst)
# rmse_list_final = [rmse_train_lst_sk, rmse_lst_sk.data]
# result_obj = pd.DataFrame(rmse_list_final, columns = learn_types_list, index = ["TrainErr","TestErr"])
# print(result_obj.astype(float))
# print("Target:", modelTarget, "\nMinimum Training Error:", result_obj.astype(float).idxmin(axis=1)[0]
#                                                 , result_obj.min(axis=1)[0], "\nMinimum Testing Error:", result_obj.astype(float).idxmin(axis=1)[1],
#                                                  result_obj.min(axis=1)[1])
#




#Modeling, Cross Validation for time series incorporated.
    #Baseline Models TODO: Add VAR and AutoARima
    #Baseline Deep Learning Models
    #State of the art Models
    #Transfer Learning Models.
















