import logging
from functools import reduce
import gc


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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
    def __init__(self,varpath_dict_args = {
                    "rootdir" : "/Users/phn1x/icdm2018_research_BB/Stress_Test_Research/TransferLearning_CapitalPredict/",
                    "Indcol": "Indnme",
                    "Industry":"Finance",
                    "datadir" : "data",
                    "CSMAR_CompProfile": "CSMAR_CompayProfiles.csv",
                    "CSMAR_BalanceSheet": "CSMAR_BalanceSheet_All.csv",
                    "CSMAR_IncomeStatement": "CSMAR_all_IncomeStatement.csv",
                    "CSMAR_BHC_Codes" : "CSMAR_BHC_Codes.csv",
                    "CSMAR_MacroData" : "China_MacroEconomicData.csv",
                    "BHC_Full_Data" : "BHC_Full_Data.csv",
                    "BHC_MacroData_Dom" : "BHC_MacroEconomicData_Domestic.csv",
                    "BHC_MacroData_Intl" : "BHC_MacroEconomicData_International.csv",
                    "CSMAR_MacroData" : "CSMAR_MacroEconomicData.csv"
                    }):
        print("Initialize Objects")
        self.file_dict = {}
        self.logger = setup_log(tag='BHCCSMAR_CapitalPredict')
        self.logger.info(f"Initialize Class CSMAR Capital Predict: Started")
        print("Set CSMAR Data Working Directories")
        for k,v in varpath_dict_args.items():
            #print(k,v)
            if k == "rootdir":

                #Changing to Root Directory
                exec('''self.''' + k + ''' = "''' + v + '''"''')
                #exec("print(self." + k + ")")
                exec("os.chdir(self." + k + ")")
            if not k.startswith(("CSMAR", "BHC")):
                #Setting variable names to class variables.
                exec('''self.''' + k + ''' = "''' + v + '''"''')
                #exec("print(self." + k + ")")
            if k.startswith(("CSMAR", "BHC")) and v.endswith(".csv"):
               self.file_dict[k] = os.path.join(varpath_dict_args["rootdir"],varpath_dict_args["datadir"],v)

            #print(self.file_dict)
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
        #tmp_dict = {}
        tmp_dict["CSMAR_Codes"] = list(tmp_test[varname_tmp[0]].unique())

        # BHC
        tmp_dict["BHC_Codes"] = list(tmp_test[varname_tmp[1]].unique())

        self.logger.info(f"CSMAR Data File Loader: Completed")
        self.Codes_Dict = tmp_dict
        return

    def CSMAR_Data_MergeLoad(self, CompanyProfile = "CSMAR_CompProfile", ref_dict = "CSMAR_BalanceSheet", idcol = "Stkcd", mergecols = ["Stkcd", "Accper"],checkfileexists = True, existspath = "./data/CSMAR_Subset.csv"):

        if checkfileexists and  os.path.exists(existspath): #TODO: Add logic to check for file
            print(existspath,"File Exists, Loading now")
            self.CSMAR_Data_Raw = pd.read_csv(existspath, sep = ',')

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

            df  = df[[idcol.upper(), "ACCPER"] + test.Codes_Dict["CSMAR_Codes"]]
            self.logger.info(f"CSMAR Data Filtering and Object Merging: Completed")

            df['ACCPER'] = pd.to_datetime(df['ACCPER'], format = '%Y%m%d')
            df['ACCPER'][(pd.to_datetime(df['ACCPER']).dt.day == 1) & (pd.to_datetime(df['ACCPER']).dt.month == 1)] = df['ACCPER'][(pd.to_datetime(df['ACCPER']).dt.day == 1) & (pd.to_datetime(df['ACCPER']).dt.month == 1)].apply(
                lambda x: pd.to_datetime(str(pd.to_datetime(x).year - 1) + '-12-31'))

            gc.collect()
            self.CSMAR_Data_Raw = df
            self.CSMAR_Data_Raw.to_csv(existspath, sep = ',', index = False)
        return

    def BHC_Data_Load(self, checkfileexists  = True, existspath = "./data/BHC_Subset.csv", id_cols = ["RSSD9001","RSSD9999","RSSD9017","BHCK3368"], consqtr = True, concecutiveqtrs = 8, sincedt = '1989-12-31', topbanks = True, topbanksorderby = 'BHCK3368', topbankslimit = 1000):
        if checkfileexists and  os.path.exists(existspath): #TODO: Add logic to check for file
            print(existspath,"File Exists, Loading now")
            self.BHC_Data_Raw = pd.read_csv(existspath, sep = ',')
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
                tmp_dict["BHC_Full_Data"] = tmp_dict["BHC_Full_Data"].loc[:,~tmp_dict["BHC_Full_Data"].columns.duplicated()]

                print("Subsetting Dataset with ID Cols and BHC_Codes")
                tmp_subset = tmp_dict["BHC_Full_Data"][id_cols + self.Codes_Dict["BHC_Codes"]]

                print("Formatting Date Column to Datetime data type")
                tmp_subset['RSSD9999'] = pd.to_datetime(tmp_subset['RSSD9999'], format="%Y%m%d")

                if consqtr:
                    print("Determining RSSD9001 Ids that have at least",concecutiveqtrs, "quarters of data." )
                    BankPerf_Merger_df_gbsum = tmp_subset
                    BankPerf_Merger_df_gbsum = BankPerf_Merger_df_gbsum.sort_values(['RSSD9001', 'RSSD9999'])

                    print("Filtering Since Date", sincedt)
                    BankPerf_Merger_df_gbsum_tmp = BankPerf_Merger_df_gbsum[BankPerf_Merger_df_gbsum["RSSD9999"] > sincedt]
                    BankPerf_Merger_df_gbsum_tmp["ConsecutivePeriods"] = BankPerf_Merger_df_gbsum_tmp.sort_values(['RSSD9001', 'RSSD9999']).groupby(['RSSD9001']).cumcount() + 1
                    BankPerf_Merger_df_gb_tmp = BankPerf_Merger_df_gbsum_tmp.groupby('RSSD9001')
                    BankPerf_Merger_df_gb_tmp = BankPerf_Merger_df_gb_tmp.agg({"ConsecutivePeriods": ['min', 'max', 'count']})
                    RSSD_ID_consecutive = list(BankPerf_Merger_df_gb_tmp[BankPerf_Merger_df_gb_tmp["ConsecutivePeriods"]["count"] >= concecutiveqtrs].index)
                    print("Subsetting Banks with Consecutive Quarters requirement from the data.")
                    BankPerf_Merger_df_gbsum = BankPerf_Merger_df_gbsum[BankPerf_Merger_df_gbsum["RSSD9001"].isin(RSSD_ID_consecutive)]
                else:
                    BankPerf_Merger_df_gbsum = tmp_subset

                if topbanks:
                    print("Filtering TopBanks based on", topbanksorderby)
                    topbanks_filter = BankPerf_Merger_df_gbsum[["RSSD9001", "RSSD9999", topbanksorderby]]

                    RSSD_IDS = (topbanks_filter.groupby('RSSD9001')
                                .agg({'RSSD9999': 'count', topbanksorderby: 'mean'})
                                .reset_index()
                                .rename(columns={'RSSD9999': 'ReportingDate_count', topbanksorderby : topbanksorderby + '_avg'})
                                ).sort_values([topbanksorderby + '_avg', 'ReportingDate_count'], ascending=False).head(
                        topbankslimit).reset_index(drop=True)["RSSD9001"]
                    BankPerf_Merger_df_gbsum = BankPerf_Merger_df_gbsum[BankPerf_Merger_df_gbsum['RSSD9001'].isin(RSSD_IDS)]

                print("Assigning BHC data to class object")
                self.BHC_Data_Raw = BankPerf_Merger_df_gbsum
                self.BHC_Data_Raw.to_csv(existspath, sep = ',', index = False)

        return
    def BHC_MarcoEcon_Load(self,exclude_columns = ["Date"], n_components = 12):
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
                    lambda x: x.replace(' Q1', "-03-31").replace(" Q2", '-06-30').replace(" Q3", '-09-30').replace(" Q4","-12-31")),errors='coerce')
                dfs.append(tmp_dict[k])

            print("Combine List with Left Merge")
            tmp_final_df = reduce(lambda left, right: pd.merge(left, right, on="Date", how="left"), dfs)

            print("Drop Columns")
            tmp_final_df = tmp_final_df[tmp_final_df.columns.difference([x for x in tmp_final_df.columns if x.startswith("Scenario Name")])]

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
            x_train_normalized = pd.DataFrame(x_train)  # It is required to have X converted into a data frame to later plotting need
            # print("Reapplying Column Names")
            x_train_normalized.columns = columns
            df_norm = x_train_normalized
            print("Recombining Excluded Columns")
            df_norm = pd.concat([tmp_final_df[exclude_columns].reset_index(drop=True), df_norm.reset_index(drop=True)],axis=1)

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

    def CSMAR_MacroEcon_Load(self, n_components = 12, exclude_columns = ['q_dates_data']): # TODO: Add arguments
        # Load EconData CSMAR
        self.logger.info(f"CSMAR EconData Loading and Cleaning: Started")
        print("Loading CSMAR Files")
        tmp_dict = {}

        for k, v in self.file_dict.items():
            if k.startswith("CSMAR_MacroData"):
                print("Initializing:", k)
                print("Loading File:", v)
                tmp_dict[k] = pd.read_csv(v, skiprows = 1)
                print("Loading Complete")


        if "CSMAR_MacroData" in tmp_dict.keys():
            print("Convert Date Column Accordingly")
            tmp_dict['CSMAR_MacroData']['q_dates_data'] = tmp_dict['CSMAR_MacroData']['q_dates_data'].astype(str).apply( lambda x: x.replace('.0', '-03-31').replace('.25', '-06-30').replace('.5', '-09-30').replace('.75',
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
            x_train_normalized = pd.DataFrame(x_train)  # It is required to have X converted into a data frame to later plotting need
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

    def Bank_Data_Subsetting(self,bhc_id_cols = ['RSSD9001', 'RSSD9999'], bhc_target_cols = ['BHCT3247'], bhc_exclude_cols = ['RSSD9017'],csmar_id_cols = ['STKCD', 'ACCPER'], csmar_target_cols = ['A003105000'], csmar_exclude_cols = [],
                             bank_dates = ['1991-12-31','2018-12-31'], econ_dates = ['1991-12-31','2018-12-31']):
        self.logger.info(f"Subsetting Bank Input and Target Data: Started")
        print("Subsetting Inputs and Targets for BHC")
        # inputs
        BHC_X = self.BHC_Data_Raw[self.BHC_Data_Raw.columns.difference(bhc_target_cols + bhc_exclude_cols)]
        BHC_X = BHC_X.rename({bhc_id_cols[0]:"ID", bhc_id_cols[1]:"DATE"}, axis = 1)

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

                self.Data_Dict[k] = self.Data_Dict[k][(self.Data_Dict[k]['DATE'] >= econ_dates[0]) & (self.Data_Dict[k]['DATE'] <= econ_dates[1])]
                print(k,'Completed')
            else:
                self.Data_Dict[k] = self.Data_Dict[k][(self.Data_Dict[k]['DATE'] >= bank_dates[0]) & (self.Data_Dict[k]['DATE'] <= bank_dates[1])]
                print(k, 'Completed')
        return

    def Pd_to_3dNumpy(self):
        return



#Loading Initi Class
test = BHC_CSMAR_TLF()

#Configure BHC Codes Method to then subset CSMAR and BHC
test.load_BHC_CSMAR_Codes()

#Load CSMAR Files , Merge (Comp Profile, Balance Sheet and Income Statement), Subset.
test.CSMAR_Data_MergeLoad(checkfileexists=False)

#Load BHC files, Filter based on consecutive qtrs, top banksm data formatting and subsetting.
test.BHC_Data_Load()

#Load EconData CSMAR
test.CSMAR_MacroEcon_Load()

#Load EconData BHC #TODO: Get newest US Macroeconomics data
test.BHC_MarcoEcon_Load()

#Transform Data into appropirate Tensor Formatting
#Extract target and domain target values and input values. #TODO: Need to isolate target and input variables accordingly.
test.Bank_Data_Subsetting()


test.Data_Dict.keys()

fulldatescombine = True
qtr_data = True
datecol = 'DATE'
idcol = 'ID'

dates = ["2002-12-31", "2018-09-30"]
econ_dates = ["2002-12-31", "2018-09-30"]
test.Data_Dict_NPY = dict()

#Tranform to 3 dimension array for EconData and Banking Data
for k in test.Data_Dict.keys():
    #print(k)
    #k = 'BHC_ECON_X'
    if 'ECON' not in k:
        df = test.Data_Dict[k]
        if fulldatescombine:
            # print("Generating Full Mesh RSSD_ID and Reporting Date to left join to.")
            RepordingDate_Df = pd.DataFrame(df[datecol].unique())
            RepordingDate_Df = RepordingDate_Df.rename({0: datecol}, axis=1)
            RepordingDate_Df["key"] = 0

            BankIds_Df = pd.DataFrame(df[idcol].unique())
            BankIds_Df = BankIds_Df.rename({0: idcol}, axis=1)
            BankIds_Df["key"] = 0

            BankID_Date_Ref = RepordingDate_Df.assign(foo=1).merge(BankIds_Df.assign(foo=1), on="foo", how="outer").drop(
                ["foo", "key_x", "key_y"], 1)

            df_pd = BankID_Date_Ref.merge(df, left_on=[datecol, idcol],
                                          right_on=[datecol, idcol], how="left")
    else:
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
        df_pvt = pd.pivot_table(df_pd, index=[datecol],dropna = False)
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
    #Save them to a Dictionary to be used for splitting and modeling later.
    print("Saving objects to file")
    np.save(k + ".npy", result_pv)
    if qtr_data:
        np.save(k + "_qtr.npy", results_quarter)

    print("Saving to class dict")
    test.Data_Dict_NPY[k] = result_pv
    if qtr_data:
        test.Data_Dict_NPY[k + '_qtr'] = results_quarter
    #return



test.Data_Dict_NPY['BHC_X'].shape
test.Data_Dict_NPY['BHC_Y'].shape
test.Data_Dict_NPY['BHC_ECON_X'].shape


test.Data_Dict_NPY['BHC_X_qtr'].shape
test.Data_Dict_NPY['BHC_Y_qtr'].shape
test.Data_Dict_NPY['BHC_ECON_X_qtr'].shape



test.Data_Dict_NPY['CSMAR_X_qtr'].shape
test.Data_Dict_NPY['CSMAR_Y_qtr'].shape
test.Data_Dict_NPY['CSMAR_ECON_X_qtr'].shape



    #Data Train Test splitting

    #Modeling, Cross Validation for time series incorporated.
















#Split the Data for Training, Validation, Testing.(Seperate Input Training\Testing from Target Training and Target Testing

#Modeling, (Linear, Deep Models, then try Transfer Learing straight forward methods







        # print("Calculating if DataFrame has appropriate length and quarters.")
        # reporting_date_lst = list()
        # for year_num in range(int(ReportingDate_Start.split(" ")[0]), int(ReportingDate_End.split(" ")[0]) + 1):
        #     for qtr_num in range(1, 5):
        #         reporting_date_lst.append(str(year_num) + " Q" + str(qtr_num))
        # ReportingDate_DF = pd.DataFrame(reporting_date_lst, columns=["ReportingDate"])
        # ReportingDate_DF = pd.DataFrame(list(ReportingDate_DF[
        #                                          (ReportingDate_DF["ReportingDate"] >= ReportingDate_Start) & (
        #                                                      ReportingDate_DF["ReportingDate"] <= ReportingDate_End)][
        #                                          "ReportingDate"].unique()),
        #                                 columns=["ReportingDate"])
        #
        # if ReportingDate_DF.shape[0] == df.shape[0] and all(
        #         elem in list(df.ReportingDate.unique()) for elem in list(ReportingDate_DF.ReportingDate)):
        #     print("Dates and Rows Validated")
        #     df_pd = df
        # else:
        #     df_pd = ReportingDate_DF.merge(df, left_on=["ReportingDate"], right_on=["ReportingDate"], how="left")
        #     df_pd.interpolate(method="polynomial", order=2).fillna(method="bfill").fillna(method="ffill").dropna(axis=1,
        #                                                                                                          how='all')

        print(df_pd.shape)
        print("Subsetting Dates from Dataframe")
        df_pd = df_pd[(df_pd["ReportingDate"] >= ReportingDate_Start) & (df_pd["ReportingDate"] <= ReportingDate_End)]
        print(df_pd.shape)

        print("Pivoting Pandas Table to be Indexed by RSSD_ID and ReportingDate")
        df_pvt = pd.pivot_table(df_pd, index=["ReportingDate"])
        print(df_pvt.shape)

        print("Preparing to Reshape and output as Numpy Array")
        # dim1 = df_pvt.index.get_level_values('RSSD_ID').nunique()
        dim1 = df_pvt.index.get_level_values('ReportingDate').nunique()
        print("Reshaping into 3 dimensional NumPy array")
        result_pv = df_pvt.values.reshape((1, dim1, df_pvt.shape[1]))
        print(result_pv.shape)

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
