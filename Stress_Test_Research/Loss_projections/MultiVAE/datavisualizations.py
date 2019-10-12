
#ECE Compare BarPlot.
#TODO: make this into a function
os.getcwd()
ECE_ResultsBarPlot()


temp1 = ECE_ResultsBarPlot(filesuffix = "Yearly_results_rmse_qtr_1", horizontal = False) # Working
temp2 = ECE_ResultsBarPlot(filesuffix = "Yearly_results_rmse_qtr", horizontal = True) #Working
temp3 = ECE_ResultsBarPlot(filesuffix = "Quarterly_results_rmse", horizontal = False)#Working

temp2 = ECE_ResultsBarPlot(filesuffix = "Yearly_results_lld_qtr", horizontal = True) #Working
temp3 = ECE_ResultsBarPlot(filesuffix = "Quarterly_results_lld", horizontal = False)#Working



def ECE_ResultsBarPlot(filedir  = '/Users/phn1x/icdm2018_research_BB/Stress_Test_Research/Loss_projections/Images/Table_Data', filesuffix = "Quarterly_results_rmse",extension = ".csv", horizontal = False):

    os.chdir(os.path.join(filedir))
    print("Getting Files to Combine and Plot")
    BarPlot_Df = pd.DataFrame()

    if  len(filesuffix.split("_")) > 4 and filesuffix.split("_")[3] == 'qtr':
        ylabel = " ".join([filesuffix.replace(".csv", "").split("_")[-2],str(int(filesuffix.replace(".csv", "").split("_")[-1]) + 1), "RMSE"]).upper()

    elif len(filesuffix.split("_")) == 4 and filesuffix.split("_")[3] == 'qtr' :
        ylabel = filesuffix.replace(".csv", "").split("_")[-2].upper()

    else:
        #ax = BarPlot_Df.transpose().plot.bar(rot = 0, title = 'Economic Conditions Estimation Model Comparision:' + "".join(filesuffix.split("_")[0]))
        ylabel = filesuffix.replace(".csv","").split("_")[-1].upper()
        #ax.set_ylabel(filesuffix.replace(".csv","").split("_")[-1].upper())


    if filesuffix.startswith("Yearly") and len(filesuffix.split("_")) == 4 and filesuffix.split("_")[3] == 'qtr' :
        ylabel = ylabel = filesuffix.replace(".csv", "").split("_")[-2].upper()
        for file in [x for x in os.listdir() if x.replace(".csv", "")[:-2].endswith(filesuffix)]:
               #file = 'ECE_Experiment_1_Yearly_results_rmse_qtr_0.csv'
                tmp_name  = "".join([file.replace(".csv", "").split("_")[-2],str(int(file.replace(".csv", "").split("_")[-1]) + 1)]).upper().replace("QTR","Q")
                tmp_name = "".join(file.split("_")[1:3]).replace("Experiment", "Exp") + " " + tmp_name
                experiment_name = tmp_name
                tmp_df = pd.read_csv(file)
                print(experiment_name)
                tmp_df = tmp_df[tmp_df.iloc[:, 0] == "mean"]
                tmp_df = tmp_df.drop("Unnamed: 0", axis=1)
                tmp_df.index = [experiment_name]
                BarPlot_Df = pd.concat([BarPlot_Df, tmp_df], axis=0)
    else:
        for file in [x for x in os.listdir() if x.replace(extension,"").endswith(filesuffix)]:
            #file = 'ECE_Experiment_1_Yearly_results_rmse_qtr_1.csv'
            #Read in File
            #file = "ECE_Experiment_3_Yearly_results_rmse_qtr_1"
            tmp_df = pd.read_csv(file)
            #Get Experiment Name
            experiment_name = " ".join(file.split("_")[1:3])

            print(experiment_name)
            #Get only Mean Data
            tmp_df = tmp_df[tmp_df.iloc[:,0] == "mean"]
            #Transpose and drop first column, index column
            tmp_df = tmp_df.drop("Unnamed: 0", axis = 1)
            #Give Column Experiment name.
            tmp_df.index = [experiment_name]
            #Combine dataframe
            BarPlot_Df = pd.concat([BarPlot_Df,tmp_df], axis = 0)

    print("Plotting the bars")
    print("Set the chart's title")

    if horizontal:
        ax = BarPlot_Df.transpose().plot.barh(rot=0, title='Economic Conditions Estimation Model Comparision: ' + "".join(filesuffix.split("_")[0]))
    else:
        ax = BarPlot_Df.transpose().plot.bar(rot=0,
                                             title='Economic Conditions Estimation Model Comparision:' + "".join(
                                                 filesuffix.split("_")[0]))



    ax.grid(color = "black")

    if horizontal:
        tmp_file_name = "_".join(filesuffix.replace(".csv","").split("_") + ["horizontal_barplot"]) + '.pdf'
        ax.set_xlabel(ylabel)
    else:
        tmp_file_name = "_".join(filesuffix.replace(".csv", "").split("_") + ["barplot"]) + '.pdf'
        ax.set_ylabel(ylabel)

    plt.savefig(os.path.join(tmp_file_name), dpi=300, format='png', bbox_inches='tight')

    return(BarPlot_Df)





#EDA of Data
#TODO: Histogram of target variables, Y Loss Combined, Y T1 CECR
#Loan Loss Combined and Tier1 Common Equity Capital Ratio
from scipy.interpolate import spline
import scipy as sp
temp = Preprocess_Dict['Y_nco']
temp["ReportingDate"] = temp["ReportingDate"].apply(lambda x: pd.to_datetime(str(int(12/int(x.split("Q")[1]))) + "/" + x.split(" ")[0]) )

temp["NCO_Combined"]
xnew = np.linspace(temp.min(),temp.max(),90) #300 represents number of points to make between T.min and T.max

power_smooth = spline(T,power,xnew)

temp2 = pd.pivot_table(temp[temp.columns.difference(["RSSD_ID"])][temp["ReportingDate"] >= "01-01-1986"], index=["ReportingDate"], aggfunc = np.mean)
#pd.pivot_table(temp[temp.columns.difference(["RSSD_ID"])][temp["ReportingDate"] >= "01-01-1986"], index=["ReportingDate"], aggfunc = np.mean).plot(subplots = False)
#Need to smooth it out between the quarters.

temp2["NCO_Combined"].plot(color = "grey")
#p = sp.polyfit(range(0,len(temp2["NCO_Combined"].index)), temp2["NCO_Combined"], deg=100)
#y_ = sp.polyval(p, temp2["NCO_Combined"])

# plot smoothened data
#plt.plot(range(0,len(temp2["NCO_Combined"].index)), y_, color='r', linewidth=2)

import seaborn as sns
sns.distplot(Preprocess_Dict['Y_nco'])

Preprocess_Dict['Y_nco'][Preprocess_Dict['Y_nco']["ReportingDate"] >= "01-01-1990"][Preprocess_Dict['Y_nco'].columns.difference(["RSSD_ID","ReportingDate"])].plot.hist()

Preprocess_Dict['CapRatios'][Preprocess_Dict['CapRatios']["ReportingDate"] >= "1990 Q1"][Preprocess_Dict['CapRatios'].columns.difference(["RSSD_ID","ReportingDate"])].plot.hist()
Preprocess_Dict['CapRatios'][Preprocess_Dict['CapRatios']["ReportingDate"] >= "1990 Q1"][Preprocess_Dict['CapRatios'].columns.difference(["RSSD_ID","ReportingDate"])].plot.hist()


Preprocess_Dict['Y_nco'].describe()






#######Histogram of remaining Net Charge Off Rates
Preprocess_Dict['Y_nco'][Preprocess_Dict['Y_nco']["ReportingDate"] >= "01-01-1990"][Preprocess_Dict['Y_nco'].columns.difference(["RSSD_ID","ReportingDate","NCO_Combined"])].plot.hist()


####Historgram Net Charge Off Combined
Preprocess_Dict['Y_nco'][Preprocess_Dict['Y_nco']["ReportingDate"] >= "01-01-1990"][Preprocess_Dict['Y_nco'].columns.difference(["RSSD_ID","ReportingDate"])]["NCO_Combined"].plot.hist(title = "Aggregate Combined Net Charge Off")
#####Regular Histogram of Common Equity Capital Ratio
Preprocess_Dict['CapRatios'][Preprocess_Dict['CapRatios']["ReportingDate"] >= "1990 Q1"][Preprocess_Dict['CapRatios'].columns.difference(["RSSD_ID","ReportingDate"])]["Other items: CapRatios_CET1CR_coalesced"].plot.hist(title = "Aggregate Tier-1 Common Equity Capital Ratio")




#Training History Data
ScenarioGenResults_dict["cgan_results_dict"]["TrainHist"]["TrainingDiscriminator_TotalLoss"].plot()
ScenarioGenResults_dict["mcvae_results_dict"]





#Line Plot with markers for Time Split BCLP
os.getcwd()
rmse_bclp = pd.read_csv('/Users/phn1x/icdm2018_research_BB/Stress_Test_Research/Loss_projections/Images/Table_Data/bclp_results_mse_xls.csv')


#Quarterly Barplot
from matplotlib import pyplot as plt


plot_bclp(interval = 'Yearly')

def plot_bclp(rawfile  = '/Users/phn1x/icdm2018_research_BB/Stress_Test_Research/Loss_projections/Images/Table_Data/bclp_results_mse_xls.csv', interval = "Qtrly", subplots = [2,2]):
    if interval in ['Qrtly']:
        rmse_bclp = pd.read_csv(rawfile)
        plot_list = []
        title_tmp_list = []
        for target in rmse_bclp.target.unique():
            #plot_list = []
            #title_tmp_list = []
            for split in rmse_bclp.split.unique():
                tmp = rmse_bclp[(rmse_bclp.interval == interval) & (rmse_bclp.split == split) & (rmse_bclp.target == target) ][['model','exp','rmse']]
                tmp.exp = ["".join(['Exp.', str(x)]) for x in tmp.exp]
                tmp = tmp.pivot(index='exp', columns='model', values='rmse').astype(float)
                tmp.index.name = 'Experiments'
                tmp = tmp.sort_index(ascending = 1)
                plot_list.append(tmp)
                title_tmp = " ".join([target[2:], interval.upper(), split.upper() + "Split"])
                print(title_tmp)
                title_tmp_list.append(title_tmp)
        fig, axes = plt.subplots(nrows=subplots[0], ncols=subplots[1])
        for idx, ax in enumerate(axes.flat):
            print(title_tmp_list[idx])
            ax_tmp = tmp.transpose().plot.bar(rot=0, title=title_tmp_list[idx], ax = ax)
            if idx == 0:
            ax_tmp.set_ylabel("RMSE")
            ax_tmp.set_xlabel("")
            ax.legend(loc='upper left')
            ax.grid()
            if idx != 0:
                ax.legend().remove()
        plt.savefig(os.path.join(interval +".png"), dpi=300, format='png', bbox_inches='tight')
    elif interval in ["Yearly",'Yrly',"Y"]:
        rmse_bclp = pd.read_csv(rawfile)

        for target in rmse_bclp.target.unique():
            #plot_list = []
            #title_tmp_list = []
            plot_list = []
            title_tmp_list = []
            for split in rmse_bclp.split.unique():
                tmp = rmse_bclp[(rmse_bclp.interval != 'Qtrly') & (rmse_bclp.split == split) & (rmse_bclp.target == target) ][['model','exp','rmse','interval']]
                tmp = tmp.replace('X', 0)
                for expnum in tmp.exp.unique():
                    tmp_e = tmp[tmp.exp == expnum]
                    tmp_e.exp = ["_".join(['Exp.'+ str(x),y]) for x,y in zip(tmp_e.exp, tmp_e.interval)]
                    tmp_e = tmp_e.pivot(index='exp', columns='model', values='rmse').astype(float)
                    tmp_e.index.name = 'Experiments'
                    tmp_e = tmp_e.sort_index(ascending = 1)
                    plot_list.append(tmp_e)
                    title_tmp = " ".join([target[2:], interval.upper(), split.upper() + "Split","Exp"+str(expnum)])
                    print(title_tmp)
                    title_tmp_list.append(title_tmp)
            fig, axes = plt.subplots(nrows=subplots[0], ncols=subplots[1])
            fig.set_size_inches(18.5, 10.5)
            fig.text(0.5, 0.04, 'Quarter', ha='center', va='center')
            fig.text(0.06, 0.5, 'RMSE', ha='center', va='center', rotation='vertical')
            for idx, ax in enumerate(axes.flat):
                print(title_tmp_list[idx])
                ax_tmp = plot_list[idx].plot(rot=0, title=title_tmp_list[idx], ax = ax, marker = 's')
                # if idx == 0:
                # ax_tmp.set_ylabel("RMSE")
                ax_tmp.set_xlabel('')
                x1=[0, 1, 2, 3]
                squad = ["Q1","Q2","Q3",'Q4']
                ax.set_xticks(x1)
                ax.set_xticklabels(squad, minor=False, rotation=45)
                ax.legend(loc='upper left')
                ax.grid()
                if idx != 0:
                    ax.legend().remove()
            filename_tmp = "_".join([target,interval + ".png"])
            plt.savefig(os.path.join(filename_tmp), dpi=300, format='png', bbox_inches='tight')
    else:
        print("Interval Not Found")
