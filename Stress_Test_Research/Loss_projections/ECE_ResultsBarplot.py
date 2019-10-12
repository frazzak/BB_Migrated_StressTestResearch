
#ECE Compare BarPlot.
#TODO: make this into a function
os.getcwd()
ECE_ResultsBarPlot()
ECE_ResultsBarPlot(filesuffix = "Quarterly_results_lld.csv", horizontal = True)

def ECE_ResultsBarPlot(filedir  = '/Users/phn1x/icdm2018_research_BB/Stress_Test_Research/Loss_projections/Images/Table_Data', filesuffix = "Quarterly_results_rmse.csv", horizontal = False):

    os.chdir(os.path.join(filedir))
    print("Getting Files to Combine and Plot")
    BarPlot_Df = pd.DataFrame()
    for file in [x for x in os.listdir() if x.endswith(filesuffix)]:
        #Read in File
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
        ax = BarPlot_Df.transpose().plot.barh(rot=0, title='Economic Conditions Estimation Model Comparision:' + "".join(
            filesuffix.split("_")[0]))
    else:
        ax = BarPlot_Df.transpose().plot.bar(rot = 0, title = 'Economic Conditions Estimation Model Comparision:' + "".join(filesuffix.split("_")[0]))
    ax.set_ylabel(filesuffix.replace(".csv","").split("_")[-1].upper())
    ax.grid(color = "black")
    plt.savefig(os.path.join("_".join(filesuffix.replace(".csv","").split("_") + ["modelbarplot"]) + '.pdf'), dpi=300, format='png', bbox_inches='tight')

