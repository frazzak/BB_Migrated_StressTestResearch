from scipy import spatial
import os, sys


dataSetI = [3, 45, 7, 2]
dataSetII = [2, 54, 13, 15]
result = 1 - spatial.distance.cosine(dataSetI, dataSetII)

base_dir = "/Users/phn1x/Google Drive/Fall 2019/TransferLearning_Research/Data/"

csmar_codes = pd.read_csv(os.path.join(base_dir,"CSMAR_Codes.csv"))
bhc_codes = pd.read_csv(os.path.join(base_dir,"BHC_Codes.csv"))

csmar_codes
bhc_codes


csmar_codes['CSMAR Variable Description']

bhc_codes['Variable Description']



from scipy.spatial.distance import cdist
from Levenshtein import ratio



arr1 = np.array(csmar_codes['CSMAR Variable Description'])
arr2 = np.array(bhc_codes['Variable Description'])

matrix = cdist(arr2.reshape(-1, 1), arr1.reshape(-1, 1), lambda x, y: ratio(x[0], y[0]))
df = pd.DataFrame(data=matrix, index=arr2, columns=arr1)
df_sim = df.transpose()



sim_score = .7
for ind in range(0, len(df_sim.index)):

    if len(df_sim.iloc[ind,:][df_sim.iloc[ind,:] > sim_score]) > 0  :
        print("CSMAR index:", df_sim.index[ind])
        print(df_sim.iloc[ind,:][df_sim.iloc[ind,:] > sim_score])






#Get relevant BHC and CSMAR Data.












