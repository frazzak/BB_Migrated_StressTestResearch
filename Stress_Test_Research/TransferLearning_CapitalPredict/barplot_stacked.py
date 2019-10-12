stackedtest = pd.DataFrame()
test_dict = {'model': ['lstm','gru','da-rnn','lstm','gru','da-rnn'],
             'type' : ['withTransfer','withTransfer','withTransfer','withoutTransfer','withoutTransfer','withoutTransfer'],
             'rmse' : [.1735,0,0,.233,.20,.21]
             }
stackedtest = pd.DataFrame.from_dict(test_dict)

stackedtest.groupby(['model','type']).sum().unstack().plot(kind='bar',stacked= False)

test1  =  stackedtest.groupby(['model','type']).sum().unstack()
test1.names

stackedtest.plot(kind = 'bar', stacked = False)
stackedtest.plot(kind = 'bar', stacked = True)
# import gc
gc.collect()

# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

