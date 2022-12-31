from readFileTrain import readFileTrain_by_path as rtp
import os
import glob
from numba import jit
import pandas as pd


#file_text = glob.glob(path_train+"/*.txt")
#file_excel = glob.glob(path_train+"/*.xlsx")
#file_csv = glob.glob(path_train+"/*.csv")


############### setting ########################
path_train = 'train/' # ใส่ path ของ training dataset
change_col_name = {'name.1':'tag','Tag':'tag', 'Paragraph_Text':'content'}
readTrain = rtp.readTrain_by_path()
df_train = pd.DataFrame(columns=['conten','tag'])


################ Function ######################
def concat_tab(listdata):
    for tab in listdata:
        print(tab)
        data = readTrain.to_pandas_tab(path_train+tab, change_col_name)
        data_concat = pd.concat([df_train, data[['content','tag']]], axis=0)
    return data_concat


################### For Users ##################

listdata = os.listdir(path_train)