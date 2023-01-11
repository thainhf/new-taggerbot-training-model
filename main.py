# -*- encoding: utf-8 -*-
from readFileTrain import readFileTrain_by_path as rtp
from numba import jit, cuda
import os
import time
import pandas as pd
import preprocess as pre
import ensemble_modeling as ensem_model
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import NearestCentroid



#file_text = glob.glob(path_train+"/*.txt")
#file_excel = glob.glob(path_train+"/*.xlsx")
#file_csv = glob.glob(path_train+"/*.csv")


############### setting ########################
print('Program is starting...')
path_train = 'train/' # ใส่ path ของ training dataset
change_col_name = {'name.1':'tag','Tag':'tag', 'Paragraph_Text':'content'}
tag_replace = {'ict  literacy': 'ict literacy', 'information lieteracy skill':'information literacy skill',
               'media literacy':'media literacy skills', 'productivity':'productivity and accountability',
               'critical thinking':'critical thinking and problem solving'}


readTrain = rtp.readTrain_by_path()
clean = pre.cleanData()
tokenize = pre.Tokenization()
vector = pre.tfidf_vector()
model = ensem_model.runModel()

################ Function ######################
#@jit
def concat_tab(listdata):
    start_time = time.time()
    data_concat = pd.DataFrame(columns=['content','tag'])
    data = pd.DataFrame(columns=['content','tag'])
    with ThreadPoolExecutor(max_workers=10) as executor :
        future_to_dat = {executor.submit(readTrain.to_pandas_tab, path_train+tab, change_col_name) : tab for tab in listdata}
        for future in concurrent.futures.as_completed(future_to_dat):
            tab = future_to_dat[future]
            
            try:
                data = future.result()
                print('Reading file {} completed!'.format(tab))
                
            except Exception as exc:
                print('Error')
            else:
                data_concat = pd.concat([data_concat, data[['content','tag']]], axis=0)
                #print('----------------------------------------------')
            """for tab in listdata:
                print(tab)
                data = readTrain.to_pandas_tab(path_train+tab, change_col_name)"""
            
        end_time = time.time()
        print('Time spent to read all file =', end_time-start_time, 'sec')
        data_concat['tag'] = data_concat['tag'].str.lower()
    return data_concat

print('Setting Completed!')

############################ For Users #########################################
print('Program is finding train datasets...')
listdata = os.listdir(path_train)
print('Finding train datasets is completed!')
print('----------------------------------------------')

print('Starting to concat training datsets...')
df_train = concat_tab(listdata)

df_train = df_train.replace({'tag':tag_replace})



df_train = clean.cleaning(df_train)
df_train = tokenize.split_token(df_train)

#df_train.drop(grouped.get_group('content').index)

categories = set(df_train.tag)
print('before : {}'.format(df_train.shape))

df = (df_train.groupby(['content'])['tag']
      .apply(list)
      .reset_index())
#df_contain = df[df['tag'].str.contains(',')]
#print('Duplicated : {}'.format(df_contain.shape))
df_train = df_train.drop_duplicates(keep=False, subset=['content'])
print('After : {}'.format(df_train.shape))

"""df_train.to_excel('check.xlsx')
df_train_vec = vector.pre_vector(df_train)
X_train, X_test, y_train, y_test = train_test_split(df_train['token'], df_train['tag'], random_state = 0)"""

#print(vector.split_to_vector(df_train)[0])
#print(vector.featureSelection(df_train))
print('Finding features...')
feature = vector.features(df_train)
print('Transform table...')
df_train_vec = vector.pre_vector(df_train)

print('We starting to select model...')
#crossVal = model.model_selection(df_train_vec, feature)
#show = model.show_model(crossVal)

#df_train = df_train.sample(int((15/100)*df_train.shape[0]))

X_train, X_test, y_train, y_test = train_test_split(df_train['token'], df_train['tag'], random_state = 42, test_size=0.23)
print('Train size = {}'.format(X_train.shape))
print('Test size = {}'.format(X_test.shape))

#model_trained = model.train_model( show, X_train, y_train )
columns = ['model_name']
columns.extend(categories)
eval_tb = pd.DataFrame(columns=columns)


list_model = (DecisionTreeClassifier().__class__.__name__,
              RandomForestClassifier().__class__.__name__ ,
              LinearSVC().__class__.__name__, 
              SGDClassifier().__class__.__name__,
              MultinomialNB().__class__.__name__,
              MLPClassifier().__class__.__name__ ,
              LogisticRegression().__class__.__name__,
              KNeighborsClassifier().__class__.__name__ ,
              NearestCentroid().__class__.__name__ ,
              AdaBoostClassifier().__class__.__name__ ,
              
              )

"""for key in list_model:
    model_trained = model.train_model( key, X_train, y_train )
    filename = "models/"+key+"_model.pickle"
    # save model
    print('Saving model...')
    pickle.dump(model_trained, open(filename, "wb"))
    print('You can load the model with path =', filename)"""
# load model
#filename = "models/LR_model.pickle" 
#loaded_model = pickle.load(open(filename, "rb"))
#model_trained = model.train_model( SGDClassifier().__class__.__name__, X_train, y_train )

#model.model_evaluate( model_trained, X_test, y_test, df_train_vec  )

#content = 'ความรู้ และประสบการณ์วิชาชีพ ตามมาตรฐานวิชาชีพครู ได้แก่ (1) วิชาชีพครู (2) วิชาการใช้ภาษาไทยเพื่อการสื่อสาร (3) วิชาการใช้ภาษาอังกฤษเพื่อการสื่อสาร (4) วิชาการใช้เทคโนโลยีดิจิทัลเพื่อการศึกษา และ (5) วิชาเอก ตามที่คณะกรรมการคุรุสภากำหนด และ (ข) การปฏิบัติงานและการปฏิบัติตน'
#model.model_predict(content, model_trained)
#num_label = 2

for key in list_model:
    model_trained = model.train_model( key, X_train, y_train )
    df_train[key+'_predict'] = df_train['content'].apply(lambda x : str(model.model_predict_table(x, model_trained)))
    model.model_evaluate( model_trained, X_test, y_test, df_train_vec  )
df_train.to_excel('check.xlsx')


############################# end #######################################3
