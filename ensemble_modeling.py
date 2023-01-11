import pandas as pd
import time
#from numba import jit,cuda
import seaborn as sns
import matplotlib as plt
from sklearn.model_selection import train_test_split
#from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score



from sklearn.model_selection import cross_val_score
from sklearn.model_selection import validation_curve

class runModel():
    def __init__( self ):
        self._input = ''
        self._models = (
                (DecisionTreeClassifier().__class__.__name__ , DecisionTreeClassifier(max_depth=None, criterion='gini')),
                (RandomForestClassifier().__class__.__name__ ,RandomForestClassifier(n_estimators=20, max_depth=3, random_state=0, n_jobs=-1)),
                (LinearSVC().__class__.__name__ , LinearSVC()),
                (SGDClassifier().__class__.__name__ , SGDClassifier(loss="modified_huber" ,n_jobs=-1)),
                (MultinomialNB().__class__.__name__ , MultinomialNB()),
                (LogisticRegression().__class__.__name__ , LogisticRegression(random_state=0, n_jobs=-1, solver='saga')),
                (MLPClassifier().__class__.__name__ , MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)),
                (KNeighborsClassifier().__class__.__name__ , KNeighborsClassifier(n_neighbors=10, n_jobs=-1)),
                (NearestCentroid().__class__.__name__ , NearestCentroid()),
                (AdaBoostClassifier().__class__.__name__ , AdaBoostClassifier(n_estimators=100)),
            )
        self._count_vect = CountVectorizer()
        self._tfidf_transformer = TfidfTransformer() 
        
        
    #@jit(target_backend='cuda')
    def model_selection( self, input, features ):
        CV = 5
        cv_df = pd.DataFrame(index=range(CV * len(self._models)))
        entries = []
        labels = input['tag_id']
        for key, model in self._models:
            model_name = key
            start_time = time.time()
            accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
            end_time = time.time()
            for fold_idx, accuracy in enumerate(accuracies):
                entries.append((model_name, fold_idx, accuracy, (end_time-start_time)/CV))
        cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy', 'runtime (second)'])
        cv_df.accuracy = cv_df.accuracy.round(2)
        #pd.options.display.float_format = '{:,.2f}'.format
        model_accuracy = cv_df.groupby(['model_name'], as_index=False)[['accuracy','runtime (second)']].mean()
        mean_tb = model_accuracy.sort_values(by=['accuracy'],ascending=False)
        return mean_tb
        
    def show_model( self, tb ):
        print(tb)
        while True:
            ans = input('Do you want to select model by yourself? [y/n] : ')
            if ans == 'y':
                print(tb)
                return tb
                #return select_tb[['model_name', 'accuracy', 'runtime (second)']]
            elif ans == 'n':
                select_model = tb.sort_values(['accuracy', 'runtime (second)'], ascending=[False, True])['model_name'].to_records(index=False)[0]
                
                print('This program will choose the best model base on accuracy.')
                print("We recommend '{}'!".format(select_model))
                return select_model
                #return tb[tb.accuracy == tb.accuracy.max()]['id']
            else:
                print('Try again!')
                
    def train_model( self, id_model, x_train, y_train):
        
        X_train = x_train.map(' '.join)
        X_train_counts = self._count_vect.fit_transform(X_train)
        model = dict(self._models)[id_model]
        X_train_tfidf = self._tfidf_transformer.fit_transform(X_train_counts)
        
        if id_model == SGDClassifier().__class__.__name__:
            model_trained = model.partial_fit(X_train_tfidf, y_train, classes=pd.unique(y_train))
        else :
            model_trained = model.fit(X_train_tfidf, y_train)
        return model_trained
                
    def model_predict_multiprob( self, content, model_trained, num_label ):
        #count_vect = CountVectorizer()
        print('Document : ', content)
        #print('Tag : ',df_train_main['tag'][num])
        prediction = model_trained.predict_proba(self._count_vect.transform([content]))
        pred_To_Class_map = pd.DataFrame(prediction, columns=model_trained.classes_)
        pred_To_Class_map = pred_To_Class_map.transpose().rename(columns={0:'Probability'})
        #print('Predict Tag : ',model_trained.predict(self._count_vect.transform([content]))[0])
        print('Predict Tag : \n{}'.format(pred_To_Class_map.nlargest(num_label, 'Probability')))
        #print('Predict Tag : '.format(accuracy_score(test[category], prediction)))
        
    def model_predict( self, content, model_trained ):
        #count_vect = CountVectorizer()
        print('Document : ', content)
        print('Predict Tag : ',model_trained.predict(self._count_vect.transform([content]))[0])
        return model_trained.predict(self._count_vect.transform([content]))[0]
    
    def model_predict_table( self, content, model_trained ):
        #count_vect = CountVectorizer()
        return model_trained.predict(self._count_vect.transform([content]))[0]
        
    def model_evaluate( self, model, X_test, y_test,data_vec  ):
        X_test = X_test.map(' '.join)
        y_pred = model.predict(self._count_vect.transform(X_test))
        data_vec['tag_id'] = data_vec['tag'].factorize()[0]
        result = dict()
        #tag_id_df = data_vec[['tag', 'tag_id']].drop_duplicates().sort_values('tag_id')
        #conf_mat = confusion_matrix(y_test, y_pred)
        #print(model.__class__.__name__)
        #print(len(set(y_pred)))
        #print("Accuracy: %1.3f " % (accuracy_score(y_test, y_pred)))
        #print(metrics.classification_report(y_test, y_pred, target_names=data_vec['tag'].unique()[:len(y_test.unique())]))
        result['model_name'] = model.__class__.__name__
        result['Accuracy'] = accuracy_score(y_test, y_pred)
        return result
        
        
    """def model_evaluated_for_table( self, model_name, accuracy ):
        X_test = X_test.map(' '.join)
        y_pred = model.predict(self._count_vect.transform(X_test))
        data_vec['tag_id'] = data_vec['tag'].factorize()[0]
        
        return """
        
        
    def model_evaluate_multilabel( self, model, X_test, y_test, data_vec  ):
        #X_test = X_test.map(' '.join)
        y_pred = model.predict(self._count_vect.transform(X_test))
        data_vec['tag_id'] = data_vec['tag'].factorize()[0]
        print(metrics.classification_report(y_test, y_pred, target_names=data_vec['tag'].unique()))
