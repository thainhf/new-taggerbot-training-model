import pandas as pd
import numpy as np

from pythainlp import thai_digits
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus.common import thai_stopwords
from pythainlp.util import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_selection import chi2

stopwords = list(thai_stopwords())

class cleanData():
    def __init__( self ):
        self._data = pd.DataFrame()
        
    def cleaning( self, data ):
        self._data = data
        return self.drop_special()
    
    def drop_special( self ):
        self._data['content'] = self._data['content'].str.replace('[\\<>\;\[\]\:\-*\*\$\,\.\?!"\'\^\()\&\/\{\}\|\£\«\»\’\“\”\•\•\■\™\~\=\+\\\]\s*', '', regex=True)
        self._data = self._data.reset_index(drop=True)
        return self.drop_na()
    
    def drop_na( self ):
        print('Number of empty content:',len(self._data[self._data['content'].isna()]))
        self._data = self._data.dropna(subset=['content'])
        return self.drop_duplicate()
    
    def drop_duplicate( self ):
        return self._data.drop_duplicates()
    
    
class Tokenization():
    def __init__( self ):
        self._datToken = pd.DataFrame()
        
        # Create Pipeline
        #self._pipeline = [self.tokenize, self.remove_stop]
        
    def tokenize(self, text):
        return word_tokenize(text, engine="newmm")

    def remove_stop(self, tokens):
        return [t for t in tokens if (t not in stopwords) & (' ' not in t) & (~t.isdigit()) & (len(t)!=1) & ('…' not in t) & ('ๆ' not in t) & (t not in thai_digits)]

    pipeline = [tokenize, remove_stop]

    """def prepare(text, pipeline):
        tokens = text
        for transform in pipeline:
            tokens = transform(tokens)
        return tokens"""
    
    def split_token( self, data):
        pd.options.mode.chained_assignment = None
        self._datToken = data
        #self._datToken['token'] = self._datToken['content'].assign(self.prepare(pipeline=pipeline))
        self._datToken['token'] = self._datToken['content'].apply(self.tokenize)
        self._datToken['token'] = self._datToken['token'].apply(self.remove_stop)
        self._datToken['token'] = [x for x in self._datToken['token'] if x not in stopwords]
        self._datToken['num_tokens'] = self._datToken['token'].map(len)
        
        return self._datToken
    
class tfidf_vector():
    def __init__( self ):
        self._datVec = pd.DataFrame()
        self._labels = list()
        self.tfidf_table = ''
        self._features = ''
        self._tfidf_vectorizer = ''
    
    def features( self, data ):
        return self.split_to_vector(data)[1]
    
    def pre_vector( self, data ):
        self.featureSelection(data)
        return self._datVec
        
    def split_to_vector( self, data ):
        self._datVec = data
        self._datVec['to_vector'] = self._datVec[['tag','token']].apply(lambda x: ';'.join(x.astype(str)), axis=1)
        return self.to_vector()

    def to_vector( self ):
        count_vec = MultiLabelBinarizer()
        mlb = count_vec.fit(self._datVec["token"])

        text = self._datVec["token"].map(' '.join)
        count_vec = CountVectorizer()
        cv = count_vec.fit(text)
        dt = cv.transform(self._datVec['to_vector'])
        
        tfidf = TfidfTransformer()
        tfidf_dt = tfidf.fit_transform(dt)
        features = tfidf.fit_transform(dt).toarray()
        
        def identity(text):
            return text
        
        tfidf_vectorizer = TfidfVectorizer(analyzer = 'word', #this is default
                                        tokenizer=identity, #does no extra tokenizing
                                        preprocessor=identity, #no extra preprocessor
                                        token_pattern=None)

        tfidf_vector= tfidf_vectorizer.fit_transform(self._datVec['token'])
        tfidf_array = np.array(tfidf_vector.todense())
        
        #print("n_samples: %d, n_features: %d" % tfidf_dt.shape)

        #แปลงเป็น DataFrame เพื่อง่ายแก่การอ่าน
        self.tfidf_table = pd.DataFrame(tfidf_array,columns=tfidf_vectorizer.get_feature_names_out())
        self._features = features
        self._tfidf_vectorizer = tfidf_vectorizer
        
        return self.tfidf_table, self._features
    
    def featureSelection( self, data ):
        
        ans = input('Show correlated features? [y/n] : ')
        
        if ans == 'y':
            while True:
                try : 
                    N = int(input('Enter showing number of correlatedunigram (default = 5) : '))
                    for tag, tag_id in sorted(tag_to_id.items()):
                        features_chi2 = chi2(features, labels == tag_id)
                        indices = np.argsort(features_chi2[0])
                        feature_names = np.array(self._tfidf_vectorizer.get_feature_names_out())[indices]
                        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
                        #bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
                        print("# '{}':".format(tag))
                        print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
                        #print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))
                    break
                except :
                    print('------Try again!------')
            
        else :
            print('Skip!')
            
        #return self._datVec
            
            
            
    

#######################################################
 
def multiLabelChecking(df):
    df = (df.groupby(['content'])
      .agg({'tag': lambda x: ",".join(x)})
      .reset_index())
    df = df[df['tag'].str.contains(',')]
    return df