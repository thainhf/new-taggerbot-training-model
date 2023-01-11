# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer
from pprint import pprint
import pandas as pd
from openpyxl import load_workbook
from pathlib import Path
import re

class readTrain_by_path():
    def __init__(self):
        self._filePath = ''
        self._changeCol = dict()
    
    def to_pandas_tab( self, pathFile, change_name ):
        self._filePath = pathFile
        self._changeCol = change_name
        
        if re.search('.txt$', self._filePath):
            return self.read_txt()
        elif re.search('.xlsx$', self._filePath):
            return self.read_excel()
        
        elif re.search('.csv$', self._filePath):
            return self.read_csv()
        
        else :
            return 'Cannot read file {} unknown file extension'.format(self._filePath)
        
    def rename_column( self, dat ):
        return dat.rename(columns=self._changeCol)
        
    
    def read_excel( self ):
        wb = load_workbook(filename = self._filePath)
        sheet_names = wb.get_sheet_names()
        name = sheet_names[0]
        sheet_ranges = wb[name]
        df = pd.DataFrame(sheet_ranges.values)


        df.columns = df.iloc[0]
        df = df[1:]

        #Rename duplicated columns
        cols=pd.Series(df.columns)

        for dup in cols[cols.duplicated()].unique(): 
            cols[cols[cols == dup].index.values.tolist()] = [dup + '.' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]

        df.columns=cols
        
        return self.rename_column(df)
    
    def read_csv( self ):
        df = pd.read_csv(self._filePath, header=0)
        #print('Reading file {} completed!'.format(self._filePath))
        return self.rename_column(df)
    
    def read_txt( self ):
        df = pd.read_csv(self._filePath, header=0, sep=' ')
        #print('Reading file {} completed!'.format(self._filePath))
        return self.rename_column(df)
    