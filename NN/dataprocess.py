import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self,  columns_to_encode : list, columns_to_drop : list):
        
        self.columns_to_encode = columns_to_encode
        self.columns_to_drop = columns_to_drop 
        self.data = None
    
    def load_data(self, file_path):
        self.data = pd.read_csv(file_path)
        return self
    
    def cleaner(self):
        self.data["Cardiac_hospitalisations"] = self.data["No..cardiac.presentations.to.ED_6months"] + self.data["No..non.cardiac.presentations.to.ED_6months"]           
        self.data = self.data.drop(columns= self.columns_to_drop) 
        self.data = pd.get_dummies(self.data, columns = self.columns_to_encode, drop_first=True)
        return self
    
    def get_data(self):
        return self.data

