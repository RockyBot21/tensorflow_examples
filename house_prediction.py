# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 08:16:12 2024

@author: Arturo
"""
from sklearn.model_selection import train_test_split
from typing import Any, NoReturn 
import tensorflow as tf
import pandas as pd
import numpy as np
import os

TF_ENABLE_ONEDNN_OPTS=1

class Excel:
    def __init__(self, excel_file_path:pd.DataFrame|None = None):
        """
        * Attributes:
            - excel_file_path (DataFrame|None) : Path excel file (Data input)
        """
        self.excel_file_path = excel_file_path

    def read_file(self) -> pd.DataFrame:
        """
        Read excel file.
        
            * Arguments:
                - None
                
            * Returns:
                -         (Dataframe | None) : Excel file output.
        """
        if os.path.isfile(self.excel_file_path):    return pd.read_csv(filepath_or_buffer=self.excel_file_path)
        else:                                       return None 


class Data:
    @staticmethod
    def split_data(data_input:Any=None) -> Any:
        """
        Split data to train & test.
        
            * Arguments:
                - data_input (Any) : Data input for train and est the model.

            * Returns:
                -            (Any | Dataframe) : Text & train data.
        """        
        X, y = train_test_split(data_input, random_state=42, shuffle=True)
        return X, y


class RNN:
    def __init__(self):
        pass
        
        
class PricePredict(Excel, RNN, Data):
    def __init__(self, excel_file_path:pd.DataFrame|None = None):
        self.excel_file_path:str = excel_file_path

    def main(self) -> NoReturn:
        """
        Execute code make a house prediction.
        
            * Arguments:
                - None
                
            * Returns:
                - None
        """
        df = Excel(excel_file_path = self.excel_file_path).read_file()
        print(f"{df.columns}\n")
        print(f"{df.head}\n")
        print(f"{df.info()}\n")

        # Split data (First time)
        if not df.empty:
            X, y = Data.split_data(data_input = df)
            
            # Remove price column (Column to predict)
            X.pop("price")
                
            X_train, X_test = Data.split_data(data_input = X)
            y_train, y_test = Data.split_data(data_input = y)
    
            # Clear values (Free memory)
            X, y = None, None
            print(True)
        
Obj = PricePredict(excel_file_path="./data_house_prediction.csv")
Obj.main()