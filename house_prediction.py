# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 08:16:12 2024

@author: Arturo
"""
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
        
            Arguments:
                - None
                
            Returns:
                -         (Dataframe | None) : Excel file output.
        """
        if os.path.isfile(self.excel_file_path):    return pd.read_csv(filepath_or_buffer=self.excel_file_path)
        else:                                       return None 
        
        
class PricePredict(Excel):
    def __init__(self, excel_file_path:pd.DataFrame|None = None):
        self.excel_file_path:str = excel_file_path

    def main(self) -> None:
        df = Excel(excel_file_path = self.excel_file_path)
        print(df)
        
Obj = PricePredict(excel_file_path="./data_house_prediction.csv")
Obj.main()