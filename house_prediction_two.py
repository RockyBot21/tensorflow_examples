# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 21:00:10 2024

@author: Arturo
"""
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from typing import Any, NoReturn
from datetime import datetime
import tensorflow as tf
import pandas as pd
import numpy as np
import os, json

TF_ENABLE_ONEDNN_OPTS = 1

class Excel:
    def __init__(self, excel_file_path: str | None = None):
        """
        * Attributes:
            - excel_file_path (str | None) : Path to the input file.
        """
        self.excel_file_path = excel_file_path

    def read_file(self) -> pd.DataFrame:
        """
        Read an Excel or CSV file.

        * Returns:
            - (pd.DataFrame): Loaded data.
        """
        if not self.excel_file_path or not os.path.isfile(self.excel_file_path):
            raise FileNotFoundError("File not found or path is invalid.")

        if self.excel_file_path.endswith('.csv'):
            return pd.read_csv(self.excel_file_path)
        elif self.excel_file_path.endswith('.xlsx'):
            return pd.read_excel(self.excel_file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")


class Data:
    @staticmethod
    def split_data(data_input: pd.DataFrame, target_col: str) -> Any:
        """
        Split data into features and target for training and testing.

        * Arguments:
            - data_input (pd.DataFrame): Input data.
            - target_col (str): Name of the target column.

        * Returns:
            - X_train, X_test, y_train, y_test: Splitted data.
        """
        if target_col not in data_input.columns:
            raise ValueError(f"Target column '{target_col}' not found in data.")

        X = data_input.drop(columns=[target_col])
        y = data_input[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
        return X_train, X_test, y_train, y_test
    

class RNN:
    def __init__(self, input_shape):
        """
        Initialize a simple dense neural network.

        * Arguments:
            - input_shape (int): Number of input features.
        """
        
        # Define the learning rate scheduler
        lr_schedule = ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.9,
            staircase=True
        )
        
        self.model = tf.keras.Sequential([
                        tf.keras.Input(shape=(input_shape,)),
                        tf.keras.layers.Dense(128, activation='relu'),
                        tf.keras.layers.Dropout(0.2),
                        tf.keras.layers.Dense(64, activation='relu'),
                        tf.keras.layers.Dropout(0.2),
                        tf.keras.layers.Dense(1, activation='linear')
                ])
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss='mean_squared_error',
            metrics=[
                'mean_absolute_error',
                tf.keras.metrics.RootMeanSquaredError(),
                'accuracy'
            ]
        )

    def train(self, X_train, y_train, epochs, batch_size):
        """
        Train the model.

        * Arguments:
            - X_train: Training features.
            - y_train: Training labels.
            - epochs: Number of epochs.
            - batch_size: Batch size.
        """
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model.

        * Arguments:
            - X_test: Testing features.
            - y_test: Testing labels.

        * Returns:
            - Loss value.
        """
        return self.model.evaluate(X_test, y_test)
    

class PricePredict:
    def __init__(self, excel_file_path: str):
        self.excel_file_path = excel_file_path

    def main(self) -> NoReturn:
        """
        Execute code to make house price predictions.
        """
        #try:
        excel_handler = Excel(self.excel_file_path)
        df = excel_handler.read_file()

        if df.empty:
            raise ValueError("The input data is empty.")

        # Handle the 'date' column
        if 'date' in df.columns:
            #Get current date
            current_date = datetime.now()
            
            # Convert entire column to datetime format
            #df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
            df['diference_date'] = (current_date - df['date'])
            df['diference_date'] = df['diference_date'].apply(lambda x: x.days)
            # Convert to timestamp or extract features
            # df['date'] = pd.to_datetime(df['date'], errors='coerce').astype('int64') / 1e9
        
            # Join columns that are related to country, city, statezip & street
            df_address = df[['street', 'city', 'statezip', 'country']]
            df_address['Location'] =  df['street'] + ' ' + df['city'] + ' ' + df['country'] + ' ' + df['statezip']
            df_address = df_address.drop(columns=['street', 'city', 'statezip', 'country'])
            
            df_main    = df[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view',
                             'condition', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'diference_date']]
            
            # Intanciate the object for to tokenize 
            tokenizer = Tokenizer()
            
            # Create vocabulary
            tokenizer.fit_on_texts(df_address['Location'])
            #print('Get all words has been tokenized before (Dictionary):  {tokenizer.word_index}')
        
            # Create a ".txt" file for all words & index (dictionary).
            json_obj = json.dumps(tokenizer.word_index, indent=4)
            
            with open('word_index_dict.json', 'w') as j_file:
                j_file.write(json_obj)
   
            """
            * Use this method if you want to load json file (Dictionary) is has been tokenizer before
            from tensorflow.keras.preprocessing.text import tokenizer_from_json
            
                - Example:
                    
                    with open('word_index_dict.json', 'r') as j_file:
                        tokenizer_json = j_file.read()

                    tokenizer = tokenizer_from_json(tokenizer_json)
            """
        
            # Convert to sequences
            sequences = tokenizer.texts_to_sequences(df_address['Location'])
            sequences = pad_sequences(sequences)
            
            # Concatenate string text to all dataframe created
            #df_combined = tf.concat([df_main, sequences], axis=1)
            df_combined = pd.concat([df_main, pd.DataFrame(sequences, index=df_main.index)], axis=1)

        print(f"Columns: {df.columns}")
        print(f"First rows:\n{df.head()}")
        print(f"Info:\n{df.info()}")

        # Split data into features and target
        target_col = 'price'
        X_train, X_test, y_train, y_test = Data.split_data(data_input=df_combined, target_col=target_col)

        # Initialize and train the model
        model = RNN(input_shape=X_train.shape[1])
        model.train(X_train=X_train, y_train=y_train, epochs=100, batch_size=8)

        # Evaluate the model
        loss = model.evaluate(X_test=X_test, y_test=y_test)
        print(f"Loss: {loss}")
        print("Execution completed successfully.")

        # Save model with file format ".h5" this method save the entire model
        model.save("price_house_prediction.h5")

        # Save model with only weights.
        model.save_weights('rnn_weights/price_house_prediction.h5')

        #except Exception as e:
        #    print(f'Error: {e}. Check model & predictions something goes wrong.')

if __name__ == "__main__":
    predictor = PricePredict(excel_file_path="./data_house_prediction.csv")
    predictor.main()
    