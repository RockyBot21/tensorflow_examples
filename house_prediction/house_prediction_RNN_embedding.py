# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 23:44:47 2024

@author: Arturo
"""
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import CSVLogger
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from datetime import datetime
from typing import Any
import tensorflow as tf
import pandas as pd
import numpy as np

TF_ENABLE_ONEDNN_OPTS=1

class Data:
    @staticmethod
    def normalize_data_in_columns(df:pd.DataFrame, target_col:str, exclude_cols:list) -> pd.DataFrame:
        """
        Normalize data of the several columns in dataframe.
            
            * Arguments:
                - df          (pd.DataFrame) : Table where is the data.
                - target_col           (str) : Column name.
                - exclude_cols   (list[str]) : List of columns.

            * Returns:
                - df          (pd.DataFrame) : Table where is the data.
        """
        #scaler = MinMaxScaler()
        scaler = StandardScaler()
        columns_to_scale = [col for col in df.columns if col not in exclude_cols and col != target_col]
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
        return df

    @staticmethod
    def normalize_data_one_col(df:pd.DataFrame, target_col:str) -> pd.DataFrame and Any:
        """
        Normalize data in specific column of dataframe.
            
            * Arguments:
                - df          (pd.DataFrame) : Table where is the data.
                - target_col           (str) : Column name.

            * Returns:
                - df          (pd.DataFrame) : Table where is the data.
                - scaler               (Any) : MinMaxScaler variable.
        """
        #scaler = MinMaxScaler()
        scaler = StandardScaler()
        df[target_col] = scaler.fit_transform(df[[target_col]])
        return df, scaler

    @staticmethod
    def split_data(data_input:pd.DataFrame, target_col:str) -> Any:
        """
        Split data in several data sets (Train & test).
            
            * Arguments:                
                - data_input   (pd.DataFrame) : Table where is the data.
                - target_col            (str) : Column name.
        
            * Returns:
                -                       (Any) : Dataframe train & test.
        """
        X = data_input.drop(columns=[target_col])
        y = data_input[target_col]
        return train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

class RnnEmbedding:
    def __init__(self, embedding_input_shape, numeric_input_shape, vocab_size):
        lr_schedule = ExponentialDecay(
            initial_learning_rate=0.0005,  # Reduced learning rate
            decay_steps=1000,
            decay_rate=0.8,
            staircase=True
        )

        # Define the embedding branch
        embedding_input = tf.keras.layers.Input(shape=embedding_input_shape)
        embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=50)(embedding_input)
        embedding_output = tf.keras.layers.LSTM(128, return_sequences=False)(embedding_layer)
        #embedding_output = tf.keras.layers.Flatten()(embedding_layer)
        #embedding_output = tf.keras.layers.GlobalMaxPooling1D()(embedding_layer)        

        # Define the numeric branch
        numeric_input = tf.keras.layers.Input(shape=numeric_input_shape)
        numeric_dense = tf.keras.layers.Dense(64, activation='relu')(numeric_input)

        # Concatenate both branches
        combined = tf.keras.layers.Concatenate()([embedding_output, numeric_dense])

        # Add dense layers on top
        dense       = tf.keras.layers.Dense(512, activation='relu')(combined)
        batch_norm  = tf.keras.layers.BatchNormalization()(dense)
        dropout     = tf.keras.layers.Dropout(0.1)(batch_norm)

        dense1      = tf.keras.layers.Dense(256, activation='relu')(dropout)
        batch_norm1 = tf.keras.layers.BatchNormalization()(dense1)
        dropout1    = tf.keras.layers.Dropout(0.1)(batch_norm1)

        dense2      = tf.keras.layers.Dense(128, activation='relu')(dropout1)
        output      = tf.keras.layers.Dense(1, activation='linear')(dense2)

        self.model = tf.keras.Model(inputs=[embedding_input, numeric_input], outputs=output)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), #lr_schedule),
            loss='mean_squared_error',
            metrics=['mean_absolute_error', tf.keras.metrics.RootMeanSquaredError()]
        )

    def train(self, X_train_embeddings, X_train_numeric, y_train, epochs, batch_size):
        """ Train the model """
        self.model.fit([X_train_embeddings, X_train_numeric],
                       y_train,
                       validation_split=0.2,
                       epochs     = epochs,
                       batch_size = batch_size,
                       callbacks  = [
                           # Create register in TensorBoard
                           tf.keras.callbacks.TensorBoard(log_dir='./logs'),

                           # Save the best model in path
                           tf.keras.callbacks.ModelCheckpoint(
                               filepath='./checkpoints/best_model.keras',
                               save_best_only=True,
                               monitor='val_loss',
                               mode='min',
                               verbose=1
                           ),

                           # Stop training model if not improbe in specific case
                           tf.keras.callbacks.EarlyStopping(
                               monitor='val_loss',
                               patience=10,
                               verbose=1
                           ),

                           # Save logs of training
                           CSVLogger('training_log.csv', separator=',', append=False)
                       ]
                   )
        
        self.model.summary()
        

    def evaluate(self, X_test_embeddings, X_test_numeric, y_test):
        """ Check accuracy of the model """
        return self.model.evaluate([X_test_embeddings, X_test_numeric], y_test)

class PricePredictWithEmbedding:
    def __init__(self, excel_file_path:str):
        """
        * Attributes:
            excel_file_path  (str) : Path of the input file (Dyrectory).
        """
        self.excel_file_path = excel_file_path

    def main(self):
        df = pd.read_csv(self.excel_file_path)

        if df.empty:
            raise ValueError("The input data is empty.")

        # Add filter for to remove empty or no common values
        df['price'] = df['price'].apply(lambda x: float(f"{x:e}")) 
        df = df[df['price'] != 0]
        
        # Check he segmen of each house (Analysis)
        # Max value
        print(f"* Max price: {max(df['price'].to_list())}")

        # Min value
        print(f"* Min price: {min(df['price'].to_list())}")

        # Average prices
        print(f"* Average prices: {round((sum((df['price'].to_list()))/len((df['price'].to_list()))), 0)}")

        # Add type of segments in each case of houses 
        df['segments'] = df['price'].apply(lambda x: 'low' if (x > 0) and (x < 600000) else 'medium' if (x > 600000) and (x < 1200000) else 'top')

        # Process date column
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
        current_date = datetime.now()
        df['diference_date'] = df['date'].apply(lambda x: (current_date - x).days)

        # Tokenize address data
        df_address = df[['street', 'city', 'statezip', 'country', 'segments']]
        df_address['Location'] = df['street'] + ' ' + df['city'] + ' ' + df['country'] + ' ' + df['statezip'] + ' ' + df['segments']
        df_address = df_address.drop(columns=['street', 'city', 'statezip', 'country', 'segments'])

        df_main = df[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view',
                      'condition', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'diference_date']]

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(df_address['Location'])

        sequences = tokenizer.texts_to_sequences(df_address['Location'])
        sequences = pad_sequences(sequences)

        # Combine main data and tokenized sequences
        df_combined = pd.concat([df_main, pd.DataFrame(sequences, index=df_main.index)], axis=1)
        df_combined.columns = df_combined.columns.astype(str)

        # Normalize target (price) first
        df_combined, target_scaler = Data.normalize_data_one_col(df_combined, target_col='price')
        
        # Normalize features (excluding price and embeddings)
        df_combined = Data.normalize_data_in_columns(
            df_combined, 
            target_col='price', 
            exclude_cols=[str(i) for i in range(sequences.shape[1])] + ['price']
        )

        # Split data
        X_train, X_test, y_train, y_test = Data.split_data(df_combined, target_col='price')

        # Separate embedding inputs and numeric inputs
        num_embedding_cols = sequences.shape[1]
        X_train_embeddings = X_train.iloc[:, -num_embedding_cols:].values
        X_test_embeddings  = X_test.iloc[:, -num_embedding_cols:].values
        X_train_numeric    = X_train.iloc[:, :-num_embedding_cols].values
        X_test_numeric     = X_test.iloc[:, :-num_embedding_cols].values

        # Train the model with embedding
        model = RnnEmbedding(
                     embedding_input_shape = (num_embedding_cols,),
                     numeric_input_shape   = (X_train_numeric.shape[1],),
                     vocab_size            = (len(tokenizer.word_index) + 1)
                )
        
        model.train(X_train_embeddings = X_train_embeddings,
                    X_train_numeric    = X_train_numeric,
                    y_train            = y_train,
                    epochs             = 70,
                    batch_size         = 32)

        # Evaluate the model
        loss = model.evaluate(X_test_embeddings=X_test_embeddings, X_test_numeric=X_test_numeric, y_test=y_test)

        # Denormalize predictions for final metrics
        predictions = model.model.predict([X_test_embeddings, X_test_numeric])
        predictions = target_scaler.inverse_transform(predictions)
        y_test = target_scaler.inverse_transform(y_test.values.reshape(-1, 1))

        print("Predictions before inverse transform:", predictions[:5].flatten())
        print("Actual values before inverse transform:", y_test[:5].flatten())

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        print(f"MAE: {mae}, RMSE: {rmse}, Loss: {loss}")

if __name__ == "__main__":
    predictor = PricePredictWithEmbedding(excel_file_path="data_house_prediction.csv")
    predictor.main()

"""
Execute that command for to see the board of model (Where is the project)
    tensorboard --logdir logs/fit
"""

"""
Open the port
    http://localhost:6006/
"""
