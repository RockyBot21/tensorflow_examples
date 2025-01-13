# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 14:56:29 2024

@author: Arturo
"""
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from typing import Any
import tensorflow as tf
import pandas as pd
import numpy as np
import os, json

TF_ENABLE_ONEDNN_OPTS=1

class Data:
    @staticmethod
    def normalize_data(df: pd.DataFrame, target_col: str, exclude_cols: list) -> pd.DataFrame:
        """
        Normalize data of the several columns in dataframe.
            
            * Arguments:
                - df          (pd.DataFrame) : Table where is the data.
                - target_col           (str) : Column name.
                - exclude_cols   (list[str]) : List of columns.

            * Returns:
                - df          (pd.DataFrame) : Table where is the data.
        """
        scaler = MinMaxScaler()
        columns_to_scale = [col for col in df.columns if col not in exclude_cols and col != target_col]
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
        return df

    @staticmethod
    def normalize_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame and Any:
        """
        Normalize data in specific column of dataframe.
            
            * Arguments:
                - df          (pd.DataFrame) : Table where is the data.
                - target_col           (str) : Column name.

            * Returns:
                - df          (pd.DataFrame) : Table where is the data.
                - scaler               (Any) : MinMaxScaler variable.
        """
        scaler = MinMaxScaler()
        df[target_col] = scaler.fit_transform(df[[target_col]])
        return df, scaler

    @staticmethod
    def split_data(data_input: pd.DataFrame, target_col: str) -> Any:
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

class RNN(Model):
    def __init__(self, input_shape, initial_lr=0.0005):
        super(RNN, self).__init__()
        self.dense1 = Dense(256, activation='relu')
        self.dropout1 = Dropout(0.3)
        self.dense2 = Dense(128, activation='relu')
        self.dropout2 = Dropout(0.3)
        self.dense3 = Dense(64, activation='relu')
        self.output_layer = Dense(1, activation='linear')

        self.lr_schedule = ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=1000,
            decay_rate=0.8,
            staircase=True
        )
        self.optimizer = Adam(learning_rate=self.lr_schedule)
        self.compile(
            optimizer=self.optimizer,
            loss='mean_squared_error',
            metrics=['mean_absolute_error', tf.keras.metrics.RootMeanSquaredError()]
        )

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        x = self.dense3(x)
        return self.output_layer(x)

class PricePredict:
    def __init__(self, excel_file_path: str):
        self.excel_file_path = excel_file_path

    def main(self):
        df = pd.read_csv(self.excel_file_path)

        if df.empty:
            raise ValueError("The input data is empty.")

        # Process date column
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
        current_date = datetime.now()
        df['diference_date'] = df['date'].apply(lambda x: (current_date - x).days)

        # Tokenize address data
        df_address = df[['street', 'city', 'statezip', 'country']]
        df_address['Location'] = df['street'] + ' ' + df['city'] + ' ' + df['country'] + ' ' + df['statezip']
        df_address = df_address.drop(columns=['street', 'city', 'statezip', 'country'])

        df_main = df[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view',
                      'condition', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'diference_date']]

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(df_address['Location'])

        # Create a ".txt" file for all words & index (dictionary).
        json_obj = json.dumps(tokenizer.word_index, indent=4)
        
        with open('word_index_dict.json', 'w') as j_file:
            j_file.write(json_obj)

        sequences = tokenizer.texts_to_sequences(df_address['Location'])
        sequences = pad_sequences(sequences)

        # Combine main data and tokenized sequences
        df_combined = pd.concat([df_main, pd.DataFrame(sequences, index=df_main.index)], axis=1)
        df_combined.columns = df_combined.columns.astype(str)

        # Normalize target and features
        df_combined, target_scaler = Data.normalize_target(df_combined, target_col='price')
        df_combined = Data.normalize_data(df_combined, target_col='price', exclude_cols=[str(i) for i in range(sequences.shape[1])])

        # Split data
        X_train, X_test, y_train, y_test = Data.split_data(df_combined, target_col='price')

        # Train the model
        model = RNN(input_shape=X_train.shape[1])
        model.fit(
            X_train, y_train, 
            validation_data=(X_test, y_test), 
            epochs=100, 
            batch_size=8,
            callbacks=[
                # Registro en TensorBoard
                tf.keras.callbacks.TensorBoard(log_dir='./logs'),

                # Guardar el mejor modelo
                tf.keras.callbacks.ModelCheckpoint(
                    filepath='./checkpoints/best_model.keras',
                    save_best_only=True,
                    monitor='val_loss',
                    mode='min',
                    verbose=1
                ),

                # Detener temprano si no mejora
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    verbose=1
                ),

                # Guardar logs del entrenamiento
                CSVLogger('training_log.csv', separator=',', append=False)
            ]
        )

        model.summary()

        # Evaluate the model
        loss = model.evaluate(X_test, y_test)

        # Save model
        os.makedirs('rnn_model', exist_ok=True)
        model.save('rnn_model/price_house_prediction.keras')

        os.makedirs('weights_model', exist_ok=True)
        model.save_weights('weights_model/.weights.h5')

        # Denormalize predictions for final metrics
        predictions = model.predict(X_test)
        predictions = target_scaler.inverse_transform(predictions)
        y_test = target_scaler.inverse_transform(y_test.values.reshape(-1, 1))

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        print(f"MAE: {mae}, RMSE: {rmse}, Loss: {loss}")

if __name__ == "__main__":
    predictor = PricePredict(excel_file_path="data_house_prediction.csv")
    predictor.main()
