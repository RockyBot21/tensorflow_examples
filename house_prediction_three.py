# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 22:16:42 2024

@author: Arturo
"""
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import tensorflow as tf
import pandas as pd
import numpy as np

class Data:
    @staticmethod
    def normalize_data(df: pd.DataFrame, target_col: str, exclude_cols: list) -> pd.DataFrame:
        scaler = MinMaxScaler()
        columns_to_scale = [col for col in df.columns if col not in exclude_cols and col != target_col]
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
        return df

    @staticmethod
    def normalize_target(df: pd.DataFrame, target_col: str):
        scaler = MinMaxScaler()
        df[target_col] = scaler.fit_transform(df[[target_col]])
        return df, scaler

    @staticmethod
    def split_data(data_input: pd.DataFrame, target_col: str):
        X = data_input.drop(columns=[target_col])
        y = data_input[target_col]
        return train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

class RNN:
    def __init__(self, input_shape):
        lr_schedule = ExponentialDecay(
            initial_learning_rate=0.0005,  # Reduced learning rate
            decay_steps=1000,
            decay_rate=0.8,
            staircase=True
        )

        self.model = tf.keras.Sequential([
            tf.keras.Input(shape=(input_shape,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            loss='mean_squared_error',
            metrics=['mean_absolute_error', tf.keras.metrics.RootMeanSquaredError()]
        )

    def train(self, X_train, y_train, epochs, batch_size):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

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
        model.train(X_train=X_train, y_train=y_train, epochs=100, batch_size=8)

        # Evaluate the model
        loss = model.evaluate(X_test=X_test, y_test=y_test)

        # Denormalize predictions for final metrics
        predictions = model.model.predict(X_test)
        predictions = target_scaler.inverse_transform(predictions)
        y_test = target_scaler.inverse_transform(y_test.values.reshape(-1, 1))

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        print(f"MAE: {mae}, RMSE: {rmse}, Loss: {loss}")

if __name__ == "__main__":
    predictor = PricePredict(excel_file_path="data_house_prediction.csv")
    predictor.main()
