# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 15:56:35 2025

@author: Arturo
"""
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras import regularizers
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from datetime import datetime
from typing import Any, Tuple
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Concatenate, BatchNormalization, Dropout
import pandas as pd
import numpy as np
import os

TF_ENABLE_ONEDNN_OPTS=1

class Data:
    @staticmethod
    def normalize_data_in_columns(df:pd.DataFrame, target_col:str, exclude_cols:list) -> pd.DataFrame:
        """Normalize data of the several columns in dataframe."""
        scaler = StandardScaler()
        columns_to_scale = [col for col in df.columns if col not in exclude_cols and col != target_col]
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
        return df

    @staticmethod
    def normalize_data_one_col(df:pd.DataFrame, target_col:str) -> Tuple[pd.DataFrame, Any]:
        """Normalize data in specific column of dataframe."""
        scaler = StandardScaler()
        df[target_col] = scaler.fit_transform(df[[target_col]])
        return df, scaler

    @staticmethod
    def split_data(data_input:pd.DataFrame, target_col:str) -> Any:
        """Split data in several data sets (Train & test)."""
        X = data_input.drop(columns=[target_col])
        y = data_input[target_col]
        return train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

class RNNEmbeddingModel(Model):
    def __init__(self, embedding_input_shape: Tuple[int], numeric_input_shape: Tuple[int], vocab_size: int):
        super(RNNEmbeddingModel, self).__init__()
        
        # Guardar parámetros
        self.embedding_input_shape = embedding_input_shape
        self.numeric_input_shape = numeric_input_shape
        self.vocab_size = vocab_size
        
        # Capas para la rama de embedding
        self.embedding_layer = Embedding(input_dim=vocab_size, output_dim=50)
        self.lstm_layer = LSTM(128, return_sequences=False, kernel_regularizer=regularizers.l2(0.0005))
        
        # Capas para la rama numérica
        self.numeric_dense = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))
        
        # Capas combinadas
        self.concat_layer = Concatenate()
        self.dense1 = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001))
        self.batch_norm1 = BatchNormalization()
        self.dropout1 = Dropout(0.3)
        
        self.dense2 = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))
        self.batch_norm2 = BatchNormalization()
        self.dropout2 = Dropout(0.2)
        
        self.dense3 = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001))
        self.output_layer = Dense(1, activation='linear')
        
        # Definir inputs
        self.embedding_input = Input(shape=embedding_input_shape)
        self.numeric_input = Input(shape=numeric_input_shape)
        
        # Compilar el modelo
        self.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='mean_squared_error',
            metrics=['mean_absolute_error', tf.keras.metrics.RootMeanSquaredError()]
        )

    def call(self, inputs):
        # Separar inputs
        embedding_input, numeric_input = inputs
        
        # Procesar rama de embedding
        x_emb = self.embedding_layer(embedding_input)
        x_emb = self.lstm_layer(x_emb)
        
        # Procesar rama numérica
        x_num = self.numeric_dense(numeric_input)
        
        # Combinar ramas
        x = self.concat_layer([x_emb, x_num])
        
        # Red fully connected
        x = self.dense1(x)
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        
        x = self.dense2(x)
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        
        x = self.dense3(x)
        return self.output_layer(x)

    def build_graph(self):
        """Método para construir el grafo del modelo explícitamente"""
        return Model(
            inputs=[self.embedding_input, self.numeric_input],
            outputs=self.call([self.embedding_input, self.numeric_input])
        )

class PricePredictWithEmbedding:
    def __init__(self, excel_file_path:str):
        self.excel_file_path = excel_file_path

    def main(self):
        # Cargar y preparar datos
        df = pd.read_csv(self.excel_file_path)
        if df.empty:
            raise ValueError("The input data is empty.")

        # Preprocesamiento
        df['price'] = df['price'].apply(lambda x: float(f"{x:e}")) 
        df = df[df['price'] != 0]
        
        # Análisis inicial
        print(f"* Max price: {max(df['price'].to_list())}")
        print(f"* Min price: {min(df['price'].to_list())}")
        print(f"* Average prices: {round((sum((df['price'].to_list()))/len((df['price'].to_list()))), 0)}")

        # Segmentación
        df['segments'] = df['price'].apply(
            lambda x: 'low' if (x > 0) and (x < 600000) else 'medium' if (x > 600000) and (x < 1200000) else 'top'
        )

        # Procesamiento de fechas
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
        current_date = datetime.now()
        df['diference_date'] = df['date'].apply(lambda x: (current_date - x).days)

        # Tokenización de direcciones
        df_address = df[['street', 'city', 'statezip', 'country', 'segments']]
        df_address['Location'] = df['street'] + ' ' + df['city'] + ' ' + df['country'] + ' ' + df['statezip'] + ' ' + df['segments']
        df_address = df_address.drop(columns=['street', 'city', 'statezip', 'country', 'segments'])

        df_main = df[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view',
                      'condition', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'diference_date']]

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(df_address['Location'])

        sequences = tokenizer.texts_to_sequences(df_address['Location'])
        sequences = pad_sequences(sequences)

        # Combinar datos
        df_combined = pd.concat([df_main, pd.DataFrame(sequences, index=df_main.index)], axis=1)
        df_combined.columns = df_combined.columns.astype(str)

        # Normalización
        df_combined, target_scaler = Data.normalize_data_one_col(df_combined, target_col='price')
        df_combined = Data.normalize_data_in_columns(
            df_combined, 
            target_col='price', 
            exclude_cols=[str(i) for i in range(sequences.shape[1])] + ['price']
        )

        # División de datos
        X_train, X_test, y_train, y_test = Data.split_data(df_combined, target_col='price')

        # Separar inputs
        num_embedding_cols = sequences.shape[1]
        X_train_embeddings = X_train.iloc[:, -num_embedding_cols:].values
        X_test_embeddings = X_test.iloc[:, -num_embedding_cols:].values
        X_train_numeric = X_train.iloc[:, :-num_embedding_cols].values
        X_test_numeric = X_test.iloc[:, :-num_embedding_cols].values

        # Crear y entrenar modelo
        model = RNNEmbeddingModel(
            embedding_input_shape=(num_embedding_cols,),
            numeric_input_shape=(X_train_numeric.shape[1],),
            vocab_size=(len(tokenizer.word_index) + 1)
        )
        
        # Crear directorios necesarios
        os.makedirs('./logs', exist_ok=True)
        os.makedirs('./checkpoints', exist_ok=True)

        # Entrenamiento
        model.fit(
            [X_train_embeddings, X_train_numeric],
            y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=32,
            callbacks=[
                tf.keras.callbacks.TensorBoard(log_dir='./logs'),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath='./checkpoints/best_model.keras',
                    save_best_only=True,
                    monitor='val_loss',
                    mode='min',
                    verbose=1
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    verbose=1
                ),
                CSVLogger('training_log.csv', separator=',', append=False)
            ]
        )

        # Evaluación
        loss = model.evaluate([X_test_embeddings, X_test_numeric], y_test)

        # Predicciones y métricas finales
        predictions = model.predict([X_test_embeddings, X_test_numeric])
        predictions = target_scaler.inverse_transform(predictions)
        y_test = target_scaler.inverse_transform(y_test.values.reshape(-1, 1))

        print("Actual values before inverse transform:", y_test[:5].flatten())
        print("Predictions before inverse transform:", predictions[:5].flatten())

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        print(f"MAE: {mae}, RMSE: {rmse}, Loss: {loss}")

if __name__ == "__main__":
    predictor = PricePredictWithEmbedding(excel_file_path="data_house_prediction.csv")
    predictor.main()
