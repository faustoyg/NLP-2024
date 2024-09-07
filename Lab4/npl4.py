# -*- coding: utf-8 -*-
"""
Análisis de Sentimientos en Reseñas de Películas con RNN, LSTM, GRU, y Naive Bayes
Autor: Carlos Jarrín y Fausto Yugcha (modificado para añadir modelos y solución de errores)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, GRU, Dense, SimpleRNN
from sklearn.naive_bayes import MultinomialNB
from keras.utils.np_utils import to_categorical

# Descargar recursos necesarios de NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Leer los datos
train_data = pd.read_csv('C:/Users/Usuario/Desktop/nlp2/train.tsv', sep='\t')

# Reducir las categorías de sentimiento a tres clases
def simplificar_sentimiento(sentimiento):
    if sentimiento in [0, 1]:
        return 'negative'
    elif sentimiento == 2:
        return 'neutral'
    else:
        return 'positive'

train_data['Simplified_Sentiment'] = train_data['Sentiment'].apply(simplificar_sentimiento)

# Filtrar las tres categorías
train_data = train_data[train_data['Simplified_Sentiment'].isin(['negative', 'neutral', 'positive'])]

# Dividir los datos
X = train_data['Phrase']
y = train_data['Simplified_Sentiment']

# Convertir las etiquetas a categorías numéricas
y = pd.get_dummies(y)

# Tokenización y padding de secuencias
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=100)

# Dividir el conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=13, stratify=y)

# Crear y entrenar un modelo RNN
def modelo_rnn():
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))
    model.add(SimpleRNN(64))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Crear y entrenar un modelo LSTM
def modelo_lstm():
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))
    model.add(LSTM(64))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Crear y entrenar un modelo GRU
def modelo_gru():
    model = Sequential()
    model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))
    model.add(GRU(64))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Entrenar los modelos y evaluar
def entrenar_y_evaluar(modelo, X_train, y_train, X_test, y_test, modelo_nombre):
    print(f"\nEntrenando el modelo {modelo_nombre}...")
    modelo.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
    y_pred = modelo.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test.values, axis=1)
    accuracy = accuracy_score(y_true, y_pred_classes)
    report = classification_report(y_true, y_pred_classes, target_names=['negative', 'neutral', 'positive'])
    print(f"\nAccuracy del modelo {modelo_nombre}: {accuracy:.4f}")
    print(f"\nReporte de clasificación para {modelo_nombre}:\n{report}")
    
    # Matriz de confusión
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['negative', 'neutral', 'positive'],
                yticklabels=['negative', 'neutral', 'positive'])
    plt.title(f'Matriz de Confusión - {modelo_nombre}')
    plt.show()

# Entrenar y evaluar los modelos de RNN, LSTM y GRU
rnn_model = modelo_rnn()
entrenar_y_evaluar(rnn_model, X_train, y_train, X_test, y_test, "RNN")

lstm_model = modelo_lstm()
entrenar_y_evaluar(lstm_model, X_train, y_train, X_test, y_test, "LSTM")

gru_model = modelo_gru()
entrenar_y_evaluar(gru_model, X_train, y_train, X_test, y_test, "GRU")

# Modelo Naive Bayes
vectorizer = CountVectorizer()
X_nb = vectorizer.fit_transform(train_data['Phrase'])
X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(X_nb, train_data['Simplified_Sentiment'], test_size=0.2, random_state=13, stratify=train_data['Simplified_Sentiment'])

naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_nb, y_train_nb)

# Predicciones con Naive Bayes
y_pred_nb = naive_bayes.predict(X_test_nb)
accuracy_nb = accuracy_score(y_test_nb, y_pred_nb)
print(f"\nAccuracy del modelo Naive Bayes: {accuracy_nb:.4f}")
report_nb = classification_report(y_test_nb, y_pred_nb, target_names=['negative', 'neutral', 'positive'])
print(f"\nReporte de clasificación para Naive Bayes:\n{report_nb}")

# Matriz de confusión Naive Bayes
conf_matrix_nb = confusion_matrix(y_test_nb, y_pred_nb)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_nb, annot=True, fmt='d', cmap='Blues', xticklabels=['negative', 'neutral', 'positive'],
            yticklabels=['negative', 'neutral', 'positive'])
plt.title('Matriz de Confusión - Naive Bayes')
plt.show()

